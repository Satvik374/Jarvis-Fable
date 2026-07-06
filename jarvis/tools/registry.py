"""Execute a parsed action against the live desktop.

The agentic loop hands us ``(action_name, args, observation)`` and we turn it
into real mouse/keyboard/OS effects, returning a human-readable result string
that gets fed back to the model as the outcome of its action.

Pointer targets are resolved here: an ``element`` id is looked up in the current
observation to get an exact centre pixel; otherwise raw ``x``/``y`` are used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import mouse, keyboard, apps, files, system
from .schema import ACTIONS_BY_NAME
from ..config import Config
from ..perception.elements import Observation


@dataclass
class ActionResult:
    ok: bool
    message: str
    # ``needs_observe`` tells the loop the screen likely changed.
    needs_observe: bool = True
    # Set for terminal actions (finish / ask).
    finished: bool = False
    ask: str | None = None


class UnknownAction(Exception):
    pass


def execute(name: str, args: dict[str, Any], obs: Observation,
            cfg: Config) -> ActionResult:
    if name not in ACTIONS_BY_NAME:
        raise UnknownAction(name)
    args = args or {}
    if cfg.brain.backend in {"gemini", "vertex"} and cfg.brain.use_vision:
        args = _denormalize_coords(args, obs)
    handler = _HANDLERS.get(name)
    if handler is None:  # pragma: no cover - schema/registry mismatch guard
        raise UnknownAction(name)
    return handler(args, obs, cfg)


def _denormalize_coords(args: dict, obs: Observation) -> dict:
    """Gemini vision emits coordinates normalized to 0-1000 (its trained
    convention, reinforced by our system instruction). Convert every raw
    coordinate pair back to real screen pixels. Values above 1000 are already
    pixels and pass through untouched.
    """
    sw, sh = obs.screen_size
    if sw <= 1000 and sh <= 1000:      # tiny screen: spaces are ambiguous
        return args
    out = dict(args)
    for xk, yk in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
        if out.get(xk) is None or out.get(yk) is None:
            continue
        try:
            x, y = float(out[xk]), float(out[yk])
        except (TypeError, ValueError):
            continue
        if x <= 1000 and y <= 1000:
            out[xk] = round(x * sw / 1000)
            out[yk] = round(y * sh / 1000)
    return out


# --------------------------------------------------------------------------- #
# target resolution
# --------------------------------------------------------------------------- #

def _resolve_point(args: dict, obs: Observation,
                   el_key: str = "element", x_key: str = "x",
                   y_key: str = "y") -> tuple[int, int] | None:
    if args.get(el_key) is not None:
        el = obs.by_id(int(args[el_key]))
        if el is not None:
            return el.center
        return None
    if args.get(x_key) is not None and args.get(y_key) is not None:
        raw_x, raw_y = int(args[x_key]), int(args[y_key])
        # Snap to the nearest element if one is within 50px — the model's
        # raw coordinate guesses from vision are often slightly off, but
        # element centres from UIA are pixel-perfect.
        best_el = None
        best_dist = 50  # snap radius in pixels
        for el in obs.elements:
            cx, cy = el.center
            dist = ((cx - raw_x) ** 2 + (cy - raw_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_el = el
        if best_el is not None:
            return best_el.center
        return raw_x, raw_y
    return None


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #

def _h_click(args, obs, cfg):
    pt = _resolve_point(args, obs)
    if pt is None:
        return ActionResult(False, "click needs a valid element id or x,y")
    count = max(1, int(args.get("count", 1)))
    return ActionResult(True, mouse.click(*pt, clicks=count))


def _h_double_click(args, obs, cfg):
    pt = _resolve_point(args, obs)
    if pt is None:
        return ActionResult(False, "double_click needs a valid element id or x,y")
    return ActionResult(True, mouse.double_click(*pt))


def _h_triple_click(args, obs, cfg):
    pt = _resolve_point(args, obs)
    if pt is None:
        return ActionResult(False, "triple_click needs a valid element id or x,y")
    return ActionResult(True, mouse.triple_click(*pt))


def _h_right_click(args, obs, cfg):
    pt = _resolve_point(args, obs)
    if pt is None:
        return ActionResult(False, "right_click needs a valid element id or x,y")
    return ActionResult(True, mouse.right_click(*pt))


def _h_move(args, obs, cfg):
    if args.get("x") is None or args.get("y") is None:
        return ActionResult(False, "move needs x and y")
    return ActionResult(True, mouse.move(int(args["x"]), int(args["y"])),
                        needs_observe=False)


def _h_drag(args, obs, cfg):
    src = _resolve_point(args, obs, "from_element", "x1", "y1")
    dst = _resolve_point(args, obs, "to_element", "x2", "y2")
    if src is None or dst is None:
        return ActionResult(False, "drag needs valid source and target points")
    return ActionResult(True, mouse.drag(*src, *dst))


def _h_scroll(args, obs, cfg):
    return ActionResult(True, mouse.scroll(int(args.get("dy", 3)),
                                           int(args.get("dx", 0))))


def _h_type(args, obs, cfg):
    text = str(args.get("text", ""))
    if not text:
        return ActionResult(False, "type needs text")
    return ActionResult(True, keyboard.type_text(text))


def _h_press(args, obs, cfg):
    keys = str(args.get("keys", ""))
    if not keys:
        return ActionResult(False, "press needs keys")
    return ActionResult(True, keyboard.press(keys))


def _h_key_sequence(args, obs, cfg):
    keys = args.get("keys", [])
    if not keys:
        return ActionResult(False, "key_sequence needs a non-empty 'keys' list")
    return ActionResult(True, keyboard.press_sequence(keys))


def _h_open_app(args, obs, cfg):
    name = str(args.get("name", ""))
    if not name:
        return ActionResult(False, "open_app needs a name")
    return ActionResult(True, apps.open_app(name))


def _h_focus_window(args, obs, cfg):
    return ActionResult(True, apps.focus_window(str(args.get("title", ""))))


def _h_open_url(args, obs, cfg):
    return ActionResult(True, system.open_url(str(args.get("url", ""))))


def _h_run_command(args, obs, cfg):
    return ActionResult(
        True,
        system.run_command(str(args.get("command", "")),
                           blocked=cfg.safety.blocked_command_patterns),
        needs_observe=False,
    )


def _h_read_file(args, obs, cfg):
    path = str(args.get("path", ""))
    from pathlib import Path
    if Path(path).name == "memory.txt":
        proj_root = Path(__file__).resolve().parent.parent.parent
        path = str(proj_root / "memory.txt")
    return ActionResult(True, files.read_file(path),
                        needs_observe=False)


def _h_write_file(args, obs, cfg):
    path = str(args.get("path", ""))
    from pathlib import Path
    if Path(path).name == "memory.txt":
        proj_root = Path(__file__).resolve().parent.parent.parent
        path = str(proj_root / "memory.txt")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")), encoding="utf-8")
            return ActionResult(True, f"wrote {len(args.get('content', ''))} chars to memory.txt",
                                needs_observe=False)
        except Exception as exc:
            return ActionResult(False, f"could not write memory: {exc}",
                                needs_observe=False)
    return ActionResult(
        True,
        files.write_file(path, str(args.get("content", "")),
                         allow=cfg.safety.allow_paths),
        needs_observe=False,
    )


def _h_list_dir(args, obs, cfg):
    return ActionResult(True, files.list_dir(str(args.get("path", "."))),
                        needs_observe=False)


def _h_make_dir(args, obs, cfg):
    return ActionResult(
        True,
        files.make_dir(str(args.get("path", "")), allow=cfg.safety.allow_paths),
        needs_observe=False,
    )


def _h_clipboard_read(args, obs, cfg):
    return ActionResult(True, "clipboard: " + system.clipboard_read(),
                        needs_observe=False)


def _h_clipboard_write(args, obs, cfg):
    return ActionResult(True, system.clipboard_write(str(args.get("text", ""))),
                        needs_observe=False)


def _h_wait(args, obs, cfg):
    secs = max(0.0, min(10.0, float(args.get("seconds", 1.0))))
    time.sleep(secs)
    return ActionResult(True, f"waited {secs}s")


def _h_observe(args, obs, cfg):
    return ActionResult(True, "re-reading the screen", needs_observe=True)


def _h_finish(args, obs, cfg):
    return ActionResult(True, str(args.get("summary", "done")),
                        needs_observe=False, finished=True)


def _h_ask(args, obs, cfg):
    q = str(args.get("question", "Could you clarify?"))
    return ActionResult(True, q, needs_observe=False, finished=True, ask=q)


_HANDLERS = {
    "click": _h_click,
    "double_click": _h_double_click,
    "triple_click": _h_triple_click,
    "right_click": _h_right_click,
    "move": _h_move,
    "drag": _h_drag,
    "scroll": _h_scroll,
    "type": _h_type,
    "press": _h_press,
    "key_sequence": _h_key_sequence,
    "open_app": _h_open_app,
    "focus_window": _h_focus_window,
    "open_url": _h_open_url,
    "run_command": _h_run_command,
    "read_file": _h_read_file,
    "write_file": _h_write_file,
    "make_dir": _h_make_dir,
    "list_dir": _h_list_dir,
    "clipboard_read": _h_clipboard_read,
    "clipboard_write": _h_clipboard_write,
    "wait": _h_wait,
    "observe": _h_observe,
    "finish": _h_finish,
    "ask": _h_ask,
}
