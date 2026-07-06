"""System prompt construction and robust parsing of the model's action reply.

The contract with the model is deliberately tiny and strict so that even a
1.5-3B model can follow it reliably: reply with exactly ONE JSON object,
nothing else.

    {"thought": "...", "action": "<name>", "args": {...}}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..tools.schema import ACTIONS, ACTIONS_BY_NAME


@dataclass
class Decision:
    thought: str
    action: str
    args: dict[str, Any]
    raw: str = ""
    # True when the parser could not extract a real action and synthesized this
    # decision (prose reply or unknown action). The loop must NOT treat such a
    # "finish" as task success - that would reward failures (fake RL signal).
    fallback: bool = False


def build_system_prompt(memory: str = "") -> str:
    prompt = _SYSTEM_HEADER
    if memory.strip():
        prompt += f"\n\n=== PERSISTENT MEMORY (persists across runs) ===\n{memory.strip()}\n================================================="
    else:
        prompt += f"\n\n=== PERSISTENT MEMORY (persists across runs) ===\n(No memory recorded yet. Use write_file on 'memory.txt' to store facts/preferences.)\n================================================="
    return prompt + "\n\n" + _action_reference() + "\n\n" + _RULES


def format_observation(active_window: str, screen_size: tuple[int, int],
                       menu: str) -> str:
    """The exact user-turn text that presents the screen to the model.

    Shared by the live loop and the dataset builder so training data matches
    inference input byte-for-byte.
    """
    return (f"ACTIVE WINDOW: {active_window or '(desktop)'}\n"
            f"SCREEN: {screen_size[0]}x{screen_size[1]}\n"
            f"ELEMENTS (each line shows [id] Role \"Label\" @ (center_x,center_y) — "
            f"use the id to click precisely):\n{menu}\n\n"
            f"IMPORTANT: To click any item above, use its element id "
            f"(e.g. {{\"action\":\"click\",\"args\":{{\"element\":0}}}}).\n"
            f"Reply with one action as JSON.")


def format_decision(thought: str, action: str, args: dict) -> str:
    """The exact assistant-turn text (a single JSON action object)."""
    return json.dumps({"thought": thought, "action": action, "args": args},
                      ensure_ascii=False)


_SYSTEM_HEADER = """\
You are JARVIS, an autonomous desktop assistant that operates a Windows computer \
on the user's behalf. You work in a loop: you are shown what is currently on the \
screen, you think one step, and you output ONE action. After the action runs you \
are shown the new screen state, and you continue until the task is done.

Each turn you receive:
  * the user's task,
  * the active window title,
  * a numbered list of interactable elements currently on screen, each with its
    role, label and centre coordinate (these coordinates are exact pixel positions),
  * optionally a screenshot image of the current screen.

You reply with a SINGLE JSON object and nothing else:
  {"thought": "<one sentence of reasoning>", "action": "<name>", "args": {<args>}}

CRITICAL CLICKING RULES:
  * ALWAYS click by element id: {"action":"click","args":{"element":3}}
  * The element list gives you EXACT pixel-perfect click targets. Each element's \
@ (x,y) coordinate is its precise centre on screen.
  * NEVER guess raw x,y coordinates from the screenshot image. Your visual coordinate \
estimates are NOT accurate enough. The element list coordinates are always correct.
  * Use raw x,y ONLY as an absolute last resort when there are zero elements on \
screen AND the target is clearly visible in the screenshot.
  * If the screenshot is provided, use it ONLY to understand context and identify \
WHICH element to click — then click that element by its id, not by guessed coordinates."""


def _action_reference() -> str:
    lines = ["Available actions:"]
    cat = None
    for a in ACTIONS:
        if a.category != cat:
            cat = a.category
            lines.append(f"\n# {cat}")
        params = ", ".join(
            f"{p.name}" + ("" if p.required else "?") for p in a.params
        )
        sig = f"{a.name}({params})"
        lines.append(f"  {sig:<34} {a.summary}")
    return "\n".join(lines)


_RULES = """\
Rules:
  1. Output ONLY the JSON object. No markdown, no code fences, no extra text.
  2. One action per turn. Do the smallest useful next step.
  3. After an action changes the screen, the new state is shown automatically -
     you do not need to call observe unless you deliberately waited.
  4. If a needed control is off-screen, scroll to find it.
  5. To open a website, use open_url with the full URL (e.g.
     {"action":"open_url","args":{"url":"https://www.youtube.com"}}).
     NEVER click the browser address bar and type a URL by hand.
  6. After clicking a text box or search field, your NEXT action must be type,
     and then press enter to submit.
  7. Never repeat an action whose RESULT said the screen did not change. If the
     same action failed twice, it will fail forever - pick a different element,
     scroll, or take a different approach.
  8. When the whole task is complete, call finish with a short summary.
  9. If you are genuinely stuck or the request is ambiguous, call ask.
  10. Never invent element ids that are not in the list.
  11. ALWAYS use element ids to click. NEVER guess x,y from the screenshot.
      The element list coordinates are pixel-perfect; your visual estimates are not.
  12. PERSISTENT MEMORY: You have a persistent memory stored in "memory.txt".
     - Its current contents are automatically shown above in the system prompt.
     - You can update this memory at any time by calling write_file(path="memory.txt", content="<updated memory>").
     - Use this memory to store key facts, user preferences, API keys, paths, or tips you want to remember across different runs. Keep it concise.

Example reply:
  {"thought": "The Save dialog is open; I'll type the filename.", "action": "type", "args": {"text": "report.txt"}}"""


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def parse_decision(text: str) -> Decision:
    """Extract a :class:`Decision` from a raw model reply, tolerantly."""
    obj = _extract_json(text)
    if obj is None:
        # Model produced prose instead of JSON. Mark as fallback so the loop
        # can push back ("reply with one JSON action") instead of treating the
        # prose as a successful finish.
        return Decision(thought="", action="finish",
                        args={"summary": text.strip()[:400]}, raw=text,
                        fallback=True)

    action = str(obj.get("action", "")).strip()
    args = obj.get("args", {})
    if not isinstance(args, dict):
        args = {}
    thought = str(obj.get("thought", "")).strip()

    # Some models nest coordinates or use synonyms; normalise a few.
    action, args = _normalise(action, args)

    if action not in ACTIONS_BY_NAME:
        # Unknown/hallucinated action: also a fallback, not a real finish.
        return Decision(thought=thought, action="finish",
                        args={"summary": thought or text.strip()[:400]}, raw=text,
                        fallback=True)
    return Decision(thought=thought, action=action, args=args, raw=text)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    # Strip code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                  flags=re.MULTILINE).strip()
    # Fast path.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Find the first balanced { ... } block.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


_SYNONYMS = {
    "left_click": "click", "leftclick": "click", "tap": "click",
    "doubleclick": "double_click", "double": "double_click",
    "tripleclick": "triple_click", "triple": "triple_click",
    "rightclick": "right_click",
    "type_text": "type", "write": "type", "keypress": "press",
    "hotkey": "press", "key": "press", "presskey": "press",
    "keys": "key_sequence", "key_seq": "key_sequence", "keyseq": "key_sequence",
    "launch": "open_app", "open": "open_app",
    "goto": "open_url", "navigate": "open_url",
    "shell": "run_command", "cmd": "run_command", "exec": "run_command",
    "done": "finish", "complete": "finish", "stop": "finish",
    "question": "ask",
}


def _normalise(action: str, args: dict) -> tuple[str, dict]:
    a = action.strip().lower()
    a = _SYNONYMS.get(a, a)

    # coordinate objects like {"coordinate": [x, y]} or {"position": {...}}
    if "coordinate" in args and isinstance(args["coordinate"], (list, tuple)):
        args["x"], args["y"] = args["coordinate"][0], args["coordinate"][1]
    if "key" in args and "keys" not in args:
        args["keys"] = args.pop("key")
    if a == "type" and "text" not in args and "value" in args:
        args["text"] = args.pop("value")
    # multi-click given as {"clicks": n} or {"times": n} -> count
    if a == "click":
        if "clicks" in args and "count" not in args:
            args["count"] = args.pop("clicks")
        if "times" in args and "count" not in args:
            args["count"] = args.pop("times")
    # a press whose keys is actually a list -> key_sequence
    if a == "press" and isinstance(args.get("keys"), list):
        a = "key_sequence"
    return a, args
