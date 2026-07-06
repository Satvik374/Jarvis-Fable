"""Interactive text console for Jarvis.

Type a task in plain English; Jarvis perceives the screen, reasons, and acts.
Special commands start with ':'.
"""

from __future__ import annotations

import sys
import time

from .config import load_config, Config
from .agent.brain import make_brain, BrainError
from .utils import logging as log
from .utils.logging import _c, _COLORS
from .agent.loop import Agent


BANNER = r"""
     _   _   ___     _____ ___
  _ | | /_\ | _ \ \ / /_ _/ __|   local agentic desktop assistant
 | || |/ _ \|   /\ V / | |\__ \   perceive - think - act
  \__/ /_/ \_\_|_\ \_/ |___|___/
"""


def _preflight(cfg: Config) -> None:
    """Warn early about the most common setup gaps, with fixes."""
    if cfg.brain.backend == "ollama":
        try:
            import requests  # type: ignore

            requests.get(f"{cfg.brain.base_url}/api/tags", timeout=3)
        except Exception:
            log.warn(f"Ollama not reachable at {cfg.brain.base_url}.")
            log.warn("  1) install: https://ollama.com/download")
            log.warn(f"  2) pull the model:  ollama pull {cfg.brain.model}")
            log.warn("  3) it serves automatically; then restart Jarvis.")


def _status_bar(cfg: Config) -> str:
    def dot(on: bool) -> str:
        return _c("on", "green") if on else _c("off", "grey")
    return (f"  {_c('brain', 'grey')} {_c(cfg.brain.backend, 'cyan')}"
            f":{cfg.brain.model}   "
            f"{_c('vision', 'grey')} {dot(cfg.brain.use_vision)}   "
            f"{_c('UIA', 'grey')} {dot(cfg.perception.use_uia)}   "
            f"{_c('OCR', 'grey')} {dot(cfg.perception.use_ocr)}   "
            f"{_c('steps', 'grey')} {cfg.safety.max_steps}")


def repl(cfg: Config | None = None) -> int:
    cfg = cfg or load_config()
    print(_c(BANNER, "cyan"))
    print(_status_bar(cfg))
    log.rule()
    _preflight(cfg)

    try:
        brain = make_brain(cfg.brain)
    except BrainError as exc:
        log.error(str(exc))
        return 1
    agent = Agent(brain, cfg)

    log.jarvis("Ready. Tell me what to do. (':help' for commands, ':quit' to exit)")
    prompt = f"\n{_COLORS['bold']}{_COLORS['cyan']}you ›{_COLORS['reset']} "
    while True:
        try:
            task = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.startswith(":"):
            if _command(task, cfg):
                break
            continue
        log.rule(task[:60], "blue")
        started = time.time()
        try:
            result = agent.run(task)
        except KeyboardInterrupt:
            log.warn("interrupted; back to prompt.")
            continue
        except Exception as exc:
            log.error(f"unexpected error: {exc}")
            continue
        log.jarvis(result)
        log.rule(f"done in {time.time() - started:.1f}s")
    log.jarvis("Goodbye.")
    return 0


def _command(cmd: str, cfg: Config) -> bool:
    """Handle ':' commands. Returns True if the REPL should exit."""
    c = cmd[1:].strip().lower()
    if c in {"quit", "exit", "q"}:
        return True
    if c in {"help", "h", "?"}:
        log.rule("commands", "cyan")
        for cmd, desc in (
            (":help", "show this help"),
            (":confirm on|off", "toggle per-action confirmation"),
            (":vision on|off", "toggle sending screenshots to the model"),
            (":steps N", "set the max steps per task"),
            (":config", "print the active configuration"),
            (":quit", "exit"),
        ):
            print(f"  {_c(f'{cmd:<16}', 'cyan')} {_c(desc, 'grey')}")
        log.rule()
    elif c.startswith("confirm"):
        cfg.safety.confirm_each_action = c.endswith("on")
        log.ok(f"confirm_each_action = {cfg.safety.confirm_each_action}")
    elif c.startswith("vision"):
        cfg.brain.use_vision = c.endswith("on")
        log.ok(f"use_vision = {cfg.brain.use_vision}")
    elif c.startswith("steps"):
        try:
            cfg.safety.max_steps = int(c.split()[1])
            log.ok(f"max_steps = {cfg.safety.max_steps}")
        except (IndexError, ValueError):
            log.warn("usage: :steps 20")
    elif c == "config":
        import json
        print(json.dumps(cfg.as_dict(), indent=2, default=str))
    else:
        log.warn(f"unknown command '{cmd}' (':help' for the list)")
    return False


if __name__ == "__main__":
    sys.exit(repl())
