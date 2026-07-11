"""Interactive text console for Jarvis.

Type a task in plain English; Jarvis perceives the screen, reasons, and acts.
Special commands start with ':'.
"""

from __future__ import annotations

import sys
import time

from pathlib import Path

from .config import load_config, Config
from .agent.brain import make_brain, BrainError
from .utils import logging as log
from .utils import voice
from .utils.logging import _c, _COLORS
from .agent.loop import Agent
from . import scheduler


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
    voice.configure(brain, cfg.voice)

    # Cron: a background thread fires scheduled jobs through the same agent.
    # Every job run holds the desktop lock so it never fights a foreground task.
    def _cron_runner(command: str) -> None:
        log.rule(f"cron: {command[:50]}", "magenta")
        with scheduler.desktop():
            result = agent.run(command)
        log.jarvis(f"[scheduled] {result}")
        if cfg.voice_enabled:
            try:
                voice.speak(result, wait=True)
            except Exception:
                pass

    sched = scheduler.Scheduler(
        Path(__file__).resolve().parent.parent / "cron_jobs.json",
        runner=_cron_runner)
    scheduler.set_default(sched)
    sched.start()

    log.jarvis("Ready. Tell me what to do. (':help' for commands, ':voice on' "
               "to talk, ':cron' to schedule, ':quit' to exit)")
    prompt = f"\n{_COLORS['bold']}{_COLORS['cyan']}you ›{_COLORS['reset']} "

    # Launched with --voice / voice_enabled: go straight to voice-only mode.
    if cfg.voice_enabled:
        try:
            _voice_loop(agent, cfg)
        except KeyboardInterrupt:
            print()
        cfg.voice_enabled = False
        log.ok("voice mode off - typed prompt. (':voice on' to resume)")

    while True:
        try:
            task = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.startswith(":"):
            c = task[1:].strip().lower()
            if c == "wake":
                try:
                    _wake_loop(agent, cfg)
                except KeyboardInterrupt:
                    print()
                    log.ok("hands-free mode off; back to the prompt.")
                continue
            if c.startswith("voice"):
                # Voice mode is voice-ONLY: entering it replaces the typed
                # prompt until Ctrl+C / "exit voice mode".
                if c.endswith("off"):
                    log.ok("voice is already off (typed prompt).")
                    continue
                cfg.voice_enabled = True
                try:
                    _voice_loop(agent, cfg)
                except KeyboardInterrupt:
                    print()
                cfg.voice_enabled = False
                log.ok("voice mode off - typed prompt. (':voice on' to resume)")
                continue
            if _command(task, cfg):
                break
            continue
        log.rule(task[:60], "blue")
        started = time.time()
        try:
            with scheduler.desktop():      # serialize with any cron job
                result = agent.run(task)
        except KeyboardInterrupt:
            log.warn("interrupted; back to prompt.")
            continue
        except Exception as exc:
            log.error(f"unexpected error: {exc}")
            continue
        log.jarvis(result)
        log.rule(f"done in {time.time() - started:.1f}s")
    sched.stop()
    scheduler.set_default(None)
    log.jarvis("Goodbye.")
    return 0


_VOICE_EXIT_PHRASES = {"exit voice mode", "stop voice mode", "voice off",
                       "stop listening", "goodbye jarvis"}


def _voice_loop(agent: Agent, cfg: Config) -> None:
    """Voice-ONLY mode: no typed prompt - listen, act, speak, repeat.
    Exit with Ctrl+C or by saying one of the exit phrases."""
    log.ok('voice mode ON - just speak. Say "exit voice mode" or press '
           "Ctrl+C to go back to typing.")
    voice.speak("Voice mode on. I am listening.", wait=True)
    while True:
        log.info("listening...")
        wav = voice.listen(start_timeout=30.0)
        if not wav:
            continue                      # silence - keep waiting
        with log.spinner("transcribing"):
            task = voice.transcribe(wav, agent.brain)
        if not task:
            voice.speak("Sorry, I couldn't understand that.", wait=True)
            continue
        log.info(f'heard: "{task}"')
        if task.strip().lower().rstrip(".!,") in _VOICE_EXIT_PHRASES:
            voice.speak("Voice mode off.", wait=True)
            return
        log.rule(task[:60], "blue")
        started = time.time()
        try:
            with scheduler.desktop():     # serialize with any cron job
                result = agent.run(task)
        except KeyboardInterrupt:
            raise                         # exit voice mode entirely
        except Exception as exc:
            log.error(f"unexpected error: {exc}")
            voice.speak("Something went wrong with that task.", wait=True)
            continue
        log.jarvis(result)
        # wait=True: never let Jarvis's own speech bleed into the next listen
        voice.speak(result, wait=True)
        log.rule(f"done in {time.time() - started:.1f}s")


def _wake_loop(agent: Agent, cfg: Config) -> None:
    """Hands-free mode: wait for "hey jarvis", listen, act, speak - repeat.
    Ctrl+C (handled by the caller) exits back to the typed prompt."""
    log.ok('hands-free mode ON - say "Hey Jarvis" to give a command '
           "(Ctrl+C to exit)")
    voice.speak("Hands free mode on. Say hey jarvis when you need me.")
    while True:
        log.info('waiting for "Hey Jarvis"...')
        if not voice.wait_for_wake():
            log.warn("wake-word listener unavailable; leaving hands-free mode.")
            return
        voice.speak("Yes?", wait=True)      # sync so it never bleeds into the mic
        log.info("listening... speak your command")
        wav = voice.listen(start_timeout=8.0)
        if not wav:
            voice.speak("I didn't catch that.")
            continue
        with log.spinner("transcribing"):
            task = voice.transcribe(wav, agent.brain)
        if not task:
            voice.speak("Sorry, I couldn't understand that.")
            continue
        log.info(f'heard: "{task}"')
        log.rule(task[:60], "blue")
        started = time.time()
        try:
            with scheduler.desktop():        # serialize with any cron job
                result = agent.run(task)
        except KeyboardInterrupt:
            raise                            # exit hands-free mode entirely
        except Exception as exc:
            log.error(f"unexpected error: {exc}")
            voice.speak("Something went wrong with that task.")
            continue
        log.jarvis(result)
        voice.speak(result)
        log.rule(f"done in {time.time() - started:.1f}s")


def _command(cmd: str, cfg: Config) -> bool:
    """Handle ':' commands. Returns True if the REPL should exit."""
    c = cmd[1:].strip().lower()
    if c in {"quit", "exit", "q"}:
        return True
    if c in {"help", "h", "?"}:
        log.rule("commands", "cyan")
        for cmd, desc in (
            (":help", "show this help"),
            (":voice on", "voice-ONLY mode: talk instead of typing"),
            (":wake", 'hands-free mode: say "Hey Jarvis" to command'),
            (":cron", "list/add/remove scheduled jobs"),
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
    elif c == "cron" or c.startswith("cron "):
        _cron_command(cmd)
    else:
        log.warn(f"unknown command '{cmd}' (':help' for the list)")
    return False


def _cron_command(raw: str) -> None:
    """Handle ':cron [list | add <schedule> | <command> | remove <id>]'.

    Uses the raw (case-preserving) text so the scheduled command isn't
    lower-cased.
    """
    sched = scheduler.get_default()
    if sched is None:
        log.warn("scheduler is not running.")
        return
    body = raw[1:].strip()                    # drop leading ':'
    body = body[4:].strip() if body[:4].lower() == "cron" else body

    if not body or body.lower() == "list":
        jobs = sched.jobs()
        if not jobs:
            log.info("no scheduled jobs. Add one, e.g. "
                     ":cron add every 30 minutes | tell me the system status")
            return
        log.rule("cron jobs", "magenta")
        for j in jobs:
            print("  " + j.describe())
        return

    low = body.lower()
    if low.startswith(("remove", "rm", "del")):
        try:
            jid = int(body.split()[1])
        except (IndexError, ValueError):
            log.warn("usage: :cron remove <id>")
            return
        log.ok(f"removed job {jid}" if sched.remove(jid) else f"no job with id {jid}")
        return

    if low.startswith("add"):
        rest = body[3:].strip()
        if "|" not in rest:
            log.warn("usage: :cron add <schedule> | <command>   e.g. "
                     ":cron add daily at 08:00 | search the web for the news")
            return
        spec, command = rest.split("|", 1)
        try:
            job = sched.add(spec.strip(), command.strip())
        except scheduler.ScheduleError as exc:
            log.warn(str(exc))
            return
        log.ok(f"scheduled job {job.id}: {job.spec} -> {job.command!r}")
        return

    log.warn("usage: :cron [list | add <schedule> | <command> | remove <id>]")


if __name__ == "__main__":
    sys.exit(repl())
