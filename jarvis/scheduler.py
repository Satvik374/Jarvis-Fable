"""Cron-style scheduled jobs for Jarvis.

A background thread fires jobs when they come due and runs each one through the
same agent the user drives. Because a job moves the real mouse/keyboard, every
desktop-controlling run - foreground OR scheduled - is serialized through one
process-wide lock (:func:`desktop`), so a cron job can never collide with a
task the user is actively running.

Schedule grammar (kept deliberately small - not full cron syntax):

    every <N> <seconds|minutes|hours|days>   -> recurring interval
    daily at <HH:MM>                          -> recurring, once a day
    in <N> <minutes|hours>                    -> one-shot, N from now
    at <HH:MM>                                 -> one-shot, next HH:MM (24h)

# ponytail: no cron-expression parser - add `croniter` only if someone actually
# needs '*/5 * * * *'. The four forms above cover reminders + recurring checks.
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from .utils import logging as log

_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


class ScheduleError(ValueError):
    """Raised when a schedule spec can't be understood."""


def parse_schedule(spec: str) -> tuple[str, int | None, int | None, int | None]:
    """Parse a spec into ``(kind, interval_seconds, hh, mm)``.

    ``kind`` is one of 'interval', 'daily', 'once_in', 'once_at'. Raises
    :class:`ScheduleError` on anything it doesn't recognise.
    """
    s = " ".join((spec or "").strip().lower().split())

    m = re.fullmatch(r"every\s+(\d+)\s*([a-z]+)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit not in _UNITS or n <= 0:
            raise ScheduleError(f"bad interval: {spec!r}")
        return ("interval", n * _UNITS[unit], None, None)

    m = re.fullmatch(r"in\s+(\d+)\s*([a-z]+)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit not in _UNITS or n <= 0:
            raise ScheduleError(f"bad delay: {spec!r}")
        return ("once_in", n * _UNITS[unit], None, None)

    m = re.fullmatch(r"daily(?:\s+at)?\s+(\d{1,2}):(\d{2})", s)
    if m:
        return ("daily", None, *_hhmm(m, spec))

    m = re.fullmatch(r"(?:at\s+)?(\d{1,2}):(\d{2})", s)
    if m:
        return ("once_at", None, *_hhmm(m, spec))

    raise ScheduleError(
        f"can't parse schedule {spec!r}; try 'every 30 minutes', "
        "'daily at 08:00', 'in 10 minutes', or 'at 14:30'")


def _hhmm(m: re.Match, spec: str) -> tuple[int, int]:
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ScheduleError(f"bad time in {spec!r} (use 24h HH:MM)")
    return hh, mm


def next_run(parsed: tuple, now: float, base: float | None = None) -> float:
    """Next fire time (epoch seconds) strictly appropriate for ``parsed``.

    ``base`` seeds the interval forms (defaults to ``now``); it lets the
    scheduler reschedule from the previous fire time if it wants.
    """
    kind, interval, hh, mm = parsed
    if kind in ("interval", "once_in"):
        return (now if base is None else base) + interval
    # daily / once_at: the next occurrence of HH:MM (today if still ahead).
    dt = datetime.fromtimestamp(now).replace(
        hour=hh, minute=mm, second=0, microsecond=0)
    if dt.timestamp() <= now:
        dt += timedelta(days=1)
    return dt.timestamp()


def is_recurring(spec: str) -> bool:
    return parse_schedule(spec)[0] in ("interval", "daily")


@dataclass
class Job:
    id: int
    spec: str          # normalized schedule text
    command: str       # what to run (natural language, via agent.run)
    next_run: float    # epoch seconds
    enabled: bool = True

    def describe(self, now: float | None = None) -> str:
        now = now if now is not None else time.time()
        when = datetime.fromtimestamp(self.next_run).strftime("%Y-%m-%d %H:%M")
        state = "" if self.enabled else " (disabled)"
        return f"[{self.id}] {self.spec} -> {self.command!r}  next: {when}{state}"


# --------------------------------------------------------------------------- #
# One process-wide lock: no scheduled job runs the desktop while the user is.
# --------------------------------------------------------------------------- #
_DESKTOP = threading.RLock()


@contextmanager
def desktop():
    """Hold this around any run that controls the mouse/keyboard (foreground
    or scheduled) so the two never overlap."""
    with _DESKTOP:
        yield


class Scheduler:
    def __init__(self, path, runner, tick: float = 15.0, clock=time.time):
        self.path = Path(path)
        self.runner = runner            # callable(command:str) -> None
        self.tick = tick
        self.clock = clock              # injectable for tests
        self._jobs: list[Job] = self._load()
        self._next_id = 1 + max((j.id for j in self._jobs), default=0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- management ---------------------------------------------------- #
    def add(self, spec: str, command: str) -> Job:
        parsed = parse_schedule(spec)                    # raises ScheduleError
        job = Job(self._next_id, " ".join(spec.strip().lower().split()),
                  command.strip(), next_run(parsed, self.clock()))
        self._next_id += 1
        self._jobs.append(job)
        self._save()
        return job

    def remove(self, job_id: int) -> bool:
        before = len(self._jobs)
        self._jobs = [j for j in self._jobs if j.id != job_id]
        if len(self._jobs) != before:
            self._save()
            return True
        return False

    def jobs(self) -> list[Job]:
        return list(self._jobs)

    # -- firing -------------------------------------------------------- #
    def run_due(self, now: float | None = None) -> list[Job]:
        now = now if now is not None else self.clock()
        fired = [j for j in self._jobs if j.enabled and j.next_run <= now]
        for job in fired:
            try:
                self.runner(job.command)
            except Exception as exc:                     # a bad job can't kill the loop
                log.warn(f"scheduled job {job.id} failed: {exc}")
            if is_recurring(job.spec):
                job.next_run = next_run(parse_schedule(job.spec), self.clock())
            else:
                job.enabled = False
        if fired:
            self._save()
        return fired

    # -- background thread --------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="jarvis-cron")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.tick):
            try:
                self.run_due()
            except Exception as exc:
                log.warn(f"scheduler tick error: {exc}")

    def stop(self) -> None:
        self._stop.set()

    # -- persistence --------------------------------------------------- #
    def _load(self) -> list[Job]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warn(f"could not read cron jobs: {exc}")
            return []
        out: list[Job] = []
        for d in data if isinstance(data, list) else []:
            try:
                job = Job(**d)
                parse_schedule(job.spec)                 # drop unparseable
                out.append(job)
            except Exception:
                continue
        return out

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps([asdict(j) for j in self._jobs], indent=2),
                encoding="utf-8")
        except Exception as exc:
            log.warn(f"could not save cron jobs: {exc}")


# A single active scheduler the agent action can reach without threading a
# handle through every call. Set by the console at startup.
_DEFAULT: Scheduler | None = None


def set_default(sched: Scheduler | None) -> None:
    global _DEFAULT
    _DEFAULT = sched


def get_default() -> Scheduler | None:
    return _DEFAULT
