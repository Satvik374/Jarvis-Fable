"""Tiny colourised console logger shared across the app.

Kept dependency-free (no ``rich``) so Jarvis stays light on an 8 GB machine.
"""

from __future__ import annotations

import shutil
import sys
import time

_COLORS = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "red": "\033[31m", "magenta": "\033[35m", "blue": "\033[34m",
    "grey": "\033[90m",
}
# ASCII fallback: only true when ANSI setup failed (legacy conhost) - then
# unicode box chars won't render either, so use plain dashes.
_ASCII = False

# Force UTF-8 output where possible so ANSI + any unicode never crash the app
# on a legacy (cp1252) Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Enable ANSI on Windows terminals that need it.
if sys.platform == "win32":  # pragma: no cover - platform specific
    try:
        import colorama  # type: ignore

        colorama.just_fix_windows_console()
    except Exception:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            for k in _COLORS:
                _COLORS[k] = ""
            _ASCII = True


def _c(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _width() -> int:
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def rule(label: str = "", color: str = "grey") -> None:
    """A full-width horizontal divider, optionally with a centred label."""
    ch = "-" if _ASCII else "─"
    w = _width()
    if label:
        tag = f" {label} "
        side = max(0, (w - len(tag)) // 2)
        line = ch * side + tag + ch * max(0, w - side - len(tag))
    else:
        line = ch * w
    print(_c(line, color))


def _stamp() -> str:
    return _c(time.strftime("%H:%M:%S"), "dim")


def info(msg: str) -> None:
    print(f"{_stamp()} {_c('*', 'cyan')} {msg}")


def step(msg: str) -> None:
    print(f"{_stamp()} {_c('>', 'blue')} {msg}")


def think(msg: str) -> None:
    print(f"{_stamp()} {_c('~', 'magenta')} {_c(msg, 'dim')}")


def act(msg: str) -> None:
    print(f"{_stamp()} {_c('->', 'yellow')} {msg}")


def ok(msg: str) -> None:
    print(f"{_stamp()} {_c('[ok]', 'green')} {msg}")


def warn(msg: str) -> None:
    print(f"{_stamp()} {_c('[!]', 'yellow')} {_c(msg, 'yellow')}")


def error(msg: str) -> None:
    print(f"{_stamp()} {_c('[x]', 'red')} {_c(msg, 'red')}")


def jarvis(msg: str) -> None:
    """A spoken/printed line attributed to Jarvis itself."""
    badge = _c(f"{_COLORS['bold']} JARVIS ", "cyan")
    print(f"\n{badge}{_COLORS['reset']} {msg}\n")


def pop(success: bool = True) -> None:
    """Short 'task finished' sound so you know Jarvis stopped without watching.

    A rising two-note chirp on success, one low note otherwise. Best-effort:
    Windows-only (winsound is stdlib there) and never raises.
    """
    if sys.platform != "win32":
        return
    try:
        import winsound
        if success:
            winsound.Beep(880, 90)
            winsound.Beep(1320, 130)
        else:
            winsound.Beep(440, 180)
    except Exception:
        pass
