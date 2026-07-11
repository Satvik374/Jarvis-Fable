"""Mouse control built on pyautogui.

Coordinates are absolute screen pixels. The agent normally passes an *element*
which the loop resolves to a centre coordinate via the current observation;
these functions take the already-resolved (x, y).
"""

from __future__ import annotations

import time


def _pg():
    import pyautogui  # type: ignore

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.05
    return pyautogui


def _safe(pg, x: int, y: int) -> tuple[int, int]:
    """Clamp coordinates to on-screen bounds so a stray value can't overflow
    the Win32 C int in SetCursorPos and kill the run."""
    sw, sh = pg.size()
    return min(max(int(x), 3), sw - 4), min(max(int(y), 3), sh - 4)


def move(x: int, y: int, duration: float = 0.2) -> str:
    pg = _pg()
    x, y = _safe(pg, x, y)
    pg.moveTo(x, y, duration=duration)
    return f"moved mouse to ({x},{y})"


def click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    pg = _pg()
    x, y = _safe(pg, x, y)
    pg.moveTo(x, y, duration=0.15)
    pg.click(x, y, clicks=clicks, interval=0.08, button=button)
    return f"{button}-clicked ({x},{y})" + (f" x{clicks}" if clicks > 1 else "")


def double_click(x: int, y: int) -> str:
    return click(x, y, button="left", clicks=2)


def triple_click(x: int, y: int) -> str:
    return click(x, y, button="left", clicks=3)


def right_click(x: int, y: int) -> str:
    return click(x, y, button="right", clicks=1)


def drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.4) -> str:
    pg = _pg()
    x1, y1 = _safe(pg, x1, y1)
    x2, y2 = _safe(pg, x2, y2)
    pg.moveTo(x1, y1, duration=0.15)
    pg.mouseDown()
    time.sleep(0.05)
    pg.moveTo(x2, y2, duration=duration)
    pg.mouseUp()
    return f"dragged ({x1},{y1}) -> ({x2},{y2})"


def scroll(dy: int = 3, dx: int = 0) -> str:
    """Positive dy scrolls down (pyautogui uses positive = up, so we invert)."""
    pg = _pg()
    if dy:
        pg.scroll(-dy * 120)
    if dx:
        pg.hscroll(dx * 120)
    return f"scrolled dy={dy} dx={dx}"


def position() -> tuple[int, int]:
    return tuple(_pg().position())
