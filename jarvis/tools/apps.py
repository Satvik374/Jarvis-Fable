"""Launch, focus, and enumerate applications and windows (Windows-first)."""

from __future__ import annotations

import os
import subprocess
import time


# Common friendly names -> launch command on Windows.
_KNOWN = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "files": "explorer.exe",
    "file explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "chrome": "chrome",
    "edge": "msedge",
    "firefox": "firefox",
    "word": "winword",
    "excel": "excel",
    "vscode": "code",
    "vs code": "code",
    "code": "code",
    "spotify": "spotify",
}


def open_app(name: str) -> str:
    """Launch (or bring up) an application by friendly name or executable."""
    key = name.strip().lower()
    target = _KNOWN.get(key, name)

    # Try to focus it first if a matching window already exists.
    if focus_window(name).startswith("focused"):
        return f"focused existing '{name}'"

    if target.startswith("ms-settings:") or target.startswith("http"):
        os.startfile(target)  # type: ignore[attr-defined]
        return f"opened '{target}'"

    try:
        if os.name == "nt":
            # ``start`` resolves App Paths / PATH like the Run dialog does.
            subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
        else:
            subprocess.Popen([target])
        time.sleep(1.0)
        return f"launched '{name}'"
    except Exception as exc:
        # Last resort: hand it to the shell / Start.
        try:
            os.startfile(target)  # type: ignore[attr-defined]
            return f"opened '{name}'"
        except Exception:
            return f"could not launch '{name}': {exc}"


def focus_window(title: str) -> str:
    """Activate the first window whose title contains ``title``."""
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return "pygetwindow unavailable"

    matches = [w for w in gw.getAllWindows()
               if w.title and title.lower() in w.title.lower()]
    if not matches:
        return f"no window matching '{title}'"
    w = matches[0]
    try:
        if w.isMinimized:
            w.restore()
        w.activate()
        return f"focused '{w.title}'"
    except Exception as exc:
        return f"found '{w.title}' but could not focus: {exc}"


def list_windows() -> list[str]:
    try:
        import pygetwindow as gw  # type: ignore

        return [w.title for w in gw.getAllWindows() if w.title.strip()]
    except Exception:
        return []
