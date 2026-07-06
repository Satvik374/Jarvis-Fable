"""System-level tools: shell commands, clipboard, web, and machine info.

Shell execution is guarded by a denylist from :class:`SafetyConfig`; anything
matching a blocked pattern is refused rather than run.
"""

from __future__ import annotations

import subprocess
import webbrowser


def run_command(command: str, blocked: tuple[str, ...] = (),
                timeout: int = 60) -> str:
    low = command.lower()
    for pat in blocked:
        if pat.lower() in low:
            return f"refused: command matches blocked pattern '{pat.strip()}'"
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"command timed out after {timeout}s"
    except Exception as exc:
        return f"failed to run: {exc}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parts = [f"exit code {proc.returncode}"]
    if out:
        parts.append("stdout:\n" + out[:6000])
    if err:
        parts.append("stderr:\n" + err[:2000])
    return "\n".join(parts)


def open_url(url: str) -> str:
    if not url.startswith(("http://", "https://", "file://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"opened {url}"


def clipboard_read() -> str:
    try:
        import pyperclip  # type: ignore

        return pyperclip.paste()
    except Exception:
        try:
            import pyautogui  # type: ignore

            return pyautogui.paste()  # type: ignore[attr-defined]
        except Exception as exc:
            return f"clipboard unavailable: {exc}"


def clipboard_write(text: str) -> str:
    try:
        import pyperclip  # type: ignore

        pyperclip.copy(text)
        return "copied to clipboard"
    except Exception as exc:
        return f"clipboard unavailable: {exc}"


def machine_info() -> str:
    import platform

    lines = [f"os: {platform.platform()}", f"python: {platform.python_version()}"]
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        lines.append(f"ram: {vm.used // 2**20} / {vm.total // 2**20} MB used")
        lines.append(f"cpu: {psutil.cpu_percent(interval=0.1)}%")
    except Exception:
        pass
    return "\n".join(lines)
