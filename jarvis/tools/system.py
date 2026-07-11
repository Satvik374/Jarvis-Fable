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


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo search, no API key. Returns a compact text block Jarvis can
    read out or act on: an instant answer (when DDG has one) plus the top
    result titles, links and snippets.

    Real web results come from the ``ddgs`` client, which handles DuckDuckGo's
    anti-bot handshake (naive requests-scraping gets a 202 challenge page and
    zero results). The JSON Instant-Answer API is layered on top for quick
    factual answers.
    """
    query = (query or "").strip()
    if not query:
        return "web_search needs a query"
    max_results = max(1, min(10, int(max_results or 5)))
    lines: list[str] = []

    answer = _instant_answer(query)
    if answer:
        lines.append(answer)

    results, lib_ok = _ddg_text(query, max_results)
    for title, url, snippet in results:
        block = f"- {title}\n  {url}"
        if snippet:
            block += f"\n  {snippet}"
        lines.append(block)

    if not lines:
        if not lib_ok:
            return ("web search needs the DuckDuckGo client - "
                    "install it with:  pip install ddgs")
        return f"no results for '{query}' (try rephrasing, or open_url a search)"
    return "\n".join(lines)


def _instant_answer(query: str) -> str:
    """DuckDuckGo Instant Answer API - a one-line factual answer when it has
    one (empty for most general queries; that's what _ddg_text covers)."""
    import requests  # type: ignore

    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": _UA}, timeout=10)
        data = r.json()
    except Exception:
        return ""
    text = (data.get("AbstractText") or data.get("Answer") or "").strip()
    if not text:
        return ""
    src = (data.get("AbstractSource") or "").strip()
    return f"Answer: {text}" + (f"  ({src})" if src else "")


def _ddg_text(query: str, max_results: int) -> tuple[list[tuple[str, str, str]], bool]:
    """Return (results, library_available). Each result is (title, url, snippet)."""
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        try:
            from duckduckgo_search import DDGS  # type: ignore  # older name
        except Exception:
            return [], False
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return [], True
    out: list[tuple[str, str, str]] = []
    for r in rows:
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or "").strip()
        if title and url:
            out.append((title, url, body[:200]))
    return out, True


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


def system_status() -> str:
    """A JARVIS-style diagnostics readout: CPU, memory, disk, battery, uptime.

    Everything past the OS line is best-effort - psutil may be missing a given
    sensor (e.g. no battery on a desktop), so each block degrades on its own.
    """
    import platform

    lines = [f"os: {platform.platform()}"]
    try:
        import psutil  # type: ignore
    except Exception:
        lines.append("(install psutil for CPU/memory/disk/battery detail)")
        return "\n".join(lines)

    def _gb(n: int) -> str:
        return f"{n / 2**30:.1f} GB"

    try:
        lines.append(f"cpu: {psutil.cpu_percent(interval=0.2):.0f}% "
                     f"over {psutil.cpu_count(logical=True)} threads")
    except Exception:
        pass
    try:
        vm = psutil.virtual_memory()
        lines.append(f"memory: {_gb(vm.used)} / {_gb(vm.total)} used ({vm.percent:.0f}%)")
    except Exception:
        pass
    try:
        du = psutil.disk_usage("/")
        lines.append(f"disk: {_gb(du.used)} / {_gb(du.total)} used ({du.percent:.0f}%)")
    except Exception:
        pass
    try:
        bat = psutil.sensors_battery()
        if bat is not None:
            plug = "charging" if bat.power_plugged else "on battery"
            lines.append(f"battery: {bat.percent:.0f}% ({plug})")
    except Exception:
        pass
    try:
        import time
        up = int(time.time() - psutil.boot_time())
        h, m = up // 3600, (up % 3600) // 60
        lines.append(f"uptime: {h}h {m}m")
    except Exception:
        pass
    return "\n".join(lines)
