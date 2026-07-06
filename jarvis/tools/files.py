"""File-system tools, sandboxed to safe locations.

Reads are unrestricted; writes/creates are confined to the user's home tree
(or the ``allow_paths`` from config) so a hallucinated path can't clobber
system files.
"""

from __future__ import annotations

import os
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _within(path: Path, allow: tuple[str, ...]) -> bool:
    roots = [Path.home()] + [_expand(a) for a in allow]
    return any(_is_relative_to(path, r) for r in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def read_file(path: str, max_chars: int = 20000) -> str:
    p = _expand(path)
    if not p.exists():
        return f"file not found: {p}"
    if not p.is_file():
        return f"not a file: {p}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"could not read {p}: {exc}"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated, {len(text)} chars total]"
    return text


def write_file(path: str, content: str, allow: tuple[str, ...] = ()) -> str:
    p = _expand(path)
    if not _within(p, allow):
        return (f"refused: {p} is outside allowed write locations "
                f"(home dir + configured allow_paths)")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {p}"
    except Exception as exc:
        return f"could not write {p}: {exc}"


def list_dir(path: str = ".") -> str:
    p = _expand(path)
    if not p.exists():
        return f"path not found: {p}"
    if p.is_file():
        return f"{p} (file, {p.stat().st_size} bytes)"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except Exception as exc:
        return f"could not list {p}: {exc}"
    lines = []
    for e in entries[:200]:
        kind = "DIR " if e.is_dir() else "FILE"
        size = "" if e.is_dir() else f"  {e.stat().st_size}b"
        lines.append(f"{kind}  {e.name}{size}")
    header = f"{p}  ({len(entries)} entries)"
    return header + "\n" + "\n".join(lines)
