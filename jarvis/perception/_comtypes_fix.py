"""Make ``uiautomation`` importable when Python lives in C:\\Program Files.

``uiautomation`` relies on ``comtypes``, which by default writes generated COM
wrapper modules into its own install directory. When the Python installation is
under ``C:\\Program Files`` that directory is not user-writable, so comtypes
raises ``PermissionError`` and reports "Can not load UIAutomationCore.dll" -
which would silently disable Jarvis's element detection.

Calling :func:`ensure` before importing ``uiautomation`` redirects that cache to
a writable per-user folder (``%LOCALAPPDATA%\\Jarvis\\comtypes_gen``), which
fixes it. It is idempotent and a no-op off Windows.
"""

from __future__ import annotations

import os
import sys
import tempfile

_DONE = False


def data_dir() -> str:
    """A writable per-user directory for Jarvis's own caches and logs."""
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    path = os.path.join(base, "Jarvis")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = tempfile.gettempdir()
    return path


def ensure() -> None:
    global _DONE
    if _DONE or os.name != "nt":
        _DONE = True
        return
    _DONE = True

    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    gen = os.path.join(base, "Jarvis", "comtypes_gen")
    try:
        os.makedirs(gen, exist_ok=True)
    except Exception:
        return

    try:
        import comtypes.client  # type: ignore

        comtypes.client.gen_dir = gen
        import comtypes.gen as _gen  # type: ignore

        if gen not in _gen.__path__:
            _gen.__path__.insert(0, gen)
    except Exception:
        # If comtypes isn't importable yet we still put the dir on sys.path so
        # a later import picks it up.
        pass

    if gen not in sys.path:
        sys.path.insert(0, gen)
