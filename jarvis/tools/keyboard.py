"""Keyboard control: typing text and pressing key combos."""

from __future__ import annotations


def _pg():
    import pyautogui  # type: ignore

    pyautogui.PAUSE = 0.02
    return pyautogui


# Friendly aliases the model might emit -> pyautogui key names.
_ALIASES = {
    "win": "winleft", "windows": "winleft", "cmd": "winleft",
    "meta": "winleft", "super": "winleft",
    "esc": "escape", "return": "enter", "del": "delete",
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "opt": "alt", "option": "alt",
    "pgup": "pageup", "pgdn": "pagedown", "ins": "insert",
    "space": "space", "spacebar": "space", "plus": "add",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "backspace": "backspace", "bksp": "backspace", "capslock": "capslock",
    "printscreen": "printscreen", "prtsc": "printscreen",
}


def type_text(text: str, interval: float = 0.01) -> str:
    _pg().typewrite(text, interval=interval)
    preview = text if len(text) <= 40 else text[:37] + "..."
    return f'typed "{preview}"'


def press(keys: str) -> str:
    """Press a single key or a '+'-joined hotkey like 'ctrl+shift+esc'."""
    pg = _pg()
    parts = [_ALIASES.get(k.strip().lower(), k.strip().lower())
             for k in keys.replace(" ", "").split("+") if k.strip()]
    if not parts:
        return "no keys given"
    if len(parts) == 1:
        pg.press(parts[0])
    else:
        pg.hotkey(*parts)
    return f"pressed {'+'.join(parts)}"


def hotkey(*keys: str) -> str:
    _pg().hotkey(*[_ALIASES.get(k.lower(), k.lower()) for k in keys])
    return f"pressed {'+'.join(keys)}"


def press_sequence(keys) -> str:
    """Press each key/combo in an ordered list, one after another."""
    if isinstance(keys, str):
        keys = [keys]
    done = []
    for combo in keys:
        press(str(combo))
        done.append(str(combo))
    return "pressed sequence: " + " -> ".join(done)
