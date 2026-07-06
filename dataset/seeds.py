"""Scenario generators for the Jarvis dataset.

Each generator yields a *trajectory*: an ordered list of
``(observation, thought, action, args)`` tuples describing one complete task,
grounded in a synthetic-but-realistic screen state. The build script turns
these into chat-format SFT examples using the SAME observation/decision
formatting the live agent uses at inference time, so the model trains on
exactly what it will later see.

An ``observation`` here is ``(active_window, [(role, name, cx, cy), ...])``.
The build script assigns element ids by list order (0..n-1), identical to how
``perception.elements`` numbers them at runtime, so ``{"element": k}`` in an
action always refers to the k-th element in that turn's list.

The variety comes from randomised app names, filenames, free text, window
chrome, coordinates and distractor elements - not from hand-written strings -
so a few generators expand into thousands of distinct grounded examples.
"""

from __future__ import annotations

import random
from typing import Callable, Iterator

# A step the model should produce: (thought, action, args)
Step = tuple[str, str, dict]
# An observation the model is shown: (active_window, elements)
# where elements is a list of (role, name, center_x, center_y)
Element = tuple[str, str, int, int]
Obs = tuple[str, list[Element]]
# A trajectory turn pairs the shown observation with the target step.
Turn = tuple[Obs, Step]
Trajectory = list[Turn]


# --------------------------------------------------------------------------- #
# small vocabularies
# --------------------------------------------------------------------------- #

_FILENAMES = ["notes", "todo", "report", "ideas", "log", "shopping",
              "budget", "draft", "meeting", "journal", "recipe", "plan"]
_SENTENCES = [
    "Buy milk, eggs and bread.",
    "Remember to call the dentist tomorrow.",
    "Meeting with the team at 3pm.",
    "The quarterly numbers look strong.",
    "Pick up the package from the post office.",
    "Water the plants before the weekend.",
    "Finish the presentation slides tonight.",
    "Book train tickets for Friday.",
]
_QUERIES = ["weather in London", "python list comprehension", "best pizza near me",
            "how to tie a tie", "latest space news", "cheap flights to Tokyo",
            "unit converter", "current time in New York"]
_SITES = [("YouTube", "https://www.youtube.com"), ("GitHub", "https://github.com"),
          ("Wikipedia", "https://www.wikipedia.org"),
          ("Gmail", "https://mail.google.com"),
          ("Reddit", "https://www.reddit.com")]
_FOLDERS = ["Downloads", "Documents", "Pictures", "Desktop", "Music", "Videos"]


def _noise(n: int, taken: set[tuple[int, int]]) -> list[Element]:
    """Generate n plausible distractor elements at unused coordinates."""
    roles = ["Button", "Text", "MenuItem", "Image", "Group", "Hyperlink"]
    names = ["File", "Edit", "View", "Help", "Settings", "Home", "Close",
             "Minimize", "Back", "Forward", "Refresh", "More", "Options"]
    out: list[Element] = []
    for _ in range(n):
        for _try in range(8):
            x, y = random.randint(20, 1260), random.randint(20, 700)
            if all(abs(x - px) + abs(y - py) > 30 for px, py in taken):
                break
        taken.add((x, y))
        out.append((random.choice(roles), random.choice(names), x, y))
    return out


def _shuffle_with_targets(targets: list[Element], distractors: int) -> tuple[list[Element], dict]:
    """Mix target elements with distractors, shuffle, and return id lookup.

    Returns (elements, {target_marker: element_id}) where target_marker is the
    element's (role, name) so scenarios can reference the right id after shuffle.
    """
    taken: set[tuple[int, int]] = {(x, y) for _, _, x, y in targets}
    elems = list(targets) + _noise(distractors, taken)
    random.shuffle(elems)
    index = {(role, name): i for i, (role, name, _, _) in enumerate(elems)}
    return elems, index


# --------------------------------------------------------------------------- #
# scenario generators
# --------------------------------------------------------------------------- #

def scn_open_app(task_word: str = "") -> tuple[str, Trajectory]:
    """Open an application from the desktop, then finish."""
    app = random.choice(["notepad", "calculator", "chrome", "file explorer",
                         "paint", "settings", "spotify", "task manager"])
    desktop_elems, _ = _shuffle_with_targets([], random.randint(3, 7))
    obs: Obs = ("(desktop)", desktop_elems)
    task = random.choice([f"open {app}", f"launch {app}", f"start {app} for me"])
    traj: Trajectory = [
        (obs, (f"I'll launch {app}.", "open_app", {"name": app})),
        ((_app_window(app), _app_elems(app)),
         (f"{app.title()} is open, the task is done.", "finish",
          {"summary": f"Opened {app}."})),
    ]
    return task, traj


def scn_notepad_write(_: str = "") -> tuple[str, Trajectory]:
    """Open Notepad, type a note, save it under a filename."""
    fname = random.choice(_FILENAMES)
    text = random.choice(_SENTENCES)
    desktop, _ = _shuffle_with_targets([], random.randint(2, 5))
    empty_doc = ("Untitled - Notepad",
                 _prepend_menu([("Document", "Text editor", 640, 400)]))
    save_dialog_targets = [("Edit", "File name:", 500, 620),
                           ("Button", "Save", 820, 660)]
    save_elems, idx = _shuffle_with_targets(save_dialog_targets, random.randint(3, 6))

    task = random.choice([
        f"open notepad, write '{text}' and save it as {fname}.txt",
        f"create a note saying {text} and save as {fname}",
        f"make a text file {fname}.txt containing: {text}",
    ])
    traj: Trajectory = [
        (("(desktop)", desktop), ("Open Notepad first.", "open_app",
                                  {"name": "notepad"})),
        (empty_doc, ("Type the note text.", "type", {"text": text})),
        (empty_doc, ("Open the Save dialog.", "press", {"keys": "ctrl+s"})),
        ((f"Save As", save_elems),
         ("Focus the filename field.", "click",
          {"element": idx[("Edit", "File name:")]})),
        ((f"Save As", save_elems),
         ("Type the filename.", "type", {"text": f"{fname}.txt"})),
        ((f"Save As", save_elems),
         ("Confirm the save.", "click", {"element": idx[("Button", "Save")]})),
        ((f"{fname}.txt - Notepad", _prepend_menu([])),
         ("The note is saved.", "finish",
          {"summary": f"Saved '{text}' to {fname}.txt."})),
    ]
    return task, traj


def scn_web_search(_: str = "") -> tuple[str, Trajectory]:
    """Open the browser and run a search."""
    query = random.choice(_QUERIES)
    targets = [("Edit", "Search or type URL", 640, 70)]
    browser_elems, idx = _shuffle_with_targets(targets, random.randint(4, 9))
    desktop, _ = _shuffle_with_targets([], random.randint(2, 5))
    task = random.choice([f"search the web for {query}",
                          f"google {query}",
                          f"look up {query} online"])
    traj: Trajectory = [
        (("(desktop)", desktop), ("Open the browser.", "open_app",
                                  {"name": "chrome"})),
        (("New Tab - Google Chrome", browser_elems),
         ("Click the address bar.", "click",
          {"element": idx[("Edit", "Search or type URL")]})),
        (("New Tab - Google Chrome", browser_elems),
         ("Type the query.", "type", {"text": query})),
        (("New Tab - Google Chrome", browser_elems),
         ("Run the search.", "press", {"keys": "enter"})),
        ((f"{query} - Google Search - Chrome", _prepend_menu([])),
         ("Results are shown.", "finish",
          {"summary": f"Searched Google for '{query}'."})),
    ]
    return task, traj


def scn_open_website(_: str = "") -> tuple[str, Trajectory]:
    name, url = random.choice(_SITES)
    desktop, _ = _shuffle_with_targets([], random.randint(2, 6))
    task = random.choice([f"open {name}", f"go to {name}", f"take me to {name.lower()}"])
    traj: Trajectory = [
        (("(desktop)", desktop), (f"Open {name} in the browser.", "open_url",
                                  {"url": url})),
        ((f"{name} - Chrome", _prepend_menu([])),
         (f"{name} is open.", "finish", {"summary": f"Opened {name}."})),
    ]
    return task, traj


def scn_calculator(_: str = "") -> tuple[str, Trajectory]:
    a, b = random.randint(2, 99), random.randint(2, 99)
    op_word, op_key = random.choice([("plus", "+"), ("times", "*"),
                                     ("minus", "-")])
    digits_a = list(str(a))
    digits_b = list(str(b))
    # calculator buttons
    def btn(d): return ("Button", d, 200 + int(d) * 10, 400)
    keys = [("Button", d, 100 + i * 40, 400) for i, d in enumerate("0123456789")]
    op_btn = ("Button", {"+": "Add", "*": "Multiply", "-": "Subtract"}[op_key],
              500, 300)
    eq_btn = ("Button", "Equals", 560, 460)
    targets = keys + [op_btn, eq_btn]
    elems, idx = _shuffle_with_targets(targets, random.randint(2, 5))
    win = "Calculator"
    task = f"calculate {a} {op_word} {b} in the calculator"
    traj: Trajectory = [(("(desktop)", []), ("Open the calculator.", "open_app",
                                             {"name": "calculator"}))]
    for d in digits_a:
        traj.append(((win, elems), (f"Press {d}.", "click",
                     {"element": idx[("Button", d)]})))
    traj.append(((win, elems), (f"Press {op_word}.", "click",
                 {"element": idx[op_btn[:2]]})))
    for d in digits_b:
        traj.append(((win, elems), (f"Press {d}.", "click",
                     {"element": idx[("Button", d)]})))
    traj.append(((win, elems), ("Press equals.", "click",
                 {"element": idx[eq_btn[:2]]})))
    traj.append(((win, elems), ("The calculation is complete.", "finish",
                 {"summary": f"Computed {a} {op_key} {b}."})))
    return task, traj


def scn_file_explorer(_: str = "") -> tuple[str, Trajectory]:
    folder = random.choice(_FOLDERS)
    targets = [("TreeItem", folder, 120, 300)]
    elems, idx = _shuffle_with_targets(
        targets + [("TreeItem", f, 120, 300 + 30 * i)
                   for i, f in enumerate(random.sample(_FOLDERS, 3)) if f != folder],
        random.randint(3, 7))
    task = random.choice([f"open my {folder} folder",
                          f"go to the {folder} folder",
                          f"show me {folder} in file explorer"])
    traj: Trajectory = [
        (("(desktop)", []), ("Open File Explorer.", "open_app",
                             {"name": "file explorer"})),
        (("File Explorer", elems),
         (f"Open the {folder} folder.", "double_click",
          {"element": idx[("TreeItem", folder)]})),
        ((f"{folder} - File Explorer", _prepend_menu([])),
         (f"{folder} is open.", "finish", {"summary": f"Opened {folder}."})),
    ]
    return task, traj


def scn_scroll_find(_: str = "") -> tuple[str, Trajectory]:
    """Target not visible; the model must scroll before it appears."""
    label = random.choice(["Privacy", "About", "Bluetooth", "Display",
                           "Sound", "Storage", "Updates"])
    before, _ = _shuffle_with_targets([], random.randint(5, 8))
    target = [("ListItem", label, 300, 500)]
    after, idx = _shuffle_with_targets(target, random.randint(4, 7))
    win = "Settings"
    task = f"open the {label} settings"
    traj: Trajectory = [
        (("(desktop)", []), ("Open Settings.", "open_app", {"name": "settings"})),
        ((win, before), (f"I don't see {label}; scroll down to find it.",
                         "scroll", {"dy": 5})),
        ((win, after), (f"Now I can see {label}; click it.", "click",
                        {"element": idx[("ListItem", label)]})),
        ((f"{label} - Settings", _prepend_menu([])),
         (f"{label} settings are open.", "finish",
          {"summary": f"Opened {label} settings."})),
    ]
    return task, traj


def scn_close_window(_: str = "") -> tuple[str, Trajectory]:
    app = random.choice(["Notepad", "Chrome", "Calculator", "Paint"])
    close = [("Button", "Close", 1260, 12)]
    elems, idx = _shuffle_with_targets(close, random.randint(3, 6))
    task = random.choice([f"close the {app.lower()} window", f"close {app.lower()}"])
    traj: Trajectory = [
        ((f"{app}", elems), (f"Close {app} with the X button.", "click",
                             {"element": idx[("Button", "Close")]})),
        (("(desktop)", []), (f"{app} is closed.", "finish",
                             {"summary": f"Closed {app}."})),
    ]
    return task, traj


def scn_run_command(_: str = "") -> tuple[str, Trajectory]:
    cmd, desc = random.choice([
        ("ipconfig", "show network configuration"),
        ("echo %USERNAME%", "show the current user"),
        ("dir", "list files in the folder"),
        ("systeminfo", "show system information"),
        ("tasklist", "list running processes"),
    ])
    task = random.choice([desc, f"run {cmd}", f"can you {desc}"])
    traj: Trajectory = [
        (("(desktop)", []), (f"Run '{cmd}' to {desc}.", "run_command",
                             {"command": cmd})),
        (("(desktop)", []), ("Command finished.", "finish",
                             {"summary": f"Ran `{cmd}`."})),
    ]
    return task, traj


def scn_write_file_direct(_: str = "") -> tuple[str, Trajectory]:
    fname = random.choice(_FILENAMES)
    text = random.choice(_SENTENCES)
    task = random.choice([f"save a file called {fname}.txt with the text: {text}",
                          f"write '{text}' to ~/{fname}.txt"])
    traj: Trajectory = [
        (("(desktop)", []),
         ("Write the file directly to disk.", "write_file",
          {"path": f"~/{fname}.txt", "content": text})),
        (("(desktop)", []), ("File written.", "finish",
                             {"summary": f"Wrote ~/{fname}.txt."})),
    ]
    return task, traj


def scn_clipboard(_: str = "") -> tuple[str, Trajectory]:
    text = random.choice(_SENTENCES)
    task = random.choice([f"copy this to the clipboard: {text}",
                          f"put '{text}' on my clipboard"])
    traj: Trajectory = [
        (("(desktop)", []), ("Copy the text to the clipboard.",
                             "clipboard_write", {"text": text})),
        (("(desktop)", []), ("Copied.", "finish",
                             {"summary": "Copied text to the clipboard."})),
    ]
    return task, traj


def scn_ask_when_ambiguous(_: str = "") -> tuple[str, Trajectory]:
    """Teach the model to ask instead of guessing when truly ambiguous."""
    files = random.sample(["report.docx", "report.pdf", "report.xlsx"], 2)
    elems, idx = _shuffle_with_targets(
        [("ListItem", files[0], 200, 300), ("ListItem", files[1], 200, 340)],
        random.randint(2, 5))
    task = "open the report"
    traj: Trajectory = [
        (("File Explorer", elems),
         (f"There are two files named report; I should not guess.", "ask",
          {"question": f"Which one - {files[0]} or {files[1]}?"})),
    ]
    return task, traj


# --------------------------------------------------------------------------- #
# keyboard-heavy scenarios (arbitrary combos, sequences, multi-click)
# --------------------------------------------------------------------------- #

def scn_literal_key(_: str = "") -> tuple[str, Trajectory]:
    """User asks for a specific key/combo; the model presses exactly that."""
    combo = random.choice([
        "ctrl+enter", "shift+j", "ctrl+k", "ctrl+shift+p", "alt+tab",
        "ctrl+w", "ctrl+t", "f5", "ctrl+l", "ctrl+shift+t", "ctrl+shift+n",
        "ctrl+b", "ctrl+f", "win+e", "win+d", "ctrl+plus", "ctrl+minus",
        "shift+enter", "ctrl+shift+esc", "alt+f4", "ctrl+`",
    ])
    win = random.choice(["Google Chrome", "Visual Studio Code", "Notepad",
                         "Discord", "Slack", "Spotify"])
    task = random.choice([f"press {combo}", f"hit {combo}", f"do {combo}",
                          f"send the shortcut {combo}", f"use {combo}"])
    traj: Trajectory = [
        ((win, _prepend_menu([])), (f"Press {combo}.", "press", {"keys": combo})),
        ((win, _prepend_menu([])),
         ("Shortcut sent.", "finish", {"summary": f"Pressed {combo}."})),
    ]
    return task, traj


def scn_select_copy(_: str = "") -> tuple[str, Trajectory]:
    doc = ("Document", "Text editor", 640, 400)
    elems, idx = _shuffle_with_targets([doc], random.randint(3, 6))
    win = random.choice(["Untitled - Notepad", "report.txt - Notepad",
                         "main.py - Visual Studio Code"])
    task = random.choice(["select all the text and copy it",
                          "copy everything in the document", "select all and copy"])
    traj: Trajectory = [
        ((win, elems), ("Click into the document.", "click",
                        {"element": idx[("Document", "Text editor")]})),
        ((win, elems), ("Select all, then copy.", "key_sequence",
                        {"keys": ["ctrl+a", "ctrl+c"]})),
        ((win, elems), ("Copied.", "finish",
                        {"summary": "Selected all text and copied it."})),
    ]
    return task, traj


def scn_send_message(_: str = "") -> tuple[str, Trajectory]:
    """Type then send with ctrl+enter."""
    msg = random.choice(_SENTENCES)
    inp = ("Edit", "Message", 640, 700)
    elems, idx = _shuffle_with_targets([inp], random.randint(4, 8))
    app = random.choice(["Discord", "Slack", "Microsoft Teams", "WhatsApp"])
    task = random.choice([f"send a message saying '{msg}' in {app}",
                          f"type '{msg}' and send it", f"message the team: {msg}"])
    traj: Trajectory = [
        ((app, elems), ("Click the message box.", "click",
                        {"element": idx[("Edit", "Message")]})),
        ((app, elems), ("Type the message.", "type", {"text": msg})),
        ((app, elems), ("Send with Ctrl+Enter.", "press", {"keys": "ctrl+enter"})),
        ((app, _prepend_menu([])),
         ("Sent.", "finish", {"summary": f"Sent '{msg}' in {app}."})),
    ]
    return task, traj


def scn_command_palette(_: str = "") -> tuple[str, Trajectory]:
    action, combo = random.choice([
        ("open the command palette", "ctrl+shift+p"),
        ("open quick search", "ctrl+k"),
        ("open a new tab", "ctrl+t"),
        ("close the current tab", "ctrl+w"),
        ("reopen the last closed tab", "ctrl+shift+t"),
        ("focus the address bar", "ctrl+l"),
        ("open an incognito window", "ctrl+shift+n"),
        ("open the integrated terminal", "ctrl+`"),
    ])
    win = random.choice(["Google Chrome", "Visual Studio Code", "Microsoft Edge"])
    traj: Trajectory = [
        ((win, _prepend_menu([])), (f"Use {combo}.", "press", {"keys": combo})),
        ((win, _prepend_menu([])),
         ("Done.", "finish", {"summary": f"Pressed {combo} to {action}."})),
    ]
    return action, traj


def scn_triple_select_replace(_: str = "") -> tuple[str, Trajectory]:
    line = ("Document", "Editor line", 600, 300)
    elems, idx = _shuffle_with_targets([line], random.randint(3, 6))
    new_text = random.choice(_SENTENCES)
    win = random.choice(["Untitled - Notepad", "notes.txt - Notepad"])
    task = random.choice([f"select the whole line and replace it with: {new_text}",
                          f"replace the current line with '{new_text}'"])
    traj: Trajectory = [
        ((win, elems), ("Triple-click to select the whole line.", "triple_click",
                        {"element": idx[("Document", "Editor line")]})),
        ((win, elems), ("Type the replacement.", "type", {"text": new_text})),
        ((win, elems), ("Replaced.", "finish", {"summary": "Replaced the line."})),
    ]
    return task, traj


def scn_double_click_word(_: str = "") -> tuple[str, Trajectory]:
    word = ("Text", "selectable word", 500, 320)
    elems, idx = _shuffle_with_targets([word], random.randint(3, 6))
    win = random.choice(["Untitled - Notepad", "document.docx - Word"])
    task = random.choice(["double click the word to select it then make it bold",
                          "select the word and bold it"])
    traj: Trajectory = [
        ((win, elems), ("Double-click the word to select it.", "double_click",
                        {"element": idx[("Text", "selectable word")]})),
        ((win, elems), ("Bold it.", "press", {"keys": "ctrl+b"})),
        ((win, elems), ("Done.", "finish",
                        {"summary": "Selected the word and bolded it."})),
    ]
    return task, traj


def scn_key_nav(_: str = "") -> tuple[str, Trajectory]:
    n = random.randint(2, 4)
    seq = ["down"] * n + ["enter"]
    win = random.choice(["Settings", "File Explorer", "Start"])
    task = f"move down {n} items in the list and open the selected one"
    traj: Trajectory = [
        ((win, _prepend_menu([])), (f"Arrow down {n} times, then Enter.",
                                    "key_sequence", {"keys": seq})),
        ((win, _prepend_menu([])),
         ("Opened.", "finish", {"summary": f"Navigated down {n} and opened it."})),
    ]
    return task, traj


def scn_multiclick_count(_: str = "") -> tuple[str, Trajectory]:
    btn = ("Button", "Add", 700, 400)
    elems, idx = _shuffle_with_targets([btn], random.randint(3, 6))
    n = random.randint(2, 5)
    win = random.choice(["Shopping Cart", "Quantity", "Counter"])
    task = random.choice([f"click the add button {n} times",
                          f"press add {n} times", f"add {n} of them"])
    traj: Trajectory = [
        ((win, elems), (f"Click Add {n} times.", "click",
                        {"element": idx[("Button", "Add")], "count": n})),
        ((win, elems), ("Done.", "finish",
                        {"summary": f"Clicked Add {n} times."})),
    ]
    return task, traj


def scn_run_any_command(_: str = "") -> tuple[str, Trajectory]:
    cmd, desc = random.choice([
        ("git status", "check the git status"),
        ("python --version", "check the python version"),
        ("pip list", "list installed python packages"),
        ("whoami", "show the current user"),
        ("date /t", "show today's date"),
        ("dir /b", "list file names"),
        ("ping -n 1 google.com", "ping google once"),
        ("where python", "find the python executable"),
        ("git log --oneline -5", "show the last 5 commits"),
        ("curl -s https://api.github.com", "call the github api"),
        ("node --version", "check the node version"),
    ])
    task = random.choice([desc, f"run '{cmd}'", f"execute {cmd}", f"can you {desc}"])
    traj: Trajectory = [
        (("(desktop)", []), (f"Run `{cmd}`.", "run_command", {"command": cmd})),
        (("(desktop)", []), ("Command finished.", "finish",
                             {"summary": f"Ran `{cmd}`."})),
    ]
    return task, traj


# --------------------------------------------------------------------------- #
# tiny helpers for window chrome
# --------------------------------------------------------------------------- #

def _prepend_menu(extra: list[Element]) -> list[Element]:
    """A generic window menubar plus any extra elements, shuffled with noise."""
    base = [("MenuItem", "File", 20, 30), ("MenuItem", "Edit", 60, 30),
            ("Button", "Close", 1260, 12)]
    elems, _ = _shuffle_with_targets(base + extra, random.randint(2, 5))
    return elems


def _app_window(app: str) -> str:
    return {
        "notepad": "Untitled - Notepad", "calculator": "Calculator",
        "chrome": "New Tab - Google Chrome", "file explorer": "File Explorer",
        "paint": "Untitled - Paint", "settings": "Settings",
        "spotify": "Spotify", "task manager": "Task Manager",
    }.get(app, f"{app.title()}")


def _app_elems(app: str) -> list[Element]:
    return _prepend_menu([])


# The registry of generators the build script samples from, with weights
# reflecting how much of each to produce.
GENERATORS: list[tuple[Callable[[str], "tuple[str, Trajectory]"], int]] = [
    (scn_open_app, 3),
    (scn_notepad_write, 4),
    (scn_web_search, 4),
    (scn_open_website, 2),
    (scn_calculator, 3),
    (scn_file_explorer, 3),
    (scn_scroll_find, 3),
    (scn_close_window, 2),
    (scn_run_command, 2),
    (scn_write_file_direct, 2),
    (scn_clipboard, 1),
    (scn_ask_when_ambiguous, 1),
    # keyboard-heavy + new capabilities (arbitrary combos, sequences, multi-click)
    (scn_literal_key, 4),
    (scn_select_copy, 2),
    (scn_send_message, 3),
    (scn_command_palette, 3),
    (scn_triple_select_replace, 2),
    (scn_double_click_word, 2),
    (scn_key_nav, 2),
    (scn_multiclick_count, 2),
    (scn_run_any_command, 3),
]


def weighted_generators() -> list[Callable[[str], "tuple[str, Trajectory]"]]:
    out = []
    for gen, weight in GENERATORS:
        out.extend([gen] * weight)
    return out
