"""Single source of truth for Jarvis's action space.

Every layer of the system references this file so they can never drift apart:

  * the runtime agentic loop (``jarvis.agent.loop``) executes these actions,
  * the system prompt (``jarvis.agent.prompts``) documents them to the model,
  * the dataset builder (``dataset/build_dataset.py``) generates training
    examples that emit exactly these actions,
  * the training pipeline (``training/``) fine-tunes a model to produce them.

An *action* is the atomic unit the brain emits each step. The model always
replies with a single JSON object of the form::

    {"thought": "<short reasoning>", "action": "<name>", "args": {...}}

Keeping this module free of heavy dependencies (no pyautogui / torch / uiautomation)
is deliberate: the dataset builder and tests can import it anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class Action:
    name: str
    summary: str
    params: tuple[Param, ...] = field(default_factory=tuple)
    # ``category`` groups actions in the docs; ``terminal`` marks actions that
    # end the loop (``finish`` / ``ask``).
    category: str = "control"
    terminal: bool = False
    # A couple of realistic example arg dicts, reused as few-shot seeds.
    examples: tuple[dict, ...] = field(default_factory=tuple)


# A pointer target may be given either as an element id from the current
# perception snapshot (preferred, accurate) or as raw x/y pixel coordinates.
_TARGET_PARAMS = (
    Param("element", "int", "Id of a labelled element from the current screen "
          "observation. Prefer this over raw coordinates.", required=False),
    Param("x", "int", "Absolute screen x pixel. Use only when no element id fits.",
          required=False),
    Param("y", "int", "Absolute screen y pixel. Use only when no element id fits.",
          required=False),
)


ACTIONS: tuple[Action, ...] = (
    # ---- pointer ---------------------------------------------------------
    Action(
        "click", "Left-click an element or screen coordinate. Pass 'count' to "
        "click several times in place (e.g. count 2 = double, 3 = triple).",
        _TARGET_PARAMS + (
            Param("count", "int", "How many clicks (default 1).",
                  required=False, default=1),
        ), category="pointer",
        examples=({"element": 4}, {"x": 640, "y": 360}, {"element": 4, "count": 3}),
    ),
    Action(
        "double_click", "Double-click an element or coordinate (open items, "
        "select a word).", _TARGET_PARAMS, category="pointer",
        examples=({"element": 7},),
    ),
    Action(
        "triple_click", "Triple-click an element or coordinate (selects a whole "
        "line or paragraph).", _TARGET_PARAMS, category="pointer",
        examples=({"element": 5},),
    ),
    Action(
        "right_click", "Right-click to open a context menu.",
        _TARGET_PARAMS, category="pointer",
        examples=({"element": 2},),
    ),
    Action(
        "move", "Move the mouse without clicking (hover to reveal menus).",
        (Param("x", "int", "Absolute screen x."),
         Param("y", "int", "Absolute screen y.")),
        category="pointer", examples=({"x": 100, "y": 200},),
    ),
    Action(
        "drag", "Press at one point/element and release at another.",
        (Param("from_element", "int", "Source element id.", required=False),
         Param("x1", "int", "Source x.", required=False),
         Param("y1", "int", "Source y.", required=False),
         Param("to_element", "int", "Target element id.", required=False),
         Param("x2", "int", "Target x.", required=False),
         Param("y2", "int", "Target y.", required=False)),
        category="pointer",
        examples=({"x1": 300, "y1": 400, "x2": 600, "y2": 400},),
    ),
    Action(
        "scroll", "Scroll the active window. Positive dy scrolls down.",
        (Param("dy", "int", "Vertical clicks; positive = down, negative = up.",
               required=False, default=3),
         Param("dx", "int", "Horizontal clicks; positive = right.",
               required=False, default=0)),
        category="pointer", examples=({"dy": 5}, {"dy": -3}),
    ),
    # ---- keyboard --------------------------------------------------------
    Action(
        "type", "Type literal text at the current keyboard focus.",
        (Param("text", "str", "The exact text to type."),),
        category="keyboard", examples=({"text": "hello world"},),
    ),
    Action(
        "press", "Press a single key or ANY hotkey combo. Join keys with '+'. "
        "Works for letters, digits, function keys (f1-f12), arrows (up/down/"
        "left/right), and modifiers (ctrl/shift/alt/win) in any combination - "
        "e.g. 'enter', 'ctrl+s', 'ctrl+enter', 'shift+j', 'ctrl+k', "
        "'ctrl+shift+p', 'alt+tab', 'ctrl+alt+delete'.",
        (Param("keys", "str", "Key name or '+'-joined combo."),),
        category="keyboard",
        examples=({"keys": "enter"}, {"keys": "ctrl+s"}, {"keys": "ctrl+enter"},
                  {"keys": "shift+j"}, {"keys": "ctrl+k"}, {"keys": "alt+tab"}),
    ),
    Action(
        "key_sequence", "Press several keys/combos one after another in a single "
        "step. Give an ordered list; each item is a key or '+'-combo.",
        (Param("keys", "list", "Ordered list of keys/combos to press in turn."),),
        category="keyboard",
        examples=({"keys": ["ctrl+a", "ctrl+c"]},
                  {"keys": ["down", "down", "enter"]}),
    ),
    # ---- apps / os -------------------------------------------------------
    Action(
        "open_app", "Launch or focus an application by name (e.g. 'notepad', "
        "'chrome', 'calculator', 'explorer').",
        (Param("name", "str", "Application name or executable."),),
        category="apps", examples=({"name": "notepad"}, {"name": "chrome"}),
    ),
    Action(
        "open_url", "Open a URL in the default web browser.",
        (Param("url", "str", "Fully-qualified URL."),),
        category="apps", examples=({"url": "https://www.google.com"},),
    ),
    Action(
        "run_command", "Run a shell command and capture its output. Use for "
        "non-GUI tasks. Refuses obviously destructive commands.",
        (Param("command", "str", "The command line to execute."),),
        category="system", examples=({"command": "ipconfig"},),
    ),
    Action(
        "focus_window", "Bring an open window matching a title substring to the "
        "foreground.",
        (Param("title", "str", "Case-insensitive substring of the window title."),),
        category="apps", examples=({"title": "Notepad"},),
    ),
    # ---- files -----------------------------------------------------------
    Action(
        "read_file", "Read a UTF-8 text file and return its content.",
        (Param("path", "str", "Absolute or user-relative file path."),),
        category="files", examples=({"path": "~/notes.txt"},),
    ),
    Action(
        "write_file", "Create or overwrite a UTF-8 text file (use for code too).",
        (Param("path", "str", "Destination path; parent folders are auto-created."),
         Param("content", "str", "Full file content.")),
        category="files",
        examples=({"path": "~/todo.txt", "content": "buy milk"},
                  {"path": "~/projects/app/main.py",
                   "content": "print('hello world')\n"}),
    ),
    Action(
        "make_dir", "Create a folder (and any missing parent folders).",
        (Param("path", "str", "Folder path to create."),),
        category="files",
        examples=({"path": "~/projects/myapp"},
                  {"path": "~/projects/myapp/src"}),
    ),
    Action(
        "list_dir", "List the entries of a directory.",
        (Param("path", "str", "Directory path.", required=False, default="."),),
        category="files", examples=({"path": "~/Downloads"},),
    ),
    # ---- clipboard -------------------------------------------------------
    Action(
        "system_status", "Report machine diagnostics (CPU, memory, disk, "
        "battery, uptime). Use to answer 'how's my system / battery / cpu / "
        "memory / disk' without opening any app.", (),
        category="system", examples=({},),
    ),
    Action(
        "web_search", "Search the web with DuckDuckGo and get back text "
        "results (an instant answer plus top links). Use to look something up "
        "and answer directly, without opening a browser.",
        (Param("query", "str", "What to search for."),
         Param("max_results", "int", "How many results (default 5).",
               required=False, default=5)),
        category="system",
        examples=({"query": "who won the 2022 world cup"},
                  {"query": "python read a file", "max_results": 3}),
    ),
    Action(
        "schedule_task", "Schedule a task to run automatically later or on a "
        "repeat (a cron job). schedule accepts 'every N minutes/hours', 'daily "
        "at HH:MM', 'in N minutes', or 'at HH:MM' (24h).",
        (Param("schedule", "str", "When to run, e.g. 'daily at 08:00'."),
         Param("command", "str", "The task to run when it fires.")),
        category="system",
        examples=({"schedule": "daily at 08:00", "command": "search the web for today's news"},
                  {"schedule": "every 30 minutes", "command": "tell me the system status"}),
    ),
    Action(
        "clipboard_read", "Read the current clipboard text.", (),
        category="system", examples=({},),
    ),
    Action(
        "clipboard_write", "Put text on the clipboard.",
        (Param("text", "str", "Text to copy."),),
        category="system", examples=({"text": "copied text"},),
    ),
    # ---- meta ------------------------------------------------------------
    Action(
        "wait", "Pause briefly to let the screen settle after an action.",
        (Param("seconds", "float", "Seconds to wait (<= 10).",
               required=False, default=1.0),),
        category="meta", examples=({"seconds": 1.5},),
    ),
    Action(
        "observe", "Take a fresh screenshot and re-read the screen. Use after an "
        "action changes the UI and you need to see the result.", (),
        category="meta", examples=({},),
    ),
    Action(
        "finish", "The task is complete. Provide a short result summary for the user.",
        (Param("summary", "str", "What was accomplished."),),
        category="meta", terminal=True,
        examples=({"summary": "Saved the note to todo.txt."},),
    ),
    Action(
        "ask", "Stop and ask the user a question when the task is ambiguous or "
        "blocked. Only use when you genuinely cannot proceed.",
        (Param("question", "str", "The question for the user."),),
        category="meta", terminal=True,
        examples=({"question": "Which file did you mean, report.docx or report.pdf?"},),
    ),
)


ACTIONS_BY_NAME: dict[str, Action] = {a.name: a for a in ACTIONS}


def action_names() -> list[str]:
    return [a.name for a in ACTIONS]


def to_json_schema() -> list[dict]:
    """Return an OpenAI/JSON-schema style description of every action.

    Handy for OpenAI-compatible function-calling backends and for docs.
    """
    out: list[dict] = []
    for a in ACTIONS:
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in a.params:
            props[p.name] = {"type": _json_type(p.type), "description": p.description}
            if p.required:
                required.append(p.name)
        out.append({
            "name": a.name,
            "description": a.summary,
            "parameters": {"type": "object", "properties": props, "required": required},
        })
    return out


def _json_type(t: str) -> str:
    return {"int": "integer", "float": "number", "str": "string",
            "bool": "boolean", "list": "array"}.get(t, "string")
