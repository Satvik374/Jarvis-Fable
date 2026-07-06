"""Turn the live screen into a numbered list of interactable elements.

This is the piece that makes accurate clicking possible on a small local model.
Rather than asking the model to guess a pixel from raw image bytes (unreliable
below ~7B vision models), we detect the real UI controls and their exact
bounding boxes, then hand the model a compact numbered menu:

    [0] Button "Save"            @ (512, 40)
    [1] Edit   "Search"          @ (300, 80)
    [2] Text   "Untitled - Notepad"

The model only has to answer "click element 0". This is the "Set-of-Marks"
technique and it works with 1.5-3B text models.

Two detectors, tried in order:
  * Windows UI Automation (``uiautomation``): rich, exact, no GPU. Primary.
  * OCR (``easyocr``): optional fallback for canvases/games with no a11y tree.

Everything degrades gracefully: if neither is available you still get an empty
element list and the loop can fall back to raw-coordinate actions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


# UIA control types worth surfacing as actionable/informative to the model.
_INTERACTIVE_ROLES = {
    "Button", "Hyperlink", "MenuItem", "ListItem", "TabItem", "CheckBox",
    "RadioButton", "ComboBox", "Edit", "Document", "TreeItem", "SplitButton",
    "Slider", "MenuBar", "Menu", "Text", "Image", "Custom", "Group",
}
# Roles we keep even when they have no name (they are still clickable targets).
_KEEP_UNNAMED = {"Edit", "Document", "Button", "ComboBox", "Custom"}


@dataclass
class Element:
    id: int
    role: str
    name: str
    bbox: tuple[int, int, int, int]   # (left, top, right, bottom)
    center: tuple[int, int]
    interactive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        name = self.name.strip().replace("\n", " ")
        if len(name) > 60:
            name = name[:57] + "..."
        label = f'"{name}"' if name else "(unlabeled)"
        cx, cy = self.center
        return f'[{self.id}] {self.role:<10} {label} @ ({cx},{cy})'


@dataclass
class Observation:
    """A full snapshot of what Jarvis can see right now."""
    elements: list[Element]
    screen_size: tuple[int, int]
    active_window: str = ""
    screenshot_path: str | None = None

    def menu(self) -> str:
        """The numbered element list shown to the model."""
        if not self.elements:
            return "(no interactable elements detected on screen)"
        return "\n".join(e.describe() for e in self.elements)

    def by_id(self, element_id: int) -> Element | None:
        for e in self.elements:
            if e.id == element_id:
                return e
        return None


def observe(max_elements: int = 60, use_uia: bool = True,
            use_ocr: bool = False) -> Observation:
    """Build an :class:`Observation` of the current desktop."""
    from .screen import screen_size

    size = screen_size()
    active = _active_window_title()
    elements: list[Element] = []

    if use_uia:
        try:
            elements = _detect_uia(max_elements, size)
        except Exception:
            elements = []

    if not elements and use_ocr:
        try:
            elements = _detect_ocr(max_elements)
        except Exception:
            elements = []

    return Observation(elements=elements, screen_size=size, active_window=active)


def _active_window_title() -> str:
    try:
        import pygetwindow as gw  # type: ignore

        w = gw.getActiveWindow()
        return w.title if w else ""
    except Exception:
        return ""


def _detect_uia(max_elements: int, size: tuple[int, int]) -> list[Element]:
    """Walk the UI Automation tree of the foreground window.

    Browsers (and Electron apps) nest the actual page content deep inside a
    ``Document`` control - far deeper than the window chrome. A naive
    breadth-first walk spends the whole element budget on chrome (tabs,
    toolbar, address bar) and never reaches the page, leaving the agent blind
    to what it actually needs to click. So:

      * chrome elements are capped (`_MAX_CHROME`) - enough for the address
        bar, tabs and window buttons;
      * as soon as a ``Document`` is found, its subtree is explored FIRST
        (depth-first via appendleft) with a much deeper limit, so the page
        content gets the remaining element budget;
      * element centres must be truly on screen (off-screen/degenerate boxes
        used to produce stacked, unclickable marks);
      * near-duplicate centres are dropped.
    """
    import os
    from collections import deque

    from ._comtypes_fix import ensure as _ensure_comtypes, data_dir
    _ensure_comtypes()
    import uiautomation as auto  # type: ignore

    # Keep uiautomation's log file out of the user's project directory.
    try:
        auto.Logger.SetLogFile(os.path.join(data_dir(), "uiautomation.log"))
    except Exception:
        pass

    import time

    _MAX_CHROME = 18        # element budget for non-document (window chrome)
    _CHROME_DEPTH = 12
    _DOC_DEPTH = 45         # web pages nest deeply
    _MAX_VISITED = 6000

    sw, sh = size

    root = auto.GetForegroundControl()
    if root is None:
        return []

    # Chromium-based browsers build the page's accessibility tree LAZILY: the
    # first UIA query on a freshly focused tab can return an empty Document.
    # If that happens, wait briefly and walk once more.
    for attempt in range(2):
        elements: list[Element] = []
        seen: set[tuple] = set()
        centers: set[tuple] = set()
        kept_names: set[str] = set()
        chrome_count = 0
        doc_found = False
        doc_kept = 0

        # queue holds (control, depth, in_document)
        queue: deque = deque([(root, 0, False)])
        visited = 0
        while queue and len(elements) < max_elements and visited < _MAX_VISITED:
            ctrl, depth, in_doc = queue.popleft()
            visited += 1
            try:
                role = ctrl.ControlTypeName.replace("Control", "")
                rect = ctrl.BoundingRectangle
                left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
            except Exception:
                continue

            is_doc = role == "Document"
            if is_doc:
                doc_found = True
            w, h = right - left, bottom - top
            cx, cy = (left + right) // 2, (top + bottom) // 2
            on_screen = (w > 3 and h > 3 and w < sw and h <= sh
                         and 0 <= cx < sw and 0 <= cy < sh)
            name = ""
            try:
                name = (ctrl.Name or "").strip()
            except Exception:
                pass

            keep = (role in _INTERACTIVE_ROLES and on_screen
                    and (name or role in _KEEP_UNNAMED))
            if keep and not in_doc and not is_doc and chrome_count >= _MAX_CHROME:
                keep = False    # chrome budget spent; still traverse to find the doc
            if keep and role == "Text" and name and name in kept_names:
                keep = False    # plain-text duplicate of an element already listed
            key = (role, name, left, top)
            ckey = (cx // 8, cy // 8)
            if keep and key not in seen and ckey not in centers:
                seen.add(key)
                centers.add(ckey)
                if name:
                    kept_names.add(name)
                elements.append(Element(
                    id=len(elements), role=role, name=name,
                    bbox=(left, top, right, bottom), center=(cx, cy),
                ))
                if in_doc:
                    doc_kept += 1
                elif not is_doc:
                    chrome_count += 1

            limit = _DOC_DEPTH if (in_doc or is_doc) else _CHROME_DEPTH
            if depth < limit:
                try:
                    children = ctrl.GetChildren()
                except Exception:
                    children = []
                if is_doc or in_doc:
                    # Page content: explore before any remaining chrome so the
                    # budget goes to what the user actually wants to click.
                    for child in reversed(children):
                        queue.appendleft((child, depth + 1, True))
                else:
                    for child in children:
                        queue.append((child, depth + 1, False))

        if doc_found and doc_kept == 0 and attempt == 0:
            time.sleep(0.6)     # let the renderer finish building the a11y tree
            continue
        break

    return elements


def _detect_ocr(max_elements: int) -> list[Element]:
    """OCR fallback: every recognised text box becomes a clickable element."""
    import easyocr  # type: ignore
    from .screen import capture

    shot = capture()
    reader = _ocr_reader()
    import numpy as np  # type: ignore

    results = reader.readtext(np.array(shot.image))
    elements: list[Element] = []
    for i, (box, text, conf) in enumerate(results[:max_elements]):
        if conf < 0.4 or not text.strip():
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        elements.append(Element(
            id=len(elements), role="Text", name=text.strip(),
            bbox=(left, top, right, bottom),
            center=((left + right) // 2, (top + bottom) // 2),
        ))
    return elements


_OCR_SINGLETON = None


def _ocr_reader():
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        import easyocr  # type: ignore

        _OCR_SINGLETON = easyocr.Reader(["en"], gpu=False)
    return _OCR_SINGLETON
