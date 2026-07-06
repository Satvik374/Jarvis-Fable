"""Draw the numbered element marks onto a screenshot.

Used for (a) the optional vision brain, which reasons over the marked image,
and (b) debugging / dataset collection, so you can eyeball what Jarvis saw.
"""

from __future__ import annotations

from pathlib import Path

from .elements import Observation


_PALETTE = [
    (255, 64, 64), (64, 160, 255), (48, 200, 96), (255, 176, 32),
    (200, 96, 255), (0, 200, 200),
]


def annotate(observation: Observation, screenshot, out_path: str | Path):
    """Return a PIL image of ``screenshot`` with each element boxed + numbered."""
    from PIL import ImageDraw, ImageFont  # type: ignore

    img = screenshot.image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for el in observation.elements:
        color = _PALETTE[el.id % len(_PALETTE)]
        left, top, right, bottom = el.bbox
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        tag = str(el.id)
        tw = draw.textlength(tag, font=font)
        draw.rectangle([left, top - 16, left + tw + 6, top], fill=color)
        draw.text((left + 3, top - 16), tag, fill=(255, 255, 255), font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img
