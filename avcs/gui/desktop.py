"""A synthetic desktop the GUI model can actually perceive.

`EyeStream(source="synthetic")` returns a flat gray buffer, which is useful for
testing plumbing and useless for showing perception: there are no controls in it
to recognise. This module renders a small desktop whose controls are drawn from
the lexicon's own vocabulary, so the reader has something real to read and a
wrong reading is a visible failure rather than an absence.

Two things it deliberately does:

- **It is not a fixture for the tests to pass on.** The words come from
  `avc.lexicon.vocab.LABELS`, so the corpus and the reader share a vocabulary by
  construction. That is the honest scope: this shows the reader working on
  software this generator draws, not generalisation to unseen applications.
- **The layout changes on demand.** `render(spec)` takes a list of button names,
  so a task can put a control on screen for one step and remove it for the next,
  which is how the "target absent" path gets exercised live instead of described.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

WIDTH = 900
HEIGHT = 520

_BG = (242, 243, 245)
_PANEL = (255, 255, 255)
_TITLEBAR = (58, 62, 74)
_TEXT = (32, 34, 40)
_MUTED = (110, 114, 126)
_BORDER = (168, 172, 182)


def _font(size: int):
    from PIL import ImageFont

    from avc.lexicon.render import discover_fonts

    fonts = discover_fonts()
    if fonts:
        try:
            return ImageFont.truetype(fonts[0], size)
        except Exception:
            pass
    return ImageFont.load_default()


def render(
    buttons: Sequence[str],
    title: str = "Untitled document",
    fields: Optional[Sequence[str]] = None,
    status: str = "ready",
    width: int = WIDTH,
    height: int = HEIGHT,
) -> np.ndarray:
    """Draw a desktop with a title bar, a toolbar of `buttons`, and a status line.

    Returns an HxWx3 uint8 array. Pure PIL, no display needed.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(image)

    # Title bar
    draw.rectangle([0, 0, width, 34], fill=_TITLEBAR)
    draw.text((14, 9), title, font=_font(16), fill=(235, 236, 240))

    # Toolbar panel
    draw.rectangle([0, 34, width, 96], fill=_PANEL)
    draw.line([0, 96, width, 96], fill=_BORDER)

    font = _font(20)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    x = 18
    y0, y1 = 48, 84
    for name in buttons:
        text_w = probe.textbbox((0, 0), name, font=font)[2]
        w = text_w + 26
        if x + w > width - 12:
            break
        draw.rectangle([x, y0, x + w, y1], fill=(248, 249, 251), outline=_BORDER)
        draw.text((x + 13, y0 + 7), name, font=font, fill=_TEXT)
        x += w + 14

    # A form area, so the screen is not one strip of controls
    if fields:
        fy = 130
        for label in fields:
            draw.text((28, fy), label, font=_font(15), fill=_MUTED)
            draw.rectangle([150, fy - 4, 430, fy + 26], fill=_PANEL, outline=_BORDER)
            fy += 46

    # Status line
    draw.rectangle([0, height - 30, width, height], fill=(232, 234, 238))
    draw.text((14, height - 24), status, font=_font(14), fill=_MUTED)

    return np.asarray(image)


class DesktopSource:
    """A frame source whose content the GUI can change between steps.

    Implements the same callable contract `LexiconPerception` accepts, so the
    GUI can swap the screen mid-conversation without rebuilding perception.
    """

    def __init__(
        self,
        buttons: Optional[Sequence[str]] = None,
        title: str = "Untitled document",
        fields: Optional[Sequence[str]] = None,
    ) -> None:
        self.buttons: List[str] = list(buttons or ["save", "close", "undo", "redo"])
        self.title = title
        self.fields = list(fields or [])
        self.status = "ready"

    def set_buttons(self, buttons: Sequence[str]) -> None:
        self.buttons = list(buttons)

    def capture(self) -> np.ndarray:
        return render(self.buttons, title=self.title, fields=self.fields, status=self.status)

    def __call__(self) -> np.ndarray:
        return self.capture()
