"""Experimental Playwright perception adapter with fail-closed motor actions."""
from __future__ import annotations

import io
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from forge.vision.cortex.sdk import ScreenEye, UnvalidatedGroundingError


class AVCSPlaywrightEye:
    """Perception helper; semantic browser input stays disabled until grounding is validated."""

    def __init__(
        self,
        page: Any,
        eye: Optional[ScreenEye] = None,
        viewport_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.page = page
        self.eye = eye or ScreenEye()
        self._viewport_size = viewport_size

    def get_viewport_size(self) -> Tuple[int, int]:
        """Returns browser viewport dimensions (width, height)."""
        if self._viewport_size:
            return self._viewport_size
        try:
            vp = self.page.viewport_size
            if vp and "width" in vp and "height" in vp:
                return (int(vp["width"]), int(vp["height"]))
        except Exception:
            pass
        return (1280, 720)

    def grab_page_frame(self) -> np.ndarray:
        """Grabs current screenshot from Playwright page as RGB numpy array."""
        try:
            raw_bytes = self.page.screenshot()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            return np.array(pil_img)
        except Exception:
            # Fallback to local capture source if page screenshot unavailable
            return self.eye.source.grab()

    def wait_until_visual_stable(self, timeout: float = 4.0) -> bool:
        """Blocks until rendering, transitions, and CSS animations settle."""
        return self.eye.wait_until_stable(timeout=timeout)

    def visual_click(
        self,
        label_or_intent: str,
        timeout: float = 5.0,
        click_delay: float = 0.05,
    ) -> Tuple[int, int]:
        """Rejects language-grounded clicks until semantic grounding is validated."""
        raise UnvalidatedGroundingError(
            "Natural-language targets cannot drive browser input: validated trained "
            "semantic grounding is unavailable."
        )

    def visual_type(
        self,
        field_label: str,
        text: str,
        clear_first: bool = True,
    ) -> Tuple[int, int]:
        """Visually locates a text field, clicks to focus, and types input string."""
        import sys
        x, y = self.visual_click(field_label)
        if clear_first and hasattr(self.page, "keyboard"):
            # Select all and delete existing text (cross-platform)
            select_key = "Meta+A" if sys.platform == "darwin" else "Control+A"
            self.page.keyboard.press(select_key)
            self.page.keyboard.press("Backspace")

        if hasattr(self.page, "keyboard") and hasattr(self.page.keyboard, "type"):
            self.page.keyboard.type(text)

        return (x, y)

    def assert_element_visible(self, label_or_intent: str) -> bool:
        """Rejects semantic assertions until grounding is validated."""
        raise UnvalidatedGroundingError(
            "Natural-language visibility assertions are unavailable until semantic "
            "grounding is validated."
        )

    # Aliases for developer convenience
    click_visual = visual_click
    input_text = visual_type
