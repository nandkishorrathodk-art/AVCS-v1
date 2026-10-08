"""Emergency Safety Fail-Safe for AVCS Motor Control.

Protects the user by instantly relinquishing cursor control whenever the
mouse enters a designated emergency breakout corner or whenever a manual
override is detected.
"""
from __future__ import annotations

from typing import Tuple


class SafetyBreakoutException(RuntimeError):
    """Raised immediately when emergency fail-safe is triggered by the user."""


class CornerBreakoutFailSafe:
    """Detects emergency cursor breakout in any of the 4 screen corners."""

    def __init__(self, corner_margin_px: int = 15, enabled: bool = True) -> None:
        self.corner_margin_px = corner_margin_px
        self.enabled = enabled

    def check(self, pixel_x: int, pixel_y: int, screen_width: int, screen_height: int) -> None:
        """Checks if the cursor is in an emergency breakout corner.

        Raises SafetyBreakoutException if triggered.
        """
        if not self.enabled:
            return

        margin = self.corner_margin_px

        top_left = pixel_x <= margin and pixel_y <= margin
        top_right = pixel_x >= (screen_width - margin) and pixel_y <= margin
        bottom_left = pixel_x <= margin and pixel_y >= (screen_height - margin)
        bottom_right = pixel_x >= (screen_width - margin) and pixel_y >= (screen_height - margin)

        if top_left or top_right or bottom_left or bottom_right:
            corner_name = (
                "top-left" if top_left else
                "top-right" if top_right else
                "bottom-left" if bottom_left else
                "bottom-right"
            )
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Emergency abort triggered: cursor entered {corner_name} "
                f"corner ({pixel_x}, {pixel_y}). Motor control relinquished to user."
            )
