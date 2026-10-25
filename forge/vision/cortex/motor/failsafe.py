"""Emergency Safety Fail-Safe for AVCS Motor Control.

Protects the user by instantly relinquishing cursor control whenever the
mouse enters a designated emergency breakout corner or whenever a manual
override is detected.
"""
from __future__ import annotations

from typing import Optional, Tuple


class SafetyBreakoutException(RuntimeError):
    """Raised immediately when emergency fail-safe is triggered by the user."""


class CornerBreakoutFailSafe:
    """Detects emergency cursor breakout in any of the 4 screen corners."""

    def __init__(
        self,
        corner_margin_px: int = 15,
        enabled: bool = True,
        dwell_frames: int = 1,
        exclude_top_right: bool = False,
        exclude_bottom_left: bool = False,
    ) -> None:
        self.corner_margin_px = corner_margin_px
        self.enabled = enabled
        self.dwell_frames = dwell_frames
        self.exclude_top_right = exclude_top_right
        self.exclude_bottom_left = exclude_bottom_left
        self._tripped = False
        self._consecutive_corner_frames = 0
        self._current_corner: Optional[str] = None

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        """Resets the safety fail-safe latch, allowing motor control to resume."""
        self._tripped = False
        self._consecutive_corner_frames = 0
        self._current_corner = None

    def check_coordinates(
        self,
        pixel_x: int,
        pixel_y: int,
        screen_width: int,
        screen_height: int,
    ) -> None:
        """Checks commanded cursor coordinates against emergency breakout corners.

        Excludes standard Windows 11 window controls (top-right close box and
        bottom-left Start button) so normal interactions with window controls do
        not permanently trip the failsafe and lock out the motor driver.
        """
        if not self.enabled:
            return

        if self._tripped:
            raise SafetyBreakoutException(
                "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                "Call failsafe.reset() to re-enable motor operations."
            )

        margin = self.corner_margin_px

        top_left = pixel_x <= margin and pixel_y <= margin
        bottom_right = pixel_x >= (screen_width - margin) and pixel_y >= (screen_height - margin)

        if top_left or bottom_right:
            self._tripped = True
            corner_name = "top-left" if top_left else "bottom-right"
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Emergency abort triggered: cursor entered {corner_name} "
                f"corner ({pixel_x}, {pixel_y}). Motor control relinquished to user."
            )

    def check_hardware_failsafe(
        self,
        pixel_x: int,
        pixel_y: int,
        screen_width: int,
        screen_height: int,
        dwell_frames: int = 3,
    ) -> None:
        """Checks physical hardware cursor position for emergency user breakout.

        Windows 11 places the maximized window Close button ('X') in the top-right
        corner (pixel_x >= screen_width - 15, pixel_y <= 15) and the Start button in
        the bottom-left corner (pixel_x <= 15, pixel_y >= screen_height - 15).
        To prevent normal interactions with these window controls from permanently
        tripping the failsafe and locking out the motor driver, emergency breakout
        in those corners requires holding across multiple frames (corner dwell).
        Top-left triggers an instant safety abort.
        """
        if not self.enabled:
            return

        if self._tripped:
            raise SafetyBreakoutException(
                "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                "Call failsafe.reset() to re-enable motor operations."
            )

        margin = self.corner_margin_px

        top_left = pixel_x <= margin and pixel_y <= margin
        top_right = pixel_x >= (screen_width - margin) and pixel_y <= margin
        bottom_left = pixel_x <= margin and pixel_y >= (screen_height - margin)
        bottom_right = pixel_x >= (screen_width - margin) and pixel_y >= (screen_height - margin)

        if not (top_left or top_right or bottom_left or bottom_right):
            self._consecutive_corner_frames = 0
            self._current_corner = None
            return

        corner_name = (
            "top-left" if top_left else
            "top-right" if top_right else
            "bottom-left" if bottom_left else
            "bottom-right"
        )

        # Primary emergency breakout corners (top-left, bottom-right) trip immediately
        if top_left or bottom_right:
            self._tripped = True
            self._consecutive_corner_frames = 0
            self._current_corner = None
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Emergency abort triggered: cursor entered {corner_name} "
                f"corner ({pixel_x}, {pixel_y}). Motor control relinquished to user."
            )

        # Window control corners (top-right close box, bottom-left Start button) require dwell
        effective_dwell = max(1, self.dwell_frames if self.dwell_frames > 1 else dwell_frames)
        if self.exclude_top_right and top_right:
            self._consecutive_corner_frames = 0
            self._current_corner = None
            return
        if self.exclude_bottom_left and bottom_left:
            self._consecutive_corner_frames = 0
            self._current_corner = None
            return

        if self._current_corner == corner_name:
            self._consecutive_corner_frames += 1
        else:
            self._current_corner = corner_name
            self._consecutive_corner_frames = 1

        if self._consecutive_corner_frames >= effective_dwell:
            self._tripped = True
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Emergency abort triggered: cursor entered {corner_name} "
                f"corner ({pixel_x}, {pixel_y}). Motor control relinquished to user."
            )

    def check(self, pixel_x: int, pixel_y: int, screen_width: int, screen_height: int) -> None:
        """Checks if the cursor is in an emergency breakout corner.

        Raises SafetyBreakoutException if triggered or already latched.
        """
        if not self.enabled:
            return

        margin = self.corner_margin_px

        top_left = pixel_x <= margin and pixel_y <= margin
        top_right = pixel_x >= (screen_width - margin) and pixel_y <= margin
        bottom_left = pixel_x <= margin and pixel_y >= (screen_height - margin)
        bottom_right = pixel_x >= (screen_width - margin) and pixel_y >= (screen_height - margin)

        if self.exclude_top_right and top_right:
            top_right = False
        if self.exclude_bottom_left and bottom_left:
            bottom_left = False

        if not (top_left or top_right or bottom_left or bottom_right):
            self._consecutive_corner_frames = 0
            self._current_corner = None
            if self._tripped:
                raise SafetyBreakoutException(
                    "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                    "Call failsafe.reset() to re-enable motor operations."
                )
            return

        corner_name = (
            "top-left" if top_left else
            "top-right" if top_right else
            "bottom-left" if bottom_left else
            "bottom-right"
        )

        if self.dwell_frames > 1 and (top_right or bottom_left):
            if self._current_corner == corner_name:
                self._consecutive_corner_frames += 1
            else:
                self._current_corner = corner_name
                self._consecutive_corner_frames = 1
            if self._consecutive_corner_frames < self.dwell_frames:
                return

        self._tripped = True
        raise SafetyBreakoutException(
            f"[AVCS Safety Fail-Safe] Emergency abort triggered: cursor entered {corner_name} "
            f"corner ({pixel_x}, {pixel_y}). Motor control relinquished to user."
        )

    def check_override(
        self,
        actual_pos: Tuple[float, float],
        expected_pos: Tuple[float, float],
        threshold_norm: float = 0.08,
    ) -> None:
        """Detects physical human resistance (user snatching mouse during servoing)."""
        if not self.enabled:
            return
        dx = actual_pos[0] - expected_pos[0]
        dy = actual_pos[1] - expected_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > threshold_norm:
            self._tripped = True
            raise SafetyBreakoutException(
                f"[AVCS Safety Fail-Safe] Manual override detected: cursor diverted by {dist:.3f} "
                f"from planned trajectory. Control relinquished to user."
            )
        if self._tripped:
            raise SafetyBreakoutException(
                "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                "Call failsafe.reset() to re-enable motor operations."
            )
