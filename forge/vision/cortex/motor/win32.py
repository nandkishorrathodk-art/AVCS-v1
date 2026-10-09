"""Native Windows Win32 OS Cursor Driver.

Direct hardware-level cursor steering using Windows user32.dll APIs.
Zero external dependencies, sub-millisecond execution latency.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import time
from typing import Tuple

from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe

# Win32 Mouse Event Flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000

SM_CXSCREEN = 0
SM_CYSCREEN = 1


class Win32CursorDriver:
    """Hardware cursor driver for Microsoft Windows."""

    def __init__(self, failsafe: bool = True) -> None:
        self.user32 = ctypes.windll.user32
        self.failsafe = CornerBreakoutFailSafe(enabled=failsafe)

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Returns primary display resolution in pixels: (width, height)."""
        width = int(self.user32.GetSystemMetrics(SM_CXSCREEN))
        height = int(self.user32.GetSystemMetrics(SM_CYSCREEN))
        return (width, height)

    def get_position_pixels(self) -> Tuple[int, int]:
        """Returns current physical mouse coordinates in screen pixels."""
        pt = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y))

    def get_position_normalized(self) -> Tuple[float, float]:
        """Returns mouse position normalized to [0.0, 1.0]."""
        px, py = self.get_position_pixels()
        w, h = self.screen_size
        nx = max(0.0, min(1.0, px / max(1, w - 1)))
        ny = max(0.0, min(1.0, py / max(1, h - 1)))
        return (nx, ny)

    def move_to_pixels(self, x: int, y: int) -> None:
        """Moves cursor to absolute pixel coordinates with fail-safe guard."""
        w, h = self.screen_size
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))

        self.failsafe.check(clamped_x, clamped_y, w, h)
        self.user32.SetCursorPos(clamped_x, clamped_y)

    def move_to_normalized(self, nx: float, ny: float) -> None:
        """Moves cursor to normalized screen coordinates [0.0, 1.0]."""
        w, h = self.screen_size
        px = int(round(nx * (w - 1)))
        py = int(round(ny * (h - 1)))
        self.move_to_pixels(px, py)

    def move_relative(self, dx: int, dy: int) -> None:
        """Moves cursor by relative pixel offsets."""
        curr_x, curr_y = self.get_position_pixels()
        self.move_to_pixels(curr_x + dx, curr_y + dy)

    def mouse_down(self, button: str = "left") -> None:
        """Presses mouse button down without releasing."""
        flag = (
            MOUSEEVENTF_RIGHTDOWN if button == "right" else
            MOUSEEVENTF_MIDDLEDOWN if button == "middle" else
            MOUSEEVENTF_LEFTDOWN
        )
        self.user32.mouse_event(flag, 0, 0, 0, 0)

    def mouse_up(self, button: str = "left") -> None:
        """Releases pressed mouse button."""
        flag = (
            MOUSEEVENTF_RIGHTUP if button == "right" else
            MOUSEEVENTF_MIDDLEUP if button == "middle" else
            MOUSEEVENTF_LEFTUP
        )
        self.user32.mouse_event(flag, 0, 0, 0, 0)

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        """Executes a physical click with natural hold duration."""
        self.mouse_down(button=button)
        if hold_duration > 0:
            time.sleep(hold_duration)
        self.mouse_up(button=button)

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        """Executes double click."""
        self.click(button=button)
        time.sleep(interval)
        self.click(button=button)

