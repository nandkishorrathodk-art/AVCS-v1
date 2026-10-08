"""Unified Cross-Platform OS Cursor Driver Interface.

Provides a polymorphic driver abstraction supporting native Windows Win32,
cross-platform fallbacks, and headless Virtual drivers.
"""
from __future__ import annotations

import sys
import time
from typing import List, Optional, Protocol, Tuple

from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe, SafetyBreakoutException


class OSCursorDriver(Protocol):
    """Abstract protocol for OS cursor drivers."""

    @property
    def screen_size(self) -> Tuple[int, int]:
        ...

    def get_position_pixels(self) -> Tuple[int, int]:
        ...

    def get_position_normalized(self) -> Tuple[float, float]:
        ...

    def move_to_pixels(self, x: int, y: int) -> None:
        ...

    def move_to_normalized(self, nx: float, ny: float) -> None:
        ...

    def move_relative(self, dx: int, dy: int) -> None:
        ...

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        ...

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        ...


class VirtualCursorDriver:
    """In-memory simulated cursor driver for headless testing and CI."""

    def __init__(
        self,
        screen_size: Tuple[int, int] = (1920, 1080),
        initial_position: Tuple[float, float] = (0.5, 0.5),
        failsafe: bool = True,
    ) -> None:
        self._screen_size = screen_size
        self._norm_x, self._norm_y = initial_position
        self.failsafe = CornerBreakoutFailSafe(enabled=failsafe)
        self.click_history: List[Tuple[float, float, str]] = []
        self.trajectory_history: List[Tuple[float, float]] = [(self._norm_x, self._norm_y)]

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._screen_size

    def get_position_pixels(self) -> Tuple[int, int]:
        w, h = self._screen_size
        return (int(round(self._norm_x * (w - 1))), int(round(self._norm_y * (h - 1))))

    def get_position_normalized(self) -> Tuple[float, float]:
        return (self._norm_x, self._norm_y)

    def move_to_pixels(self, x: int, y: int) -> None:
        w, h = self._screen_size
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))

        self.failsafe.check(clamped_x, clamped_y, w, h)
        self._norm_x = clamped_x / max(1, w - 1)
        self._norm_y = clamped_y / max(1, h - 1)
        self.trajectory_history.append((self._norm_x, self._norm_y))

    def move_to_normalized(self, nx: float, ny: float) -> None:
        w, h = self._screen_size
        clamped_nx = max(0.0, min(1.0, nx))
        clamped_ny = max(0.0, min(1.0, ny))
        px = int(round(clamped_nx * (w - 1)))
        py = int(round(clamped_ny * (h - 1)))
        self.failsafe.check(px, py, w, h)
        self._norm_x = clamped_nx
        self._norm_y = clamped_ny
        self.trajectory_history.append((self._norm_x, self._norm_y))

    def move_relative(self, dx: int, dy: int) -> None:
        px, py = self.get_position_pixels()
        self.move_to_pixels(px + dx, py + dy)

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        self.click_history.append((self._norm_x, self._norm_y, button))

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.click(button=button)
        self.click(button=button)


def get_default_driver(force_virtual: bool = False, failsafe: bool = True) -> OSCursorDriver:
    """Factory: selects optimal hardware driver for current environment."""
    if not force_virtual and sys.platform == "win32":
        try:
            from forge.vision.cortex.motor.win32 import Win32CursorDriver
            return Win32CursorDriver(failsafe=failsafe)
        except Exception:
            pass

    return VirtualCursorDriver(failsafe=failsafe)
