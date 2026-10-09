"""Native Windows Win32 OS Cursor Driver.

Hardened hardware-level cursor steering using Windows user32.dll & SendInput.
Features Per-Monitor V2 DPI awareness, high-precision timers, and multi-monitor support.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import time
from typing import Tuple

from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe

# Input Types
INPUT_MOUSE = 0

# Win32 Mouse Event Flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_CXSCREEN = 0
SM_CYSCREEN = 1
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Ctypes 64-bit SendInput Structures
ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_u", _INPUT_UNION),
    ]


class Win32CursorDriver:
    """Hardware cursor driver for Microsoft Windows with SendInput & DPI awareness."""

    def __init__(self, failsafe: bool = True) -> None:
        self.user32 = ctypes.windll.user32
        self._init_dpi_awareness()
        self._setup_ctypes_signatures()
        self._init_high_res_timer()
        self.failsafe = CornerBreakoutFailSafe(enabled=failsafe)

    def _init_dpi_awareness(self) -> None:
        """Enables Per-Monitor V2 DPI awareness to prevent coordinate scaling drift."""
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    self.user32.SetProcessDPIAware()
                except Exception:
                    pass

    def _setup_ctypes_signatures() -> None:
        pass

    def _setup_ctypes_signatures(self) -> None:
        """Declares strict 64-bit argument and return types for user32 APIs."""
        self.user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        self.user32.GetSystemMetrics.restype = ctypes.c_int

        self.user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self.user32.GetCursorPos.restype = wintypes.BOOL

        self.user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        self.user32.SetCursorPos.restype = wintypes.BOOL

        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT

    def _init_high_res_timer(self) -> None:
        """Requests 1ms OS scheduling resolution via winmm.dll."""
        try:
            self.winmm = ctypes.windll.winmm
            self.winmm.timeBeginPeriod.argtypes = [wintypes.UINT]
            self.winmm.timeBeginPeriod.restype = wintypes.UINT
            self.winmm.timeEndPeriod.argtypes = [wintypes.UINT]
            self.winmm.timeEndPeriod.restype = wintypes.UINT
            self.winmm.timeBeginPeriod(1)
        except Exception:
            self.winmm = None

    def __del__(self) -> None:
        if getattr(self, "winmm", None):
            try:
                self.winmm.timeEndPeriod(1)
            except Exception:
                pass

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Returns primary display resolution in physical pixels: (width, height)."""
        width = int(self.user32.GetSystemMetrics(SM_CXSCREEN))
        height = int(self.user32.GetSystemMetrics(SM_CYSCREEN))
        return (width, height)

    @property
    def virtual_screen_rect(self) -> Tuple[int, int, int, int]:
        """Returns virtual desktop bounds (x, y, width, height) across all monitors."""
        vx = int(self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        vy = int(self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        vw = int(self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        vh = int(self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        return (vx, vy, vw, vh)

    def get_position_pixels(self) -> Tuple[int, int]:
        """Returns current physical mouse coordinates in screen pixels."""
        pt = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y))

    def get_position_normalized(self) -> Tuple[float, float]:
        """Returns mouse position normalized to primary display [0.0, 1.0]."""
        px, py = self.get_position_pixels()
        w, h = self.screen_size
        nx = max(0.0, min(1.0, px / max(1, w - 1)))
        ny = max(0.0, min(1.0, py / max(1, h - 1)))
        return (nx, ny)

    def check_hardware_failsafe(self) -> None:
        """Queries physical mouse position and verifies safety fail-safe."""
        hw_x, hw_y = self.get_position_pixels()
        w, h = self.screen_size
        self.failsafe.check(hw_x, hw_y, w, h)

    def move_to_pixels(self, x: int, y: int) -> None:
        """Moves cursor to absolute pixel coordinates with fail-safe guard."""
        # 1. Guard against user breakout before dispatching movement
        self.check_hardware_failsafe()

        w, h = self.screen_size
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))

        # 2. Guard against commanding into breakout corner
        self.failsafe.check(clamped_x, clamped_y, w, h)
        self.user32.SetCursorPos(clamped_x, clamped_y)

    def move_to_normalized(self, nx: float, ny: float) -> None:
        w, h = self.screen_size
        px = int(round(nx * (w - 1)))
        py = int(round(ny * (h - 1)))
        self.move_to_pixels(px, py)

    def move_relative(self, dx: int, dy: int) -> None:
        curr_x, curr_y = self.get_position_pixels()
        self.move_to_pixels(curr_x + dx, curr_y + dy)

    def _send_mouse_input(self, flags: int, data: int = 0) -> None:
        """Dispatches serialized mouse event via SendInput."""
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi = MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=data,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def mouse_down(self, button: str = "left") -> None:
        self.check_hardware_failsafe()
        flag = (
            MOUSEEVENTF_RIGHTDOWN if button == "right" else
            MOUSEEVENTF_MIDDLEDOWN if button == "middle" else
            MOUSEEVENTF_LEFTDOWN
        )
        self._send_mouse_input(flag)

    def mouse_up(self, button: str = "left") -> None:
        flag = (
            MOUSEEVENTF_RIGHTUP if button == "right" else
            MOUSEEVENTF_MIDDLEUP if button == "middle" else
            MOUSEEVENTF_LEFTUP
        )
        self._send_mouse_input(flag)

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        self.check_hardware_failsafe()
        self.mouse_down(button=button)
        if hold_duration > 0:
            time.sleep(hold_duration)
        self.mouse_up(button=button)

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.click(button=button)
        time.sleep(interval)
        self.click(button=button)
