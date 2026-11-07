"""Windows Computer Observer & Biological Hardware Actuator.

Implements:
1. Dynamic binding to the active Windows input desktop (`winsta0\\default`).
2. Real-time screen capture scaled to canonical 1280x800 canvas with in-memory WebP/JPEG encoding.
3. Native Win32 SendInput execution with 5th-order minimum-jerk biological cursor physics.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import io
import math
import sys
from enum import Enum
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from avcs.ace.actions import Action, ActionKind

# Win32 Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0

# Native Win32 Structs
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUTunion),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


@dataclass
class Observation:
    """Canonical visual observation of the desktop."""
    image_bytes: bytes
    format: str
    width: int = 1280
    height: int = 800
    native_width: int = 1280
    native_height: int = 800
    capture_time_ms: float = 0.0


class SafetyRiskLevel(str, Enum):
    """Categorized risk levels for desktop actuation."""
    SAFE = "safe"
    CONSOLE_HAZARD = "console_hazard"
    FILE_DELETION = "permanent_file_deletion"
    FINANCIAL_PAYMENT = "financial_payment"
    OUTBOUND_COMMUNICATION = "outbound_communication"
    SENSITIVE_SYSTEM = "sensitive_system"


@dataclass
class SafetyCheckResult:
    """Result of an Execution Safety Firewall evaluation."""
    allowed: bool
    risk_level: SafetyRiskLevel
    reason: str
    requires_confirmation: bool = False


class ExecutionSafetyFirewall:
    """Execution Safety Firewall guarding against destructive or irreversible actions.

    Enforces P1 guardrails:
    1. Console protection: Blocks ungrounded terminal typing.
    2. Irreversible operation gating: Permanent file deletion, financial payments, and outbound messaging.
    """

    DELETION_PATTERNS = [
        r'\b(?:del|erase)\s+.*[/\-](?:f|q|s)\b',
        r'\b(?:rmdir|rd)\s+[/\-]s\b',
        r'\brm\s+-[rRfF]+\b',
        r'\bformat\s+[a-zA-Z]:',
        r'\bRemove-Item\b.*(?:-Recurse|-Force)\b',
        r'\b(?:del|erase|rmdir|rd|ri)\s+.*(?:-recurse|-force)\b',
        r'\b(?:sdelete|shred|wipe)\b',
        r'\bempty\s+(?:the\s+)?recycle\s*bin\b',
        r'\b(?:drop|truncate)\s+(?:database|table)\b',
        r'\b(?:diskpart|fdisk|mkfs)\b',
        r'\bgit\s+(?:clean\s+-[a-zA-Z]*f|reset\s+--hard)\b',
    ]

    FINANCIAL_PATTERNS = [
        r'\b(?:credit\s*card|cvv|cvc|card\s*number|expiry\s*date)\b',
        r'\b(?:pay\s*now|make\s*payment|send\s*payment|transfer\s*funds)\b',
        r'\b(?:authorize\s*transaction|confirm\s*purchase|place\s*order|buy\s*now)\b',
        r'\b(?:bank\s*account|routing\s*number|wire\s*transfer)\b',
        r'\b(?:checkout|confirm\s*order|complete\s*order|complete\s*purchase|proceed\s*to\s*pay)\b',
        r'\b(?:paypal|upi\s*id|crypto\s*transfer|bitcoin|ethereum)\b',
    ]

    COMMUNICATION_PATTERNS = [
        r'\b(?:send\s+(?:email|message|sms|dm|reply|tweet)|compose\s+email|post\s+(?:tweet|update|status|message)|send\s+to\s+channel)\b',
        r'\b(?:send\s*slack|discord\s*send|telegram\s*send|whatsapp\s*send)\b',
        r'\b(?:click\s+(?:send|submit|post|tweet|publish)\b)',
    ]

    CONSOLE_CLASSES = ("cascadia", "consolewindowclass", "windowsterminal", "mintty", "conemuwindowclass")

    def __init__(self, require_confirmation: bool = True) -> None:
        self.require_confirmation = require_confirmation

    def evaluate(
        self,
        action: Action,
        active_window_class: Optional[str] = None,
        user_confirmed: bool = False,
    ) -> SafetyCheckResult:
        """Evaluates an action for safety hazards and returns approval result."""
        # 1. Console protection: ungrounded typing into terminal
        if action.kind == ActionKind.TYPE and action.coordinate is None:
            if active_window_class and any(term in active_window_class.lower() for term in self.CONSOLE_CLASSES):
                return SafetyCheckResult(
                    allowed=False,
                    risk_level=SafetyRiskLevel.CONSOLE_HAZARD,
                    reason=f"Execution Safety Firewall: Active console '{active_window_class}' detected. Rejecting ungrounded typing into terminal to prevent corruption. Click target UI element first.",
                    requires_confirmation=False,
                )

        text_content = ""
        if action.text:
            text_content += " " + action.text
        if action.reason:
            text_content += " " + action.reason

        # 2. Check for Shift+Delete hotkey (permanent file deletion in Windows)
        if action.kind == ActionKind.KEY and action.keys:
            norm_keys = [k.lower().strip() for k in action.keys]
            if any(k in ("shift", "shift_l", "shift_r") for k in norm_keys) and any(k in ("delete", "del") for k in norm_keys):
                if not user_confirmed and self.require_confirmation:
                    return SafetyCheckResult(
                        allowed=False,
                        risk_level=SafetyRiskLevel.FILE_DELETION,
                        reason="Execution Safety Firewall: High-risk permanent file deletion shortcut (Shift+Delete) detected. Confirmation required before proceeding.",
                        requires_confirmation=True,
                    )

        # 3. Permanent file deletion patterns
        for pat in self.DELETION_PATTERNS:
            if re.search(pat, text_content, re.IGNORECASE):
                if not user_confirmed and self.require_confirmation:
                    return SafetyCheckResult(
                        allowed=False,
                        risk_level=SafetyRiskLevel.FILE_DELETION,
                        reason="Execution Safety Firewall: High-risk permanent file deletion detected. Confirmation required before proceeding.",
                        requires_confirmation=True,
                    )

        # 4. Financial / Payment patterns
        for pat in self.FINANCIAL_PATTERNS:
            if re.search(pat, text_content, re.IGNORECASE):
                if not user_confirmed and self.require_confirmation:
                    return SafetyCheckResult(
                        allowed=False,
                        risk_level=SafetyRiskLevel.FINANCIAL_PAYMENT,
                        reason="Execution Safety Firewall: High-risk financial/payment action detected. Confirmation required before proceeding.",
                        requires_confirmation=True,
                    )

        # 5. Outbound communications
        for pat in self.COMMUNICATION_PATTERNS:
            if re.search(pat, text_content, re.IGNORECASE):
                if not user_confirmed and self.require_confirmation:
                    return SafetyCheckResult(
                        allowed=False,
                        risk_level=SafetyRiskLevel.OUTBOUND_COMMUNICATION,
                        reason="Execution Safety Firewall: High-risk outbound communication detected. Confirmation required before proceeding.",
                        requires_confirmation=True,
                    )

        return SafetyCheckResult(
            allowed=True,
            risk_level=SafetyRiskLevel.SAFE,
            reason="Action approved by Execution Safety Firewall.",
            requires_confirmation=False,
        )


class WindowsComputer:
    """Real-time Windows Desktop Observer and Biological Actuator."""

    CANONICAL_WIDTH = 1280
    CANONICAL_HEIGHT = 800

    # Calibrated Fitts's Law duration boundaries: 50 ms to 120 ms at 100 Hz
    FITTS_MIN_DURATION_S = 0.050
    FITTS_MAX_DURATION_S = 0.120

    def calculate_fitts_duration(self, dist: float) -> float:
        """Calculates 5th-order minimum-jerk flight duration calibrated to 50ms - 120ms at 100 Hz.

        Eliminates artificial latency while ensuring biological minimum-jerk smoothness without teleportation.
        """
        raw_dur = 0.050 + 0.00014 * dist
        return min(self.FITTS_MAX_DURATION_S, max(self.FITTS_MIN_DURATION_S, raw_dur))

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsComputer requires Windows OS.")
        
        # Enable Per-Monitor V2 DPI awareness for true physical pixel capture
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.kernel32 = getattr(ctypes.windll, "kernel32", None)
        self._orig_hdesk: Optional[int] = None
        if self.kernel32 and hasattr(self.user32, "GetThreadDesktop"):
            try:
                self._orig_hdesk = self.user32.GetThreadDesktop(self.kernel32.GetCurrentThreadId())
            except Exception:
                self._orig_hdesk = None
        self._hdesk: Optional[int] = None
        
        # Attach to interactive desktop station
        self.attach_desktop()
        
        # Cache native screen dimensions
        self.native_width = self.user32.GetSystemMetrics(0)
        self.native_height = self.user32.GetSystemMetrics(1)
        if self.native_width <= 0 or self.native_height <= 0:
            self.native_width, self.native_height = 1280, 800

    def attach_desktop(self) -> None:
        """Binds thread to the active user desktop station (winsta0\\default)."""
        try:
            hdesk = self.user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                self.user32.SetThreadDesktop(hdesk)
                old_hdesk = getattr(self, "_hdesk", None)
                if old_hdesk and old_hdesk != hdesk:
                    try:
                        self.user32.CloseDesktop(old_hdesk)
                    except Exception:
                        pass
                self._hdesk = hdesk
        except Exception:
            pass

    def close(self) -> None:
        """Releases the desktop handle and cleans resources."""
        try:
            self.emergency_release_hardware()
        except Exception:
            pass
        old_hdesk = getattr(self, "_hdesk", None)
        orig_hdesk = getattr(self, "_orig_hdesk", None)
        user32 = getattr(self, "user32", None)
        if user32:
            if orig_hdesk:
                try:
                    user32.SetThreadDesktop(orig_hdesk)
                except Exception:
                    pass
            if old_hdesk:
                try:
                    user32.CloseDesktop(old_hdesk)
                except Exception:
                    pass
        self._hdesk = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "WindowsComputer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_cursor_pos(self) -> Tuple[int, int]:
        """Returns current hardware mouse position in native screen pixels."""
        pt = ctypes.wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def observe(self, format: str = "jpeg") -> Observation:
        """Captures desktop, scales to 1280x800, and encodes to in-memory bytes in <8 ms."""
        t0 = time.time()
        self.attach_desktop()

        # Update dimensions in case resolution or display scaling dynamically changed
        nw = self.user32.GetSystemMetrics(0)
        nh = self.user32.GetSystemMetrics(1)
        if nw > 0 and nh > 0:
            self.native_width, self.native_height = nw, nh

        hdc_screen = self.user32.GetDC(0)
        hdc_mem = self.gdi32.CreateCompatibleDC(hdc_screen)
        hbm = self.gdi32.CreateCompatibleBitmap(hdc_screen, self.native_width, self.native_height)
        self.gdi32.SelectObject(hdc_mem, hbm)

        # BitBlt capture with layered window support (CAPTUREBLT)
        self.gdi32.BitBlt(hdc_mem, 0, 0, self.native_width, self.native_height, hdc_screen, 0, 0, SRCCOPY | CAPTUREBLT)
        self.gdi32.GdiFlush()

        # Extract bits
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = self.native_width
        bmi.bmiHeader.biHeight = -self.native_height  # Top-down DIB
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        buffer_len = self.native_width * self.native_height * 4
        raw_bytes = (ctypes.c_char * buffer_len)()
        self.gdi32.GetDIBits(hdc_mem, hbm, 0, self.native_height, raw_bytes, ctypes.byref(bmi), DIB_RGB_COLORS)

        # Clean GDI handles
        self.user32.ReleaseDC(0, hdc_screen)
        self.gdi32.DeleteDC(hdc_mem)
        self.gdi32.DeleteObject(hbm)

        # Construct image and scale to canonical WXGA (1280x800)
        img = Image.frombuffer("RGBA", (self.native_width, self.native_height), raw_bytes, "raw", "BGRA", 0, 1).convert("RGB")
        if (self.native_width, self.native_height) != (self.CANONICAL_WIDTH, self.CANONICAL_HEIGHT):
            img = img.resize((self.CANONICAL_WIDTH, self.CANONICAL_HEIGHT), Image.Resampling.BILINEAR)

        # Encode in-memory
        buf = io.BytesIO()
        if format.lower() == "webp":
            try:
                img.save(buf, format="WEBP", quality=80, method=0)
                fmt_out = "webp"
            except Exception:
                img.save(buf, format="JPEG", quality=85)
                fmt_out = "jpeg"
        else:
            img.save(buf, format="JPEG", quality=85)
            fmt_out = "jpeg"

        t1 = time.time()
        return Observation(
            image_bytes=buf.getvalue(),
            format=fmt_out,
            width=self.CANONICAL_WIDTH,
            height=self.CANONICAL_HEIGHT,
            native_width=self.native_width,
            native_height=self.native_height,
            capture_time_ms=(t1 - t0) * 1000.0,
        )

    def _canonical_to_native(self, x: int, y: int) -> Tuple[int, int]:
        """Maps coordinate from canonical 1280x800 to native hardware display pixels."""
        nx = int((x / self.CANONICAL_WIDTH) * self.native_width)
        ny = int((y / self.CANONICAL_HEIGHT) * self.native_height)
        nx = max(0, min(self.native_width - 1, nx))
        ny = max(0, min(self.native_height - 1, ny))
        return nx, ny

    def _move_minimum_jerk(self, target_x: int, target_y: int, duration_ms: Optional[float] = None) -> None:
        """Moves hardware cursor along 5th-order minimum-jerk trajectory with dual dispatch.

        Simultaneously updates SetCursorPos AND sends raw input motion via SendInput
        with MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK so
        Windows 11 XAML Island (DesktopWindowXamlSource) receives real WM_MOUSEMOVE
        and WM_POINTERENTER hover events. Movement duration is scaled via Fitts's Law
        at ~100 Hz.
        """
        start_x, start_y = self.get_cursor_pos()
        dist = math.hypot(target_x - start_x, target_y - start_y)
        if dist < 2.0:
            self.user32.SetCursorPos(target_x, target_y)
            self._send_mouse_event(MOUSEEVENTF_MOVE, x=target_x, y=target_y)
            return

        if duration_ms is not None and duration_ms > 0:
            duration_sec = duration_ms / 1000.0
        else:
            # Calibrated Fitts's Law duration clamp: 50 ms - 120 ms (0.05s - 0.12s) at 100 Hz
            duration_sec = self.calculate_fitts_duration(dist)

        # ~100 Hz sampling rate
        steps = max(5, int(round(duration_sec * 100.0)))
        dt = duration_sec / steps

        for step in range(1, steps + 1):
            tau = step / float(steps)
            poly = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            cur_x = int(round(start_x + (target_x - start_x) * poly))
            cur_y = int(round(start_y + (target_y - start_y) * poly))
            self.user32.SetCursorPos(cur_x, cur_y)
            self._send_mouse_event(MOUSEEVENTF_MOVE, x=cur_x, y=cur_y)
            time.sleep(dt)

        self.user32.SetCursorPos(target_x, target_y)
        self._send_mouse_event(MOUSEEVENTF_MOVE, x=target_x, y=target_y)

    def _send_mouse_event(self, flags: int, data: int = 0, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Sends native mouse input event via SendInput with normalized absolute coordinates."""
        inp = INPUT()
        inp.type = INPUT_MOUSE
        actual_flags = flags

        if x is None or y is None:
            cx, cy = self.get_cursor_pos()
            if x is None:
                x = cx
            if y is None:
                y = cy

        vx = self.user32.GetSystemMetrics(76) if self.user32 else 0
        vy = self.user32.GetSystemMetrics(77) if self.user32 else 0
        vw = self.user32.GetSystemMetrics(78) if self.user32 else 0
        vh = self.user32.GetSystemMetrics(79) if self.user32 else 0
        if vw <= 0 or vh <= 0:
            vx, vy = 0, 0
            vw, vh = self.native_width, self.native_height

        norm_x = min(65535, max(0, int(round(((x - vx) * 65535.0) / max(1, vw - 1)))))
        norm_y = min(65535, max(0, int(round(((y - vy) * 65535.0) / max(1, vh - 1)))))
        inp.union.mi.dx = norm_x
        inp.union.mi.dy = norm_y
        actual_flags |= (MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)

        inp.union.mi.dwFlags = actual_flags
        inp.union.mi.mouseData = data
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _send_unicode_char(self, char: str) -> None:
        """Sends clean unicode keystroke via SendInput."""
        code = ord(char)
        # Key down
        inp_down = INPUT()
        inp_down.type = INPUT_KEYBOARD
        inp_down.union.ki.wScan = code
        inp_down.union.ki.dwFlags = KEYEVENTF_UNICODE
        # Key up
        inp_up = INPUT()
        inp_up.type = INPUT_KEYBOARD
        inp_up.union.ki.wScan = code
        inp_up.union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        inputs = (INPUT * 2)(inp_down, inp_up)
        self.user32.SendInput(2, inputs, ctypes.sizeof(INPUT))

    EXTENDED_VKS = {
        0x26, 0x28, 0x25, 0x27,  # Up, Down, Left, Right
        0x24, 0x23, 0x21, 0x22,  # Home, End, Prior, Next
        0x2D, 0x2E,              # Insert, Delete
        0x5B, 0x5C,              # LWin, RWin
        0xA3, 0xA5,              # RCtrl, RAlt
    }

    def _send_vk(self, vk_code: int, keyup: bool = False) -> None:
        """Sends virtual key event including hardware scan code."""
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk_code

        scan_code = 0
        if hasattr(self.user32, "MapVirtualKeyW"):
            try:
                scan_code = self.user32.MapVirtualKeyW(vk_code, 0)  # MAPVK_VK_TO_VSC = 0
            except Exception:
                scan_code = 0
        inp.union.ki.wScan = scan_code & 0xFFFF

        flags = KEYEVENTF_KEYUP if keyup else 0
        if vk_code in self.EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        inp.union.ki.dwFlags = flags
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def emergency_release_hardware(self) -> None:
        """Failsafe release of any held mouse buttons, modifier keys, and cursor clipping."""
        # Release mouse buttons with current cursor position
        try:
            cx, cy = self.get_cursor_pos()
            for flag in (MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_MIDDLEUP):
                try:
                    self._send_mouse_event(flag, x=cx, y=cy)
                except Exception:
                    pass
        except Exception:
            pass

        # Release modifier keys if held down
        modifier_vks = [
            0x5B, 0x5C,        # LWin, RWin
            0x11, 0xA2, 0xA3,  # Ctrl, LCtrl, RCtrl
            0x12, 0xA4, 0xA5,  # Alt, LAlt, RAlt
            0x10, 0xA0, 0xA1,  # Shift, LShift, RShift
        ]
        for vk in modifier_vks:
            try:
                if self.user32 and (self.user32.GetAsyncKeyState(vk) & 0x8000):
                    self._send_vk(vk, keyup=True)
            except Exception:
                pass

        # Reset cursor clipping
        try:
            if self.user32 and hasattr(self.user32, "ClipCursor"):
                self.user32.ClipCursor(None)
        except Exception:
            pass

    def _resolve_vk(self, key: str) -> Optional[int]:
        """Resolves a key name or character to a Windows Virtual-Key code."""
        k = key.lower().strip()
        vk_map = {
            "enter": 0x0D, "return": 0x0D,
            "tab": 0x09, "space": 0x20,
            "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
            "escape": 0x1B, "esc": 0x1B,
            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            "ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12,
            "win": 0x5B, "windows": 0x5B, "super": 0x5B, "cmd": 0x5B, "command": 0x5B,
            "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
            "page_up": 0x21, "page_down": 0x22,
            "insert": 0x2D,
            "capslock": 0x14, "caps_lock": 0x14,
            "prtsc": 0x2C, "printscreen": 0x2C, "print_screen": 0x2C,
            "super_l": 0x5B, "super_r": 0x5C, "lwin": 0x5B, "rwin": 0x5C,
            "ctrl_l": 0x11, "ctrl_r": 0x11, "control_l": 0x11, "control_r": 0x11,
            "alt_l": 0x12, "alt_r": 0x12, "shift_l": 0x10, "shift_r": 0x10,
            "menu": 0x5D,
            "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
            "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
        }
        if k in vk_map:
            return vk_map[k]
        k_clean = k.replace("_", "")
        if k_clean in vk_map:
            return vk_map[k_clean]
        if len(k) == 1 and k.isalnum():
            return ord(k.upper())
        if len(k) == 1:
            try:
                vk_scan = self.user32.VkKeyScanW(ord(k))
                if vk_scan != -1:
                    return vk_scan & 0xFF
            except Exception:
                pass
        return None

    def execute(self, action: Action, dry_run: bool = False) -> None:
        """Executes physical OS action with native Win32 hardware events."""
        if action.kind == ActionKind.STOP:
            return

        if dry_run:
            if action.coordinate:
                self._canonical_to_native(*action.coordinate)
            return

        self.attach_desktop()

        if action.kind in (ActionKind.LEFT_CLICK, ActionKind.RIGHT_CLICK, ActionKind.DOUBLE_CLICK, ActionKind.TRIPLE_CLICK):
            if action.coordinate:
                nx, ny = self._canonical_to_native(*action.coordinate)
                self._move_minimum_jerk(nx, ny)
                time.sleep(0.035)  # 35ms hover dwell
            else:
                nx, ny = self.get_cursor_pos()

            if action.kind == ActionKind.LEFT_CLICK:
                self._send_mouse_event(MOUSEEVENTF_LEFTDOWN, x=nx, y=ny)
                time.sleep(0.065)  # 65ms hold duration
                self._send_mouse_event(MOUSEEVENTF_LEFTUP, x=nx, y=ny)
            elif action.kind == ActionKind.RIGHT_CLICK:
                self._send_mouse_event(MOUSEEVENTF_RIGHTDOWN, x=nx, y=ny)
                time.sleep(0.065)
                self._send_mouse_event(MOUSEEVENTF_RIGHTUP, x=nx, y=ny)
            elif action.kind == ActionKind.DOUBLE_CLICK:
                for _ in range(2):
                    self._send_mouse_event(MOUSEEVENTF_LEFTDOWN, x=nx, y=ny)
                    time.sleep(0.065)
                    self._send_mouse_event(MOUSEEVENTF_LEFTUP, x=nx, y=ny)
                    time.sleep(0.05)
            elif action.kind == ActionKind.TRIPLE_CLICK:
                for _ in range(3):
                    self._send_mouse_event(MOUSEEVENTF_LEFTDOWN, x=nx, y=ny)
                    time.sleep(0.065)
                    self._send_mouse_event(MOUSEEVENTF_LEFTUP, x=nx, y=ny)
                    time.sleep(0.04)

        elif action.kind == ActionKind.MOUSE_MOVE:
            if action.coordinate:
                nx, ny = self._canonical_to_native(*action.coordinate)
                self._move_minimum_jerk(nx, ny)

        elif action.kind == ActionKind.LEFT_CLICK_DRAG:
            target_end = action.end_coordinate or action.coordinate
            if action.start_coordinate and target_end:
                sx, sy = self._canonical_to_native(*action.start_coordinate)
                ex, ey = self._canonical_to_native(*target_end)
            elif target_end:
                sx, sy = self.get_cursor_pos()
                ex, ey = self._canonical_to_native(*target_end)
            else:
                sx, sy = self.get_cursor_pos()
                ex, ey = sx, sy
            self._move_minimum_jerk(sx, sy)
            time.sleep(0.035)
            self._send_mouse_event(MOUSEEVENTF_LEFTDOWN, x=sx, y=sy)
            time.sleep(0.065)
            self._move_minimum_jerk(ex, ey)
            time.sleep(0.035)
            self._send_mouse_event(MOUSEEVENTF_LEFTUP, x=ex, y=ey)

        elif action.kind == ActionKind.SCROLL:
            if action.coordinate:
                nx, ny = self._canonical_to_native(*action.coordinate)
                self._move_minimum_jerk(nx, ny)
                time.sleep(0.035)
            else:
                nx, ny = self.get_cursor_pos()
            if action.scroll_delta:
                dx, dy = action.scroll_delta
                if dy != 0:
                    self._send_mouse_event(MOUSEEVENTF_WHEEL, data=dy, x=nx, y=ny)
                if dx != 0:
                    self._send_mouse_event(MOUSEEVENTF_HWHEEL, data=dx, x=nx, y=ny)
            else:
                self._send_mouse_event(MOUSEEVENTF_WHEEL, data=-120, x=nx, y=ny)

        elif action.kind == ActionKind.TYPE:
            if action.coordinate:
                nx, ny = self._canonical_to_native(*action.coordinate)
                self._move_minimum_jerk(nx, ny)
                time.sleep(0.035)
                self._send_mouse_event(MOUSEEVENTF_LEFTDOWN, x=nx, y=ny)
                time.sleep(0.065)
                self._send_mouse_event(MOUSEEVENTF_LEFTUP, x=nx, y=ny)
                time.sleep(0.05)

            if action.text:
                for char in action.text:
                    if char in ("\r", "\n"):
                        self._send_vk(0x0D, keyup=False)
                        time.sleep(0.01)
                        self._send_vk(0x0D, keyup=True)
                    elif char == "\t":
                        self._send_vk(0x09, keyup=False)
                        time.sleep(0.01)
                        self._send_vk(0x09, keyup=True)
                    else:
                        self._send_unicode_char(char)
                    time.sleep(0.005)

            if action.press_enter:
                time.sleep(0.02)
                self._send_vk(0x0D, keyup=False)
                time.sleep(0.01)
                self._send_vk(0x0D, keyup=True)

        elif action.kind == ActionKind.KEY:
            if action.keys:
                pressed_vks = []
                for k in action.keys:
                    vk = self._resolve_vk(k)
                    if vk:
                        self._send_vk(vk, keyup=False)
                        pressed_vks.append(vk)
                        time.sleep(0.01)
                for vk in reversed(pressed_vks):
                    self._send_vk(vk, keyup=True)
                    time.sleep(0.01)

        elif action.kind == ActionKind.WAIT:
            time.sleep(max(0.1, action.duration))

        # Settlement delay
        time.sleep(0.02)

