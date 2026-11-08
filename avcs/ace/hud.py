"""AVCS Native Real-Time Desktop HUD & On-Screen Visual Feedback.

Provides an ultra-aesthetic, Apple MacBook / iPhone-grade Dynamic Island:
1. Dynamic Size & Fluid Spring Morphing:
   - Compact Idle / Ready (~220px wide, ~36px high): `(🟢) AVCS ACE • READY`
   - Thinking / Planning (~380-420px wide, ~38px high): `(⚡) PLANNING │ Decomposing goal into steps...`
   - Active Motor Execution (~620-660px wide, ~42px high): `🖱️ CLICK │ Step 1/3: Open Start Menu │ [LIVE] 45ms (●)`
   - Verified / Done (~290px wide, ~38px high): `(✓) COMPLETE │ Verified Successfully`
   - Halted / Error (~440px wide, ~38px high): `(✕) HALTED │ Error message`
   - Smooth exponential spring interpolation on timer ticks:
     `current_w += (target_w - current_w) * 0.22`
2. MacBook Obsidian Glass Aesthetics:
   - Deep obsidian black core (`#060709`) with true capsule curvature (`border_radius = height // 2`).
   - Frosted subtle rim hairline (`#2E3038`) plus an ultra-thin top-edge curved glass light highlight.
   - Double-buffered GDI rendering (`CreateCompatibleDC` + `CreateCompatibleBitmap`) for 100% flicker-free rendering.
3. Living Pulsing Breathing Orb:
   - Smooth sinusoidal glowing indicator dot (`sin(time * 4)`) pulsing with state color:
     Emerald for LIVE/READY, Cyan for PLANNING, Electric Amber for MOTOR/CLICK, Violet for VERIFY, Crimson for ABORT.
4. Sub-Goal Progress Micro-Dots & Hairline Track:
   - 2px hairline progress track along the bottom edge with subtle glowing micro-dots showing step progress (`●──○──○`).
5. Modern ClearType Typography:
   - `Segoe UI` fonts with `CLEARTYPE_QUALITY` for crisp, publication-grade text rendering.
6. Fail-Closed Hardware Invariants:
   - Click-through (`WS_EX_TRANSPARENT`), non-activating (`WS_EX_NOACTIVATE`), topmost layered overlay.
   - 100% invisible to BitBlt/AVCS screen perception via `WDA_EXCLUDEFROMCAPTURE`.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import math
import sys
import threading
import time
from typing import Optional, Tuple, Union

logger = logging.getLogger("avcs.ace.hud")

# Win32 Window Styles & Extended Styles
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000

LWA_COLORKEY = 0x00000001
LWA_ALPHA = 0x00000002
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10 2004+ / Windows 11 capture exclusion
SW_SHOWNOACTIVATE = 4
SW_HIDE = 0

WM_PAINT = 0x000F
WM_ERASEBKGND = 0x0014
WM_TIMER = 0x0113
WM_DESTROY = 0x0002

# GDI DrawText Flags
DT_LEFT = 0x0000
DT_CENTER = 0x0001
DT_RIGHT = 0x0002
DT_VCENTER = 0x0004
DT_SINGLELINE = 0x0020
DT_NOPREFIX = 0x0800
DT_END_ELLIPSIS = 0x8000

# Color Constants (Win32 BGR: 0x00BBGGRR)
COLOR_TRANSPARENT_KEY = 0x00000001  # Near-black transparent colorkey (RGB 1, 0, 0)
COLOR_OBSIDIAN = 0x00090706         # Core: #060709 (R=6, G=7, B=9)
COLOR_RIM = 0x0038302E              # Frosted rim: #2E3038 (R=46, G=48, B=56)
COLOR_HIGHLIGHT = 0x00554A44        # Subtle top-edge curved glass light highlight
COLOR_TRACK_BG = 0x00241E1A         # Progress track background: #1A1E24
COLOR_TRACK_DOT_OFF = 0x00453B34    # Upcoming subgoal dot: #343B45
COLOR_DIVIDER = 0x004A3C34          # Subtle vertical separator line

# Typography Text Colors
COLOR_TEXT_PRIMARY = 0x00FFFFFF     # Crisp White: #FFFFFF
COLOR_TEXT_SECONDARY = 0x00D6D3D1   # Soft Silver: #D1D3D6
COLOR_TEXT_MUTED = 0x009CA3AF       # Muted Gray: #AFA39C

# State Accent Colors (BGR format: primary, core, glow)
STATE_COLORS = {
    "LIVE": {
        "primary": 0x0081B910,      # Emerald #10B981
        "core": 0x0099D334,         # Bright Emerald #34D399
        "glow": 0x003B4E06,         # Dark Emerald glow
    },
    "PLANNING": {
        "primary": 0x00D4B606,      # Electric Cyan #06B6D4
        "core": 0x00F8BD38,         # Bright Cyan #38BDF8
        "glow": 0x0090740E,         # Cyan glow
    },
    "MOTOR": {
        "primary": 0x000B9EF5,      # Electric Amber #F59E0B
        "core": 0x0024BFFB,         # Bright Amber #FBBF24
        "glow": 0x000953B4,         # Amber glow
    },
    "VERIFIED": {
        "primary": 0x00F755A8,      # Electric Violet #A855F7
        "core": 0x00FC84C0,         # Bright Violet #C084FC
        "glow": 0x00D9286D,         # Violet glow
    },
    "VOICE": {
        "primary": 0x00E040B0,      # Siri Electric Pink/Purple #B040E0
        "core": 0x00FF80D0,         # Bright Magenta-Pink #D080FF
        "glow": 0x00802060,         # Deep Magenta glow
    },
    "ERROR": {
        "primary": 0x004444EF,      # Crimson Red #EF4444
        "core": 0x007171F8,         # Bright Crimson #F87171
        "glow": 0x001B1B99,         # Crimson glow
    },
    "SIMULATION": {
        "primary": 0x0008B3EA,      # Warm Gold #EAB308
        "core": 0x0015CCFA,         # Bright Gold #FACC15
        "glow": 0x000762A1,         # Gold glow
    },
}


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", ctypes.c_void_p),
        ("fErase", ctypes.c_int),
        ("rcPaint", ctypes.wintypes.RECT),
        ("fRestore", ctypes.c_int),
        ("fIncUpdate", ctypes.c_int),
        ("rgbReserved", ctypes.c_byte * 32),
    ]


def _setup_win32_signatures() -> None:
    """Configures 64-bit argument and return types for Win32 User32 and GDI32 APIs."""
    if sys.platform != "win32":
        return
    try:
        u32 = ctypes.windll.user32
        gdi = ctypes.windll.gdi32
        k32 = ctypes.windll.kernel32

        u32.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t]
        u32.DefWindowProcW.restype = ctypes.c_size_t

        k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        k32.GetModuleHandleW.restype = ctypes.c_void_p

        u32.BeginPaint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        u32.BeginPaint.restype = ctypes.c_void_p
        u32.EndPaint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        u32.EndPaint.restype = ctypes.c_bool

        u32.FillRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        u32.FillRect.restype = ctypes.c_int

        u32.DrawTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        u32.DrawTextW.restype = ctypes.c_int

        u32.InvalidateRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        u32.InvalidateRect.restype = ctypes.c_bool

        u32.CreateWindowExW.argtypes = [
            ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        u32.CreateWindowExW.restype = ctypes.c_void_p

        u32.RegisterClassExW.argtypes = [ctypes.c_void_p]
        u32.RegisterClassExW.restype = ctypes.c_ushort

        gdi.CreateSolidBrush.argtypes = [ctypes.c_uint32]
        gdi.CreateSolidBrush.restype = ctypes.c_void_p

        gdi.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
        gdi.CreatePen.restype = ctypes.c_void_p

        gdi.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi.SelectObject.restype = ctypes.c_void_p

        gdi.GetStockObject.argtypes = [ctypes.c_int]
        gdi.GetStockObject.restype = ctypes.c_void_p

        gdi.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi.DeleteObject.restype = ctypes.c_bool

        gdi.DeleteDC.argtypes = [ctypes.c_void_p]
        gdi.DeleteDC.restype = ctypes.c_bool

        gdi.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        gdi.CreateCompatibleDC.restype = ctypes.c_void_p

        gdi.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gdi.CreateCompatibleBitmap.restype = ctypes.c_void_p

        gdi.BitBlt.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32
        ]
        gdi.BitBlt.restype = ctypes.c_bool

        gdi.RoundRect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        gdi.RoundRect.restype = ctypes.c_bool

        gdi.Ellipse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        gdi.Ellipse.restype = ctypes.c_bool

        gdi.MoveToEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        gdi.MoveToEx.restype = ctypes.c_bool

        gdi.LineTo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gdi.LineTo.restype = ctypes.c_bool

        gdi.SetBkMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        gdi.SetBkMode.restype = ctypes.c_int

        gdi.SetTextColor.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        gdi.SetTextColor.restype = ctypes.c_uint32

        u32.SetLayeredWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_byte, ctypes.c_ulong]
        u32.SetLayeredWindowAttributes.restype = ctypes.c_bool

        u32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u32.ShowWindow.restype = ctypes.c_bool

        u32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
        u32.SetTimer.restype = ctypes.c_size_t

        u32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        u32.KillTimer.restype = ctypes.c_bool

        u32.GetMessageW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        u32.GetMessageW.restype = ctypes.c_int

        u32.TranslateMessage.argtypes = [ctypes.c_void_p]
        u32.TranslateMessage.restype = ctypes.c_bool

        u32.DispatchMessageW.argtypes = [ctypes.c_void_p]
        u32.DispatchMessageW.restype = ctypes.c_size_t

        u32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64, ctypes.c_int64]
        u32.PostMessageW.restype = ctypes.c_bool

        u32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        u32.LoadCursorW.restype = ctypes.c_void_p

        if hasattr(u32, "SetWindowDisplayAffinity"):
            u32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            u32.SetWindowDisplayAffinity.restype = ctypes.c_bool

        u32.SetWindowPos.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint
        ]
        u32.SetWindowPos.restype = ctypes.c_bool

        u32.UpdateWindow.argtypes = [ctypes.c_void_p]
        u32.UpdateWindow.restype = ctypes.c_bool

        u32.GetSystemMetrics.argtypes = [ctypes.c_int]
        u32.GetSystemMetrics.restype = ctypes.c_int

        u32.GetDC.argtypes = [ctypes.c_void_p]
        u32.GetDC.restype = ctypes.c_void_p

        u32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        u32.ReleaseDC.restype = ctypes.c_int

        gdi.GetDIBits.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint
        ]
        gdi.GetDIBits.restype = ctypes.c_int

        gdi.CreateFontW.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_wchar_p
        ]
        gdi.CreateFontW.restype = ctypes.c_void_p
    except Exception:
        pass


def _unpack_bgr(color_int: int) -> Tuple[int, int, int]:
    """Unpacks a Win32 0x00BBGGRR integer into (R, G, B) components."""
    r = color_int & 0xFF
    g = (color_int >> 8) & 0xFF
    b = (color_int >> 16) & 0xFF
    return r, g, b


def _pack_bgr(r: int, g: int, b: int) -> int:
    """Packs (R, G, B) into a Win32 0x00BBGGRR integer."""
    return (int(r) & 0xFF) | ((int(g) & 0xFF) << 8) | ((int(b) & 0xFF) << 16)


def _blend_bgr(c1: int, c2: int, factor: float) -> int:
    """Blends between two BGR colors. factor=0.0 returns c2, factor=1.0 returns c1."""
    r1, g1, b1 = _unpack_bgr(c1)
    r2, g2, b2 = _unpack_bgr(c2)
    f = max(0.0, min(1.0, float(factor)))
    r = int(r2 + (r1 - r2) * f)
    g = int(g2 + (g1 - g2) * f)
    b = int(b2 + (b1 - b2) * f)
    return _pack_bgr(r, g, b)


_setup_win32_signatures()


def play_acoustic_earcon(kind: str) -> None:
    """Dispatches asynchronous non-blocking audio earcons via winsound on Windows."""
    if sys.platform != "win32":
        return

    def _worker():
        try:
            import winsound
            if kind == "start":
                winsound.Beep(520, 60)
                winsound.Beep(780, 80)
            elif kind == "action":
                winsound.Beep(1200, 25)
            elif kind == "verify":
                winsound.Beep(880, 40)
                winsound.Beep(1100, 50)
            elif kind == "success":
                winsound.Beep(523, 70)
                winsound.Beep(659, 70)
                winsound.Beep(784, 120)
            elif kind == "alert":
                winsound.Beep(330, 180)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


class AVCSVisualHUD:
    """Native Windows 11 Hardware Dynamic Island HUD & Live Desktop Feedback Overlay."""

    def __init__(self, title: str = "AVCS Autopilot HUD", exclude_from_capture: bool = True) -> None:
        self.title = title
        self.exclude_from_capture = exclude_from_capture
        self.hwnd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # State Telemetry
        self.mode_label = "LIVE"
        self.status_text = "READY"
        self.subgoal_text = ""
        self.step_info = "0/0"
        self.latency_ms = 0.0

        # Dynamic Island Capsule Morphing State
        self.current_w: float = 220.0
        self.current_h: float = 36.0
        self.target_w: float = 220.0
        self.target_h: float = 36.0
        self.morph_spring_k: float = 0.22

        # Click reticle animation state
        self.reticle_coord: Optional[Tuple[int, int]] = None
        self.reticle_start_time: float = 0.0
        self.reticle_duration: float = 0.35  # 350ms pulse
        self.reticle_kind = "click"

        # Double Buffering & GDI Handles
        self._mem_dc: Optional[int] = None
        self._mem_bm: Optional[int] = None
        self._old_bm: Optional[int] = None
        self._cached_w: int = 0
        self._cached_h: int = 0

        # Segoe UI Font Handles
        self._hfont_bold: Optional[int] = None
        self._hfont_medium: Optional[int] = None
        self._hfont_chip: Optional[int] = None

        # Display dimensions & multi-monitor virtual desktop bounds
        self.screen_x = 0
        self.screen_y = 0
        self.screen_w = 1920
        self.screen_h = 1200
        if sys.platform == "win32":
            try:
                u32 = ctypes.windll.user32
                vx = u32.GetSystemMetrics(76)
                vy = u32.GetSystemMetrics(77)
                vw = u32.GetSystemMetrics(78)
                vh = u32.GetSystemMetrics(79)
                if vw > 0 and vh > 0:
                    self.screen_x = vx
                    self.screen_y = vy
                    self.screen_w = vw
                    self.screen_h = vh
                else:
                    self.screen_w = u32.GetSystemMetrics(0) or 1920
                    self.screen_h = u32.GetSystemMetrics(1) or 1200
            except Exception:
                pass

    @staticmethod
    def classify_state(status: str) -> str:
        """Classifies the raw autopilot status string into a Dynamic Island visual state."""
        s = (status or "").upper().strip()
        if any(k in s for k in ("HALT", "ERROR", "FAIL", "ABORT")):
            return "ERROR"
        if any(k in s for k in ("VERIF", "COMPLETE", "SUCCESS", "DONE", "FINISHED")):
            return "VERIFIED"
        if any(k in s for k in ("SPEAK", "VOICE", "TALK", "LISTEN", "AUDIO")):
            return "VOICE"
        if any(k in s for k in ("CLICK", "TYPE", "KEY", "HOTKEY", "SCROLL", "DRAG", "PRESS", "MOVE", "ACTUAT")):
            return "MOTOR"
        if any(k in s for k in ("PLAN", "THINK", "OBSERV", "LOOKUP", "RESOLV")):
            return "PLANNING"
        if s in ("READY", "IDLE", "STANDBY", "WAITING", "INITIALIZING", "STARTING", ""):
            return "IDLE"
        return "PLANNING" if len(s) > 8 else "IDLE"

    @staticmethod
    def calculate_target_size(state: str, subgoal: str = "") -> Tuple[int, int]:
        """Calculates the target pixel width and height for each Dynamic Island state."""
        sub_len = len((subgoal or "").strip())
        if state == "IDLE":
            return 220, 36
        elif state == "PLANNING":
            w = max(380, min(520, 200 + sub_len * 7)) if sub_len > 0 else 380
            return w, 38
        elif state == "MOTOR":
            w = max(620, min(690, 380 + sub_len * 6)) if sub_len > 0 else 640
            return w, 42
        elif state == "VERIFIED":
            return 290, 38
        elif state == "VOICE":
            w = max(460, min(580, 260 + sub_len * 6)) if sub_len > 0 else 480
            return w, 38
        elif state == "ERROR":
            w = max(420, min(560, 220 + sub_len * 7)) if sub_len > 0 else 440
            return w, 38
        return 380, 38

    @staticmethod
    def parse_step_progress(step_str: str) -> Tuple[int, int]:
        """Parses 'current/total' step telemetry into integer indices."""
        if "/" in step_str:
            parts = step_str.split("/")
            try:
                curr = int(parts[0].strip())
                total = int(parts[1].strip())
                if curr <= 0 and total <= 0:
                    return 0, 0
                return max(0, curr), max(1, total)
            except ValueError:
                pass
        return 0, 0

    def step_spring(self) -> bool:
        """Applies fluid exponential spring morphing interpolation towards target dimensions."""
        diff_w = self.target_w - self.current_w
        diff_h = self.target_h - self.current_h
        animating = False

        if abs(diff_w) > 0.5:
            self.current_w += diff_w * self.morph_spring_k
            animating = True
        else:
            self.current_w = float(self.target_w)

        if abs(diff_h) > 0.5:
            self.current_h += diff_h * self.morph_spring_k
            animating = True
        else:
            self.current_h = float(self.target_h)

        return animating

    def start(self) -> None:
        """Starts the native transparent overlay in a background daemon thread."""
        if sys.platform != "win32":
            logger.info("AVCSVisualHUD: Non-Windows environment, visual overlay disabled.")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        ready_evt = threading.Event()
        self._thread = threading.Thread(target=self._window_worker, args=(ready_evt,), daemon=True)
        self._thread.start()
        ready_evt.wait(timeout=2.0)
        play_acoustic_earcon("start")

    def stop(self) -> None:
        """Closes the overlay window and terminates background thread."""
        self._stop_event.set()
        if self.hwnd and sys.platform == "win32":
            try:
                ctypes.windll.user32.PostMessageW(self.hwnd, 0x0010, 0, 0)  # WM_CLOSE
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.hwnd = None

    def destroy(self) -> None:
        """Alias for stop() to support standard UI lifecycle destruction."""
        self.stop()

    def update_status(
        self,
        status: str,
        subgoal: str = "",
        step: str = "",
        mode: str = "LIVE",
        latency_ms: float = 0.0,
    ) -> None:
        """Updates status telemetry and adjusts Dynamic Island target morphing dimensions."""
        with self._lock:
            self.status_text = status.upper().strip()
            if subgoal:
                self.subgoal_text = subgoal.strip()
            if step:
                self.step_info = step.strip()
            self.mode_label = mode.upper().strip()
            self.latency_ms = latency_ms

            state = self.classify_state(self.status_text)
            tw, th = self.calculate_target_size(state, self.subgoal_text)
            self.target_w = float(tw)
            self.target_h = float(th)

        self._request_repaint()

    def show_click_reticle(self, nx: int, ny: int, kind: str = "click") -> None:
        """Flashes an animated targeting reticle at physical desktop coordinate (nx, ny)."""
        with self._lock:
            self.reticle_coord = (nx, ny)
            self.reticle_start_time = time.time()
            self.reticle_kind = kind

        play_acoustic_earcon("action")
        self._request_repaint()

    def _request_repaint(self) -> None:
        if self.hwnd and sys.platform == "win32":
            try:
                ctypes.windll.user32.InvalidateRect(self.hwnd, None, False)
            except Exception:
                pass

    def _create_fonts(self, gdi) -> None:
        """Initializes high-fidelity Segoe UI fonts with CLEARTYPE_QUALITY."""
        try:
            # -12px Bold (FW_BOLD = 700, CLEARTYPE_QUALITY = 5)
            self._hfont_bold = gdi.CreateFontW(
                -12, 0, 0, 0, 700, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI"
            )
            # -11px Medium (FW_SEMIBOLD = 600, CLEARTYPE_QUALITY = 5)
            self._hfont_medium = gdi.CreateFontW(
                -11, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI"
            )
            # -10px Chip (FW_SEMIBOLD = 600, CLEARTYPE_QUALITY = 5)
            self._hfont_chip = gdi.CreateFontW(
                -10, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI"
            )
        except Exception:
            pass

    def _delete_fonts(self, gdi) -> None:
        """Releases Segoe UI font handles on window destruction."""
        for font in (self._hfont_bold, self._hfont_medium, self._hfont_chip):
            if font:
                try:
                    gdi.DeleteObject(font)
                except Exception:
                    pass
        self._hfont_bold = None
        self._hfont_medium = None
        self._hfont_chip = None

    def _window_worker(self, ready_evt: threading.Event) -> None:
        """Background Win32 message loop managing the topmost transparent layered window."""
        import os
        if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
            ready_evt.set()
            return

        try:
            u32 = ctypes.windll.user32
            gdi = ctypes.windll.gdi32
            k32 = ctypes.windll.kernel32

            # Configure strict 64-bit argument types and return types
            u32.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64, ctypes.c_int64]
            u32.DefWindowProcW.restype = ctypes.c_int64

            k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            k32.GetModuleHandleW.restype = ctypes.c_void_p

            u32.BeginPaint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            u32.BeginPaint.restype = ctypes.c_void_p
            u32.EndPaint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            gdi.CreateSolidBrush.argtypes = [ctypes.c_uint32]
            gdi.CreateSolidBrush.restype = ctypes.c_void_p
            gdi.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
            gdi.CreatePen.restype = ctypes.c_void_p
            gdi.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            gdi.SelectObject.restype = ctypes.c_void_p
            gdi.GetStockObject.argtypes = [ctypes.c_int]
            gdi.GetStockObject.restype = ctypes.c_void_p
            gdi.DeleteObject.argtypes = [ctypes.c_void_p]
            gdi.DeleteObject.restype = ctypes.c_bool
            gdi.DeleteDC.argtypes = [ctypes.c_void_p]
            gdi.DeleteDC.restype = ctypes.c_bool

            gdi.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
            gdi.CreateCompatibleDC.restype = ctypes.c_void_p
            gdi.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            gdi.CreateCompatibleBitmap.restype = ctypes.c_void_p
            gdi.BitBlt.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32
            ]
            gdi.BitBlt.restype = ctypes.c_bool

            u32.FillRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            u32.DrawTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
            u32.InvalidateRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            gdi.RoundRect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            gdi.Ellipse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            gdi.MoveToEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            gdi.LineTo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            gdi.SetBkMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            gdi.SetTextColor.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

            self._create_fonts(gdi)

            def _wnd_proc(hwnd, msg, wparam, lparam):
                if msg == WM_PAINT:
                    ps = PAINTSTRUCT()
                    hdc = u32.BeginPaint(hwnd, ctypes.byref(ps))
                    try:
                        # Double-buffered GDI rendering for 100% flicker-free graphics
                        if (
                            self._mem_dc is None
                            or self._cached_w != self.screen_w
                            or self._cached_h != self.screen_h
                        ):
                            if self._mem_dc:
                                gdi.SelectObject(self._mem_dc, self._old_bm)
                                gdi.DeleteObject(self._mem_bm)
                                gdi.DeleteDC(self._mem_dc)
                            self._mem_dc = gdi.CreateCompatibleDC(hdc)
                            self._mem_bm = gdi.CreateCompatibleBitmap(hdc, self.screen_w, self.screen_h)
                            self._old_bm = gdi.SelectObject(self._mem_dc, self._mem_bm)
                            self._cached_w = self.screen_w
                            self._cached_h = self.screen_h

                        target_dc = self._mem_dc if self._mem_dc else hdc
                        self._draw_hud(target_dc)
                        if self._mem_dc:
                            gdi.BitBlt(
                                hdc, 0, 0, self.screen_w, self.screen_h,
                                self._mem_dc, 0, 0, 0x00CC0020  # SRCCOPY = 0x00CC0020
                            )
                    finally:
                        u32.EndPaint(hwnd, ctypes.byref(ps))
                    return 0
                elif msg == WM_ERASEBKGND:
                    return 1  # Bypasses background erase, eliminating GDI flicker
                elif msg == WM_TIMER:
                    now = time.time()
                    with self._lock:
                        self.step_spring()
                        if self.reticle_coord and (now - self.reticle_start_time) >= self.reticle_duration:
                            self.reticle_coord = None
                    # Continuous pulse & smooth spring interpolation redraw
                    u32.InvalidateRect(hwnd, None, False)
                    return 0
                elif msg == WM_DESTROY:
                    if self._mem_dc:
                        gdi.SelectObject(self._mem_dc, self._old_bm)
                        gdi.DeleteObject(self._mem_bm)
                        gdi.DeleteDC(self._mem_dc)
                        self._mem_dc = None
                        self._mem_bm = None
                    self._delete_fonts(gdi)
                    u32.PostQuitMessage(0)
                    return 0
                return u32.DefWindowProcW(hwnd, msg, wparam, lparam)

            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_int64, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64, ctypes.c_int64
            )
            self._proc_ref = WNDPROC(_wnd_proc)

            class WNDCLASSEXW(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_uint),
                    ("style", ctypes.c_uint),
                    ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int),
                    ("hInstance", ctypes.c_void_p),
                    ("hIcon", ctypes.c_void_p),
                    ("hCursor", ctypes.c_void_p),
                    ("hbrBackground", ctypes.c_void_p),
                    ("lpszMenuName", ctypes.c_wchar_p),
                    ("lpszClassName", ctypes.c_wchar_p),
                    ("hIconSm", ctypes.c_void_p),
                ]

            hinst = k32.GetModuleHandleW(None)
            cls_name = f"AVCS_HUD_{int(time.time()*1000)}"

            wc = WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
            wc.style = 3  # CS_HREDRAW | CS_VREDRAW
            wc.lpfnWndProc = self._proc_ref
            wc.hInstance = hinst
            wc.hCursor = u32.LoadCursorW(None, 32512)  # IDC_ARROW
            wc.lpszClassName = cls_name

            u32.RegisterClassExW.argtypes = [ctypes.c_void_p]
            u32.RegisterClassExW.restype = ctypes.c_ushort

            atom = u32.RegisterClassExW(ctypes.byref(wc))
            if not atom:
                logger.warning(f"AVCSVisualHUD: RegisterClassExW failed ({k32.GetLastError()}).")
                ready_evt.set()
                return

            ex_style = (
                WS_EX_TOPMOST
                | WS_EX_LAYERED
                | WS_EX_TRANSPARENT
                | WS_EX_NOACTIVATE
                | WS_EX_TOOLWINDOW
            )
            style = WS_POPUP | WS_VISIBLE

            u32.CreateWindowExW.argtypes = [
                ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ]
            u32.CreateWindowExW.restype = ctypes.c_void_p

            hwnd = u32.CreateWindowExW(
                ex_style,
                cls_name,
                self.title,
                style,
                self.screen_x,
                self.screen_y,
                self.screen_w,
                self.screen_h,
                None,
                None,
                hinst,
                None,
            )

            if not hwnd:
                logger.warning(f"AVCSVisualHUD: CreateWindowExW failed ({k32.GetLastError()}).")
                ready_evt.set()
                return

            self.hwnd = hwnd

            # Set layered window color key: pixels painted with COLOR_TRANSPARENT_KEY are 100% transparent & click-through
            u32.SetLayeredWindowAttributes(hwnd, COLOR_TRANSPARENT_KEY, 255, LWA_COLORKEY)

            # Exclude from BitBlt screen capture so HUD never interferes with AVCS vision (unless capture is permitted)
            if self.exclude_from_capture:
                try:
                    if hasattr(u32, "SetWindowDisplayAffinity"):
                        u32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
                except Exception as e:
                    logger.debug(f"SetWindowDisplayAffinity notice: {e}")

            u32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
            # Ensure window is placed topmost and immediately painted
            HWND_TOPMOST = ctypes.c_void_p(-1)
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            SWP_SHOWWINDOW = 0x0040
            u32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
            u32.UpdateWindow(hwnd)
            u32.InvalidateRect(hwnd, None, False)
            ready_evt.set()

            # High-fluidity 30 Hz timer (33ms ticks) for living pulse & smooth spring morphing
            u32.SetTimer(hwnd, 1, 33, None)

            msg = ctypes.wintypes.MSG()
            while not self._stop_event.is_set():
                ret = u32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:
                    break
                u32.TranslateMessage(ctypes.byref(msg))
                u32.DispatchMessageW(ctypes.byref(msg))
        except Exception as e:
            logger.debug(f"AVCSVisualHUD worker error: {e}")
        finally:
            ready_evt.set()

    def _draw_hud(self, hdc: int) -> None:
        """Double-buffered GDI rendering routine for the Apple Dynamic Island capsule and reticle."""
        u32 = ctypes.windll.user32
        gdi = ctypes.windll.gdi32

        # 1. Fill entire screen buffer with transparent key
        rc_screen = ctypes.wintypes.RECT(0, 0, self.screen_w, self.screen_h)
        hbr_key = gdi.CreateSolidBrush(COLOR_TRANSPARENT_KEY)
        u32.FillRect(hdc, ctypes.byref(rc_screen), hbr_key)
        gdi.DeleteObject(hbr_key)

        with self._lock:
            mode = self.mode_label
            status = self.status_text
            subgoal = self.subgoal_text
            step = self.step_info
            lat = self.latency_ms
            reticle = self.reticle_coord
            r_start = self.reticle_start_time
            r_dur = self.reticle_duration
            pill_w = int(self.current_w)
            pill_h = int(self.current_h)

        state = self.classify_state(status)

        # 2. Dynamic Island Positioning (Symmetrically Top-Centered)
        primary_w = u32.GetSystemMetrics(0) if u32 else self.screen_w
        if primary_w <= 0:
            primary_w = self.screen_w
        primary_offset_x = max(0, -self.screen_x)
        pill_x = primary_offset_x + (primary_w - pill_w) // 2
        pill_y = 12
        border_radius = pill_h // 2  # True capsule curvature (height // 2)

        # 3. MacBook Obsidian Glass Capsule Core & Frosted Rim
        hbr_obsidian = gdi.CreateSolidBrush(COLOR_OBSIDIAN)
        hpen_rim = gdi.CreatePen(0, 1, COLOR_RIM)
        old_brush = gdi.SelectObject(hdc, hbr_obsidian)
        old_pen = gdi.SelectObject(hdc, hpen_rim)

        # RoundRect with ellipse width & height = pill_h forms a true capsule
        gdi.RoundRect(hdc, pill_x, pill_y, pill_x + pill_w, pill_y + pill_h, pill_h, pill_h)

        gdi.SelectObject(hdc, old_brush)
        gdi.SelectObject(hdc, old_pen)
        gdi.DeleteObject(hbr_obsidian)
        gdi.DeleteObject(hpen_rim)

        # 4. Ultra-Thin Top-Edge Curved Glass Light Highlight
        hpen_hl = gdi.CreatePen(0, 1, COLOR_HIGHLIGHT)
        old_pen = gdi.SelectObject(hdc, hpen_hl)
        hl_start_x = pill_x + border_radius - 2
        hl_end_x = pill_x + pill_w - border_radius + 2
        if hl_end_x > hl_start_x:
            gdi.MoveToEx(hdc, hl_start_x, pill_y + 1, None)
            gdi.LineTo(hdc, hl_end_x, pill_y + 1)
        gdi.SelectObject(hdc, old_pen)
        gdi.DeleteObject(hpen_hl)

        # 5. Living Pulsing Breathing Orb State Calculation
        accent_cfg = STATE_COLORS.get(state, STATE_COLORS["LIVE"])
        if mode == "SIMULATION" and state not in ("ERROR", "MOTOR"):
            accent_cfg = STATE_COLORS["SIMULATION"]

        # Sinusoidal pulse: ranges smoothly from 0.0 to 1.0 at 4 rad/s
        pulse = 0.5 + 0.5 * math.sin(time.time() * 4.0)
        halo_radius = int(5 + pulse * 2.5)  # 5px to 7px glow radius
        core_radius = 3                     # Solid 3px core

        gdi.SetBkMode(hdc, 1)  # TRANSPARENT = 1

        # 6. Render State Content
        if state == "IDLE":
            # [1. COMPACT / IDLE STATE] (🟢) AVCS ACE • READY (~220px)
            orb_cx = pill_x + 24
            orb_cy = pill_y + pill_h // 2
            self._draw_breathing_orb(hdc, orb_cx, orb_cy, core_radius, halo_radius, accent_cfg)

            # Segoe UI text: AVCS ACE • READY
            old_font = gdi.SelectObject(hdc, self._hfont_bold or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_PRIMARY)
            rc_title = ctypes.wintypes.RECT(pill_x + 38, pill_y, pill_x + 115, pill_y + pill_h)
            u32.DrawTextW(hdc, "AVCS ACE", -1, ctypes.byref(rc_title), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            gdi.SelectObject(hdc, self._hfont_medium or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_MUTED)
            rc_dot = ctypes.wintypes.RECT(pill_x + 116, pill_y, pill_x + 128, pill_y + pill_h)
            u32.DrawTextW(hdc, "•", -1, ctypes.byref(rc_dot), DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            gdi.SetTextColor(hdc, accent_cfg["core"])
            rc_ready = ctypes.wintypes.RECT(pill_x + 130, pill_y, pill_x + pill_w - 14, pill_y + pill_h)
            u32.DrawTextW(hdc, "READY", -1, ctypes.byref(rc_ready), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)
            gdi.SelectObject(hdc, old_font)

        elif state == "PLANNING":
            # [2. THINKING / PLANNING STATE] (⚡) PLANNING │ Decomposing goal into steps... (~380-420px)
            orb_cx = pill_x + 22
            orb_cy = pill_y + pill_h // 2
            self._draw_breathing_orb(hdc, orb_cx, orb_cy, core_radius, halo_radius, accent_cfg)

            old_font = gdi.SelectObject(hdc, self._hfont_bold or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, accent_cfg["core"])
            rc_badge = ctypes.wintypes.RECT(pill_x + 36, pill_y, pill_x + 104, pill_y + pill_h)
            badge_text = "PLANNING" if "PLAN" in status else ("THINKING" if "THINK" in status else "OBSERVING")
            u32.DrawTextW(hdc, badge_text, -1, ctypes.byref(rc_badge), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            # Hairline Divider
            self._draw_divider(hdc, pill_x + 108, pill_y + 8, pill_y + pill_h - 8)

            # Sub-Goal / Planning details
            gdi.SelectObject(hdc, self._hfont_medium or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_SECONDARY)
            detail = subgoal if subgoal else "Decomposing goal into steps..."
            rc_subgoal = ctypes.wintypes.RECT(pill_x + 118, pill_y, pill_x + pill_w - 18, pill_y + pill_h)
            u32.DrawTextW(hdc, detail, -1, ctypes.byref(rc_subgoal), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX)
            gdi.SelectObject(hdc, old_font)

        elif state == "MOTOR":
            # [3. ACTIVE MOTOR EXECUTION STATE] 🖱️ CLICK │ Step 1/3: Open Start Menu │ [LIVE] 45ms (●) (~640px)
            # Left Action Badge
            old_font = gdi.SelectObject(hdc, self._hfont_bold or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, accent_cfg["core"])
            action_label = "CLICK" if "CLICK" in status else ("TYPE" if "TYPE" in status or "KEY" in status else "ACTION")
            rc_act = ctypes.wintypes.RECT(pill_x + 20, pill_y, pill_x + 88, pill_y + pill_h - 4)
            u32.DrawTextW(hdc, f"🖱️ {action_label}", -1, ctypes.byref(rc_act), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            # Divider 1
            self._draw_divider(hdc, pill_x + 94, pill_y + 8, pill_y + pill_h - 12)

            # Middle: Step index & active sub-goal
            gdi.SelectObject(hdc, self._hfont_medium or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_PRIMARY)
            step_disp = f"Step {step}: " if step and step != "0/0" else ""
            task_str = f"{step_disp}{subgoal}" if subgoal else f"{step_disp}Executing active action"
            rc_step = ctypes.wintypes.RECT(pill_x + 104, pill_y, pill_x + pill_w - 130, pill_y + pill_h - 4)
            u32.DrawTextW(hdc, task_str, -1, ctypes.byref(rc_step), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX)

            # Divider 2
            self._draw_divider(hdc, pill_x + pill_w - 124, pill_y + 8, pill_y + pill_h - 12)

            # Right: Mode chip & latency
            gdi.SelectObject(hdc, self._hfont_chip or gdi.GetStockObject(17))
            mode_col = STATE_COLORS["LIVE"]["core"] if mode == "LIVE" else STATE_COLORS["SIMULATION"]["core"]
            gdi.SetTextColor(hdc, mode_col)
            rc_mode = ctypes.wintypes.RECT(pill_x + pill_w - 118, pill_y, pill_x + pill_w - 74, pill_y + pill_h - 4)
            u32.DrawTextW(hdc, f"[{mode}]", -1, ctypes.byref(rc_mode), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            if lat > 0:
                gdi.SetTextColor(hdc, COLOR_TEXT_MUTED)
                rc_lat = ctypes.wintypes.RECT(pill_x + pill_w - 72, pill_y, pill_x + pill_w - 36, pill_y + pill_h - 4)
                u32.DrawTextW(hdc, f"{lat:.0f}ms", -1, ctypes.byref(rc_lat), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            # Right Living Breathing Orb
            orb_cx = pill_x + pill_w - 20
            orb_cy = pill_y + (pill_h - 4) // 2
            self._draw_breathing_orb(hdc, orb_cx, orb_cy, core_radius, halo_radius, accent_cfg)
            gdi.SelectObject(hdc, old_font)

            # Bottom Sub-Goal Progress Micro-Dots & Hairline Track
            self._draw_subgoal_progress(hdc, pill_x, pill_y, pill_w, pill_h, border_radius, step, accent_cfg["core"])

        elif state == "VERIFIED":
            # (✓) COMPLETE │ Verified Successfully (~290px)
            orb_cx = pill_x + 22
            orb_cy = pill_y + pill_h // 2
            self._draw_breathing_orb(hdc, orb_cx, orb_cy, core_radius, halo_radius, accent_cfg)

            old_font = gdi.SelectObject(hdc, self._hfont_bold or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, accent_cfg["core"])
            rc_comp = ctypes.wintypes.RECT(pill_x + 36, pill_y, pill_x + 112, pill_y + pill_h)
            u32.DrawTextW(hdc, "COMPLETE", -1, ctypes.byref(rc_comp), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            self._draw_divider(hdc, pill_x + 116, pill_y + 8, pill_y + pill_h - 8)

            gdi.SelectObject(hdc, self._hfont_medium or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_PRIMARY)
            rc_vtext = ctypes.wintypes.RECT(pill_x + 126, pill_y, pill_x + pill_w - 18, pill_y + pill_h)
            ver_text = subgoal if subgoal else "Verified Successfully"
            u32.DrawTextW(hdc, ver_text, -1, ctypes.byref(rc_vtext), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX)
            gdi.SelectObject(hdc, old_font)

        elif state == "VOICE":
            # 🗣️ VOICE │ "Spoken text preview..." │ ▂▃▅▆▇▅▃ (~500px)
            orb_cx = pill_x + 22
            orb_cy = pill_y + pill_h // 2
            self._draw_breathing_orb(hdc, orb_cx, orb_cy, core_radius, halo_radius, accent_cfg)

            old_font = gdi.SelectObject(hdc, self._hfont_bold or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, accent_cfg["core"])
            is_listen = "LISTEN" in status
            voice_label = "LISTENING" if is_listen else "VOICE"
            voice_icon = "🎙️" if is_listen else "🗣️"
            rc_vbadge = ctypes.wintypes.RECT(pill_x + 36, pill_y, pill_x + 114, pill_y + pill_h)
            u32.DrawTextW(hdc, f"{voice_icon} {voice_label}", -1, ctypes.byref(rc_vbadge), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            self._draw_divider(hdc, pill_x + 120, pill_y + 8, pill_y + pill_h - 8)

            gdi.SelectObject(hdc, self._hfont_medium or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_PRIMARY)
            rc_spk = ctypes.wintypes.RECT(pill_x + 130, pill_y, pill_x + pill_w - 60, pill_y + pill_h)
            spk_text = subgoal if subgoal else ("Listening to your voice..." if is_listen else "Speaking output...")
            u32.DrawTextW(hdc, spk_text, -1, ctypes.byref(rc_spk), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX)
            gdi.SelectObject(hdc, old_font)

            # Siri-Style Audio Waveform on Right
            wave_cx = pill_x + pill_w - 32
            wave_cy = pill_y + pill_h // 2
            self._draw_voice_waveform(hdc, wave_cx, wave_cy, accent_cfg["core"])

        elif state == "ERROR":
            # (✕) HALTED │ Error details (~440px)
            orb_cx = pill_x + 22
            orb_cy = pill_y + pill_h // 2
            self._draw_breathing_orb(hdc, orb_cx, orb_cy, core_radius, halo_radius, accent_cfg)

            old_font = gdi.SelectObject(hdc, self._hfont_bold or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, accent_cfg["core"])
            rc_err = ctypes.wintypes.RECT(pill_x + 36, pill_y, pill_x + 98, pill_y + pill_h)
            u32.DrawTextW(hdc, "HALTED", -1, ctypes.byref(rc_err), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX)

            self._draw_divider(hdc, pill_x + 102, pill_y + 8, pill_y + pill_h - 8)

            gdi.SelectObject(hdc, self._hfont_medium or gdi.GetStockObject(17))
            gdi.SetTextColor(hdc, COLOR_TEXT_PRIMARY)
            rc_err_text = ctypes.wintypes.RECT(pill_x + 112, pill_y, pill_x + pill_w - 18, pill_y + pill_h)
            err_msg = subgoal if subgoal else status
            u32.DrawTextW(hdc, err_msg, -1, ctypes.byref(rc_err_text), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX)
            gdi.SelectObject(hdc, old_font)

        # 7. Animated Precision Click Reticle
        if reticle:
            rx, ry = reticle
            rx = rx - self.screen_x
            ry = ry - self.screen_y
            elapsed = time.time() - r_start
            if 0 <= elapsed <= r_dur:
                progress = elapsed / r_dur
                radius = int(10 + progress * 26)  # Expanding ripple from 10px to 36px
                ret_col = STATE_COLORS["MOTOR"]["core"] if "click" in self.reticle_kind.lower() else STATE_COLORS["PLANNING"]["core"]

                # Central targeting dot
                hbr_dot = gdi.CreateSolidBrush(ret_col)
                old_b = gdi.SelectObject(hdc, hbr_dot)
                old_p = gdi.SelectObject(hdc, gdi.GetStockObject(8))  # NULL_PEN = 8
                gdi.Ellipse(hdc, rx - 3, ry - 3, rx + 4, ry + 4)
                gdi.SelectObject(hdc, old_b)
                gdi.DeleteObject(hbr_dot)

                # Outer expanding ripple ring
                hpen_ret = gdi.CreatePen(0, 2, ret_col)
                gdi.SelectObject(hdc, hpen_ret)
                gdi.SelectObject(hdc, gdi.GetStockObject(5))  # NULL_BRUSH = 5
                gdi.Ellipse(hdc, rx - radius, ry - radius, rx + radius, ry + radius)

                # Precision crosshair notches
                notch = 6
                gdi.MoveToEx(hdc, rx - radius - notch, ry, None)
                gdi.LineTo(hdc, rx - radius + 2, ry)
                gdi.MoveToEx(hdc, rx + radius - 2, ry, None)
                gdi.LineTo(hdc, rx + radius + notch, ry)
                gdi.MoveToEx(hdc, rx, ry - radius - notch, None)
                gdi.LineTo(hdc, rx, ry - radius + 2)
                gdi.MoveToEx(hdc, rx, ry + radius - 2, None)
                gdi.LineTo(hdc, rx, ry + radius + notch)

                gdi.SelectObject(hdc, old_p)
                gdi.DeleteObject(hpen_ret)

    def _draw_breathing_orb(
        self,
        hdc: int,
        cx: int,
        cy: int,
        core_r: int,
        halo_r: int,
        accent: dict,
    ) -> None:
        """Draws a multi-layered living breathing neon orb with soft glow halo."""
        gdi = ctypes.windll.gdi32

        # 1. Outer Soft Glow Halo
        hbr_halo = gdi.CreateSolidBrush(accent["glow"])
        old_b = gdi.SelectObject(hdc, hbr_halo)
        old_p = gdi.SelectObject(hdc, gdi.GetStockObject(8))  # NULL_PEN = 8
        gdi.Ellipse(hdc, cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r)
        gdi.SelectObject(hdc, old_b)
        gdi.DeleteObject(hbr_halo)

        # 2. Solid High-Intensity Center Core
        hbr_core = gdi.CreateSolidBrush(accent["core"])
        gdi.SelectObject(hdc, hbr_core)
        gdi.Ellipse(hdc, cx - core_r, cy - core_r, cx + core_r, cy + core_r)
        gdi.SelectObject(hdc, old_b)
        gdi.SelectObject(hdc, old_p)
        gdi.DeleteObject(hbr_core)

    def _draw_divider(self, hdc: int, x: int, y1: int, y2: int) -> None:
        """Renders a subtle 1px vertical hairline divider inside the capsule."""
        gdi = ctypes.windll.gdi32
        hpen_div = gdi.CreatePen(0, 1, COLOR_DIVIDER)
        old_p = gdi.SelectObject(hdc, hpen_div)
        gdi.MoveToEx(hdc, x, y1, None)
        gdi.LineTo(hdc, x, y2)
        gdi.SelectObject(hdc, old_p)
        gdi.DeleteObject(hpen_div)

    def _draw_voice_waveform(self, hdc: int, cx: int, cy: int, accent_color: int) -> None:
        """Draws Apple Siri-style animated dancing soundwave bars."""
        gdi = ctypes.windll.gdi32
        t = time.time()
        num_bars = 6
        bar_w = 3
        bar_gap = 2
        total_w = num_bars * bar_w + (num_bars - 1) * bar_gap
        start_x = cx - total_w // 2

        hbr_bar = gdi.CreateSolidBrush(accent_color)
        old_b = gdi.SelectObject(hdc, hbr_bar)
        old_p = gdi.SelectObject(hdc, gdi.GetStockObject(8))  # NULL_PEN = 8

        for i in range(num_bars):
            phase = i * 0.9
            amp = 0.5 * math.sin(t * 7.0 + phase) + 0.5 * math.cos(t * 4.0 - phase)
            bar_h = max(4, int(4 + 10 * abs(amp)))
            bx = start_x + i * (bar_w + bar_gap)
            by1 = cy - bar_h // 2
            by2 = cy + bar_h // 2
            gdi.RoundRect(hdc, bx, by1, bx + bar_w, by2, 2, 2)

        gdi.SelectObject(hdc, old_b)
        gdi.SelectObject(hdc, old_p)
        gdi.DeleteObject(hbr_bar)

    def _draw_subgoal_progress(
        self,
        hdc: int,
        pill_x: int,
        pill_y: int,
        pill_w: int,
        pill_h: int,
        border_radius: int,
        step_str: str,
        accent_color: int,
    ) -> None:
        """Renders a 2px hairline progress track with glowing micro-dots along the bottom edge."""
        curr_step, total_steps = self.parse_step_progress(step_str)
        if total_steps <= 1:
            return

        gdi = ctypes.windll.gdi32
        track_y = pill_y + pill_h - 4
        track_pad = border_radius + 14
        track_x1 = pill_x + track_pad
        track_x2 = pill_x + pill_w - track_pad
        track_len = track_x2 - track_x1
        if track_len <= 30:
            return

        # 1. Background Hairline Track (Dark Slate: #1A1E24)
        hpen_bg = gdi.CreatePen(0, 2, COLOR_TRACK_BG)
        old_p = gdi.SelectObject(hdc, hpen_bg)
        gdi.MoveToEx(hdc, track_x1, track_y, None)
        gdi.LineTo(hdc, track_x2, track_y)
        gdi.SelectObject(hdc, old_p)
        gdi.DeleteObject(hpen_bg)

        # 2. Active Progress Hairline
        progress_ratio = min(1.0, max(0.0, curr_step / total_steps))
        active_x2 = track_x1 + int(track_len * progress_ratio)
        if active_x2 > track_x1:
            hpen_fg = gdi.CreatePen(0, 2, accent_color)
            old_p = gdi.SelectObject(hdc, hpen_fg)
            gdi.MoveToEx(hdc, track_x1, track_y, None)
            gdi.LineTo(hdc, active_x2, track_y)
            gdi.SelectObject(hdc, old_p)
            gdi.DeleteObject(hpen_fg)

        # 3. Micro-Dots (●──○──○) for sub-goals (up to 8 steps)
        if 2 <= total_steps <= 8:
            spacing = track_len / (total_steps - 1)
            for i in range(total_steps):
                dot_cx = int(track_x1 + i * spacing)
                if i < curr_step:
                    # Completed/Active Sub-Goal: Filled Glowing Accent Dot (●)
                    hbr_dot = gdi.CreateSolidBrush(accent_color)
                    old_b = gdi.SelectObject(hdc, hbr_dot)
                    old_p = gdi.SelectObject(hdc, gdi.GetStockObject(8))  # NULL_PEN = 8
                    gdi.Ellipse(hdc, dot_cx - 2, track_y - 2, dot_cx + 3, track_y + 3)
                    gdi.SelectObject(hdc, old_b)
                    gdi.SelectObject(hdc, old_p)
                    gdi.DeleteObject(hbr_dot)
                else:
                    # Upcoming Sub-Goal: Dim Hollow Circle (○)
                    hpen_dot = gdi.CreatePen(0, 1, COLOR_TRACK_DOT_OFF)
                    hbr_dot = gdi.CreateSolidBrush(COLOR_OBSIDIAN)
                    old_b = gdi.SelectObject(hdc, hbr_dot)
                    old_p = gdi.SelectObject(hdc, hpen_dot)
                    gdi.Ellipse(hdc, dot_cx - 2, track_y - 2, dot_cx + 3, track_y + 3)
                    gdi.SelectObject(hdc, old_b)
                    gdi.SelectObject(hdc, old_p)
                    gdi.DeleteObject(hbr_dot)
                    gdi.DeleteObject(hpen_dot)

    def __enter__(self) -> "AVCSVisualHUD":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
