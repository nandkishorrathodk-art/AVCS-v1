"""[4] hand_motor.py: Biological Human Mouse Motion & Visual Ripple Verification.

Implements human-like 5th-order minimum-jerk kinematics (Flash & Hogan 1985)
and closed-loop visual ripple verification to confirm button depression.
Includes hardened 4-corner latching safety failsafe breakout.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np


# Pre-bind Win32 DLLs and APIs at module level to eliminate runtime lookup latency
if sys.platform == "win32":
    try:
        _user32 = ctypes.windll.user32
        _winmm = ctypes.windll.winmm
        _timeBeginPeriod = _winmm.timeBeginPeriod
        _timeEndPeriod = _winmm.timeEndPeriod
        _GetCursorPos = _user32.GetCursorPos
        _SetCursorPos = _user32.SetCursorPos
        _GetSystemMetrics = _user32.GetSystemMetrics
        _mouse_event = _user32.mouse_event
        _keybd_event = _user32.keybd_event
        _VkKeyScanW = _user32.VkKeyScanW
    except Exception:
        _user32 = None
        _winmm = None
        _timeBeginPeriod = None
        _timeEndPeriod = None
        _GetCursorPos = None
        _SetCursorPos = None
        _GetSystemMetrics = None
        _mouse_event = None
        _keybd_event = None
        _VkKeyScanW = None
else:
    _user32 = None
    _winmm = None
    _timeBeginPeriod = None
    _timeEndPeriod = None
    _GetCursorPos = None
    _SetCursorPos = None
    _GetSystemMetrics = None
    _mouse_event = None
    _keybd_event = None
    _VkKeyScanW = None


@dataclass(frozen=True)
class RippleResult:
    """Outcome of visual ripple click verification."""
    verified: bool            # True if screen confirmed reaction (photometric change)
    surprise_magnitude: float # Magnitude of detected luminance ripple
    latency_ms: float         # Duration of verification check
    window_title_changed: bool# True if foreground window mutated


class SafetyBreakoutException(RuntimeError):
    """Raised immediately when emergency fail-safe is triggered by the user."""


class _CheckDispatcher:
    """Dispatches .check() to instance method or class default instance."""

    def __get__(self, instance, owner):
        if instance is None:
            def _caller(*args, **kwargs):
                return owner._get_default_instance().check(*args, **kwargs)
            return _caller
        else:
            def _caller(*args, **kwargs):
                return instance._check_impl(*args, **kwargs)
            return _caller


class CornerBreakoutFailSafe:
    """Detects emergency cursor breakout in any of the 4 screen corners."""

    _global_instance: Optional["CornerBreakoutFailSafe"] = None

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
        self._latched = False
        self._consecutive_corner_frames = 0
        self._current_corner: Optional[str] = None

    @classmethod
    def _get_default_instance(cls) -> "CornerBreakoutFailSafe":
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @property
    def is_tripped(self) -> bool:
        return self._latched or self._tripped

    @property
    def is_latched(self) -> bool:
        return self._latched

    def reset(self) -> None:
        """Resets the safety fail-safe latch, allowing motor control to resume."""
        self._latched = False
        self._tripped = False
        self._consecutive_corner_frames = 0
        self._current_corner = None

    @staticmethod
    def get_screen_dimensions() -> Tuple[int, int]:
        """Returns screen dimensions (width, height) in physical pixels."""
        if sys.platform == "win32":
            try:
                if _GetSystemMetrics is not None:
                    w = int(_GetSystemMetrics(0))
                    h = int(_GetSystemMetrics(1))
                else:
                    user32 = ctypes.windll.user32
                    w = int(user32.GetSystemMetrics(0))
                    h = int(user32.GetSystemMetrics(1))
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass
        return 1920, 1080

    @staticmethod
    def get_hardware_cursor_pos() -> Optional[Tuple[int, int]]:
        """Returns current hardware cursor coordinates, or None if unavailable."""
        if sys.platform == "win32":
            try:
                point = ctypes.wintypes.POINT()
                user32 = ctypes.windll.user32
                try:
                    h_desk = user32.OpenInputDesktop(0, False, 0x01FF)
                    if h_desk:
                        user32.SetThreadDesktop(h_desk)
                except Exception:
                    pass

                user32.GetCursorPos(ctypes.byref(point))
                return int(point.x), int(point.y)
            except Exception:
                pass
        return None

    @staticmethod
    def is_corner(
        pixel_x: int,
        pixel_y: int,
        screen_width: int,
        screen_height: int,
        margin: int = 15,
        exclude_top_right: bool = False,
        exclude_bottom_left: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether coordinates fall in any of the 4 screen corners."""
        top_left = pixel_x <= margin and pixel_y <= margin
        top_right = pixel_x >= (screen_width - margin) and pixel_y <= margin
        bottom_left = pixel_x <= margin and pixel_y >= (screen_height - margin)
        bottom_right = pixel_x >= (screen_width - margin) and pixel_y >= (screen_height - margin)

        if exclude_top_right and top_right:
            top_right = False
        if exclude_bottom_left and bottom_left:
            bottom_left = False

        if not (top_left or top_right or bottom_left or bottom_right):
            return False, None

        corner_name = (
            "top-left" if top_left else
            "top-right" if top_right else
            "bottom-left" if bottom_left else
            "bottom-right"
        )
        return True, corner_name

    def trigger(self, corner_name: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Permanently latches the fail-safe and raises RuntimeError."""
        self._latched = True
        self._tripped = True
        self._current_corner = corner_name
        raise SafetyBreakoutException("Fail-safe engaged: cursor moved to corner breakout region.")

    def check_coordinates(
        self,
        pixel_x: int,
        pixel_y: int,
        screen_width: int,
        screen_height: int,
    ) -> None:
        """Checks commanded cursor coordinates against emergency breakout corners."""
        if not self.enabled:
            return

        if self._latched or self._tripped:
            raise SafetyBreakoutException("Fail-safe engaged: cursor moved to corner breakout region.")

        hit, corner_name = self.is_corner(
            pixel_x, pixel_y, screen_width, screen_height, self.corner_margin_px
        )
        if hit:
            self.trigger(corner_name, pixel_x, pixel_y)

    def check_hardware_failsafe(
        self,
        pixel_x: int,
        pixel_y: int,
        screen_width: int,
        screen_height: int,
        dwell_frames: int = 1,
    ) -> None:
        """Checks physical hardware cursor position for emergency user breakout."""
        if not self.enabled:
            return

        if self._latched or self._tripped:
            raise SafetyBreakoutException("Fail-safe engaged: cursor moved to corner breakout region.")

        hit, corner_name = self.is_corner(
            pixel_x, pixel_y, screen_width, screen_height, self.corner_margin_px,
            exclude_top_right=self.exclude_top_right,
            exclude_bottom_left=self.exclude_bottom_left,
        )
        if hit:
            self.trigger(corner_name, pixel_x, pixel_y)

    def check_override(
        self,
        actual_pos: Tuple[float, float],
        expected_pos: Tuple[float, float],
        threshold_norm: float = 0.08,
    ) -> None:
        """Detects physical human resistance."""
        if not self.enabled:
            return
        if self._latched or self._tripped:
            raise SafetyBreakoutException("Fail-safe engaged: cursor moved to corner breakout region.")
        dx = actual_pos[0] - expected_pos[0]
        dy = actual_pos[1] - expected_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > threshold_norm:
            self._latched = True
            self._tripped = True
            raise SafetyBreakoutException("Fail-safe engaged: cursor moved to corner breakout region.")

    def _check_impl(
        self,
        pixel_x: Optional[int] = None,
        pixel_y: Optional[int] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
    ) -> bool:
        if not self.enabled:
            return False

        if pixel_x is None or pixel_y is None:
            # Polling hardware failsafe check without arguments: returns bool
            if sys.platform != "win32":
                return False
            hw_pos = self.get_hardware_cursor_pos()
            if hw_pos is None:
                return False
            cur_x, cur_y = hw_pos
            w, h = self.get_screen_dimensions()
            hit, corner_name = self.is_corner(
                cur_x,
                cur_y,
                w,
                h,
                margin=self.corner_margin_px,
                exclude_top_right=self.exclude_top_right,
                exclude_bottom_left=self.exclude_bottom_left,
            )
            if hit:
                self._latched = True
                self._tripped = True
                self._current_corner = corner_name
                return True
            return False

        # Commanded coordinate validation with explicit coordinates: raises SafetyBreakoutException on breach
        if self._latched or self._tripped:
            raise SafetyBreakoutException("Fail-safe engaged: cursor moved to corner breakout region.")

        w = screen_width if screen_width is not None else 1920
        h = screen_height if screen_height is not None else 1080

        hit, corner_name = self.is_corner(
            pixel_x,
            pixel_y,
            w,
            h,
            margin=self.corner_margin_px,
            exclude_top_right=self.exclude_top_right,
            exclude_bottom_left=self.exclude_bottom_left,
        )

        if hit:
            self.trigger(corner_name, pixel_x, pixel_y)

        return False

    check = _CheckDispatcher()


class HandMotor:
    """Biological minimum-jerk motor controller for natural human mouse trajectories."""

    def __init__(self, step_duration_ms: float = 12.0, failsafe: Optional[CornerBreakoutFailSafe] = None) -> None:
        self.step_duration_ms = step_duration_ms
        self.failsafe = failsafe if failsafe is not None else CornerBreakoutFailSafe()
        self._latched = False
        self._keybd_event = _keybd_event
        self._VkKeyScanW = _VkKeyScanW

    @property
    def is_latched(self) -> bool:
        return self._latched or self.failsafe.is_latched

    @property
    def is_failsafe_tripped(self) -> bool:
        return self.is_latched

    def reset(self) -> None:
        """Resets the safety fail-safe latch, allowing motor control to resume."""
        self._latched = False
        self.failsafe.reset()

    def _check_failsafe(self, check_pos: Optional[Tuple[int, int]] = None) -> None:
        """Verifies fail-safe latch state and commanded target coordinates against all 4 corners."""
        if self._latched or self.failsafe.is_latched:
            self._latched = True
            raise RuntimeError("Fail-safe engaged: cursor moved to corner breakout region.")

        if sys.platform != "win32":
            return

        w, h = CornerBreakoutFailSafe.get_screen_dimensions()

        # Check commanded/target coordinate if provided
        if check_pos is not None:
            tx, ty = check_pos
            hit_target, target_corner = CornerBreakoutFailSafe.is_corner(tx, ty, w, h, margin=15)
            if hit_target:
                self._latched = True
                self.failsafe.trigger(target_corner, tx, ty)

    @staticmethod
    def get_screen_size() -> Tuple[int, int]:
        return CornerBreakoutFailSafe.get_screen_dimensions()

    @staticmethod
    def get_cursor_pos() -> Tuple[int, int]:
        pos = CornerBreakoutFailSafe.get_hardware_cursor_pos()
        if pos is not None:
            return pos
        return 0, 0

    @staticmethod
    def set_cursor_pos(x: int, y: int) -> None:
        if sys.platform == "win32":
            if _SetCursorPos is not None:
                _SetCursorPos(x, y)
            else:
                ctypes.windll.user32.SetCursorPos(x, y)

    @staticmethod
    def get_foreground_title() -> str:
        """Pure-RGB pixel invariant: OS window-title scraping is disabled."""
        return ""

    def generate_trajectory(
        self,
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        duration_s: float = 0.35,
    ) -> List[Tuple[int, int]]:
        """Generates 5th-order minimum-jerk trajectory (Flash & Hogan 1985)."""
        screen_w, screen_h = self.get_screen_size()
        x0, y0 = start_pt
        xf, yf = end_pt
        dx = xf - x0
        dy = yf - y0
        dist = math.hypot(dx, dy)

        duration = max(0.18, min(0.65, duration_s * (dist / 600.0) ** 0.5))
        steps = max(5, int(duration / (self.step_duration_ms / 1000.0)))

        waypoints: List[Tuple[int, int]] = []
        for i in range(steps + 1):
            tau = i / float(steps)
            s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            tremor_window = math.sin(math.pi * tau)
            tremor_x = math.sin(tau * 18.0) * 0.8 * tremor_window
            tremor_y = math.cos(tau * 15.0) * 0.8 * tremor_window
            cur_x = int(round(x0 + dx * s + tremor_x))
            cur_y = int(round(y0 + dy * s + tremor_y))
            cur_x = max(0, min(max(1, screen_w - 1), cur_x))
            cur_y = max(0, min(max(1, screen_h - 1), cur_y))
            waypoints.append((cur_x, cur_y))

        return waypoints

    def move_to(self, target_x: int, target_y: int, duration_s: float = 0.35) -> bool:
        """Glides cursor along minimum-jerk curve to target.

        Returns False on non-Windows platforms instead of sleeping through the
        whole trajectory and then claiming success.
        """
        if self._latched or self.failsafe.is_latched:
            self._latched = True
            raise RuntimeError("Fail-safe engaged: cursor moved to corner breakout region.")

        self._check_failsafe(check_pos=(target_x, target_y))

        if CornerBreakoutFailSafe.check():
            self._latched = True
            print("[SAFETY] Corner breakout triggered! Motor execution halted.")
            return False

        if sys.platform != "win32":
            return False

        start_pt = self.get_cursor_pos()
        waypoints = self.generate_trajectory(start_pt, (target_x, target_y), duration_s)

        for pt in waypoints:
            if CornerBreakoutFailSafe.check() or self._latched or self.failsafe.is_latched:
                self._latched = True
                print("[SAFETY] Corner breakout triggered! Motor execution halted.")
                return False
            self.set_cursor_pos(pt[0], pt[1])
            time.sleep(self.step_duration_ms / 1000.0)

        self.set_cursor_pos(target_x, target_y)
        return True

    def _click_button(
        self,
        target_x: int,
        target_y: int,
        down_flag: int,
        up_flag: int,
        clicks: int = 1,
    ) -> bool:
        """Move cursor to target and issue one or more native Win32 mouse clicks."""
        self._check_failsafe(check_pos=(target_x, target_y))

        if sys.platform != "win32":
            return False

        if not self.move_to(target_x, target_y):
            return False

        self._check_failsafe()

        if _mouse_event is None:
            return False

        for idx in range(clicks):
            if idx > 0:
                time.sleep(0.04)
            else:
                time.sleep(0.04)
            _mouse_event(down_flag, 0, 0, 0, 0)
            time.sleep(0.06)
            _mouse_event(up_flag, 0, 0, 0, 0)
        return True

    def click(self, target_x: int, target_y: int) -> bool:
        """Moves cursor to target and issues native OS mouse click.

        Returns False on non-Windows platforms: this controller drives the
        native Win32 cursor API, so it cannot actuate a pointer elsewhere. It
        never reports success for an action it did not perform.
        """
        return self._click_button(target_x, target_y, 0x0002, 0x0004, clicks=1)

    def double_click(self, target_x: int, target_y: int) -> bool:
        """Moves cursor to target and issues a native OS double-click."""
        return self._click_button(target_x, target_y, 0x0002, 0x0004, clicks=2)

    def triple_click(self, target_x: int, target_y: int) -> bool:
        """Moves cursor to target and issues a native OS triple-click."""
        return self._click_button(target_x, target_y, 0x0002, 0x0004, clicks=3)

    def right_click(self, target_x: int, target_y: int) -> bool:
        """Moves cursor to target and issues a native OS secondary (right) click."""
        return self._click_button(target_x, target_y, 0x0008, 0x0010, clicks=1)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_s: float = 0.45,
    ) -> bool:
        """Drags with primary mouse button held along a minimum-jerk trajectory."""
        self._check_failsafe(check_pos=(start_x, start_y))
        self._check_failsafe(check_pos=(end_x, end_y))

        if sys.platform != "win32" or _mouse_event is None:
            return False

        if not self.move_to(start_x, start_y):
            return False

        self._check_failsafe()
        waypoints = self.generate_trajectory((start_x, start_y), (end_x, end_y), duration_s)
        _mouse_event(0x0002, 0, 0, 0, 0)  # Left down
        try:
            time.sleep(0.04)
            for pt in waypoints:
                if CornerBreakoutFailSafe.check() or self._latched or self.failsafe.is_latched:
                    self._latched = True
                    return False
                self.set_cursor_pos(pt[0], pt[1])
                time.sleep(self.step_duration_ms / 1000.0)
            self.set_cursor_pos(end_x, end_y)
            time.sleep(0.04)
        finally:
            _mouse_event(0x0004, 0, 0, 0, 0)  # Always release left button
        return True

    def scroll(
        self,
        delta: int = -3,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> bool:
        """Scrolls the mouse wheel vertically (negative = down, positive = up)."""
        check_pt = (int(x), int(y)) if (x is not None and y is not None) else None
        self._check_failsafe(check_pos=check_pt)

        if sys.platform != "win32" or _mouse_event is None:
            return False

        if check_pt is not None and not self.move_to(check_pt[0], check_pt[1]):
            return False

        wheel_delta = int(delta)
        if wheel_delta != 0 and abs(wheel_delta) <= 10:
            wheel_delta *= 120
        _mouse_event(0x0800, 0, 0, wheel_delta, 0)  # MOUSEEVENTF_WHEEL
        return True

    @staticmethod
    def _resolve_vk(key_name: str) -> Tuple[Optional[int], bool]:
        """Resolve a key name or single character to (vk_code, needs_shift)."""
        aliases = {
            "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
            "space": 0x20, "spacebar": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
            "insert": 0x2D, "home": 0x24, "end": 0x23, "page_up": 0x21, "pageup": 0x21,
            "pgup": 0x21, "page_down": 0x22, "pagedown": 0x22, "pgdn": 0x22,
            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            "ctrl": 0x11, "control": 0x11, "lctrl": 0xA2, "left_ctrl": 0xA2,
            "shift": 0x10, "lshift": 0xA0, "left_shift": 0xA0,
            "alt": 0x12, "option": 0x12, "lalt": 0xA4, "left_alt": 0xA4,
            "win": 0x5B, "windows": 0x5B, "super": 0x5B, "cmd": 0x5B, "left_gui": 0x5B,
            "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
            "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
        }
        clean = str(key_name).strip().lower()
        if clean in aliases:
            return aliases[clean], False
        if len(clean) == 1 and _VkKeyScanW is not None:
            vk = _VkKeyScanW(ord(clean))
            if vk != -1:
                return vk & 0xFF, bool((vk >> 8) & 1)
        return None, False

    def key(self, key_name: str) -> bool:
        """Presses a single key or '+'-delimited key chord (e.g. 'enter', 'ctrl+s')."""
        parts = [p.strip() for p in str(key_name).split("+") if p.strip()]
        if not parts:
            return False
        if len(parts) > 1:
            return self.hotkey(*parts)

        self._check_failsafe()
        if sys.platform != "win32" or _keybd_event is None:
            return False

        vk, needs_shift = self._resolve_vk(parts[0])
        if vk is None:
            return False

        if needs_shift:
            _keybd_event(0x10, 0, 0, 0)
        try:
            _keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)
            _keybd_event(vk, 0, 0x0002, 0)
        finally:
            if needs_shift:
                _keybd_event(0x10, 0, 0x0002, 0)
        return True

    def hotkey(self, *keys: Union[str, Sequence[str]]) -> bool:
        """Presses a simultaneous key chord (e.g. hotkey('ctrl', 's') or hotkey(['ctrl', 's']))."""
        flat: List[str] = []
        for item in keys:
            if isinstance(item, (list, tuple)):
                flat.extend(str(k).strip() for k in item if str(k).strip())
            else:
                flat.extend(p.strip() for p in str(item).split("+") if p.strip())
        if not flat:
            return False

        self._check_failsafe()
        if sys.platform != "win32" or _keybd_event is None:
            return False

        resolved: List[int] = []
        for name in flat:
            vk, needs_shift = self._resolve_vk(name)
            if vk is None:
                return False
            if needs_shift and 0x10 not in resolved:
                resolved.append(0x10)
            resolved.append(vk)

        held: List[int] = []
        try:
            for vk in resolved:
                _keybd_event(vk, 0, 0, 0)
                held.append(vk)
                time.sleep(0.01)
            time.sleep(0.02)
        finally:
            for vk in reversed(held):
                _keybd_event(vk, 0, 0x0002, 0)
        return True

    def type_text(self, text: str, typing_speed_cps: float = 25.0) -> bool:
        """Types characters with human inter-key jitter."""
        self._check_failsafe()

        if sys.platform != "win32" or _VkKeyScanW is None or _keybd_event is None:
            return False
        base_delay = 1.0 / max(1.0, typing_speed_cps)

        for char in text:
            self._check_failsafe()
            vk = _VkKeyScanW(ord(char))
            if vk != -1:
                key_code = vk & 0xFF
                shift = (vk >> 8) & 1
                if shift:
                    _keybd_event(0x10, 0, 0, 0)
                _keybd_event(key_code, 0, 0, 0)
                time.sleep(0.02)
                _keybd_event(key_code, 0, 2, 0)
                if shift:
                    _keybd_event(0x10, 0, 2, 0)
            time.sleep(np.random.uniform(0.7, 1.3) * base_delay)
        return True

    def verify_ripple(
        self,
        capture_fn: Callable[[], np.ndarray],
        target_norm: Tuple[float, float],
        pre_frame: Optional[np.ndarray] = None,
        timeout_ms: float = 300.0,
        min_ripple_energy: float = 0.015,
    ) -> RippleResult:
        """Monitors local ROI around target for pure-pixel click reaction flash."""
        t0 = time.perf_counter()

        if pre_frame is None:
            pre_frame = capture_fn()

        h, w = pre_frame.shape[:2]
        if h == 0 or w == 0:
            return RippleResult(
                verified=False,
                surprise_magnitude=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                window_title_changed=False,
            )

        cx, cy = int(target_norm[0] * w), int(target_norm[1] * h)
        roi = 120
        x1, y1 = max(0, cx - roi // 2), max(0, cy - roi // 2)
        x2, y2 = min(w, x1 + roi), min(h, y1 + roi)
        pre_roi = pre_frame[y1:y2, x1:x2].astype(np.float32)
        diff_buf = np.empty_like(pre_roi)
        roi_size = pre_roi.size
        inv_norm = (1.0 / (255.0 * roi_size)) if roi_size > 0 else 0.0

        max_diff = 0.0
        verified = False

        if _timeBeginPeriod is not None:
            try:
                _timeBeginPeriod(1)
            except Exception:
                pass

        try:
            # Immediate capture before entering sleep loop (eliminating 8ms initial blind lag)
            post_frame = capture_fn()
            post_crop = post_frame[y1:y2, x1:x2]
            if post_crop.shape == pre_roi.shape:
                np.subtract(post_crop, pre_roi, out=diff_buf, casting="unsafe")
                np.abs(diff_buf, out=diff_buf)
                energy = float(np.sum(diff_buf)) * inv_norm
            else:
                post_roi = post_crop.astype(np.float32)
                diff = np.abs(post_roi - pre_roi) / 255.0
                energy = float(np.mean(diff))

            if energy > max_diff:
                max_diff = energy

            if max_diff >= min_ripple_energy:
                verified = True

            # If reaction is not caught on the immediate sample, enter adaptive polling loop
            while not verified and (time.perf_counter() - t0) * 1000.0 < timeout_ms:
                time.sleep(0.004)
                post_frame = capture_fn()
                post_crop = post_frame[y1:y2, x1:x2]
                if post_crop.shape == pre_roi.shape:
                    np.subtract(post_crop, pre_roi, out=diff_buf, casting="unsafe")
                    np.abs(diff_buf, out=diff_buf)
                    energy = float(np.sum(diff_buf)) * inv_norm
                else:
                    post_roi = post_crop.astype(np.float32)
                    diff = np.abs(post_roi - pre_roi) / 255.0
                    energy = float(np.mean(diff))

                if energy > max_diff:
                    max_diff = energy

                if max_diff >= min_ripple_energy:
                    verified = True
                    break
        finally:
            if _timeEndPeriod is not None:
                try:
                    _timeEndPeriod(1)
                except Exception:
                    pass

        return RippleResult(
            verified=verified,
            surprise_magnitude=round(max_diff, 4),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            window_title_changed=False,
        )
