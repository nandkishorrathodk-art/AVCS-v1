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
from typing import Callable, List, Optional, Tuple

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
        _GetForegroundWindow = _user32.GetForegroundWindow
        _GetWindowTextLengthW = _user32.GetWindowTextLengthW
        _GetWindowTextW = _user32.GetWindowTextW
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
        _GetForegroundWindow = None
        _GetWindowTextLengthW = None
        _GetWindowTextW = None
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
    _GetForegroundWindow = None
    _GetWindowTextLengthW = None
    _GetWindowTextW = None
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
        if sys.platform != "win32":
            return ""
        try:
            if _GetForegroundWindow is not None and _GetWindowTextW is not None and _GetWindowTextLengthW is not None:
                hwnd = _GetForegroundWindow()
                length = _GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                _GetWindowTextW(hwnd, buff, length + 1)
                return buff.value
            else:
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                return buff.value
        except Exception:
            return ""

    def generate_trajectory(
        self,
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        duration_s: float = 0.35,
    ) -> List[Tuple[int, int]]:
        """Generates 5th-order minimum-jerk trajectory (Flash & Hogan 1985)."""
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

    def click(self, target_x: int, target_y: int) -> bool:
        """Moves cursor to target and issues native OS mouse click.

        Returns False on non-Windows platforms: this controller drives the
        native Win32 cursor API, so it cannot actuate a pointer elsewhere. It
        never reports success for an action it did not perform.
        """
        self._check_failsafe(check_pos=(target_x, target_y))

        if sys.platform != "win32":
            return False

        if not self.move_to(target_x, target_y):
            return False

        self._check_failsafe()

        if sys.platform == "win32" and _mouse_event is not None:
            time.sleep(0.04)
            _mouse_event(0x0002, 0, 0, 0, 0) # Left down
            time.sleep(0.06)
            _mouse_event(0x0004, 0, 0, 0, 0) # Left up
            return True
        return False

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
        """Monitors local ROI around target for click reaction flash or window change."""
        t0 = time.perf_counter()
        initial_title = self.get_foreground_title()

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
        title_changed = False

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

            curr_title = self.get_foreground_title()
            if curr_title and curr_title != initial_title:
                title_changed = True
                verified = True
            elif max_diff >= min_ripple_energy:
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

                curr_title = self.get_foreground_title()
                if curr_title and curr_title != initial_title:
                    title_changed = True
                    verified = True
                    break

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
            window_title_changed=title_changed,
        )
