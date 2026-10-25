"""[4] hand_motor.py: Biological Human Mouse Motion & Visual Ripple Verification.

Implements human-like 5th-order minimum-jerk kinematics (Flash & Hogan 1985)
and closed-loop visual ripple verification to confirm button depression.
"""
from __future__ import annotations

import ctypes
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RippleResult:
    """Outcome of visual ripple click verification."""
    verified: bool            # True if screen confirmed reaction (photometric change)
    surprise_magnitude: float # Magnitude of detected luminance ripple
    latency_ms: float         # Duration of verification check
    window_title_changed: bool# True if foreground window mutated


class CornerBreakoutFailSafe:
    """Instant emergency breakout if cursor is moved to screen corner."""

    @staticmethod
    def check() -> bool:
        if sys.platform != "win32":
            return False
        point = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        return point.x <= 2 and point.y <= 2


class HandMotor:
    """Biological minimum-jerk motor controller for natural human mouse trajectories."""

    def __init__(self, step_duration_ms: float = 12.0) -> None:
        self.step_duration_ms = step_duration_ms

    @staticmethod
    def get_cursor_pos() -> Tuple[int, int]:
        if sys.platform == "win32":
            point = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            return int(point.x), int(point.y)
        return 0, 0

    @staticmethod
    def set_cursor_pos(x: int, y: int) -> None:
        if sys.platform == "win32":
            ctypes.windll.user32.SetCursorPos(x, y)

    @staticmethod
    def get_foreground_title() -> str:
        if sys.platform != "win32":
            return ""
        try:
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
        """Glides cursor along minimum-jerk curve to target."""
        if sys.platform != "win32":
            return False

        start_pt = self.get_cursor_pos()
        waypoints = self.generate_trajectory(start_pt, (target_x, target_y), duration_s)

        for pt in waypoints:
            if CornerBreakoutFailSafe.check():
                print("[SAFETY] Corner breakout triggered! Motor execution halted.")
                return False
            self.set_cursor_pos(pt[0], pt[1])
            time.sleep(self.step_duration_ms / 1000.0)

        self.set_cursor_pos(target_x, target_y)
        return True

    def click(self, target_x: int, target_y: int) -> bool:
        """Moves cursor to target and issues native OS mouse click."""
        if not self.move_to(target_x, target_y):
            return False

        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            time.sleep(0.04)
            user32.mouse_event(0x0002, 0, 0, 0, 0) # Left down
            time.sleep(0.06)
            user32.mouse_event(0x0004, 0, 0, 0, 0) # Left up
            return True
        return False

    def type_text(self, text: str, typing_speed_cps: float = 25.0) -> bool:
        """Types characters with human inter-key jitter."""
        if sys.platform != "win32":
            return False
        user32 = ctypes.windll.user32
        base_delay = 1.0 / max(1.0, typing_speed_cps)

        for char in text:
            if CornerBreakoutFailSafe.check():
                return False
            vk = user32.VkKeyScanW(ord(char))
            if vk != -1:
                key_code = vk & 0xFF
                shift = (vk >> 8) & 1
                if shift:
                    user32.keybd_event(0x10, 0, 0, 0)
                user32.keybd_event(key_code, 0, 0, 0)
                time.sleep(0.02)
                user32.keybd_event(key_code, 0, 2, 0)
                if shift:
                    user32.keybd_event(0x10, 0, 2, 0)
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

        if sys.platform == "win32":
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass

        try:
            while (time.perf_counter() - t0) * 1000.0 < timeout_ms:
                time.sleep(0.008)
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
            if sys.platform == "win32":
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass

        return RippleResult(
            verified=verified,
            surprise_magnitude=round(max_diff, 4),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            window_title_changed=title_changed,
        )
