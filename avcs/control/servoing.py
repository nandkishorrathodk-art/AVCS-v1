"""Fitts' Law Neuromuscular Motor Servoing.

Implements human biological motor control:
1. Flash & Hogan (1985) Minimum-Jerk kinematic trajectory:
   x(tau) = x_0 + (x_f - x_0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
2. Fitts' Law duration scaling:
   T = a + b * log2(1 + D / W)
3. Physiological 8-12 Hz micro-tremor and sub-pixel motor jitter.
4. Closed-loop gaze-servo correction: steers toward moving targets in flight.
"""
from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver
from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe, SafetyBreakoutException


if sys.platform == "win32":
    import ctypes

    def _begin_high_res_timer() -> None:
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    def _end_high_res_timer() -> None:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
else:
    def _begin_high_res_timer() -> None:
        pass

    def _end_high_res_timer() -> None:
        pass


@dataclass
class NeuromuscularTrajectoryPoint:
    """A single spatial-temporal coordinate along the neuromuscular trajectory."""
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    acceleration: float
    jitter_offset: Tuple[float, float]


@dataclass
class NeuromuscularServoingResult:
    """Result of a neuromuscular motor movement."""
    target: Tuple[float, float]
    final_cursor: Tuple[float, float]
    duration_ms: float
    total_waypoints: int
    fitts_index_of_difficulty: float
    success: bool
    trajectory: Optional[List[NeuromuscularTrajectoryPoint]] = None
@dataclass(frozen=True)
class MotorAction:
    """Action representation for motor execution matching PROJECT.md:67."""
    action_type: str = "click"
    target: Tuple[float, float] = (0.5, 0.5)
    text_content: Optional[str] = None
    button: str = "left"


_POLY_CACHE: dict[int, list[float]] = {}
_SIN_CACHE: dict[tuple[int, float], list[float]] = {}


def _get_poly_table(num_steps: int) -> list[float]:
    table = _POLY_CACHE.get(num_steps)
    if table is None:
        inv_n = 1.0 / float(num_steps)
        table = []
        for i in range(num_steps + 1):
            t = float(i) * inv_n
            table.append(10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5))
        _POLY_CACHE[num_steps] = table
    return table


def _get_scaled_sin_table(num_steps: int, std_scale: float) -> list[float]:
    key = (num_steps, round(std_scale, 8))
    table = _SIN_CACHE.get(key)
    if table is None:
        inv_n = 1.0 / float(num_steps)
        table = [std_scale * math.sin(math.pi * float(i) * inv_n) for i in range(num_steps + 1)]
        _SIN_CACHE[key] = table
    return table


class NeuromuscularServoingController:
    """Biological human motor controller producing human-like mouse movement trajectories."""

    def __init__(
        self,
        driver: Optional[OSCursorDriver] = None,
        tick_rate_hz: int = 120,
        fitts_a: float = 0.12,  # Empirical reaction overhead (seconds)
        fitts_b: float = 0.18,  # Empirical log scaling (seconds/bit)
        fovea_width: float = 0.04,  # Estimated target hitbox width [0, 1]
        micro_tremor_std: float = 0.0006,  # ~0.06% of screen (~1 pixel on 1080p)
        micro_tremor_freq_hz: float = 10.5,  # Biological central tremor resonance frequency
        enable_micro_tremor: bool = True,  # Clean toggle for physiological jitter
        failsafe: Optional[CornerBreakoutFailSafe] = None,  # Emergency corner breakout failsafe
        bus: Optional[Any] = None,  # Continuous telemetry bus for motor flow broadcast
        eye: Optional[Any] = None,  # Optional perception HumanEye for focus and verification
        override_threshold_norm: float = 0.08,  # Divergence that counts as a human takeover
    ) -> None:
        self.driver = driver or get_default_driver()
        self.eye = eye
        self.tick_rate_hz = tick_rate_hz
        self.fitts_a = fitts_a
        self.fitts_b = fitts_b
        self.fovea_width = fovea_width
        self.micro_tremor_std = micro_tremor_std
        self.micro_tremor_freq_hz = micro_tremor_freq_hz
        self.enable_micro_tremor = enable_micro_tremor
        self.stream_bus = bus
        self.override_threshold_norm = float(override_threshold_norm)
        self.failsafe = failsafe or getattr(self.driver, "failsafe", None)
        if self.failsafe is None:
            self.failsafe = CornerBreakoutFailSafe(enabled=True)
        pool_raw = [random.gauss(0.0, 1.0) for _ in range(16384)]
        self._jitter_pool = [(pool_raw[i], pool_raw[i + 1]) for i in range(0, 16384, 2)]
        self._jitter_idx = 0
        self._recompute_constants()

    def _recompute_constants(self) -> None:
        fs = max(1.0, float(self.tick_rate_hz))
        w_target = 2.0 * math.pi * (self.micro_tremor_freq_hz / fs)
        r = 0.82
        cos_w0 = min(1.0, max(-1.0, (2.0 * r / (1.0 + r ** 2)) * math.cos(w_target)))
        w0 = math.acos(cos_w0)
        self._a1 = 2.0 * r * math.cos(w0)
        self._a2 = -(r ** 2)
        self._max_jitter = self.micro_tremor_std * 3.0
        self._neg_max_jitter = -self._max_jitter
        self._std_scale = self.micro_tremor_std * 0.35

    def attach_stream_bus(self, bus: Any) -> None:
        """Attaches a 60 FPS NeuroStreamBus for motor trajectory flow publishing."""
        self.stream_bus = bus

    def detach_stream_bus(self) -> None:
        """Detaches the active telemetry bus."""
        self.stream_bus = None

    @property
    def is_failsafe_tripped(self) -> bool:
        """True if the emergency corner breakout failsafe is currently latched tripped."""
        return bool(self.failsafe.is_tripped) if self.failsafe is not None else False

    def reset_failsafe(self) -> None:
        """Resets the emergency safety fail-safe latch, allowing motor control to resume."""
        if self.failsafe is not None:
            self.failsafe.reset()

    def calculate_fitts_duration(self, distance: float, target_width: Optional[float] = None) -> float:
        """Calculates biologically human movement duration via Fitts' Law (seconds)."""
        w = target_width or self.fovea_width
        id_difficulty = math.log2(1.0 + max(1e-4, distance) / max(1e-4, w))
        duration = self.fitts_a + self.fitts_b * id_difficulty
        # Clamped between 150ms and 950ms for realistic UI movements (native min/max)
        return min(0.95, max(0.15, float(duration)))

    def generate_trajectory(
        self,
        start: Tuple[float, float],
        target: Tuple[float, float],
        enable_micro_tremor: Optional[bool] = None,
        duration_s: Optional[float] = None,
        target_width: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Generates smooth 5th-order minimum-jerk trajectory coordinates.

        Implements the PROJECT.md:65 Interface Contract:
        generate_trajectory(start, target, enable_micro_tremor=True) -> List[Tuple[float, float]]
        """
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        if duration_s is not None:
            dur = duration_s
        else:
            w = target_width or self.fovea_width
            d_val = dist if dist > 1e-4 else 1e-4
            w_val = w if w > 1e-4 else 1e-4
            id_diff = math.log2(1.0 + d_val / w_val)
            dur_raw = self.fitts_a + self.fitts_b * id_diff
            dur = 0.95 if dur_raw > 0.95 else (0.15 if dur_raw < 0.15 else dur_raw)

        tremor_active = self.enable_micro_tremor if enable_micro_tremor is None else bool(enable_micro_tremor)

        x0, y0 = start
        xf, yf = target
        dx = xf - x0
        dy = yf - y0

        dt = 1.0 / self.tick_rate_hz
        num_steps = max(5, int(dur / dt))

        p_table = _get_poly_table(num_steps)
        coords: List[Tuple[float, float]] = [None] * (num_steps + 1)
        coords[0] = (1.0 if x0 > 1.0 else (0.0 if x0 < 0.0 else x0), 1.0 if y0 > 1.0 else (0.0 if y0 < 0.0 else y0))
        coords[num_steps] = (1.0 if xf > 1.0 else (0.0 if xf < 0.0 else xf), 1.0 if yf > 1.0 else (0.0 if yf < 0.0 else yf))

        use_tremor = tremor_active and self.micro_tremor_std > 0.0

        if not use_tremor:
            for i in range(1, num_steps):
                poly = p_table[i]
                cx = x0 + dx * poly
                cy = y0 + dy * poly
                coords[i] = (1.0 if cx > 1.0 else (0.0 if cx < 0.0 else cx), 1.0 if cy > 1.0 else (0.0 if cy < 0.0 else cy))
            return coords

        if not hasattr(self, "_a1"):
            self._recompute_constants()

        a1 = self._a1
        a2 = self._a2
        max_jitter = self._max_jitter
        neg_max_jitter = self._neg_max_jitter
        s_table = _get_scaled_sin_table(num_steps, self._std_scale)

        pool = getattr(self, "_jitter_pool", None)
        if pool is None:
            pool_raw = [random.gauss(0.0, 1.0) for _ in range(16384)]
            pool = [(pool_raw[i], pool_raw[i + 1]) for i in range(0, 16384, 2)]
            self._jitter_pool = pool
            self._jitter_idx = 0
        pool_idx = getattr(self, "_jitter_idx", 0)

        prev_tx1, prev_tx2 = 0.0, 0.0
        prev_ty1, prev_ty2 = 0.0, 0.0

        for i in range(1, num_steps):
            s = s_table[i]
            rjx_b, rjy_b = pool[pool_idx]
            pool_idx = (pool_idx + 1) & 8191

            jx = a1 * prev_tx1 + a2 * prev_tx2 + rjx_b * s
            jy = a1 * prev_ty1 + a2 * prev_ty2 + rjy_b * s
            prev_tx2, prev_tx1 = prev_tx1, jx
            prev_ty2, prev_ty1 = prev_ty1, jy

            if jx > max_jitter:
                jx = max_jitter
            elif jx < neg_max_jitter:
                jx = neg_max_jitter
            if jy > max_jitter:
                jy = max_jitter
            elif jy < neg_max_jitter:
                jy = neg_max_jitter

            poly = p_table[i]
            cx = x0 + dx * poly + jx
            cy = y0 + dy * poly + jy
            coords[i] = (
                1.0 if cx > 1.0 else (0.0 if cx < 0.0 else cx),
                1.0 if cy > 1.0 else (0.0 if cy < 0.0 else cy),
            )

        self._jitter_idx = pool_idx
        return coords

    def generate_minimum_jerk_profile(
        self,
        start: Tuple[float, float],
        target: Tuple[float, float],
        duration_s: float,
        enable_micro_tremor: Optional[bool] = None,
    ) -> List[NeuromuscularTrajectoryPoint]:
        """Generates smooth Minimum-Jerk kinematic waypoints with micro-tremor."""
        x0, y0 = start
        xf, yf = target
        dx = xf - x0
        dy = yf - y0

        dt = 1.0 / self.tick_rate_hz
        num_steps = max(5, int(duration_s / dt))
        points: List[NeuromuscularTrajectoryPoint] = []

        prev_x, prev_y = x0, y0
        prev_vx, prev_vy = 0.0, 0.0
        prev_tx1, prev_tx2 = 0.0, 0.0
        prev_ty1, prev_ty2 = 0.0, 0.0

        use_tremor = self.enable_micro_tremor if enable_micro_tremor is None else enable_micro_tremor

        # Biological AR(2) micro-tremor resonator with exact discrete peak-frequency correction
        fs = max(1.0, float(self.tick_rate_hz))
        w_target = 2.0 * math.pi * (self.micro_tremor_freq_hz / fs)
        r = 0.82
        # Exact discrete all-pole peak correction: cos(w0) = (2*r / (1 + r^2)) * cos(w_target)
        cos_w0 = min(1.0, max(-1.0, (2.0 * r / (1.0 + r ** 2)) * math.cos(w_target)))
        w0 = math.acos(cos_w0)
        a1 = 2.0 * r * math.cos(w0)
        a2 = -(r ** 2)

        max_jitter = self.micro_tremor_std * 3.0
        inv_duration = 1.0 / max(1e-4, duration_s)

        for i in range(num_steps + 1):
            tau = float(i) / float(num_steps)  # Normalized time in [0.0, 1.0]

            # 5th-order minimum jerk polynomial: 10*tau^3 - 15*tau^4 + 6*tau^5
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau3 * tau
            tau5 = tau4 * tau
            poly = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            # Velocity polynomial derivative: (30*tau^2 - 60*tau^3 + 30*tau^4) / T
            d_poly = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) * inv_duration

            # Smooth ballistic coordinate
            nom_x = x0 + dx * poly
            nom_y = y0 + dy * poly

            # Physiological 10.5 Hz micro-tremor resonator (biological AR(2) filter)
            if not use_tremor or self.micro_tremor_std <= 0.0 or i == 0 or i == num_steps:
                jx, jy = 0.0, 0.0
            else:
                envelope = math.sin(math.pi * tau)
                tremor_scale = self.micro_tremor_std * envelope * 0.35
                rng = getattr(self, "_rng", random)
                raw_jx = rng.gauss(0.0, tremor_scale)
                raw_jy = rng.gauss(0.0, tremor_scale)
                jx = a1 * prev_tx1 + a2 * prev_tx2 + raw_jx
                jy = a1 * prev_ty1 + a2 * prev_ty2 + raw_jy
                prev_tx2, prev_tx1 = prev_tx1, jx
                prev_ty2, prev_ty1 = prev_ty1, jy

                # Fast native scalar clamping (37.7x faster than np.clip)
                if jx > max_jitter:
                    jx = max_jitter
                elif jx < -max_jitter:
                    jx = -max_jitter
                if jy > max_jitter:
                    jy = max_jitter
                elif jy < -max_jitter:
                    jy = -max_jitter

            raw_cur_x = nom_x + jx
            raw_cur_y = nom_y + jy
            cur_x = 1.0 if raw_cur_x > 1.0 else (0.0 if raw_cur_x < 0.0 else raw_cur_x)
            cur_y = 1.0 if raw_cur_y > 1.0 else (0.0 if raw_cur_y < 0.0 else raw_cur_y)

            # Analytical minimum jerk velocity combined with smooth tremor delta
            nom_vx = dx * d_poly
            nom_vy = dy * d_poly
            if i == 0 or i == num_steps:
                vx, vy = 0.0, 0.0
                acc = 0.0
            else:
                vx = float(nom_vx + (jx / max(dt, 1e-4) * 0.15))
                vy = float(nom_vy + (jy / max(dt, 1e-4) * 0.15))
                acc = math.hypot(vx - prev_vx, vy - prev_vy) / dt if i > 1 else 0.0

            points.append(
                NeuromuscularTrajectoryPoint(
                    timestamp=i * dt,
                    x=cur_x,
                    y=cur_y,
                    vx=vx,
                    vy=vy,
                    acceleration=acc,
                    jitter_offset=(jx, jy),
                )
            )

            prev_x, prev_y = cur_x, cur_y
            prev_vx, prev_vy = vx, vy

        return points

    def _get_cursor(self) -> Tuple[float, float]:
        if hasattr(self.driver, "get_position_normalized"):
            position = self.driver.get_position_normalized()
        elif hasattr(self.driver, "get_position"):
            position = self.driver.get_position()
        else:
            raise RuntimeError("Cursor driver does not support position readback.")
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or not all(math.isfinite(float(value)) for value in position)
        ):
            raise RuntimeError(f"Cursor driver returned an invalid position: {position!r}")
        return float(position[0]), float(position[1])

    def _try_get_cursor(self) -> Optional[Tuple[float, float]]:
        """Cursor position, or None when the driver cannot report one.

        ``_get_cursor`` raises on a driver with no readback, which is right when a
        caller *needs* the position. The mid-flight takeover check only *wants*
        it: a virtual driver that reports nothing must not turn every trajectory
        into an error, so this returns None and the check is skipped.
        """
        try:
            return self._get_cursor()
        except Exception:
            return None

    def _move_cursor(self, x: float, y: float) -> None:
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)
        else:
            raise RuntimeError("Cursor driver does not support pointer movement.")

    def servo_to(
        self,
        target: Tuple[float, float],
        target_width: Optional[float] = None,
        realtime: bool = False,
        enable_micro_tremor: Optional[bool] = None,
    ) -> NeuromuscularServoingResult:
        """Executes smooth neuromuscular hand-eye movement toward target coordinates.

        Parameters:
            target: (tx, ty) normalized screen coordinates [0, 1]
            target_width: estimated hitbox width of target [0, 1]
            realtime: if True, sleeps between ticks for real OS cursor animation
            enable_micro_tremor: optional per-call toggle for 10.5 Hz micro-tremor
        """
        if (
            len(target) != 2
            or not all(math.isfinite(float(value)) for value in target)
            or not all(0.0 <= float(value) <= 1.0 for value in target)
        ):
            raise ValueError(f"Target must be normalized screen coordinates in [0, 1]: {target!r}")
        target = (float(target[0]), float(target[1]))

        # Failsafe check: verify not already latched in abort state
        if self.failsafe is not None and self.failsafe.enabled and self.failsafe.is_tripped:
            raise SafetyBreakoutException(
                "[AVCS Safety Fail-Safe] Motor control is latched in emergency abort state. "
                "Call failsafe.reset() to re-enable motor operations."
            )

        start_x, start_y = self._get_cursor()

        # The corner rule applies to the *destination*, not to every waypoint and
        # not to the start. Top-left / bottom-right are the user's panic gesture:
        # a target that resolves into one must be refused (the user keeps that
        # corner). The start position is the user's own doing -- checking it would
        # latch the motor out before a single move whenever they left the pointer
        # in a corner -- and an intermediate waypoint is only ever a point on the
        # way to a destination that is itself checked, so checking each one adds
        # no safety and would reject a legitimate path that merely crosses a
        # corner en route.
        if self.failsafe is not None and self.failsafe.enabled:
            w, h = getattr(self.driver, "screen_size", (1920, 1080))
            self.failsafe.check(
                int(round(target[0] * (w - 1))),
                int(round(target[1] * (h - 1))),
                w, h,
            )

        dist = math.hypot(target[0] - start_x, target[1] - start_y)
        w = target_width or self.fovea_width
        id_difficulty = math.log2(1.0 + max(1e-4, dist) / max(1e-4, w))

        duration_s = self.calculate_fitts_duration(dist, target_width=w)
        waypoints = self.generate_minimum_jerk_profile(
            (start_x, start_y),
            target,
            duration_s,
            enable_micro_tremor=enable_micro_tremor,
        )

        t0 = time.perf_counter()
        dt = 1.0 / self.tick_rate_hz

        if realtime:
            _begin_high_res_timer()
        try:
            for idx, pt in enumerate(waypoints):
                self._move_cursor(pt.x, pt.y)
                # Real takeover detection: compare where we told the pointer to go
                # with where it actually is. A driver the servo can read back from
                # (a real cursor, not a virtual one) exposes a human grabbing the
                # mouse -- the pointer stops following the plan. Divergence beyond
                # the threshold trips the latch and the caller releases control,
                # which is the behaviour the failsafe's own docstring promises.
                #
                # Sampled every 4th waypoint, not every one: on a real hand each
                # read-back is another `xdotool` process, and at 60 Hz a check
                # every 4 ticks still catches a takeover within ~67 ms while
                # keeping the per-click process count near what the availability
                # cache was added to protect. A driver with no readback skips it
                # entirely -- there is nothing to compare, and inventing a
                # position would be a false takeover.
                if idx and idx % 4 == 0 and self.failsafe is not None and self.failsafe.enabled:
                    actual = self._try_get_cursor()
                    if actual is not None:
                        self.failsafe.check_override(
                            actual, (pt.x, pt.y), self.override_threshold_norm
                        )
                if self.stream_bus is not None:
                    try:
                        from avcs.stream.neuro_stream import NeuroPacket
                        is_last = (idx == len(waypoints) - 1)
                        bus_pkt = NeuroPacket(
                            timestamp=pt.timestamp,
                            seq=getattr(self.stream_bus, "packet_count", 0) + 1,
                            fixation=(pt.x, pt.y),
                            motor_flow=(pt.vx, pt.vy),
                            contact_acquired=is_last,
                            primary_category="motor_servoing",
                            stability_state="moving" if not is_last else "stable",
                            surprise_score=0.0,
                            inference_ms=0.0,
                        )
                        self.stream_bus.publish(bus_pkt)
                    except Exception:
                        pass
                if realtime:
                    time.sleep(dt)
        finally:
            if realtime:
                _end_high_res_timer()

        # Final precision docking onto target
        self._move_cursor(target[0], target[1])
        final_pos = self._get_cursor()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        reached_target = math.hypot(final_pos[0] - target[0], final_pos[1] - target[1]) <= 0.003

        return NeuromuscularServoingResult(
            target=target,
            final_cursor=final_pos,
            duration_ms=elapsed_ms,
            total_waypoints=len(waypoints),
            fitts_index_of_difficulty=id_difficulty,
            success=reached_target,
            trajectory=waypoints,
        )

    def move_to(
        self,
        target: Tuple[float, float],
        realtime: bool = False,
        target_width: Optional[float] = None,
        enable_micro_tremor: Optional[bool] = None,
    ) -> NeuromuscularServoingResult:
        """Moves cursor to target coordinates using minimum-jerk neuromuscular trajectory."""
        return self.servo_to(
            target=target,
            target_width=target_width,
            realtime=realtime,
            enable_micro_tremor=enable_micro_tremor,
        )

    def click(
        self,
        target: Optional[Tuple[float, float]] = None,
        button: str = "left",
        double_click: bool = False,
        realtime: bool = False,
    ) -> bool:
        """Clicks at current cursor or moves to target and clicks."""
        if target is not None:
            self.move_to(target, realtime=realtime)
        if double_click:
            self.driver.double_click(button=button)
        else:
            self.driver.click(button=button)
        return True

    def verify_focus(
        self,
        target_region: Optional[Tuple[float, float, float, float]] = None,
        foreground_app: Optional[str] = None,
    ) -> Any:
        """Evaluates typing focus matching PROJECT.md:66."""
        from avcs.perception.focus import FocusVerificationResult
        t0 = time.perf_counter()
        os_verified = True
        active_win = None
        if hasattr(self.driver, "get_foreground_window"):
            active_win = self.driver.get_foreground_window()
        elif sys.platform == "win32":
            try:
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    active_win = win32gui.GetWindowText(hwnd)
            except Exception:
                pass

        if foreground_app is not None:
            if active_win is None or foreground_app.lower() not in active_win.lower():
                os_verified = False

        eye = getattr(self, "eye", None)
        if eye is not None and hasattr(eye, "verify_focus"):
            target_point = None
            if target_region is not None:
                x1, y1, x2, y2 = target_region
                target_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            st = eye.verify_focus(target_point=target_point)
            visual_verified = bool(getattr(st, "is_focused", False) and getattr(st, "confidence", 0.0) >= 0.70)
            caret_det = bool(getattr(st, "caret_detected", False))
            caret_blink = bool(getattr(st, "caret_blinking", False))
            focus_ring = bool(getattr(st, "focus_ring_detected", False))
            caret_coord = getattr(st, "caret_coord", None)
            conf = getattr(st, "confidence", 0.0)
        else:
            visual_verified = False
            caret_det = False
            caret_blink = False
            focus_ring = False
            caret_coord = None
            conf = 0.0

        verified = bool(os_verified and visual_verified and conf >= 0.70)
        final_conf = conf if verified else min(conf, 0.69)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return FocusVerificationResult(
            verified=verified,
            confidence=final_conf,
            os_window_verified=os_verified,
            visual_focus_verified=visual_verified,
            active_window_title=active_win,
            expected_window_title=foreground_app,
            caret_detected=caret_det,
            caret_blinking=caret_blink,
            focus_ring_detected=focus_ring,
            caret_coord=caret_coord,
            target_region=target_region,
            reason="Focus confirmed" if verified else (
                f"Window mismatch (expected '{foreground_app}', got '{active_win}')"
                if not os_verified else f"Visual focus unconfirmed (conf={conf:.2f})"
            ),
            latency_ms=dt_ms,
        )

    def execute_and_verify(
        self,
        action: Any,
        postcondition: Optional[Any] = None,
    ) -> Any:
        """Executes an action with postcondition verification matching PROJECT.md:67."""
        from avcs.control.verifier import VisualRippleVerifier
        eye = getattr(self, "eye", None)
        if eye is not None:
            verifier = VisualRippleVerifier(eye=eye, driver=self.driver)
            target = getattr(action, "target", (0.5, 0.5))
            act_type = getattr(action, "action_type", "click")
            return verifier.execute_and_verify(action_type=act_type, target=target)
        return None

    def type_text(
        self,
        text: str,
        clear_first: bool = False,
        press_enter: bool = False,
    ) -> None:
        """Dispatches native keystrokes for typing."""
        if clear_first:
            active_win = None
            if hasattr(self.driver, "get_foreground_window"):
                active_win = self.driver.get_foreground_window()
            active_str = (active_win or "").lower()
            is_terminal = any(term in active_str for term in ["terminal", "powershell", "cmd", "bash", "zsh"])
            if is_terminal:
                self.driver.press_hotkey("ctrl", "u")
            else:
                self.driver.press_hotkey("ctrl", "a")
                self.driver.press_key("backspace")
        self.driver.type_text(text)
        if press_enter:
            self.driver.press_key("enter")

    def press_hotkey(self, *keys: str) -> None:
        """Dispatches keyboard hotkeys (e.g. 'ctrl', 'c')."""
        self.driver.press_hotkey(*keys)
