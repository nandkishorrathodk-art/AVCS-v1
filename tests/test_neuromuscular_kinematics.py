"""Rigorous Unit Tests for AVCS Neuromuscular Kinematics & Flash & Hogan (1985) Trajectories.

Verifies:
1. Flash & Hogan (1985) 5th-order minimum-jerk trajectory generation:
   - Zero start/end velocity: vx=0, vy=0 at t=0 and t=T.
   - Zero start/end acceleration: acc=0 at t=0 and t=T.
   - Natural symmetric bell-shaped velocity curve peaking at tau=0.5 with vmax ~ 1.875 * D / T.
   - Boundary position precision: exact start and target coordinates docking.
2. Biological micro-tremor (10.5 Hz):
   - Jitter envelope bounds within human physiological jitter (~1-2 pixels on 1080p).
   - Clean configurability (enable_micro_tremor toggle, micro_tremor_std scaling, micro_tremor_freq_hz tuning).
   - Tremor smoothly vanishes to zero at start and landing.
3. Emergency corner breakout failsafe:
   - All 4 screen corners trip SafetyBreakoutException.
   - Dispatch halts immediately upon corner breach; subsequent waypoints are discarded.
   - Fail-safe latches until explicitly reset via reset_failsafe().
   - Fallback protection for drivers without native failsafe handling.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from avcs.servoing import (
    NeuromuscularServoingController,
    NeuromuscularServoingResult,
    NeuromuscularTrajectoryPoint,
)
from forge.vision.cortex.motor.driver import VirtualCursorDriver
from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe, SafetyBreakoutException


class TestFlashHoganMinimumJerkKinematics:
    """Verifies Flash & Hogan (1985) minimum-jerk trajectory generation and physical kinematics."""

    def test_zero_boundary_velocity_and_acceleration(self):
        """Ensure strictly zero start and end velocity and acceleration."""
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=120)

        start = (0.15, 0.25)
        target = (0.85, 0.75)
        duration = 0.50

        waypoints = controller.generate_minimum_jerk_profile(start, target, duration)
        assert len(waypoints) > 20

        # Start boundary conditions: v(0) = 0, a(0) = 0
        p_start = waypoints[0]
        assert p_start.vx == 0.0, f"Start vx must be strictly 0.0, got {p_start.vx}"
        assert p_start.vy == 0.0, f"Start vy must be strictly 0.0, got {p_start.vy}"
        assert p_start.acceleration == 0.0, f"Start acc must be strictly 0.0, got {p_start.acceleration}"
        assert p_start.x == start[0]
        assert p_start.y == start[1]
        assert p_start.jitter_offset == (0.0, 0.0)

        # End boundary conditions: v(T) = 0, a(T) = 0
        p_end = waypoints[-1]
        assert p_end.vx == 0.0, f"End vx must be strictly 0.0, got {p_end.vx}"
        assert p_end.vy == 0.0, f"End vy must be strictly 0.0, got {p_end.vy}"
        assert p_end.acceleration == 0.0, f"End acc must be strictly 0.0, got {p_end.acceleration}"
        assert p_end.x == target[0]
        assert p_end.y == target[1]
        assert p_end.jitter_offset == (0.0, 0.0)

    def test_bell_shaped_velocity_profile_and_peak_speed(self):
        """Verify natural bell-shaped velocity curve with theoretical peak at tau=0.5."""
        driver = VirtualCursorDriver()
        # Disable micro-tremor to isolate clean analytical minimum-jerk velocity curve
        controller = NeuromuscularServoingController(
            driver=driver,
            tick_rate_hz=120,
            enable_micro_tremor=False,
        )

        start = (0.2, 0.3)
        target = (0.8, 0.3)
        duration = 0.40  # seconds
        dist = 0.60  # dx = 0.6, dy = 0.0

        waypoints = controller.generate_minimum_jerk_profile(start, target, duration)
        speeds = [math.hypot(p.vx, p.vy) for p in waypoints]
        mid_idx = len(waypoints) // 2

        # 1. Peak speed must occur at the midpoint (tau ~ 0.5)
        peak_idx = int(np.argmax(speeds))
        assert abs(peak_idx - mid_idx) <= 1, f"Peak velocity at step {peak_idx}, expected near {mid_idx}"

        # 2. Flash & Hogan analytical peak speed is 1.875 * (D / T)
        # For D = 0.6, T = 0.4 -> 1.875 * (0.6 / 0.4) = 2.8125
        theoretical_vmax = 1.875 * (dist / duration)
        assert pytest.approx(speeds[mid_idx], rel=0.01) == theoretical_vmax

        # 3. Monotonic rise in first half, monotonic fall in second half
        # Check that speed at tau=0.25 is less than peak, and speed at tau=0.75 is less than peak
        idx_q1 = len(waypoints) // 4
        idx_q3 = (len(waypoints) * 3) // 4
        assert speeds[0] < speeds[idx_q1] < speeds[mid_idx]
        assert speeds[mid_idx] > speeds[idx_q3] > speeds[-1]

        # 4. Bell-curve symmetry: v(0.5 - dt) ~ v(0.5 + dt)
        for offset in [2, 5, 10]:
            left_speed = speeds[mid_idx - offset]
            right_speed = speeds[mid_idx + offset]
            assert pytest.approx(left_speed, rel=0.02) == right_speed

    def test_trajectory_end_to_end_smoothness(self):
        """Verify no teleports or discontinuous position jumps along the trajectory."""
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=120)

        waypoints = controller.generate_minimum_jerk_profile((0.1, 0.1), (0.9, 0.9), duration_s=0.5)
        dt = 1.0 / 120.0

        for i in range(1, len(waypoints)):
            p0 = waypoints[i - 1]
            p1 = waypoints[i]
            step_distance = math.hypot(p1.x - p0.x, p1.y - p0.y)
            # Max step distance in 8.3ms at peak speed (vmax ~ 4.2) is ~0.035
            assert step_distance < 0.05, f"Step distance jump too large: {step_distance}"


class TestBiologicalMicroTremor:
    """Verifies physiological 10.5 Hz micro-tremor and sub-pixel human jitter bounds."""

    def test_jitter_within_human_pixel_bounds(self):
        """Verify micro-tremor stays within human physical bounds (~1-2 pixels on 1080p)."""
        driver = VirtualCursorDriver()
        # Default micro_tremor_std is 0.0006 (~1.15 px on 1920x1080)
        controller = NeuromuscularServoingController(
            driver=driver,
            tick_rate_hz=120,
            micro_tremor_std=0.0006,
            enable_micro_tremor=True,
        )

        waypoints = controller.generate_minimum_jerk_profile((0.2, 0.2), (0.8, 0.8), duration_s=0.5)

        # On 1920x1080: 2 pixels = 2 / 1080 ~ 0.00185
        # 3 * micro_tremor_std = 0.0018
        max_jitter_allowed = 0.0006 * 3.0

        for pt in waypoints:
            jx, jy = pt.jitter_offset
            assert abs(jx) <= max_jitter_allowed + 1e-6, f"Horizontal jitter {jx} exceeded limit {max_jitter_allowed}"
            assert abs(jy) <= max_jitter_allowed + 1e-6, f"Vertical jitter {jy} exceeded limit {max_jitter_allowed}"

    def test_micro_tremor_cleanly_configurable_and_toggleable(self):
        """Verify enabling/disabling micro-tremor completely removes jitter."""
        driver = VirtualCursorDriver()
        # Disabled
        controller_off = NeuromuscularServoingController(
            driver=driver,
            enable_micro_tremor=False,
        )
        pts_off = controller_off.generate_minimum_jerk_profile((0.1, 0.1), (0.7, 0.7), 0.3)
        assert all(pt.jitter_offset == (0.0, 0.0) for pt in pts_off)

        # Enabled
        controller_on = NeuromuscularServoingController(
            driver=driver,
            enable_micro_tremor=True,
            micro_tremor_std=0.001,
        )
        pts_on = controller_on.generate_minimum_jerk_profile((0.1, 0.1), (0.7, 0.7), 0.3)
        in_flight_jitter = [pt.jitter_offset for pt in pts_on[1:-1]]
        assert any(jx != 0.0 or jy != 0.0 for jx, jy in in_flight_jitter)

    def test_micro_tremor_frequency_adaptability(self):
        """Verify AR(2) resonator pole computation adapts cleanly across sample rates."""
        # Test varying tick rates: 60Hz, 100Hz, 120Hz, 240Hz
        for fs in [60, 100, 120, 240]:
            ctrl = NeuromuscularServoingController(
                tick_rate_hz=fs,
                micro_tremor_freq_hz=10.5,
            )
            pts = ctrl.generate_minimum_jerk_profile((0.2, 0.2), (0.8, 0.8), duration_s=0.4)
            assert len(pts) >= int(0.4 * fs)
            assert pts[0].jitter_offset == (0.0, 0.0)
            assert pts[-1].jitter_offset == (0.0, 0.0)


class TestEmergencyCornerBreakoutFailsafe:
    """Verifies emergency corner breakout failsafe halting motor dispatch."""

    @pytest.mark.parametrize(
        "corner_coord,corner_name",
        [
            ((0.001, 0.001), "top-left"),
            ((0.999, 0.001), "top-right"),
            ((0.001, 0.999), "bottom-left"),
            ((0.999, 0.999), "bottom-right"),
        ],
        ids=["top-left", "top-right", "bottom-left", "bottom-right"],
    )
    def test_corner_breakout_trips_all_four_corners(self, corner_coord, corner_name):
        """Verify steering toward any corner raises SafetyBreakoutException."""
        driver = VirtualCursorDriver(screen_size=(1920, 1080), initial_position=(0.5, 0.5))
        controller = NeuromuscularServoingController(driver=driver)

        with pytest.raises(SafetyBreakoutException) as exc_info:
            controller.servo_to(corner_coord, realtime=False)

        assert corner_name in str(exc_info.value).lower()
        assert controller.is_failsafe_tripped is True

    def test_failsafe_halts_motor_dispatch_immediately(self):
        """Verify motor execution ceases instantly and does not finish remaining waypoints."""
        driver = VirtualCursorDriver(screen_size=(1920, 1080), initial_position=(0.5, 0.5))
        controller = NeuromuscularServoingController(driver=driver)

        # Pre-compute total planned waypoints
        duration = controller.calculate_fitts_duration(math.hypot(0.5, 0.5))
        planned_waypoints = controller.generate_minimum_jerk_profile((0.5, 0.5), (0.0, 0.0), duration)
        total_planned = len(planned_waypoints)
        assert total_planned > 50

        with pytest.raises(SafetyBreakoutException):
            controller.servo_to((0.0, 0.0), realtime=False)

        # Failsafe must have aborted early along the trajectory
        final_history_len = len(driver.trajectory_history)
        # Waypoints inside corner margin must not have been reached, and final docking never ran
        assert final_history_len < total_planned
        assert driver.trajectory_history[-1] != (0.0, 0.0)
        assert controller.is_failsafe_tripped is True

        # Furthermore, moving from near corner trips within just a few steps
        driver2 = VirtualCursorDriver(screen_size=(1920, 1080), initial_position=(0.02, 0.02))
        controller2 = NeuromuscularServoingController(driver=driver2)
        with pytest.raises(SafetyBreakoutException):
            controller2.servo_to((0.0, 0.0), realtime=False)
        assert len(driver2.trajectory_history) < 25

    def test_failsafe_latching_and_reset(self):
        """Verify failsafe latches and blocks all subsequent motor commands until reset."""
        driver = VirtualCursorDriver(screen_size=(1920, 1080), initial_position=(0.5, 0.5))
        controller = NeuromuscularServoingController(driver=driver)

        # Trip failsafe
        with pytest.raises(SafetyBreakoutException):
            controller.servo_to((0.0, 0.0), realtime=False)

        assert controller.is_failsafe_tripped is True

        # Subsequent commands to safe coordinates must be rejected while latched
        with pytest.raises(SafetyBreakoutException, match="latched in emergency abort state"):
            controller.servo_to((0.5, 0.5), realtime=False)

        # Reset failsafe
        controller.reset_failsafe()
        assert controller.is_failsafe_tripped is False

        # Driver position should be repositioned to safe center to test subsequent motion
        driver._norm_x, driver._norm_y = 0.5, 0.5
        result = controller.servo_to((0.6, 0.6), realtime=False)
        assert result.success is True

    def test_failsafe_protects_custom_driver_without_native_failsafe(self):
        """Verify controller failsafe protects minimal drivers lacking built-in failsafe."""
        class BareDriver:
            def __init__(self):
                self.pos = [0.5, 0.5]
                self.screen_size = (1920, 1080)
                self.moves = []

            def get_position_normalized(self):
                return (self.pos[0], self.pos[1])

            def move_to_normalized(self, x, y):
                self.pos = [x, y]
                self.moves.append((x, y))

        bare_driver = BareDriver()
        controller = NeuromuscularServoingController(driver=bare_driver)

        # Moving to corner coordinate must be caught by controller's failsafe
        with pytest.raises(SafetyBreakoutException):
            controller.servo_to((0.001, 0.001), realtime=False)

        assert controller.is_failsafe_tripped is True
