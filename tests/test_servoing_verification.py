"""Unit Tests for Phase 4: Closed-Loop Physical Hand-Eye Coordination & Visual Verification."""
from __future__ import annotations

import math
import numpy as np
import pytest

from avcs.servoing import (
    NeuromuscularServoingController,
    NeuromuscularServoingResult,
    NeuromuscularTrajectoryPoint,
)
from avcs.verifier import (
    VerificationResult,
    VerificationStatus,
    VisualRippleVerifier,
)
from avcs.human_eye import HumanEye
from avcs.sdk import HumanEyeClient
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def test_fitts_law_duration_scaling():
    """Verify Fitts' Law movement duration scales logarithmically with distance."""
    controller = NeuromuscularServoingController()

    dur_short = controller.calculate_fitts_duration(distance=0.05, target_width=0.04)
    dur_medium = controller.calculate_fitts_duration(distance=0.30, target_width=0.04)
    dur_long = controller.calculate_fitts_duration(distance=0.80, target_width=0.04)

    assert 0.15 <= dur_short < dur_medium < dur_long <= 0.95


def test_minimum_jerk_trajectory_kinematics():
    """Verify Minimum-Jerk trajectory produces smooth acceleration/deceleration bell curves."""
    driver = VirtualCursorDriver()
    controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=100)

    start = (0.1, 0.1)
    target = (0.8, 0.6)
    duration = 0.40  # 400ms

    waypoints = controller.generate_minimum_jerk_profile(start, target, duration)
    assert len(waypoints) >= 30

    # Start and end bounds
    assert pytest.approx(waypoints[0].x, abs=0.02) == start[0]
    assert pytest.approx(waypoints[0].y, abs=0.02) == start[1]
    assert pytest.approx(waypoints[-1].x, abs=0.02) == target[0]
    assert pytest.approx(waypoints[-1].y, abs=0.02) == target[1]

    # Peak velocity should occur near the midpoint (tau ~ 0.5)
    mid_idx = len(waypoints) // 2
    peak_speed = math.hypot(waypoints[mid_idx].vx, waypoints[mid_idx].vy)
    start_speed = math.hypot(waypoints[1].vx, waypoints[1].vy)
    end_speed = math.hypot(waypoints[-2].vx, waypoints[-2].vy)
    assert peak_speed > start_speed
    assert peak_speed > end_speed

    # Physiological micro-tremor non-zero
    assert any(pt.jitter_offset != (0.0, 0.0) for pt in waypoints)


def test_neuromuscular_servo_execution():
    """Verify NeuromuscularServoingController steers driver to exact target coordinate."""
    driver = VirtualCursorDriver()
    driver.move_to_normalized(0.2, 0.2)
    controller = NeuromuscularServoingController(driver=driver)

    target = (0.75, 0.82)
    res = controller.servo_to(target, realtime=False)

    assert isinstance(res, NeuromuscularServoingResult)
    assert res.success is True
    assert pytest.approx(res.final_cursor[0], abs=1e-3) == target[0]
    assert pytest.approx(res.final_cursor[1], abs=1e-3) == target[1]
    assert res.duration_ms > 0.0
    assert res.fitts_index_of_difficulty > 0.0


def test_local_optical_ripple_energy():
    """Verify VisualRippleVerifier calculates kinetic energy around the target region."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, ripple_radius=0.10)

    # Synthetic optical flow field (16 rows, 24 cols, 2 channels)
    motion_field = np.zeros((16, 24, 2), dtype=np.float32)

    # Inject motion burst at target (0.5, 0.5) -> row 8, col 12
    motion_field[7:10, 11:14, 0] = 0.5
    motion_field[7:10, 11:14, 1] = 0.5

    ripple_target = verifier.calculate_local_ripple((0.5, 0.5), motion_field)
    ripple_distant = verifier.calculate_local_ripple((0.1, 0.1), motion_field)

    assert ripple_target > 0.05
    assert ripple_distant == 0.0


def test_visual_verification_and_self_healing_ladder():
    """Verify 5-tier self-healing escalation ladder on simulated click actions."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, min_ripple_threshold=0.01)

    target = (0.45, 0.55)

    # Execute action and verify through self-healing ladder
    res = verifier.execute_and_verify(
        action_type="click",
        target=target,
        max_recovery_tier=3,
    )

    assert isinstance(res, VerificationResult)
    assert res.status in (
        VerificationStatus.VERIFIED,
        VerificationStatus.RETRY_MICRO_JITTER,
        VerificationStatus.KEYBOARD_FALLBACK,
    )
    assert res.elapsed_ms > 0.0


def test_client_closed_loop_motor_and_verification():
    """Verify HumanEyeClient move_to, click_and_verify, and type_and_verify integration."""
    client = HumanEyeClient(source="synthetic", force_virtual_motor=True)
    with client:
        # 1. Neuromuscular smooth move to coordinate
        servo_res = client.move_to((0.6, 0.7), realtime=False)
        assert isinstance(servo_res, NeuromuscularServoingResult)
        assert pytest.approx(servo_res.final_cursor[0], abs=1e-3) == 0.6
        assert pytest.approx(servo_res.final_cursor[1], abs=1e-3) == 0.7

        # 2. Click and verify
        verif_res = client.click_and_verify(target=(0.6, 0.7))
        assert isinstance(verif_res, VerificationResult)
        assert verif_res.action_name == "click@(0.6, 0.7)"

        # 3. Type and verify
        type_res = client.type_and_verify(target=(0.3, 0.3), text="hello world", clear_first=False)
        assert isinstance(type_res, VerificationResult)
        assert "type" in type_res.action_name
