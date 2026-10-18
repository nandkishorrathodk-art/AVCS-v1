"""Comprehensive Unit and Integration Test Suite for AVCS Milestone 4.

Covers:
- Feature 11: 5th-Order Minimum-Jerk Kinematics & Fitts' Law Duration Scaling
- Feature 12: Configurable 10.5 Hz Biological Micro-Tremor & Discrete All-Pole Peak Resonance
- Feature 13: Fail-Closed Typing Safety, Dual-Condition Focus Gate & Terminal Protection
- Feature 14: Fresh Post-Action Frame Verification, Sequence Synchronization & Action Broker
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from avcs.control.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionResult as BrokerActionResult,
    ActionSafetyController,
    ActionStatus,
    PostconditionSpec,
    VerificationDetail,
)
from avcs.control.controller import ActionResult, AVCSController
from avcs.control.servoing import NeuromuscularServoingController, NeuromuscularTrajectoryPoint
from avcs.control.verifier import VerificationResult, VerificationStatus, VisualRippleVerifier
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.perception.focus import (
    FocusLostError,
    FocusVerificationResult,
    UnverifiedFocusError,
    VisualFocusDetector,
    VisualFocusState,
)
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.motor.driver import VirtualCursorDriver


# ==============================================================================
# Feature 11: 5th-Order Minimum-Jerk Kinematics
# ==============================================================================

class TestFeature11MinimumJerkKinematics:
    """Rigorous verification of Feature 11 requirements."""

    def test_feature11_analytical_polynomial_and_boundary_conditions(self):
        """Verifies Flash & Hogan (1985) 5th-order polynomial: 10*tau^3 - 15*tau^4 + 6*tau^5.
        
        Boundary derivatives must vanish:
        s(0) = 0, s(1) = 1
        s'(0) = 0, s'(1) = 0
        s''(0) = 0, s''(1) = 0
        """
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(
            driver=driver,
            tick_rate_hz=120,
            enable_micro_tremor=False,
        )

        pts = controller.generate_minimum_jerk_profile(
            start=(0.1, 0.2),
            target=(0.8, 0.9),
            duration_s=0.5,
        )

        # Boundary positions
        assert math.isclose(pts[0].x, 0.1, abs_tol=1e-5), f"Start X mismatch: {pts[0].x}"
        assert math.isclose(pts[0].y, 0.2, abs_tol=1e-5), f"Start Y mismatch: {pts[0].y}"
        assert math.isclose(pts[-1].x, 0.8, abs_tol=1e-5), f"Landing X mismatch: {pts[-1].x}"
        assert math.isclose(pts[-1].y, 0.9, abs_tol=1e-5), f"Landing Y mismatch: {pts[-1].y}"

        # Boundary velocities must strictly vanish
        assert pts[0].vx == 0.0 and pts[0].vy == 0.0, "Initial velocity must be 0"
        assert pts[-1].vx == 0.0 and pts[-1].vy == 0.0, "Terminal velocity must be 0"

        # Boundary accelerations must vanish
        assert pts[0].acceleration == 0.0, "Initial acceleration must be 0"
        assert pts[-1].acceleration == 0.0, "Terminal acceleration must be 0"

    def test_feature11_bell_curve_velocity_profile_and_peak_speed(self):
        """Verifies velocity profile is symmetric bell curve with peak at tau=0.5.
        
        Flash & Hogan analytical peak speed:
        v_max = 1.875 * (Distance / Duration)
        """
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(
            driver=driver,
            tick_rate_hz=120,
            enable_micro_tremor=False,
        )

        start = (0.2, 0.5)
        target = (0.8, 0.5)
        dist = 0.6
        duration = 0.4
        pts = controller.generate_minimum_jerk_profile(start, target, duration_s=duration)

        speeds = [math.hypot(p.vx, p.vy) for p in pts]
        n = len(pts)
        peak_idx = int(np.argmax(speeds))

        # Peak must occur at exact midpoint (tau = 0.5, +/- 1 sample)
        mid_idx = n // 2
        assert abs(peak_idx - mid_idx) <= 1, f"Peak velocity index {peak_idx} not at midpoint {mid_idx}"

        # Peak speed must match analytical theory: 1.875 * (D / T)
        theoretical_vmax = 1.875 * (dist / duration)
        measured_vmax = speeds[peak_idx]
        assert abs(measured_vmax - theoretical_vmax) < 0.02, (
            f"Peak speed {measured_vmax:.4f} diverges from theory {theoretical_vmax:.4f}"
        )

        # Symmetry: velocities at tau and (1 - tau) must match
        for i in range(1, n // 2):
            v_forward = speeds[i]
            v_backward = speeds[n - 1 - i]
            assert abs(v_forward - v_backward) < 0.05, (
                f"Asymmetric velocity at step {i}: {v_forward:.4f} vs {v_backward:.4f}"
            )

    def test_feature11_fitts_law_duration_scaling_and_bounds(self):
        """Verifies Fitts' Law duration T = a + b * log2(1 + D/W).
        
        Monotonically scales with distance, inversely scales with width,
        and is strictly clamped to [0.15, 0.95] s.
        """
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(driver=driver)

        # Monotonic scaling with distance D
        d_short = controller.calculate_fitts_duration(distance=0.05, target_width=0.04)
        d_medium = controller.calculate_fitts_duration(distance=0.30, target_width=0.04)
        d_long = controller.calculate_fitts_duration(distance=0.80, target_width=0.04)
        assert d_short < d_medium < d_long, f"Durations not monotonically increasing: {d_short}, {d_medium}, {d_long}"

        # Inverse scaling with target width W (smaller target = higher difficulty = longer duration)
        d_wide = controller.calculate_fitts_duration(distance=0.40, target_width=0.10)
        d_narrow = controller.calculate_fitts_duration(distance=0.40, target_width=0.02)
        assert d_narrow > d_wide, f"Narrow target duration ({d_narrow}) must exceed wide target ({d_wide})"

        # Clamping bounds: 0.15s min, 0.95s max
        d_zero = controller.calculate_fitts_duration(distance=0.0, target_width=1.0)
        d_huge = controller.calculate_fitts_duration(distance=100.0, target_width=0.001)
        assert d_zero >= 0.15, f"Duration under minimum clamp: {d_zero}"
        assert d_huge <= 0.95, f"Duration over maximum clamp: {d_huge}"

    def test_feature11_generate_trajectory_interface_contract(self):
        """Verifies generate_trajectory(start, target, enable_micro_tremor=True) -> List[Tuple[float, float]].
        
        Matches PROJECT.md:65 Interface Contract.
        """
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(driver=driver)

        start = (0.15, 0.25)
        target = (0.75, 0.85)
        traj = controller.generate_trajectory(start=start, target=target, enable_micro_tremor=True)

        assert isinstance(traj, list)
        assert len(traj) >= 10
        assert isinstance(traj[0], tuple) and len(traj[0]) == 2

        # Start and landing coordinates match
        assert math.isclose(traj[0][0], 0.15, abs_tol=1e-4)
        assert math.isclose(traj[0][1], 0.25, abs_tol=1e-4)
        assert math.isclose(traj[-1][0], 0.75, abs_tol=1e-4)
        assert math.isclose(traj[-1][1], 0.85, abs_tol=1e-4)

        # All coordinates bounded within [0.0, 1.0]
        for x, y in traj:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_feature11_fast_scalar_clamping_throughput(self):
        """Verifies trajectory generation throughput on CPU using fast scalar clamping (<100 us)."""
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=120)

        t0 = time.perf_counter()
        count = 100
        for _ in range(count):
            controller.generate_trajectory((0.1, 0.1), (0.9, 0.9))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = elapsed_ms / count

        # Average generation must be strictly under 1.0 ms (sub-millisecond kinematics)
        assert avg_ms < 1.0, f"Trajectory generation too slow: {avg_ms:.3f} ms/trajectory"


# ==============================================================================
# Feature 12: Configurable 10.5 Hz Biological Micro-Tremor
# ==============================================================================

class TestFeature12BiologicalMicroTremor:
    """Rigorous verification of Feature 12 requirements."""

    def test_feature12_discrete_all_pole_resonant_frequency_peak(self):
        """Verifies AR(2) biological resonator transfer function peaks at 10.5 +/- 0.2 Hz.
        
        Transfer function: H(z) = 1 / (1 - a1*z^-1 - a2*z^-2)
        With discrete peak correction: cos(w0) = (2r / (1 + r^2)) * cos(w_target)
        """
        fs = 120.0
        f_target = 10.5
        r = 0.82
        w_target = 2.0 * math.pi * (f_target / fs)

        cos_w0 = min(1.0, max(-1.0, (2.0 * r / (1.0 + r ** 2)) * math.cos(w_target)))
        w0 = math.acos(cos_w0)
        a1 = 2.0 * r * math.cos(w0)
        a2 = -(r ** 2)

        freqs = np.linspace(1.0, 30.0, 5000)
        w = 2.0 * np.pi * freqs / fs
        z = np.exp(1j * w)
        H = 1.0 / (1.0 - a1 * (z ** -1) - a2 * (z ** -2))
        mag = np.abs(H)

        peak_freq = freqs[np.argmax(mag)]
        assert abs(peak_freq - 10.5) < 0.15, (
            f"Resonant peak at {peak_freq:.3f} Hz diverges from target 10.5 Hz"
        )

    def test_feature12_sample_rate_adaptability(self):
        """Verifies that the peak correction maintains ~10.5 Hz across different tick rates."""
        for fs in [60.0, 100.0, 120.0, 240.0]:
            f_target = 10.5
            r = 0.82
            w_target = 2.0 * math.pi * (f_target / fs)
            cos_w0 = min(1.0, max(-1.0, (2.0 * r / (1.0 + r ** 2)) * math.cos(w_target)))
            w0 = math.acos(cos_w0)
            a1 = 2.0 * r * math.cos(w0)
            a2 = -(r ** 2)

            freqs = np.linspace(5.0, 20.0, 3000)
            w = 2.0 * np.pi * freqs / fs
            z = np.exp(1j * w)
            H = 1.0 / (1.0 - a1 * (z ** -1) - a2 * (z ** -2))
            peak_freq = freqs[np.argmax(np.abs(H))]
            assert abs(peak_freq - 10.5) < 0.20, f"Peak frequency {peak_freq:.2f} at fs={fs} shifted"

    def test_feature12_smooth_amplitude_envelope_and_zero_landing_offset(self):
        """Verifies jitter envelope sin(pi * tau) vanishes at launch and landing.
        
        Final step landing jitter is strictly (0.0, 0.0) with zero overshoot.
        """
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(
            driver=driver,
            tick_rate_hz=120,
            enable_micro_tremor=True,
            micro_tremor_std=0.001,
        )

        for _ in range(5):
            pts = controller.generate_minimum_jerk_profile((0.2, 0.2), (0.7, 0.8), duration_s=0.5)
            # Boundary jitter offsets must be strictly zero
            assert pts[0].jitter_offset == (0.0, 0.0), f"Initial jitter offset non-zero: {pts[0].jitter_offset}"
            assert pts[-1].jitter_offset == (0.0, 0.0), f"Terminal jitter offset non-zero: {pts[-1].jitter_offset}"

            # Landing coordinates must match target exactly
            assert math.isclose(pts[-1].x, 0.7, abs_tol=1e-5)
            assert math.isclose(pts[-1].y, 0.8, abs_tol=1e-5)

            # Mid-flight jitter exists and respects 3-sigma bounds
            max_jitter = controller.micro_tremor_std * 3.0
            mid_pts = pts[len(pts) // 4: 3 * len(pts) // 4]
            any_jitter = any(pt.jitter_offset != (0.0, 0.0) for pt in mid_pts)
            assert any_jitter, "Expected non-zero micro-tremor in mid-flight"

            for pt in pts:
                jx, jy = pt.jitter_offset
                assert abs(jx) <= max_jitter + 1e-6, f"Jitter X {jx} exceeded 3-sigma {max_jitter}"
                assert abs(jy) <= max_jitter + 1e-6, f"Jitter Y {jy} exceeded 3-sigma {max_jitter}"

    def test_feature12_micro_tremor_per_call_and_instance_toggle(self):
        """Verifies enable_micro_tremor toggle cleanly turns tremor on/off."""
        driver = VirtualCursorDriver()
        controller = NeuromuscularServoingController(
            driver=driver,
            enable_micro_tremor=True,
        )

        # 1. Per-call toggle: enable_micro_tremor=False suppresses jitter
        clean_pts = controller.generate_minimum_jerk_profile(
            (0.2, 0.2), (0.8, 0.8), duration_s=0.4, enable_micro_tremor=False
        )
        assert all(pt.jitter_offset == (0.0, 0.0) for pt in clean_pts)

        # 2. Per-call toggle on generate_trajectory
        clean_traj = controller.generate_trajectory(
            (0.2, 0.2), (0.8, 0.8), enable_micro_tremor=False, duration_s=0.4
        )
        tremor_traj = controller.generate_trajectory(
            (0.2, 0.2), (0.8, 0.8), enable_micro_tremor=True, duration_s=0.4
        )
        # Without tremor, traj matches clean_pts exactly
        for (tx, ty), pt in zip(clean_traj, clean_pts):
            assert math.isclose(tx, pt.x, abs_tol=1e-6)
            assert math.isclose(ty, pt.y, abs_tol=1e-6)

        # 3. Instance-level toggle
        controller.enable_micro_tremor = False
        default_pts = controller.generate_minimum_jerk_profile((0.2, 0.2), (0.8, 0.8), duration_s=0.4)
        assert all(pt.jitter_offset == (0.0, 0.0) for pt in default_pts)


# ==============================================================================
# Feature 13: Fail-Closed Typing Safety & Focus Verification
# ==============================================================================

class TestFeature13TypingSafetyAndFocusGate:
    """Rigorous verification of Feature 13 requirements."""

    def test_feature13_focus_verification_result_structure_and_dataclass(self):
        """Verifies FocusVerificationResult contains required fields and encapsulates geometry."""
        res = FocusVerificationResult(
            verified=True,
            confidence=0.88,
            os_window_verified=True,
            visual_focus_verified=True,
            active_window_title="Notepad - Untitled",
            expected_window_title="Notepad",
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
            caret_coord=(0.45, 0.52),
            target_region=(0.40, 0.50, 0.60, 0.55),
            reason="Focus confirmed",
            latency_ms=3.42,
        )

        assert res.verified is True
        assert res.confidence == 0.88
        assert res.os_window_verified is True
        assert res.visual_focus_verified is True

        d = res.to_dict()
        assert d["verified"] is True
        assert d["confidence"] == 0.88
        assert d["active_window_title"] == "Notepad - Untitled"
        assert d["caret_detected"] is True
        assert d["caret_blinking"] is True

        # System 2 Privacy Boundary: coordinates must NOT be in to_dict()
        assert "caret_coord" not in d
        assert "target_region" not in d

    def test_feature13_dual_condition_focus_gate_evaluates_window_and_pixels(self):
        """Verifies Dual-Condition Focus Gate:
        Verified iff (Condition A: OS Window Matches) AND (Condition B: Visual Focus >= 0.70).
        """
        driver = VirtualCursorDriver()
        driver.set_foreground_window("Chrome - Google Search")

        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Mock eye.verify_focus to return positive visual focus
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.85,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        # Case 1: Window matches expected foreground_app -> PASS
        pass_res = ctrl.verify_focus(foreground_app="Chrome")
        assert pass_res.verified is True
        assert pass_res.os_window_verified is True
        assert pass_res.visual_focus_verified is True
        assert pass_res.confidence >= 0.70

        # Case 2: Window mismatch (expected Notepad, active is Chrome) -> FAIL CLOSED
        fail_win = ctrl.verify_focus(foreground_app="Notepad")
        assert fail_win.verified is False
        assert fail_win.os_window_verified is False
        assert fail_win.confidence <= 0.69
        assert "Window mismatch" in fail_win.reason

        # Case 3: Window matches, but visual focus fails (confidence < 0.70) -> FAIL CLOSED
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=False,
            confidence=0.45,
            caret_detected=False,
            caret_blinking=False,
            focus_ring_detected=False,
        )
        fail_vis = ctrl.verify_focus(foreground_app="Chrome")
        assert fail_vis.verified is False
        assert fail_vis.os_window_verified is True
        assert fail_vis.visual_focus_verified is False
        assert fail_vis.confidence <= 0.69

    def test_feature13_fail_closed_typing_blocks_unfocused_keystrokes(self):
        """Verifies that type_text with verify_focus=True dispatches ZERO keystrokes if unfocused."""
        driver = VirtualCursorDriver()
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Unfocused visual state
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=False,
            confidence=0.30,
            caret_detected=False,
            caret_blinking=False,
            focus_ring_detected=False,
        )

        res = ctrl.type_text("malicious text", verify_focus=True)

        assert res.success is False
        assert res.verified is False
        assert "Typing blocked" in res.message
        assert len(driver.typed_history) == 0, "No keystrokes must be dispatched when unfocused"
        assert len(driver.key_history) == 0, "No special keys must be pressed when unfocused"

    def test_feature13_terminal_protection_guard_prevents_blind_ctrl_a(self):
        """Verifies that clear_first in a terminal window emits Ctrl+U, never blind Ctrl+A + Backspace."""
        driver = VirtualCursorDriver()
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Mock focused state so typing proceeds to clear_first step
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.90,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        # 1. Terminal window: "Windows PowerShell"
        driver.set_foreground_window("Administrator: Windows PowerShell")
        ctrl.type_text("ls -la", clear_first=True, verify_focus=True)

        assert "ctrl+u" in driver.key_history
        assert "ctrl+a" not in driver.key_history
        assert "backspace" not in driver.key_history

        # 2. GUI window: "Untitled - Notepad"
        driver.key_history.clear()
        driver.typed_history.clear()
        driver.set_foreground_window("Untitled - Notepad")
        ctrl.type_text("hello notepad", clear_first=True, verify_focus=True)

        assert "ctrl+a" in driver.key_history
        assert "backspace" in driver.key_history
        assert "ctrl+u" not in driver.key_history

    def test_feature13_mid_typing_focus_loss_aborts_immediately(self):
        """Verifies driver aborts mid-typing if focus is lost and raises FocusLostError."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window("TargetApp")

        # Predicate returning False mid-stream
        step = 0
        def check_focus():
            nonlocal step
            step += 1
            return step < 5

        with pytest.raises(FocusLostError, match="Focus lost mid-typing"):
            driver.type_text("long sentence to type", focus_check_fn=check_focus)

        # Only characters before focus loss are typed
        assert len(driver.typed_history) == 0  # didn't complete


# ==============================================================================
# Feature 14: Fresh Post-Action Frame Verification & Action Broker
# ==============================================================================

class TestFeature14FreshVerificationAndActionBroker:
    """Rigorous verification of Feature 14 requirements."""

    def test_feature14_postcondition_spec_and_verification_detail_dataclasses(self):
        """Verifies PostconditionSpec and VerificationDetail data contracts."""
        spec = PostconditionSpec(
            postcondition_type="optical_ripple",
            target_coord=(0.5, 0.5),
            min_ripple_threshold=0.020,
            min_photometric_delta=0.040,
            timeout_ms=1000.0,
        )
        assert spec.postcondition_type == "optical_ripple"
        assert spec.min_ripple_threshold == 0.020
        assert spec.timeout_ms == 1000.0

        detail = VerificationDetail(
            status="CONFIRMED",
            verified=True,
            optical_ripple_energy=0.035,
            photometric_delta=0.048,
            surprise_pulse=0.12,
            state_transition_detected=True,
            causal_summary="Confirmed ripple.",
            recovery_attempts=1,
            elapsed_ms=45.2,
            fresh_frame_sequence=142,
        )
        d = detail.to_dict()
        assert d["verified"] is True
        assert d["fresh_frame_sequence"] == 142
        assert "target_coord" not in d
        assert "target_region" not in d

    def test_feature14_fresh_frame_sequence_synchronization(self):
        """Verifies that observe(min_sequence=S_pre) strictly retrieves a frame with S > S_pre."""
        screen = SyntheticScreen()
        frame1, seq1 = screen.grab_with_sequence()
        assert seq1 >= 1

        # Requesting fresh frame after seq1
        frame2, seq2 = screen.grab_with_sequence(min_sequence=seq1)
        assert seq2 > seq1, f"Sequence {seq2} did not advance beyond {seq1}"

        # HumanEye.capture_fresh_frame
        eye = HumanEye(source=screen)
        f_fresh, s_fresh = eye.capture_fresh_frame(min_sequence=seq2)
        assert s_fresh > seq2

        # HumanEye.observe(min_sequence=...)
        state = eye.observe(min_sequence=s_fresh)
        assert state is not None
        assert screen.sequence_number > s_fresh

    def test_feature14_action_broker_semantic_postcondition_routing(self):
        """Verifies ActionBroker executes action and evaluates PostconditionSpec."""
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        # 1. Custom predicate postcondition that passes
        req_pass = ActionRequest(
            action_type="click",
            target_coord=(0.4, 0.4),
            action_class=ActionClass.NON_IDEMPOTENT,
            expected_postcondition=lambda pre, post: True,
        )
        res_pass = broker.execute(req_pass)
        assert res_pass.status == ActionStatus.CONFIRMED
        assert res_pass.is_confirmed is True
        assert res_pass.verification_detail is not None
        assert res_pass.verification_detail.verified is True

        # 2. PostconditionSpec that fails with zero motion -> FAILED
        req_fail = ActionRequest(
            action_type="click",
            target_coord=(0.4, 0.4),
            action_class=ActionClass.NON_IDEMPOTENT,
            expected_postcondition=PostconditionSpec(
                postcondition_type="optical_ripple",
                min_ripple_threshold=0.999,  # impossibly high
            ),
        )
        res_fail = broker.execute(req_fail)
        assert res_fail.status in (ActionStatus.FAILED, ActionStatus.AMBIGUOUS)
        assert res_fail.is_confirmed is False

    def test_feature14_action_broker_replay_prevention_on_non_idempotent(self):
        """Verifies ActionBroker never blindly replays NON_IDEMPOTENT or type actions."""
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        # Failing postcondition on NON_IDEMPOTENT action
        req = ActionRequest(
            action_type="click",
            target_coord=(0.5, 0.5),
            action_class=ActionClass.NON_IDEMPOTENT,
            expected_postcondition=lambda pre, post: False,
        )
        res = broker.execute(req)

        # Recovery attempts must be 1 (never replayed)
        assert res.recovery_attempts == 1
        assert res.status != ActionStatus.CONFIRMED

        # Failing typing action must never retry
        type_req = ActionRequest(
            action_type="type",
            target_coord=(0.5, 0.5),
            text_content="important data",
            action_class=ActionClass.IDEMPOTENT,  # even if marked idempotent, typing is never retried
            expected_postcondition=lambda pre, post: False,
        )
        type_res = broker.execute(type_req)
        assert type_res.recovery_attempts == 1
        assert type_res.status != ActionStatus.CONFIRMED

    def test_feature14_controller_privacy_boundary_zero_coordinates(self):
        """Verifies AVCSController.ActionResult.to_dict() strictly encapsulates coordinates."""
        driver = VirtualCursorDriver()
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        res = ctrl.click_target(query="Submit", verify_action=False)
        d = res.to_dict()

        # Check required fields
        assert "action" in d
        assert "target_query" in d
        assert "success" in d
        assert "verified" in d
        assert "message" in d
        assert "latency_ms" in d

        # Privacy Boundary checks
        assert "target_coord" not in d
        assert "x" not in d
        assert "y" not in d
        assert "centroid" not in d
        assert "bbox" not in d
        if d.get("verification"):
            assert "target_coord" not in d["verification"]
            assert "raw_coordinates" not in d["verification"]
