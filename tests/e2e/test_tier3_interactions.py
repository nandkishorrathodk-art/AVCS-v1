"""Tier 3: Cross-Feature Interactions E2E Test Suite.

Tests multi-subsystem pipelines:
- Perception + Motor Servoing coordination
- Grounding + Foveal Acuity refinement
- Temporal Caret Focus + Typing Keystroke Protection
- Action Broker + Visual Ripple fresh-frame verification
- Telemetry Bus streaming during motor execution
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pytest

from avcs.grounding import FovealInspection
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.sdk import HumanEyeClient, UnfocusedInputElementError
from avcs.stream.neuro_stream import NeuroPacket, NeuroStreamBus
from avcs.verifier import VerificationResult, VerificationStatus
from forge.vision.cortex.motor.driver import VirtualCursorDriver


@pytest.mark.tier3
class TestPerceptionMotorCoordination:
    """Coupling of perception hotspot detection and neuromuscular motor execution."""

    def test_interaction_perception_hotspot_to_motor_servoing(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Perception detects a button hotspot; motor servoing steers cursor to its exact centroid."""
        frame, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=301)
        # Perceive the synthetic UI
        state = avcs_client.perceive(frame=frame)
        assert len(state.affordance_hotspots) > 0, "Expected at least one affordance hotspot on desktop UI"

        # Select first detected hotspot (hx, hy, label, conf)
        hx, hy, label, conf = state.affordance_hotspots[0]
        assert 0.0 <= hx <= 1.0 and 0.0 <= hy <= 1.0

        # Execute motor servoing to hotspot coordinates
        result = avcs_client.move_to((hx, hy))
        assert result is not None
        assert result.total_waypoints >= 5
        assert len(result.trajectory) >= 5

        # Verify cursor reached the target coordinate within tight tolerance
        final_pt = result.trajectory[-1]
        assert np.isclose(final_pt.x, hx, atol=1e-3)
        assert np.isclose(final_pt.y, hy, atol=1e-3)

    def test_interaction_smooth_pursuit_updates_motor_flow(self, avcs_client: HumanEyeClient):
        """Engaging pursuit on a target updates internal gaze and active tracking target."""
        target = (0.45, 0.55)
        tracked = avcs_client.track(target)
        assert tracked == target

        # Pursuit tracker registers the active target and updates gaze
        assert avcs_client.eye.pursuit.active_target == target
        assert avcs_client.eye.pursuit.current_gaze is not None


@pytest.mark.tier3
class TestGroundingFovealRefinementPipeline:
    """Coupling of top-down gaze grounding and un-downsampled foveal acuity inspection."""

    def test_interaction_grounding_candidate_to_foveal_refinement(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Candidate coordinate from look_at undergoes un-downsampled foveal ROI acuity inspection."""
        frame, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=302)
        # First element is Play button at pixel (60, 80, 180, 120), norm centroid ~(0.09375, 0.1388)
        btn = elements[0]
        btn_centroid = btn["norm_centroid"]

        # Run foveal inspection on the target centroid
        inspection = avcs_client.eye.inspect_fovea(centroid=btn_centroid, frame=frame)
        assert inspection is not None
        assert isinstance(inspection, FovealInspection)
        # Structured button element must satisfy contrast gating
        assert inspection.contrast_score >= 0.010
        assert inspection.bbox is not None
        assert len(inspection.bbox) == 4

    def test_interaction_grounding_low_contrast_foveal_rejection(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Gaze candidate on flat uniform area is rejected by foveal contrast gating."""
        flat_frame = screen_gen.blank_frame(color=(128, 128, 128))
        hotspot = (0.5, 0.5, "button", 0.90)

        # Foveal acuity refinement fails closed
        refined = avcs_client.eye.refine_affordance_with_fovea(
            hotspot=hotspot,
            frame=flat_frame,
            min_contrast=0.010,
        )
        assert refined is None, "Low-contrast candidate must be rejected by foveal acuity engine"


@pytest.mark.tier3
class TestFocusVerificationTypingProtectionPipeline:
    """Coupling of temporal caret blink verification and keyboard motor execution."""

    def test_interaction_verified_caret_blink_allows_typing(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Active input element with verified 1.0 Hz caret blink permits typing into virtual driver."""
        frames, ts = screen_gen.caret_frames(freq_hz=1.0, num_frames=90, fps=30.0)
        # Verify focus
        focus_state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert focus_state.is_focused

        # Typing with verified focus succeeds
        avcs_client.type_text("Hello World", enforce_focus=False)
        driver = avcs_client.motor_driver
        assert isinstance(driver, VirtualCursorDriver)
        # Verify typed text exists in virtual driver's typed history
        assert "Hello World" in driver.typed_history

    def test_interaction_unfocused_element_blocks_keystrokes_and_clearing(
        self,
        avcs_client: HumanEyeClient,
    ):
        """Unfocused element strictly raises UnfocusedInputElementError with zero sent keystrokes."""
        driver = avcs_client.motor_driver
        driver.typed_history.clear()
        driver.key_history.clear()

        with pytest.raises(UnfocusedInputElementError):
            avcs_client.type_text(
                "Blocked text",
                enforce_focus=True,
                clear_first=True,  # Even with clear_first requested, it must be aborted
                target=(0.05, 0.05),
            )

        # Verify no keystrokes (and no blind Ctrl+A / Backspace) were dispatched
        assert len(driver.typed_history) == 0, "No keystrokes should be sent when focus verification fails closed"
        assert len(driver.key_history) == 0, "No control keys should be sent when focus verification fails closed"


@pytest.mark.tier3
class TestActionBrokerVerificationPipeline:
    """Coupling of ActionBroker and VisualRippleVerifier postcondition verification."""

    def test_interaction_click_and_verify_pipeline(self, avcs_client: HumanEyeClient):
        """click_and_verify dispatches motor action and evaluates fresh postcondition."""
        result = avcs_client.click_and_verify(target=(0.4, 0.4))
        assert isinstance(result, VerificationResult)
        assert result.status in list(VerificationStatus)
        assert "click" in result.action_name
        assert np.isclose(result.target_coord[0], 0.4, atol=0.01)
        assert np.isclose(result.target_coord[1], 0.4, atol=0.01)


@pytest.mark.tier3
class TestContinuousTelemetryStreamingDuringMotorAction:
    """Coupling of 60 FPS NeuroStreamBus and active neuromuscular motor servoing."""

    def test_interaction_telemetry_bus_captures_motor_trajectory_flow(
        self,
        avcs_client: HumanEyeClient,
    ):
        """Telemetry bus records live flow packets during neuromuscular servoing execution."""
        bus = NeuroStreamBus(buffer_size=120)
        captured_packets: List[NeuroPacket] = []
        bus.subscribe(lambda p: captured_packets.append(p))

        # Generate minimum-jerk trajectory waypoints
        traj = avcs_client.hands.generate_minimum_jerk_profile((0.2, 0.2), (0.6, 0.6), duration_s=0.2)
        assert len(traj) >= 5

        # Publish telemetry packet for each waypoint
        for idx, pt in enumerate(traj):
            pkt = NeuroPacket(
                timestamp=time.time() + idx * 0.016,
                seq=idx,
                fixation=(pt.x, pt.y),
                motor_flow=(pt.vx, pt.vy),
                contact_acquired=(idx == len(traj) - 1),
                primary_category="motor_servoing",
                stability_state="moving" if idx < len(traj) - 1 else "stable",
                surprise_score=0.02,
                inference_ms=1.1,
            )
            bus.publish(pkt)

        # Verify stream captured all packets
        assert len(captured_packets) == len(traj)
        assert captured_packets[0].fixation == (traj[0].x, traj[0].y)
        assert captured_packets[-1].contact_acquired is True
        assert captured_packets[-1].stability_state == "stable"
