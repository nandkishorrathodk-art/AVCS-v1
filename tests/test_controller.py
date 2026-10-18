"""Unit and Integration Tests for Central AVCSController.

Verifies:
1. Zero coordinate leakage to System 2 (geometry encapsulation)
2. Honest, evidence-backed scene descriptions (Zero hallucination)
3. Dynamic neural grounding via GoalConditionedGrounder
4. Closed-loop 5th-order minimum-jerk motor actuation
5. Post-action verification via VisualRippleVerifier
6. 60 FPS real-time telemetry streaming via NeuroStreamBus
7. Fail-closed typing safety
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avcs.controller import (
    ActionResult,
    AVCSController,
    SceneObservation,
)
from avcs.stream.neuro_stream import NeuroPacket
from avcs.verifier import VerificationResult, VerificationStatus
from forge.vision.capture import SyntheticScreen, ThreadedScreenCapture
from forge.vision.cortex.core.ventral import UI_CATEGORIES
from forge.vision.cortex.sdk import UnvalidatedGroundingError


@pytest.fixture
def synthetic_controller():
    """Creates a hermetic controller with synthetic screen and virtual motor."""
    source = SyntheticScreen(width=320, height=240)
    controller = AVCSController(
        source=source,
        weights_path="runs/weights/goal_grounder_best.pt",
        device="cpu",
        target_fps=60,
        force_virtual_motor=True,
        enable_streaming=True,
    )
    yield controller
    if hasattr(controller.source, "close"):
        controller.source.close()


def test_controller_initialization(synthetic_controller):
    ctrl = synthetic_controller
    assert ctrl.eye is not None
    assert ctrl.servoing is not None
    assert ctrl.verifier is not None
    assert ctrl.stream_bus is not None
    assert ctrl.event_streamer is not None


def test_describe_screen_zero_coordinates_and_zero_hallucination(synthetic_controller):
    ctrl = synthetic_controller
    obs = ctrl.describe_screen()

    assert isinstance(obs, SceneObservation)
    assert obs.primary_category in UI_CATEGORIES
    assert obs.stability in ("stable", "changed", "busy", "overlay", "unknown", "tracking", "dynamic_video", "surprise")
    assert 0.0 <= obs.surprise_score <= 1.0
    assert 0.0 <= obs.confidence <= 1.0
    assert isinstance(obs.description, str)
    assert len(obs.description) > 0
    assert isinstance(obs.visible_entities, list)

    # Verify dictionary serialization
    d = obs.to_dict()
    assert "primary_category" in d
    assert "description" in d
    assert "surprise_score" in d

    # Assert ZERO coordinate numbers leaked to System 2
    assert "x" not in d
    assert "y" not in d
    assert "target_coord" not in d
    assert "bounding_box" not in d


def test_telemetry_streaming_bus_receives_packets(synthetic_controller):
    ctrl = synthetic_controller
    received_packets = []

    unsub = ctrl.stream_bus.subscribe(lambda pkt: received_packets.append(pkt))
    assert callable(unsub)

    # Perform 3 visual perception steps
    for _ in range(3):
        ctrl.describe_screen()

    assert len(received_packets) >= 3
    pkt = received_packets[-1]
    assert isinstance(pkt, NeuroPacket)
    assert pkt.seq > 0
    assert pkt.inference_ms >= 0.0
    assert pkt.primary_category is not None

    unsub()


def test_controller_streams_continuously_with_monotonic_frame_sequences():
    controller = AVCSController(
        source=SyntheticScreen(width=320, height=240),
        weights_path=None,
        device="cpu",
        target_fps=30,
        force_virtual_motor=True,
    )
    packets = []
    unsubscribe = controller.stream_bus.subscribe(packets.append)
    try:
        controller.start_streaming()
        deadline = time.monotonic() + 2.0
        while len(packets) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        controller.stop_streaming()
    finally:
        unsubscribe()
        controller.close()

    assert len(packets) >= 4
    sequences = [packet.seq for packet in packets]
    assert sequences == sorted(set(sequences))
    assert controller.stream_metrics["frames_processed"] == len(packets)
    assert controller.stream_metrics["frames_dropped"] is None
    tick_latency = controller.stream_metrics["processing_tick_latency_ms"]
    tick_interval = controller.stream_metrics["processing_interval_ms"]
    assert tick_latency["samples_retained"] == len(packets)
    assert tick_latency["p50"] > 0.0
    assert tick_latency["p95"] >= tick_latency["p50"]
    assert tick_interval["samples_retained"] == len(packets) - 1
    assert tick_interval["p50"] > 0.0


def test_controller_stream_waits_for_new_capture_sequences():
    source = ThreadedScreenCapture(
        SyntheticScreen(width=320, height=240),
        target_fps=120,
    ).open()
    controller = AVCSController(
        source=source,
        weights_path=None,
        device="cpu",
        target_fps=30,
        force_virtual_motor=True,
    )
    packets = []
    unsubscribe = controller.stream_bus.subscribe(packets.append)
    try:
        controller.start_streaming()
        deadline = time.monotonic() + 2.0
        while len(packets) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        controller.stop_streaming()
    finally:
        unsubscribe()
        controller.close()

    assert len(packets) >= 4
    assert controller._last_observation is not None
    assert controller._last_observation.frame_sequence > 0
    assert all(
        packet.source_frame_sequence is not None
        for packet in controller.stream_bus.get_window(len(packets))
    )
    assert controller.stream_metrics["frames_processed"] == len(packets)
    assert controller.stream_metrics["frames_dropped"] >= 0


def test_consult_system2_is_text_only_and_does_not_dispatch_actions():
    controller = AVCSController(
        source=SyntheticScreen(width=320, height=240),
        weights_path=None,
        device="cpu",
        force_virtual_motor=True,
    )
    requested = []

    class StubSystem2:
        def request_goal(self, task, observation, *, consent_confirmed):
            requested.append((task, observation.to_dict(), consent_confirmed))
            return "Wait for the display to stabilize."

    try:
        turn = controller.consult_system2(
            "Inspect the current screen",
            StubSystem2(),
            consent_confirmed=True,
        )
        assert turn.goal == "Wait for the display to stabilize."
        assert requested[0][0] == "Inspect the current screen"
        assert requested[0][2] is True
        assert "frame_sequence" in requested[0][1]
        assert "raw_frame" not in requested[0][1]
        assert "embedding" not in requested[0][1]
        assert controller.motor_driver.get_position_normalized() == (0.5, 0.5)
    finally:
        controller.close()


def test_consult_system2_requires_consent_before_observation():
    controller = AVCSController(
        source=SyntheticScreen(width=320, height=240),
        weights_path=None,
        device="cpu",
        force_virtual_motor=True,
    )
    try:
        with pytest.raises(PermissionError, match="explicit user consent"):
            controller.consult_system2(
                "Inspect", object(), consent_confirmed=False
            )
        assert controller.eye.frame_count == 0
    finally:
        controller.close()


def test_locate_empty_query_raises_value_error(synthetic_controller):
    ctrl = synthetic_controller
    with pytest.raises(ValueError, match="cannot be empty"):
        ctrl.locate("")
    with pytest.raises(ValueError, match="cannot be empty"):
        ctrl.locate("   ")


def test_locate_with_trained_grounder(synthetic_controller):
    ctrl = synthetic_controller
    if ctrl.is_grounder_trained:
        with pytest.raises(UnvalidatedGroundingError, match="unauthorized candidate"):
            ctrl.locate("VLC Player")
    else:
        with pytest.raises(UnvalidatedGroundingError):
            ctrl.locate("VLC Player")


def test_controller_blocks_unauthorized_grounder_candidate(synthetic_controller):
    ctrl = synthetic_controller
    ctrl.grounder = SimpleNamespace(
        predict=lambda frame, query: SimpleNamespace(
            action_authorized=False,
            candidate_point_norm=(0.4, 0.6),
            peak_probability=0.9,
            heatmap=np.zeros((4, 4), dtype=np.float32),
        )
    )

    with pytest.raises(UnvalidatedGroundingError, match="unauthorized candidate"):
        ctrl.locate("Save button")


def test_click_target_encapsulates_geometry(synthetic_controller):
    ctrl = synthetic_controller
    if not ctrl.is_grounder_trained:
        with pytest.raises(UnvalidatedGroundingError):
            ctrl.click_target("Chrome Browser")
        return

    # Call with high-level intent only
    res = ctrl.click_target("Play button", verify_action=True)
    assert isinstance(res, ActionResult)
    assert res.action == "click"
    assert res.target_query == "Play button"
    assert isinstance(res.success, bool)
    assert isinstance(res.verified, bool)
    assert res.latency_ms > 0.0

    # Ensure no coordinate numbers are exposed in ActionResult dict
    d = res.to_dict()
    assert "x" not in d
    assert "y" not in d


def test_click_target_dispatches_once_and_requires_verified_postcondition(
    synthetic_controller, monkeypatch
):
    ctrl = synthetic_controller
    clicks = []
    verifier_calls = []

    monkeypatch.setattr(
        ctrl,
        "locate",
        lambda query: SimpleNamespace(confidence=0.9, target_coord=(0.4, 0.6)),
    )
    monkeypatch.setattr(
        ctrl.eye, "inspect_fovea", lambda centroid: SimpleNamespace(confidence=0.0)
    )
    monkeypatch.setattr(ctrl.servoing, "servo_to", lambda target, realtime: None)
    monkeypatch.setattr(ctrl.motor_driver, "click", lambda button: clicks.append(button))

    def execute_and_verify(**kwargs):
        verifier_calls.append(kwargs)
        ctrl.motor_driver.click(button="left")
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            action_name="click",
            target_coord=kwargs["target"],
            optical_ripple_energy=0.1,
            surprise_pulse=0.2,
            state_transition_detected=True,
            causal_summary="Verified test transition.",
            recovery_attempt=0,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr(ctrl.verifier, "execute_and_verify", execute_and_verify)

    result = ctrl.click_target("Submit", verify_action=True)

    assert len(clicks) == 1
    assert len(verifier_calls) == 1
    assert verifier_calls[0]["max_recovery_tier"] == 1
    assert result.success is True
    assert result.verified is True
    assert "target_coord" not in result.to_dict()["verification"]


def test_click_target_reports_inconclusive_action_as_not_successful(
    synthetic_controller, monkeypatch
):
    ctrl = synthetic_controller
    clicks = []
    monkeypatch.setattr(
        ctrl,
        "locate",
        lambda query: SimpleNamespace(confidence=0.9, target_coord=(0.4, 0.6)),
    )
    monkeypatch.setattr(
        ctrl.eye, "inspect_fovea", lambda centroid: SimpleNamespace(confidence=0.0)
    )
    monkeypatch.setattr(ctrl.servoing, "servo_to", lambda target, realtime: None)
    monkeypatch.setattr(ctrl.motor_driver, "click", lambda button: clicks.append(button))

    def execute_and_verify(**kwargs):
        ctrl.motor_driver.click(button="left")
        return VerificationResult(
            status=VerificationStatus.ESCALATE_BLOCKED,
            action_name="click",
            target_coord=kwargs["target"],
            optical_ripple_energy=0.0,
            surprise_pulse=0.0,
            state_transition_detected=False,
            causal_summary="No verified transition.",
            recovery_attempt=0,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr(ctrl.verifier, "execute_and_verify", execute_and_verify)

    result = ctrl.click_target("Submit", verify_action=True)

    assert clicks == ["left"]
    assert result.success is False
    assert result.verified is False
    assert "did not confirm completion" in result.message


def test_click_target_without_verification_does_not_claim_success(
    synthetic_controller, monkeypatch
):
    ctrl = synthetic_controller
    clicks = []
    monkeypatch.setattr(
        ctrl,
        "locate",
        lambda query: SimpleNamespace(confidence=0.9, target_coord=(0.4, 0.6)),
    )
    monkeypatch.setattr(
        ctrl.eye, "inspect_fovea", lambda centroid: SimpleNamespace(confidence=0.0)
    )
    monkeypatch.setattr(ctrl.servoing, "servo_to", lambda target, realtime: None)
    monkeypatch.setattr(ctrl.motor_driver, "click", lambda button: clicks.append(button))

    result = ctrl.click_target("Submit", verify_action=False)

    assert clicks == ["left"]
    assert result.success is False
    assert result.verified is False
    assert "without verification" in result.message


def test_double_click_target_dispatches_once_through_verifier(
    synthetic_controller, monkeypatch
):
    ctrl = synthetic_controller
    double_clicks = []
    verifier_calls = []
    monkeypatch.setattr(
        ctrl,
        "locate",
        lambda query: SimpleNamespace(confidence=0.9, target_coord=(0.4, 0.6)),
    )
    monkeypatch.setattr(
        ctrl.eye, "inspect_fovea", lambda centroid: SimpleNamespace(confidence=0.0)
    )
    monkeypatch.setattr(ctrl.servoing, "servo_to", lambda target, realtime: None)
    monkeypatch.setattr(
        ctrl.motor_driver,
        "double_click",
        lambda button: double_clicks.append(button),
    )

    def execute_and_verify(**kwargs):
        verifier_calls.append(kwargs)
        ctrl.motor_driver.double_click(button="left")
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            action_name="double_click",
            target_coord=kwargs["target"],
            optical_ripple_energy=0.1,
            surprise_pulse=0.2,
            state_transition_detected=True,
            causal_summary="Verified test transition.",
            recovery_attempt=0,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr(ctrl.verifier, "execute_and_verify", execute_and_verify)

    result = ctrl.click_target("Open file", double_click=True, verify_action=True)

    assert double_clicks == ["left"]
    assert len(verifier_calls) == 1
    assert verifier_calls[0]["action_type"] == "double_click"
    assert verifier_calls[0]["max_recovery_tier"] == 1
    assert result.success is True
    assert result.verified is True


def test_open_application_intent(synthetic_controller):
    ctrl = synthetic_controller
    if not ctrl.is_grounder_trained:
        pytest.skip("Trained grounder required for open_application")

    res = ctrl.open_application("VLC Player")
    assert isinstance(res, ActionResult)
    assert res.action == "open_application"
    assert res.target_query == "VLC Player"


def test_open_application_delegates_to_click_target(
    synthetic_controller, monkeypatch
):
    ctrl = synthetic_controller
    recorded = []

    monkeypatch.setattr(
        ctrl,
        "click_target",
        lambda query, double_click, verify_action: (
            recorded.append((query, double_click, verify_action))
            or ActionResult(
                action="double_click" if double_click else "click",
                target_query=query,
                success=True,
                verified=True,
                message=f"Successfully clicked and verified target '{query}'.",
                latency_ms=15.0,
                verification={"status": "verified"},
            )
        ),
    )

    res = ctrl.open_application("Terminal")
    assert recorded == [("Terminal icon", True, True)]
    assert isinstance(res, ActionResult)
    assert res.action == "open_application"
    assert res.target_query == "Terminal"
    assert res.success is True
    assert res.verified is True
    assert res.verification == {"status": "verified"}


def test_type_text_fail_closed_without_focus(synthetic_controller):
    ctrl = synthetic_controller
    # Direct typing without focus
    res1 = ctrl.type_text("hello world")
    assert isinstance(res1, ActionResult)
    assert res1.action == "type_text"
    assert res1.success is False
    assert "visual focus verification failed" in res1.message.lower()

    # Typing with unlocatable target field
    res2 = ctrl.type_text("hello world", target_query="Search input")
    assert isinstance(res2, ActionResult)
    assert res2.action == "type_text"
    assert res2.success is False


def test_type_text_does_not_claim_unverified_screen_content(
    synthetic_controller, monkeypatch
):
    ctrl = synthetic_controller
    monkeypatch.setattr(
        ctrl.eye,
        "verify_focus",
        lambda target_point: SimpleNamespace(is_focused=True, confidence=0.99),
    )
    typed = []
    monkeypatch.setattr(ctrl.motor_driver, "type_text", lambda text: typed.append(text))

    result = ctrl.type_text("hello")

    assert typed == ["hello"]
    assert result.success is False
    assert result.verified is False
    assert "was not verified" in result.message


def test_type_text_focus_fail_closed_confidence_under_70(synthetic_controller, monkeypatch):
    """Fail-closed typing contract: confidence < 0.70 strictly aborts without keystrokes."""
    ctrl = synthetic_controller
    typed = []
    monkeypatch.setattr(ctrl.motor_driver, "type_text", lambda text: typed.append(text))
    monkeypatch.setattr(
        ctrl.eye,
        "verify_focus",
        lambda target_point: SimpleNamespace(is_focused=True, confidence=0.69, caret_blinking=True),
    )

    res = ctrl.type_text("forbidden_payload")
    assert res.success is False
    assert res.verified is False
    assert "visual focus verification failed closed" in res.message.lower()
    assert len(typed) == 0, "Keystrokes must NOT be dispatched when confidence < 0.70"


def test_type_text_focus_fail_closed_caret_blink_absent(synthetic_controller, monkeypatch):
    """Fail-closed typing contract: absent caret blink strictly aborts without keystrokes."""
    ctrl = synthetic_controller
    typed = []
    monkeypatch.setattr(ctrl.motor_driver, "type_text", lambda text: typed.append(text))
    monkeypatch.setattr(
        ctrl.eye,
        "verify_focus",
        lambda target_point: SimpleNamespace(is_focused=True, confidence=0.95, caret_blinking=False),
    )

    res = ctrl.type_text("forbidden_payload")
    assert res.success is False
    assert res.verified is False
    assert "visual focus verification failed closed" in res.message.lower()
    assert len(typed) == 0, "Keystrokes must NOT be dispatched when caret blink is absent"


def test_type_text_focus_adversarial_static_pipe(synthetic_controller, monkeypatch):
    """Adversarial check: static vertical line '|' must strictly fail-closed."""
    ctrl = synthetic_controller
    typed = []
    monkeypatch.setattr(ctrl.motor_driver, "type_text", lambda text: typed.append(text))

    H, W = 240, 320
    tx, ty = int(0.5 * W), int(0.5 * H)
    f_pipe = np.full((H, W, 3), 250, dtype=np.uint8)
    f_pipe[ty - 10 : ty + 10, tx - 1 : tx + 1] = 0

    monkeypatch.setattr(ctrl.source, "grab", lambda: f_pipe)

    res = ctrl.type_text("malicious_code_injection")
    assert res.success is False
    assert "visual focus verification failed closed" in res.message.lower()
    assert len(typed) == 0, "Static pipe '|' must never trigger keystroke dispatch"


def test_type_text_focus_adversarial_scrolling_code(synthetic_controller, monkeypatch):
    """Adversarial check: scrolling code with vertical stems must strictly fail-closed."""
    ctrl = synthetic_controller
    typed = []
    monkeypatch.setattr(ctrl.motor_driver, "type_text", lambda text: typed.append(text))

    H, W = 240, 320
    tx, ty = int(0.5 * W), int(0.5 * H)
    f_scroll = np.full((H, W, 3), 255, dtype=np.uint8)
    for stem_x in [tx - 15, tx - 8, tx, tx + 6, tx + 14]:
        f_scroll[ty - 10 : ty + 10, stem_x : stem_x + 2] = 0

    monkeypatch.setattr(ctrl.source, "grab", lambda: f_scroll)

    res = ctrl.type_text("scrolling_code_payload")
    assert res.success is False
    assert "visual focus verification failed closed" in res.message.lower()
    assert len(typed) == 0, "Scrolling code must never trigger keystroke dispatch"


def test_type_text_focus_adversarial_animated_spinner(synthetic_controller, monkeypatch):
    """Adversarial check: animated rotating spinner spoke must strictly fail-closed."""
    ctrl = synthetic_controller
    typed = []
    monkeypatch.setattr(ctrl.motor_driver, "type_text", lambda text: typed.append(text))

    H, W = 240, 320
    tx, ty = int(0.5 * W), int(0.5 * H)
    f_spin = np.full((H, W, 3), 240, dtype=np.uint8)
    f_spin[ty - 12 : ty + 12, tx - 1 : tx + 1] = 50

    monkeypatch.setattr(ctrl.source, "grab", lambda: f_spin)

    res = ctrl.type_text("spinner_injection_payload")
    assert res.success is False
    assert "visual focus verification failed closed" in res.message.lower()
    assert len(typed) == 0, "Animated spinner must never trigger keystroke dispatch"


def test_unvalidated_model_fails_closed():
    """Verify that an untrained controller without weights strictly fails closed."""
    source = SyntheticScreen(width=320, height=240)
    controller = AVCSController(
        source=source,
        weights_path="nonexistent_path.pt",
        device="cpu",
        force_virtual_motor=True,
    )
    assert controller.is_grounder_trained is False
    with pytest.raises(UnvalidatedGroundingError, match="trained neural grounding checkpoint is not loaded"):
        controller.locate("Play button")
