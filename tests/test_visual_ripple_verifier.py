"""Comprehensive test suite for VisualRippleVerifier in AVCS System 1.

Validates:
1. Photometric patch delta and optical flow ripple detection in fresh frames.
2. Positive and negative control assertions: clicks are verified=True ONLY when a real
   optical change or state transition is detected on screen.
3. Pre-existing motion does not produce false-positive verification when post-action motion is absent.
4. Distant UI mutations do not produce false-positive verification for local clicks.
5. Non-replaying semantics: destructive / non-idempotent clicks are never automatically retried blindly.
6. Execution latency (<3ms verifier evaluation overhead).
"""
import time
import numpy as np
import pytest

from avcs.verifier import (
    VerificationResult,
    VerificationStatus,
    VisualRippleVerifier,
)
from avcs.action_broker import ActionClass
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.memory.types import BoundingBox, UIMLElement, UIMLTree, VisualDiff
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def _make_dummy_state(
    motion_field=None,
    luminance_grid=None,
    surprise_score=0.0,
    visual_diff=None,
    scene_tree=None,
) -> HumanEyeState:
    if motion_field is None:
        motion_field = np.zeros((16, 24, 2), dtype=np.float32)
    if luminance_grid is None:
        luminance_grid = np.zeros((16, 24), dtype=np.float32)

    return HumanEyeState(
        timestamp=time.perf_counter(),
        frame_index=1,
        latency_ms=1.0,
        fps=60.0,
        gaze_fixation=(0.5, 0.5),
        motion_magnitude=0.0,
        motion_field=motion_field,
        moving_objects=[],
        pursuit_target=None,
        pursuit_velocity=(0.0, 0.0),
        motor_flow=(0.0, 0.0),
        contact_acquired=True,
        stability="stable",
        surprise_score=surprise_score,
        affordance_hotspots=[],
        primary_category="editor_code",
        category_confidence=0.95,
        scene_tree=scene_tree,
        visual_diff=visual_diff,
        luminance_grid=luminance_grid,
    )


# ==============================================================================
# 1. Photometric Patch Delta & Optical Ripple Unit Tests
# ==============================================================================


def test_calculate_local_ripple_energy():
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, ripple_radius=0.10)

    motion_field = np.zeros((16, 24, 2), dtype=np.float32)
    # Inject kinetic burst near (0.5, 0.5) -> row 8, col 12
    motion_field[7:10, 11:14, 0] = 0.4
    motion_field[7:10, 11:14, 1] = 0.3

    ripple_local = verifier.calculate_local_ripple((0.5, 0.5), motion_field)
    ripple_distant = verifier.calculate_local_ripple((0.1, 0.1), motion_field)

    assert ripple_local > 0.04
    assert ripple_distant == 0.0


def test_calculate_local_photometric_delta():
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, ripple_radius=0.10)

    pre_luma = np.zeros((16, 24), dtype=np.float32)
    post_luma = np.zeros((16, 24), dtype=np.float32)

    # Inject luminance change in the patch near (0.5, 0.5)
    post_luma[7:10, 11:14] = 0.5

    delta_target = verifier.calculate_local_photometric_delta((0.5, 0.5), pre_luma, post_luma)
    delta_distant = verifier.calculate_local_photometric_delta((0.1, 0.1), pre_luma, post_luma)

    assert delta_target > 0.04
    assert delta_distant == 0.0


# ==============================================================================
# 2. Positive & Negative Control: Verified ONLY on Real Optical / State Change
# ==============================================================================


def test_unreactive_click_is_not_verified():
    """Negative control: clicking a dead/unreactive button must fail verification."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    pre_state = _make_dummy_state()
    post_state = _make_dummy_state()
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    assert not res.is_successful
    assert not res.verified
    assert res.status == VerificationStatus.RETRY_MICRO_JITTER
    assert res.optical_ripple_energy == 0.0
    assert not res.state_transition_detected


def test_pre_existing_motion_does_not_falsely_verify_dead_click():
    """Regression: pre_state had motion (e.g. prior cursor move), post_state has none.
    Must NOT be verified.
    """
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, min_ripple_threshold=0.015)

    # Pre-state has substantial motion at target
    pre_mf = np.zeros((16, 24, 2), dtype=np.float32)
    pre_mf[7:10, 11:14, :] = 0.5
    pre_state = _make_dummy_state(motion_field=pre_mf)

    # Post-state has zero motion and zero photometric change (dead click)
    post_state = _make_dummy_state()
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    assert not res.is_successful
    assert not res.verified
    assert res.status == VerificationStatus.RETRY_MICRO_JITTER


def test_ongoing_unchanged_background_motion_does_not_verify():
    """Ongoing background animation (e.g. video) with no change from click must not verify."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, min_ripple_threshold=0.015)

    # Same continuous motion before and after
    mf = np.zeros((16, 24, 2), dtype=np.float32)
    mf[7:10, 11:14, :] = 0.05
    pre_state = _make_dummy_state(motion_field=mf.copy())
    post_state = _make_dummy_state(motion_field=mf.copy())
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    assert not res.is_successful
    assert not res.verified


def test_real_optical_flow_ripple_verifies_click():
    """Positive control: new kinetic optical ripple at target coordinates verifies click."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, min_ripple_threshold=0.015)

    pre_state = _make_dummy_state()

    # Post-state has optical flow burst at target (0.5, 0.5)
    post_mf = np.zeros((16, 24, 2), dtype=np.float32)
    post_mf[7:10, 11:14, 0] = 0.3
    post_mf[7:10, 11:14, 1] = 0.3
    post_state = _make_dummy_state(motion_field=post_mf)
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    assert res.is_successful
    assert res.verified
    assert res.status == VerificationStatus.VERIFIED
    assert res.optical_ripple_energy >= 0.015


def test_real_photometric_patch_delta_verifies_click():
    """Positive control: button color / luminance change without large motion verifies click."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    pre_luma = np.zeros((16, 24), dtype=np.float32)
    post_luma = np.zeros((16, 24), dtype=np.float32)
    post_luma[7:10, 11:14] = 0.6  # Luminance inversion/color change

    pre_state = _make_dummy_state(luminance_grid=pre_luma)
    post_state = _make_dummy_state(luminance_grid=post_luma)
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    assert res.is_successful
    assert res.verified
    assert res.status == VerificationStatus.VERIFIED
    assert res.photometric_delta >= 0.035


def test_localized_structural_transition_verifies_click():
    """Positive control: element modified in target vicinity verifies click."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, ripple_radius=0.08)

    pre_state = _make_dummy_state()

    # Create visual diff with element near (0.5, 0.5)
    elem = UIMLElement(
        element_id="#btn_submit",
        role=AffordanceType.CLICKABLE if hasattr(AffordanceType := __import__("avcs.memory.types", fromlist=["AffordanceType"]).AffordanceType, "CLICKABLE") else None,
        label="Submitted",
        bbox=BoundingBox(xmin=0.46, ymin=0.48, xmax=0.54, ymax=0.52),
    )
    diff = VisualDiff(
        turn_index=1,
        prior_action="click",
        surprise_score=0.1,
        modified_elements=[(elem, 'label="Submit" -> "Submitted"')],
        causal_summary="Submit button activated.",
    )
    post_state = _make_dummy_state(visual_diff=diff)
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    assert res.is_successful
    assert res.verified
    assert res.status == VerificationStatus.VERIFIED
    assert res.state_transition_detected is True


def test_distant_structural_transition_does_not_verify_click():
    """Negative control: element change on opposite side of screen must NOT verify local click."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver, ripple_radius=0.08)

    pre_state = _make_dummy_state()

    # Element at (0.95, 0.95), clicked at (0.2, 0.2)
    elem = UIMLElement(
        element_id="#clock",
        role=__import__("avcs.memory.types", fromlist=["AffordanceType"]).AffordanceType.STATIC,
        label="12:01",
        bbox=BoundingBox(xmin=0.90, ymin=0.90, xmax=1.00, ymax=1.00),
    )
    diff = VisualDiff(
        turn_index=1,
        prior_action="click",
        surprise_score=0.05,
        modified_elements=[(elem, 'label="12:00" -> "12:01"')],
        causal_summary="Clock ticked in corner.",
    )
    post_state = _make_dummy_state(visual_diff=diff)
    eye.observe = lambda *args, **kwargs: post_state

    res = verifier.verify_action("click", (0.2, 0.2), pre_state=pre_state, settle_ms=0.0)

    assert not res.is_successful
    assert not res.verified
    assert res.status == VerificationStatus.RETRY_MICRO_JITTER
    assert res.state_transition_detected is False


# ==============================================================================
# 3. Non-Replaying Semantics for Destructive Actions
# ==============================================================================


def test_destructive_click_never_blindly_retried():
    """Destructive clicks that fail verification must NEVER execute blind micro-jitter retries."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    clicks = []
    orig_click = driver.click

    def record_click(button="left"):
        clicks.append(button)
        return orig_click(button=button)

    driver.click = record_click
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    # Force verification failure (unreactive button)
    verifier.verify_action = lambda *args, **kwargs: VerificationResult(
        status=VerificationStatus.RETRY_MICRO_JITTER,
        action_name="click@(0.4, 0.4)",
        target_coord=(0.4, 0.4),
        optical_ripple_energy=0.0,
        surprise_pulse=0.0,
        state_transition_detected=False,
        causal_summary="No response",
        recovery_attempt=1,
        elapsed_ms=0.1,
    )

    res = verifier.execute_and_verify(
        action_type="click",
        target=(0.4, 0.4),
        is_destructive=True,
        max_recovery_tier=5,
    )

    # Dispatched exactly ONCE; no blind micro-jitter retries
    assert len(clicks) == 1
    assert res.status == VerificationStatus.ESCALATE_BLOCKED
    assert res.recovery_attempt == 1
    assert "blind replay blocked" in res.causal_summary


def test_action_class_non_idempotent_blocks_replay():
    """Passing action_class=ActionClass.NON_IDEMPOTENT prevents replay."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    clicks = []
    driver.click = lambda button="left": clicks.append(button)
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    verifier.verify_action = lambda *args, **kwargs: VerificationResult(
        status=VerificationStatus.RETRY_MICRO_JITTER,
        action_name="click@(0.5, 0.5)",
        target_coord=(0.5, 0.5),
        optical_ripple_energy=0.0,
        surprise_pulse=0.0,
        state_transition_detected=False,
        causal_summary="No reaction",
        recovery_attempt=1,
        elapsed_ms=0.1,
    )

    res = verifier.execute_and_verify(
        action_type="click",
        target=(0.5, 0.5),
        action_class=ActionClass.NON_IDEMPOTENT,
        max_recovery_tier=5,
    )

    assert len(clicks) == 1
    assert res.status == VerificationStatus.ESCALATE_BLOCKED
    assert res.recovery_attempt == 1


def test_idempotent_click_executes_micro_jitter_recovery():
    """Non-destructive idempotent clicks proceed to Tier 2 micro-jitter recovery when unverified."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    clicks = []
    driver.click = lambda button="left": clicks.append(button)
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    call_count = 0

    def mock_verify(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return VerificationResult(
                status=VerificationStatus.RETRY_MICRO_JITTER,
                action_name="click",
                target_coord=(0.5, 0.5),
                optical_ripple_energy=0.0,
                surprise_pulse=0.0,
                state_transition_detected=False,
                causal_summary="missed deadzone",
                recovery_attempt=1,
                elapsed_ms=0.1,
            )
        # Recovers on second attempt (micro-jitter)
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            action_name="retry_click",
            target_coord=(0.503, 0.503),
            optical_ripple_energy=0.03,
            surprise_pulse=0.1,
            state_transition_detected=True,
            causal_summary="Recovered via Tier 2",
            recovery_attempt=2,
            elapsed_ms=0.2,
        )

    verifier.verify_action = mock_verify

    res = verifier.execute_and_verify(
        action_type="click",
        target=(0.5, 0.5),
        is_destructive=False,
        max_recovery_tier=3,
    )

    assert res.is_successful
    assert res.status == VerificationStatus.VERIFIED
    assert res.recovery_attempt == 2
    assert len(clicks) == 2


# ==============================================================================
# 4. Latency Benchmark (<3ms Overhead)
# ==============================================================================


def test_verifier_evaluation_latency():
    """Verifier must complete post-condition assertions in <3ms when settle_ms=0."""
    eye = HumanEye(source=SyntheticScreen())
    driver = VirtualCursorDriver()
    verifier = VisualRippleVerifier(eye=eye, driver=driver)

    pre_state = _make_dummy_state()
    post_state = _make_dummy_state()
    eye.observe = lambda *args, **kwargs: post_state

    # Warmup
    for _ in range(5):
        verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)

    durations = []
    for _ in range(30):
        t0 = time.perf_counter()
        verifier.verify_action("click", (0.5, 0.5), pre_state=pre_state, settle_ms=0.0)
        durations.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.median(durations))
    p95 = float(np.percentile(durations, 95))

    # Assert sub-3ms evaluation overhead
    assert p50 < 3.0, f"Verifier p50 latency {p50:.2f}ms exceeds 3.0ms budget"
    assert p95 < 10.0, f"Verifier p95 latency {p95:.2f}ms exceeds 10.0ms budget"
