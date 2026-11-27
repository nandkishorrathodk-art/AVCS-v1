"""AVCS Milestone 2: Action Safety Controller & Bounded Windows Action Broker.

Replaces blind click-and-assume-success with a formal action lifecycle state machine:
States: PENDING -> DISPATCHED -> (CONFIRMED | FAILED | AMBIGUOUS | BLOCKED)
Features:
- Action risk classification (IDEMPOTENT, NON_IDEMPOTENT, HIGH_RISK)
- Focus and target freshness pre-validation before SendInput
- Blind retry prevention on non-idempotent actions
- Semantic postcondition evaluation with local photometric delta evidence
"""
from __future__ import annotations

import enum
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from avcs.perception.human_eye import HumanEye, HumanEyeState
from avcs.control.verifier import (
    VerificationResult,
    VerificationStatus,
    VisualRippleVerifier,
)
from forge.vision.cortex.motor.driver import OSCursorDriver, get_default_driver


class ActionClass(str, enum.Enum):
    """Risk and replay categorization of agent operations."""
    IDEMPOTENT = "idempotent"                         # Safe to repeat (e.g. switch tab, open dropdown)
    CONDITIONALLY_REPEATABLE = "cond_repeatable"     # Repeatable only if precondition still holds
    NON_IDEMPOTENT = "non_idempotent"                 # Must never blind-retry (e.g. Submit, Delete, Pay)
    HIGH_RISK = "high_risk"                           # Requires explicit caller authorization


class ActionStatus(str, enum.Enum):
    """Discrete terminal states of an action execution lifecycle."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class PostconditionSpec:
    """Formal specification of an expected visual postcondition."""
    postcondition_type: str = "any"  # "optical_ripple", "text_change", "state_transition", "window_launch", "custom"
    target_coord: Optional[Tuple[float, float]] = None
    target_region: Optional[Tuple[float, float, float, float]] = None
    expected_text: Optional[str] = None
    min_ripple_threshold: float = 0.015
    min_photometric_delta: float = 0.035
    timeout_ms: float = 1500.0
    predicate: Optional[Callable[[HumanEyeState, HumanEyeState], bool]] = None


@dataclass(frozen=True)
class VerificationDetail:
    """Detailed visual verification evidence formatted for System 2 privacy boundary."""
    status: str
    verified: bool
    optical_ripple_energy: float
    photometric_delta: float
    surprise_pulse: float
    state_transition_detected: bool
    causal_summary: str
    recovery_attempts: int
    elapsed_ms: float
    fresh_frame_sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Encapsulated representation: zero raw coordinates or tensors."""
        return {
            "status": self.status,
            "verified": self.verified,
            "optical_ripple_energy": round(self.optical_ripple_energy, 4),
            "photometric_delta": round(self.photometric_delta, 4),
            "surprise_pulse": round(self.surprise_pulse, 4),
            "state_transition_detected": self.state_transition_detected,
            "causal_summary": self.causal_summary,
            "recovery_attempts": self.recovery_attempts,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "fresh_frame_sequence": self.fresh_frame_sequence,
        }


@dataclass(frozen=True)
class ActionRequest:
    """Bounded, validated interaction request."""
    action_type: str                                       # "click", "double_click", "type", "press_key"
    target_coord: Tuple[float, float]                      # Normalized (x, y) coordinates
    action_class: ActionClass = ActionClass.NON_IDEMPOTENT
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:10]}")
    text_content: Optional[str] = None
    expected_postcondition: Optional[Any] = None          # Callable or PostconditionSpec
    precondition_validator: Optional[Callable[[HumanEyeState], bool]] = None
    authorized: bool = False                               # Required true for HIGH_RISK actions
    target_window_title: Optional[str] = None
    timeout_ms: float = 1500.0
    button: str = "left"


@dataclass(frozen=True)
class ActionResult:
    """Auditable, immutable result of an action execution."""
    action_id: str
    status: ActionStatus
    action_type: str
    target_coord: Tuple[float, float]
    elapsed_ms: float
    evidence: str
    optical_ripple_energy: float
    photometric_delta: float
    recovery_attempts: int
    error_category: Optional[str] = None
    verification_detail: Optional[VerificationDetail] = None

    @property
    def is_confirmed(self) -> bool:
        return self.status == ActionStatus.CONFIRMED

    def to_dict(self) -> Dict[str, Any]:
        """Encapsulated System 2 representation (zero raw coordinates or tensors)."""
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "action_type": self.action_type,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "evidence": self.evidence,
            "optical_ripple_energy": round(self.optical_ripple_energy, 4),
            "photometric_delta": round(self.photometric_delta, 4),
            "recovery_attempts": self.recovery_attempts,
            "error_category": self.error_category,
            "is_confirmed": self.is_confirmed,
            "verification_detail": (
                self.verification_detail.to_dict() if self.verification_detail else None
            ),
        }


class ActionSafetyController:
    """Validates target freshness, window focus, and authorization boundaries."""

    def __init__(self, allow_destructive: bool = False) -> None:
        self.allow_destructive = allow_destructive

    def validate_preconditions(
        self,
        request: ActionRequest,
        current_state: HumanEyeState,
    ) -> Tuple[bool, Optional[str]]:
        """Checks target bounds, focus, and security authorizations."""
        tx, ty = request.target_coord
        if request.action_type not in {"click", "double_click", "type", "press_key"}:
            return False, f"Unsupported action type: {request.action_type}"
        if request.button not in {"left", "right", "middle"}:
            return False, f"Unsupported mouse button: {request.button}"
        if (
            isinstance(request.timeout_ms, bool)
            or not isinstance(request.timeout_ms, (int, float))
            or not math.isfinite(request.timeout_ms)
            or request.timeout_ms <= 0
        ):
            return False, f"timeout_ms must be a positive finite number: {request.timeout_ms}"
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            return False, f"Target coordinate ({tx:.3f}, {ty:.3f}) out of bounds [0, 1]."

        # Validate High-Risk authorization
        if request.action_class == ActionClass.HIGH_RISK:
            if not request.authorized and not self.allow_destructive:
                return False, f"Action '{request.action_id}' is HIGH_RISK and lacks authorization grant."

        # Validate Window Focus if specified
        if request.target_window_title:
            if current_state.scene_tree is None:
                # No scene tree available — fail closed, cannot confirm focus.
                return False, (
                    f"Focus verification blocked: scene_tree is unavailable, "
                    f"cannot confirm expected window '{request.target_window_title}'."
                )
            current_win = current_state.scene_tree.window_title or ""
            if request.target_window_title.lower() not in current_win.lower():
                return False, f"Focus mismatch: expected window '{request.target_window_title}', but active is '{current_win}'."

        # Validate Precondition Validator if specified on request
        if request.precondition_validator is not None:
            try:
                pre_ok = request.precondition_validator(current_state)
                if not pre_ok:
                    return False, "Precondition validator evaluated to False."
            except Exception as exc:
                return False, f"Precondition validator raised exception: {exc}"

        return True, None


class ActionBroker:
    """Executes actions through the bounded lifecycle state machine with verified postconditions."""

    def __init__(
        self,
        eye: HumanEye,
        driver: Optional[OSCursorDriver] = None,
        safety_controller: Optional[ActionSafetyController] = None,
        settle_delay_s: float = 0.04,
    ) -> None:
        self.eye = eye
        self.driver = driver or get_default_driver()
        self.verifier = VisualRippleVerifier(eye=self.eye, driver=self.driver)
        self.safety = safety_controller or ActionSafetyController()
        self.settle_delay_s = max(0.0, float(settle_delay_s))

    def observe(self) -> HumanEyeState:
        """Captures and returns current visual state through the eye."""
        return self.eye.observe()

    def execute(self, request: ActionRequest) -> ActionResult:
        """Executes a bounded action request through the state machine."""
        t0 = time.perf_counter()
        if (
            isinstance(request.timeout_ms, bool)
            or not isinstance(request.timeout_ms, (int, float))
            or not math.isfinite(request.timeout_ms)
            or request.timeout_ms <= 0
        ):
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.BLOCKED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                evidence=(
                    "timeout_ms must be a positive finite number: "
                    f"{request.timeout_ms!r}"
                ),
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="INVALID_TIMEOUT",
            )
        deadline = t0 + request.timeout_ms / 1000.0

        # 1. Capture Pre-State
        try:
            pre_state = self.eye.observe()
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=f"Observation capture failed: {exc}",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="CAPTURE_ERROR",
            )

        # 2. Safety Controller Precondition Validation
        valid, reason = self.safety.validate_preconditions(request, pre_state)
        if not valid:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.BLOCKED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=reason or "Precondition validation failed.",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="PRECONDITION_BLOCKED",
            )

        if time.perf_counter() >= deadline:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence="Action deadline expired before input dispatch.",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="TIMEOUT_BEFORE_DISPATCH",
            )

        # 3. Dispatch Physical Input to OS
        pre_seq = getattr(self.eye.source, "sequence_number", getattr(self.eye, "frame_count", 0))
        tx, ty = request.target_coord
        dispatch_error: Optional[Exception] = None
        try:
            self._dispatch_input(request, tx, ty)
        except Exception as exc:
            dispatch_error = exc

        if dispatch_error is not None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=f"Input dispatch failed: {dispatch_error}",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="DISPATCH_FAILED",
            )

        if time.perf_counter() >= deadline:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=(
                    "Input dispatch exceeded timeout_ms; the physical action may "
                    "have occurred and was not replayed."
                ),
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="TIMEOUT_AFTER_DISPATCH",
            )

        # Allow settlement for UI animation
        if self.settle_delay_s > 0:
            time.sleep(min(self.settle_delay_s, max(0.0, deadline - time.perf_counter())))

        # 4. Capture Post-State & Verify Outcome (Strictly fresh frame S > S_pre)
        try:
            post_state = self.eye.observe(min_sequence=pre_seq)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=f"Post-action capture failed: {exc}",
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="CAPTURE_ERROR",
            )

        if time.perf_counter() >= deadline:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ActionResult(
                action_id=request.action_id,
                status=ActionStatus.AMBIGUOUS,
                action_type=request.action_type,
                target_coord=request.target_coord,
                elapsed_ms=elapsed,
                evidence=(
                    "Post-action observation completed after timeout_ms; outcome "
                    "is ambiguous and the action was not replayed."
                ),
                optical_ripple_energy=0.0,
                photometric_delta=0.0,
                recovery_attempts=0,
                error_category="TIMEOUT_AFTER_DISPATCH",
            )

        ripple_energy = self.verifier.calculate_local_ripple(request.target_coord, post_state.motion_field)

        # Evaluate Photometric Local Delta
        photo_delta = 0.0
        if pre_state.motion_field is not None and post_state.motion_field is not None:
            pre_local = self.verifier.calculate_local_ripple(request.target_coord, pre_state.motion_field)
            photo_delta = abs(ripple_energy - pre_local)

        # Structural UI transition detected
        diff = post_state.visual_diff
        structural_change = bool(diff and (diff.added_elements or diff.removed_ids or diff.modified_elements))

        # 5. Semantic Postcondition Evaluation
        postcondition_passed = False
        if request.expected_postcondition is not None:
            postcondition_passed = self._evaluate_postcondition(
                request.expected_postcondition,
                pre_state,
                post_state,
                request.target_coord,
                ripple_energy,
                photo_delta,
                structural_change,
            )

        # Decision Logic: Determine Terminal State
        status = ActionStatus.FAILED
        evidence_msg = "No reaction observed."

        if request.expected_postcondition is not None:
            if postcondition_passed:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Semantic postcondition explicitly verified."
            elif ripple_energy > 0.015 or structural_change:
                status = ActionStatus.AMBIGUOUS
                evidence_msg = "Visual motion detected, but expected semantic postcondition was not satisfied."
            else:
                status = ActionStatus.FAILED
                evidence_msg = "Neither semantic postcondition nor local optical motion occurred."
        else:
            # General UI action verification (requires localized ripple or state diff)
            if (ripple_energy >= 0.015 or photo_delta >= 0.015) and post_state.surprise_score <= 0.4:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Physical visual ripple confirmed at target coordinate."
            elif structural_change and post_state.surprise_score <= 0.4:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Structural UI state change confirmed."
            elif post_state.surprise_score > 0.3:
                status = ActionStatus.AMBIGUOUS
                evidence_msg = "Screen underwent global shift; target state confirmation is ambiguous."
            else:
                status = ActionStatus.FAILED
                evidence_msg = "Target showed zero optical reaction."

        # 6. Recovery Policy for Actions
        recovery_attempts = 1
        can_retry = False
        if status in (ActionStatus.FAILED, ActionStatus.AMBIGUOUS):
            can_retry = request.action_type in {"click", "double_click"}
            if not can_retry:
                evidence_msg += " (Non-idempotent input is never replayed.)"
            elif request.action_class == ActionClass.IDEMPOTENT:
                can_retry = True
            elif request.action_class == ActionClass.CONDITIONALLY_REPEATABLE:
                can_retry = (
                    request.precondition_validator is not None
                    and request.precondition_validator(post_state)
                )
                if not can_retry:
                    evidence_msg += " (Conditionally repeatable action: precondition validator did not pass on post-state)."
            elif request.action_class in (ActionClass.NON_IDEMPOTENT, ActionClass.HIGH_RISK):
                can_retry = False
                evidence_msg += " (Non-idempotent action: blind replay blocked by safety controller)."

        if can_retry:
            # Interior Micro-Jitter Retry (Offset by 3px within interior hitbox)
            offsets = [(0.003, 0.003), (-0.003, -0.003)]
            for ox, oy in offsets:
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    status = ActionStatus.AMBIGUOUS
                    evidence_msg = "Action deadline expired; no further input was dispatched."
                    break
                recovery_attempts += 1
                rx = float(np.clip(tx + ox, 0.0, 1.0))
                ry = float(np.clip(ty + oy, 0.0, 1.0))
                try:
                    pre_retry = self.eye.observe()
                    self._dispatch_input(request, rx, ry)
                    if self.settle_delay_s > 0:
                        time.sleep(
                            min(self.settle_delay_s, max(0.0, deadline - time.perf_counter()))
                        )
                    post_retry = self.eye.observe()
                    if time.perf_counter() >= deadline:
                        status = ActionStatus.AMBIGUOUS
                        evidence_msg = (
                            "Action deadline expired during recovery; outcome is "
                            "ambiguous and no further inputs were dispatched."
                        )
                        break
                    retry_ripple = self.verifier.calculate_local_ripple((rx, ry), post_retry.motion_field)
                except Exception:
                    break

                if request.expected_postcondition is not None:
                    passed = self._evaluate_postcondition(
                        request.expected_postcondition,
                        pre_retry,
                        post_retry,
                        (rx, ry),
                        retry_ripple,
                        photo_delta,
                        bool(post_retry.visual_diff and (post_retry.visual_diff.added_elements or post_retry.visual_diff.removed_ids)),
                    )
                    if passed:
                        status = ActionStatus.CONFIRMED
                        evidence_msg = f"Recovered via interior micro-jitter ({ox:+.3f}, {oy:+.3f}) and verified semantic postcondition."
                        ripple_energy = retry_ripple
                        break
                else:
                    if retry_ripple >= 0.015 or bool(post_retry.visual_diff and (post_retry.visual_diff.added_elements or post_retry.visual_diff.removed_ids)):
                        status = ActionStatus.CONFIRMED
                        evidence_msg = f"Recovered via interior micro-jitter ({ox:+.3f}, {oy:+.3f})."
                        ripple_energy = retry_ripple
                        break

        elapsed = (time.perf_counter() - t0) * 1000.0
        v_detail = VerificationDetail(
            status=status.value,
            verified=status == ActionStatus.CONFIRMED,
            optical_ripple_energy=ripple_energy,
            photometric_delta=photo_delta,
            surprise_pulse=float(getattr(post_state, "surprise_score", 0.0)),
            state_transition_detected=structural_change,
            causal_summary=evidence_msg,
            recovery_attempts=recovery_attempts,
            elapsed_ms=elapsed,
            fresh_frame_sequence=(
                getattr(post_state, "frame_sequence", None)
                or getattr(post_state, "frame_index", None)
                or getattr(getattr(self, "eye", None), "source", None)
                and getattr(self.eye.source, "sequence_number", None)
            ),
        )
        return ActionResult(
            action_id=request.action_id,
            status=status,
            action_type=request.action_type,
            target_coord=request.target_coord,
            elapsed_ms=elapsed,
            evidence=evidence_msg,
            optical_ripple_energy=ripple_energy,
            photometric_delta=photo_delta,
            recovery_attempts=recovery_attempts,
            error_category=None if status == ActionStatus.CONFIRMED else status.value.upper(),
            verification_detail=v_detail,
        )

    def _evaluate_postcondition(
        self,
        expected: Any,
        pre_state: HumanEyeState,
        post_state: HumanEyeState,
        target_coord: Tuple[float, float],
        ripple_energy: float,
        photo_delta: float,
        structural_change: bool,
    ) -> bool:
        if expected is None:
            return False
        if isinstance(expected, PostconditionSpec):
            if expected.predicate is not None:
                try:
                    return bool(expected.predicate(pre_state, post_state))
                except Exception:
                    return False
            ptype = expected.postcondition_type
            if ptype == "optical_ripple":
                tc = expected.target_coord or target_coord
                r = self.verifier.calculate_local_ripple(tc, post_state.motion_field)
                return r >= expected.min_ripple_threshold
            elif ptype == "text_change":
                return photo_delta >= expected.min_photometric_delta or bool(
                    post_state.visual_diff and (post_state.visual_diff.modified_elements or post_state.visual_diff.added_elements)
                )
            elif ptype == "state_transition":
                return structural_change
            elif ptype == "window_launch":
                return structural_change or getattr(post_state, "surprise_score", 0.0) > 0.3
            else:
                return (
                    ripple_energy >= expected.min_ripple_threshold
                    or photo_delta >= expected.min_photometric_delta
                    or structural_change
                )
        elif callable(expected):
            try:
                return bool(expected(pre_state, post_state))
            except Exception:
                return False
        return False

    def _dispatch_input(self, request: ActionRequest, x: float, y: float) -> None:
        """Dispatches concrete mouse/keyboard events through cursor driver."""
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)
        else:
            raise RuntimeError("Cursor driver does not support pointer movement.")

        if request.action_type == "click":
            self.driver.click(button=request.button)
        elif request.action_type == "double_click":
            self.driver.double_click(button=request.button)
        elif request.action_type == "type":
            self.driver.click(button="left")
            time.sleep(0.02)
            if request.text_content and hasattr(self.driver, "type_text"):
                self.driver.type_text(request.text_content)
            elif request.text_content:
                raise RuntimeError("Cursor driver does not support text entry.")
        elif request.action_type == "press_key":
            if request.text_content and hasattr(self.driver, "press_key"):
                self.driver.press_key(request.text_content)
            else:
                raise RuntimeError("press_key action requires text_content and driver support.")
        else:
            raise RuntimeError(f"Unsupported action type: {request.action_type}")

        # Notify interactive capture source of simulated input event
        cap = getattr(self.eye, "source", None)
        while cap is not None:
            if hasattr(cap, "trigger_motion"):
                cap.trigger_motion(x, y)
                break
            cap = getattr(cap, "source", None)
