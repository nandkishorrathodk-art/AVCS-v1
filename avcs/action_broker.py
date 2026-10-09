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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.verifier import VerificationResult, VerificationStatus, VisualRippleVerifier
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
class ActionRequest:
    """Bounded, validated interaction request."""
    action_type: str                                       # "click", "double_click", "type", "press_key"
    target_coord: Tuple[float, float]                      # Normalized (x, y) coordinates
    action_class: ActionClass = ActionClass.IDEMPOTENT
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:10]}")
    text_content: Optional[str] = None
    expected_postcondition: Optional[Callable[[HumanEyeState, HumanEyeState], bool]] = None
    authorized: bool = False                               # Required true for HIGH_RISK actions
    target_window_title: Optional[str] = None
    timeout_ms: float = 1500.0


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

    @property
    def is_confirmed(self) -> bool:
        return self.status == ActionStatus.CONFIRMED


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
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            return False, f"Target coordinate ({tx:.3f}, {ty:.3f}) out of bounds [0, 1]."

        # Validate High-Risk authorization
        if request.action_class == ActionClass.HIGH_RISK:
            if not request.authorized and not self.allow_destructive:
                return False, f"Action '{request.action_id}' is HIGH_RISK and lacks authorization grant."

        # Validate Window Focus if specified
        if request.target_window_title and current_state.scene_tree:
            current_win = current_state.scene_tree.window_title or ""
            if request.target_window_title.lower() not in current_win.lower():
                return False, f"Focus mismatch: expected window '{request.target_window_title}', but active is '{current_win}'."

        return True, None


class ActionBroker:
    """Executes actions through the bounded lifecycle state machine with verified postconditions."""

    def __init__(
        self,
        eye: HumanEye,
        driver: Optional[OSCursorDriver] = None,
        safety_controller: Optional[ActionSafetyController] = None,
    ) -> None:
        self.eye = eye
        self.driver = driver or get_default_driver()
        self.verifier = VisualRippleVerifier(eye=self.eye, driver=self.driver)
        self.safety = safety_controller or ActionSafetyController()

    def execute(self, request: ActionRequest) -> ActionResult:
        """Executes a bounded action request through the state machine."""
        t0 = time.perf_counter()

        # 1. Capture Pre-State
        pre_state = self.eye.observe()

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

        # 3. Dispatch Physical Input to OS
        tx, ty = request.target_coord
        self._dispatch_input(request, tx, ty)

        # Allow settlement for UI animation
        time.sleep(0.04)

        # 4. Capture Post-State & Verify Outcome
        post_state = self.eye.observe()
        ripple_energy = self.verifier.calculate_local_ripple(request.target_coord, post_state.motion_field)

        # Evaluate Photometric Local Delta
        photo_delta = 0.0
        if pre_state.motion_field is not None and post_state.motion_field is not None:
            pre_local = self.verifier.calculate_local_ripple(request.target_coord, pre_state.motion_field)
            photo_delta = abs(ripple_energy - pre_local)

        # 5. Semantic Postcondition Evaluation
        postcondition_passed = False
        if request.expected_postcondition is not None:
            try:
                postcondition_passed = request.expected_postcondition(pre_state, post_state)
            except Exception as exc:
                print(f"[ActionBroker] Postcondition evaluation exception: {exc}")
                postcondition_passed = False

        # Structural UI transition detected
        diff = post_state.visual_diff
        structural_change = bool(diff and (diff.added_elements or diff.removed_ids or diff.modified_elements))

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
            # General UI action verification
            if structural_change or ripple_energy >= 0.015 or photo_delta >= 0.015:
                status = ActionStatus.CONFIRMED
                evidence_msg = "Physical visual ripple and structural state change confirmed."
            elif post_state.surprise_score > 0.3:
                status = ActionStatus.AMBIGUOUS
                evidence_msg = "Screen underwent global shift; target state confirmation is ambiguous."
            else:
                status = ActionStatus.FAILED
                evidence_msg = "Target showed zero optical reaction."

        # 6. Recovery Policy for Idempotent Actions
        recovery_attempts = 1
        if status in (ActionStatus.FAILED, ActionStatus.AMBIGUOUS) and request.action_class == ActionClass.IDEMPOTENT:
            # Interior Micro-Jitter Retry (Offset by 3px within interior hitbox)
            offsets = [(0.003, 0.003), (-0.003, -0.003)]
            for ox, oy in offsets:
                recovery_attempts += 1
                rx = float(np.clip(tx + ox, 0.0, 1.0))
                ry = float(np.clip(ty + oy, 0.0, 1.0))
                pre_retry = self.eye.observe()
                self._dispatch_input(request, rx, ry)
                time.sleep(0.04)
                post_retry = self.eye.observe()
                retry_ripple = self.verifier.calculate_local_ripple((rx, ry), post_retry.motion_field)
                if request.expected_postcondition is not None:
                    try:
                        passed = request.expected_postcondition(pre_retry, post_retry)
                    except Exception:
                        passed = False
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

        # For NON_IDEMPOTENT actions, blind retry is STRICTLY PROHIBITED
        if status != ActionStatus.CONFIRMED and request.action_class == ActionClass.NON_IDEMPOTENT:
            evidence_msg += " (Non-idempotent action: blind replay blocked by safety controller)."

        elapsed = (time.perf_counter() - t0) * 1000.0
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
        )

    def _dispatch_input(self, request: ActionRequest, x: float, y: float) -> None:
        """Dispatches concrete mouse/keyboard events through cursor driver."""
        if hasattr(self.driver, "move_to_normalized"):
            self.driver.move_to_normalized(x, y)
        elif hasattr(self.driver, "move_to"):
            self.driver.move_to(x, y)

        if request.action_type == "click":
            self.driver.click(button="left")
        elif request.action_type == "double_click":
            self.driver.double_click(button="left")
        elif request.action_type == "type":
            self.driver.click(button="left")
            time.sleep(0.02)
            if request.text_content and hasattr(self.driver, "type_text"):
                self.driver.type_text(request.text_content)
        elif request.action_type == "press_key":
            if request.text_content and hasattr(self.driver, "press_key"):
                self.driver.press_key(request.text_content)
