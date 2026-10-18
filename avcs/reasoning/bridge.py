"""AVCS System 1 <-> System 2 Reasoning Bridge & Orchestration State Machine.

Implements the complete observation-reasoning loop:
OBSERVE -> INTERPRET -> PLAN -> VALIDATE -> PROPOSE_ACTION -> VERIFY_RESULT -> OBSERVE_AGAIN

Operating Mandates:
1. Dry-Run / Shadow Mode Only: Generates validated proposed actions but strictly inhibits
   any live desktop input dispatch (no Win32 SendInput, mouse clicks, or keystrokes).
2. Fail-Closed Authorization: action_authorized = False by default.
3. Strict Coordinate Encapsulation: System 2 expresses intent using natural-language target
   descriptions; all coordinates are derived exclusively by AVCSSystem1V1 neural grounding.
4. Anti-Coordinate Injection: If an external LLM attempts to output explicit coordinates,
   they are stripped and rejected.
5. Non-Hallucinatory Provenance: All actions bind to the current frame sequence.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from avcs.perception.system1_v1 import AVCSSystem1V1
from avcs.perception.visual_state import VisualStateObservation
from avcs.reasoning.contracts import (
    ActionIntent,
    ActionOp,
    IntentValidationResult,
    LoopExecutionReport,
    RejectionCategory,
    StepExecutionRecord,
    System2Decision,
)
from avcs.reasoning.provider import MockReasoningProvider, ReasoningProvider

logger = logging.getLogger("avcs.reasoning.bridge")


class System2Bridge:
    """Orchestrates the closed-loop reasoning bridge between System 1 perception and System 2."""

    def __init__(
        self,
        perception_model: Optional[AVCSSystem1V1] = None,
        checkpoint_path: str = "runs/weights/candidates/avcs_system1_v1.pt",
        provider: Optional[ReasoningProvider] = None,
        mode: str = "shadow",
        max_steps: int = 10,
        presence_threshold: float = 0.50,
        proposal_threshold: float = 0.20,
        device: str = "cpu",
    ) -> None:
        self.mode = mode  # "shadow" (dry-run only)
        self.max_steps = max_steps
        self.presence_threshold = presence_threshold
        self.proposal_threshold = proposal_threshold
        self.device = device

        # Load or bind perception model
        if perception_model is not None:
            self.model = perception_model
        else:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"AVCSSystem1V1 checkpoint not found: {checkpoint_path}")
            self.model = AVCSSystem1V1.load_from_checkpoint(checkpoint_path, device=device)

        self.model.eval()
        self.provider = provider or MockReasoningProvider()
        self.step_history: List[StepExecutionRecord] = []
        self._session_frame_sequence: int = 0

    @property
    def model_parameters(self) -> int:
        return self.model.total_parameters

    def run_task(
        self,
        task: str,
        frame_source: Union[np.ndarray, Image.Image, List[np.ndarray], Callable[[], np.ndarray]],
        initial_query: Optional[str] = None,
    ) -> LoopExecutionReport:
        """Executes the complete observation-reasoning loop for a user task in shadow mode."""
        t_start_total = time.perf_counter()
        self.step_history.clear()
        self.provider.reset()
        self._session_frame_sequence = 0

        active_query: Optional[str] = initial_query
        final_status = "IN_PROGRESS"
        summary_msg = ""
        rejection_count = 0
        replan_count = 0
        authorized_count = 0

        # Helper to get current frame from various source types
        def get_frame(step_idx: int) -> np.ndarray:
            if callable(frame_source):
                f = frame_source()
            elif isinstance(frame_source, list):
                idx = min(step_idx - 1, len(frame_source) - 1)
                f = frame_source[idx]
            elif isinstance(frame_source, Image.Image):
                f = np.array(frame_source.convert("RGB"), dtype=np.uint8)
            else:
                f = frame_source
            if isinstance(f, Image.Image):
                f = np.array(f.convert("RGB"), dtype=np.uint8)
            return f

        for step in range(1, self.max_steps + 1):
            t_step_start = time.perf_counter()
            self._session_frame_sequence += 1
            curr_seq = self._session_frame_sequence

            # -------------------------------------------------------------
            # 1. OBSERVE
            # -------------------------------------------------------------
            raw_frame = get_frame(step)
            t_perc_0 = time.perf_counter()
            obs: VisualStateObservation = self.model.observe_frame(
                raw_frame,
                active_query=active_query,
                frame_sequence=curr_seq,
                presence_threshold=self.presence_threshold,
                proposal_threshold=self.proposal_threshold,
            )
            t_perc_ms = (time.perf_counter() - t_perc_0) * 1000.0

            # Convert to System-2 consumable dictionary (Zero raw RGB pixels or tensor embeddings)
            obs_dict = obs.to_dict()

            # -------------------------------------------------------------
            # 2. INTERPRET
            # -------------------------------------------------------------
            # The observation is structured, truthful, and coordinate-free.
            # Record observation event
            rec = StepExecutionRecord(
                step_index=step,
                timestamp=round(time.time(), 4),
                state_phase="OBSERVE",
                observation_summary={
                    "frame_sequence": obs.frame_sequence,
                    "primary_category": obs.primary_category,
                    "target_presence": obs.target_presence,
                    "presence_confidence": obs.presence_confidence,
                    "active_query": obs.active_query,
                    "detected_regions_count": len(obs.detected_regions),
                    "stability_state": obs.temporal_change.stability_state if obs.temporal_change else "stable",
                    "uncertainty_flags": obs.uncertainty_flags,
                },
                perception_latency_ms=t_perc_ms,
            )

            # -------------------------------------------------------------
            # 3. PLAN
            # -------------------------------------------------------------
            t_prov_0 = time.perf_counter()
            try:
                decision: System2Decision = self.provider.plan_step(
                    task=task,
                    observation=obs_dict,
                    history=[s.__dict__ for s in self.step_history],
                    step_index=step,
                )
                t_prov_ms = (time.perf_counter() - t_prov_0) * 1000.0
            except Exception as prov_err:
                t_prov_ms = (time.perf_counter() - t_prov_0) * 1000.0
                rec.state_phase = "PROVIDER_ERROR"
                rec.provider_latency_ms = t_prov_ms
                rec.step_latency_ms = (time.perf_counter() - t_step_start) * 1000.0
                self.step_history.append(rec)
                final_status = "PROVIDER_ERROR"
                summary_msg = f"System 2 reasoning provider raised exception: {prov_err}"
                break

            rec.llm_decision = {
                "thought": decision.thought,
                "task_status": decision.task_status,
                "op": decision.next_intent.op.value,
                "target": decision.next_intent.target_description,
                "expected_outcome": decision.next_intent.expected_outcome,
            }
            rec.provider_latency_ms = t_prov_ms
            rec.token_usage = getattr(self.provider, "last_usage", None)

            # -------------------------------------------------------------
            # 4. VALIDATE
            # -------------------------------------------------------------
            validation: IntentValidationResult = self._validate_intent(
                decision.next_intent,
                obs,
                raw_frame,
            )
            rec.validation = validation.to_dict()

            if not validation.is_valid:
                rejection_count += 1
                replan_count += 1
                rec.state_phase = "VALIDATE_REJECTED"
                # Pass feedback so provider replans on next turn
                active_query = None
            else:
                if validation.action_authorized:
                    authorized_count += 1

            # -------------------------------------------------------------
            # 5. PROPOSE_ACTION (Shadow Mode Only)
            # -------------------------------------------------------------
            rec.state_phase = "PROPOSE_ACTION"
            rec.proposed_action = {
                "op": decision.next_intent.op.value,
                "target": decision.next_intent.target_description,
                "grounded_target_px": list(validation.grounded_target_px) if validation.grounded_target_px else None,
                "action_authorized": validation.action_authorized,
                "mode": self.mode,
                "executed_to_desktop": False,  # Strict shadow-mode invariant
            }

            # -------------------------------------------------------------
            # 6. VERIFY_RESULT & STATE TERMINATION
            # -------------------------------------------------------------
            rec.step_latency_ms = (time.perf_counter() - t_step_start) * 1000.0
            self.step_history.append(rec)

            op = decision.next_intent.op
            if op == ActionOp.FINISH:
                final_status = "SUCCESS"
                summary_msg = decision.thought or "Task completed successfully."
                break
            elif op == ActionOp.STOP:
                final_status = "ABORTED"
                summary_msg = decision.thought or "Session safely halted by System 2."
                break
            elif op == ActionOp.ASK_USER:
                final_status = "NEEDS_CLARIFICATION"
                summary_msg = decision.clarification_prompt or "Clarification required from user."
                break
            elif op == ActionOp.OBSERVE_TARGET and decision.next_intent.target_description:
                # Set active query for the subsequent observation pass
                active_query = decision.next_intent.target_description
            elif not validation.is_valid:
                # Next turn should replan
                active_query = None

        if final_status == "IN_PROGRESS":
            final_status = "MAX_STEPS_REACHED"
            summary_msg = f"Reached maximum allowed steps ({self.max_steps}) without termination."

        total_latency_ms = (time.perf_counter() - t_start_total) * 1000.0
        n_steps = max(1, len(self.step_history))
        mean_step_lat = total_latency_ms / n_steps
        mean_prov_lat = sum(s.provider_latency_ms for s in self.step_history) / n_steps
        tot_tokens = getattr(self.provider, "total_usage", None)

        return LoopExecutionReport(
            task=task,
            mode=self.mode,
            total_steps=len(self.step_history),
            final_status=final_status,
            step_history=self.step_history,
            total_latency_ms=total_latency_ms,
            mean_step_latency_ms=mean_step_lat,
            mean_provider_latency_ms=mean_prov_lat,
            rejection_count=rejection_count,
            replan_count=replan_count,
            action_authorized_count=authorized_count,
            summary_message=summary_msg,
            total_token_usage=tot_tokens,
        )

    def _validate_intent(
        self,
        intent: ActionIntent,
        observation: VisualStateObservation,
        raw_frame: np.ndarray,
    ) -> IntentValidationResult:
        """Validates an ActionIntent against the current visual state and fail-closed safety policy."""
        # 1. Coordinate injection detection
        params = intent.parameters or {}
        coord_keys = {"x", "y", "coords", "coord", "injected_x", "injected_y"}
        if any(k in params for k in coord_keys):
            return IntentValidationResult(
                is_valid=False,
                reason="Unauthorized coordinate injection attempt detected. System 2 must not output coordinates.",
                rejection_category=RejectionCategory.COORDINATE_INJECTION,
                action_authorized=False,
                frame_sequence=observation.frame_sequence,
            )

        # 2. Stale frame detection
        if intent.frame_sequence is not None and intent.frame_sequence < observation.frame_sequence:
            return IntentValidationResult(
                is_valid=False,
                reason=f"Stale frame reference: intent based on frame {intent.frame_sequence}, current is {observation.frame_sequence}.",
                rejection_category=RejectionCategory.STALE_FRAME,
                action_authorized=False,
                frame_sequence=observation.frame_sequence,
            )

        # 3. Informational or non-mutating ops
        if intent.op in (ActionOp.OBSERVE_TARGET, ActionOp.WAIT, ActionOp.ASK_USER, ActionOp.STOP, ActionOp.FINISH):
            return IntentValidationResult(
                is_valid=True,
                reason=f"Operation '{intent.op.value}' accepted as non-mutating control.",
                action_authorized=False,  # Not a motor action
                frame_sequence=observation.frame_sequence,
                validated_intent=intent,
            )

        # 4. Mutating actions (CLICK, DOUBLE_CLICK, TYPE_TEXT, SCROLL, PRESS_KEY)
        if intent.op in (ActionOp.CLICK, ActionOp.DOUBLE_CLICK, ActionOp.TYPE_TEXT):
            if not intent.target_description:
                return IntentValidationResult(
                    is_valid=False,
                    reason=f"Operation '{intent.op.value}' requires a non-empty target_description.",
                    rejection_category=RejectionCategory.MALFORMED,
                    action_authorized=False,
                    frame_sequence=observation.frame_sequence,
                )

            # Check if current observation already grounded this exact target
            current_obs = observation
            if observation.active_query != intent.target_description:
                # Run on-demand focused grounding pass for the target description
                current_obs = self.model.observe_frame(
                    raw_frame,
                    active_query=intent.target_description,
                    frame_sequence=observation.frame_sequence,
                    presence_threshold=self.presence_threshold,
                    proposal_threshold=self.proposal_threshold,
                )

            # Target Presence Check
            if current_obs.target_presence == "ABSENT":
                return IntentValidationResult(
                    is_valid=False,
                    reason=f"Target '{intent.target_description}' is verified ABSENT on screen.",
                    rejection_category=RejectionCategory.UNGROUNDED,
                    action_authorized=False,
                    frame_sequence=current_obs.frame_sequence,
                    grounded_confidence=current_obs.presence_confidence,
                )

            if current_obs.target_presence == "UNCERTAIN" or current_obs.presence_confidence < self.presence_threshold:
                return IntentValidationResult(
                    is_valid=False,
                    reason=f"Target '{intent.target_description}' presence confidence ({current_obs.presence_confidence:.2f}) is below safe threshold {self.presence_threshold}.",
                    rejection_category=RejectionCategory.UNCERTAIN,
                    action_authorized=False,
                    frame_sequence=current_obs.frame_sequence,
                    grounded_confidence=current_obs.presence_confidence,
                )

            # Ambiguity Check
            if current_obs.is_ambiguous or current_obs.target_margin < 0.10:
                return IntentValidationResult(
                    is_valid=False,
                    reason=f"Target '{intent.target_description}' has multimodal spatial ambiguity (margin {current_obs.target_margin:.2f} < 0.10).",
                    rejection_category=RejectionCategory.UNCERTAIN,
                    action_authorized=False,
                    frame_sequence=current_obs.frame_sequence,
                    grounded_confidence=current_obs.presence_confidence,
                )

            # Temporal Stability Check
            if current_obs.temporal_change and current_obs.temporal_change.stability_state in ("busy", "surprise"):
                return IntentValidationResult(
                    is_valid=False,
                    reason=f"Screen motion state is '{current_obs.temporal_change.stability_state}'. Mutating actions prohibited until stable.",
                    rejection_category=RejectionCategory.SAFETY_VIOLATION,
                    action_authorized=False,
                    frame_sequence=current_obs.frame_sequence,
                )

            # Valid and authorized for shadow-mode proposal
            return IntentValidationResult(
                is_valid=True,
                reason="Neural grounding and safety preconditions satisfied.",
                action_authorized=True,
                validated_intent=intent,
                grounded_target_px=current_obs.selected_target_px,
                grounded_target_norm=current_obs.selected_target_norm,
                grounded_confidence=current_obs.presence_confidence,
                frame_sequence=current_obs.frame_sequence,
            )

        # Default fallback
        return IntentValidationResult(
            is_valid=True,
            reason=f"Operation '{intent.op.value}' passed basic validation.",
            action_authorized=False,
            frame_sequence=observation.frame_sequence,
            validated_intent=intent,
        )

