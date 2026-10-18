"""AVCS System 2 Reasoning Contracts & Typed Intent Specifications.

Defines the typed data structures bridging AVCS System 1 (Neural Perception & Motor Actuation)
and System 2 (High-Level Reasoning & Planning).

Core Mandates:
1. Strict Coordinate Encapsulation: System 2 expresses intent using natural-language target
   descriptions; all spatial coordinate generation and foveal anchoring belongs to System 1.
2. Anti-Coordinate Injection: If an external LLM attempts to output explicit coordinates
   (e.g. x, y, bbox), they are stripped and rejected by the validator.
3. Fail-Closed Invariant: action_authorized = False by default. Actions can only be authorized
   when neural grounding, temporal stability, and non-ambiguity criteria are satisfied.
4. Non-Hallucinatory Provenance: All intents bind to an explicit frame_sequence and observation
   timestamp. Stale frames (> max_frame_lag) trigger re-observation.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


class ActionOp(str, Enum):
    """Supported high-level operations for System 2 intent."""
    OBSERVE_TARGET = "observe_target"   # Request focused visual grounding on a specific target
    CLICK = "click"                     # Click grounded target
    DOUBLE_CLICK = "double_click"       # Double click grounded target
    TYPE_TEXT = "type_text"             # Type text into focused target field
    PRESS_KEY = "press_key"             # Send hotkey or keyboard navigation key
    SCROLL = "scroll"                   # Scroll window up or down
    WAIT = "wait"                       # Wait for UI transition or asynchronous completion
    ASK_USER = "ask_user"               # Ask user for clarification or intervention
    STOP = "stop"                       # Stop/abort execution due to unrecoverable condition
    FINISH = "finish"                   # Task objective achieved


class RejectionCategory(str, Enum):
    """Categorization for rejected action intents."""
    MALFORMED = "malformed"
    STALE_FRAME = "stale_frame"
    UNGROUNDED = "ungrounded"
    UNCERTAIN = "uncertain"
    SAFETY_VIOLATION = "safety_violation"
    UNSUPPORTED_OP = "unsupported_op"
    COORDINATE_INJECTION = "coordinate_injection"


@dataclasses.dataclass(frozen=True)
class ActionIntent:
    """Structured machine-readable action intent formulated by System 2.

    System 2 emits what it wants done, why, and what evidence it expects to observe afterwards.
    It NEVER emits (x, y) coordinates.
    """
    intent_id: str
    step_index: int
    op: ActionOp
    target_description: Optional[str] = None
    text_input: Optional[str] = None
    key_name: Optional[str] = None
    scroll_delta: Optional[int] = None
    expected_outcome: str = ""
    rationale: str = ""
    frame_sequence: Optional[int] = None
    required_evidence: Optional[str] = None
    parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(
        cls,
        step_index: int,
        op: ActionOp | str,
        target_description: Optional[str] = None,
        text_input: Optional[str] = None,
        key_name: Optional[str] = None,
        scroll_delta: Optional[int] = None,
        expected_outcome: str = "",
        rationale: str = "",
        frame_sequence: Optional[int] = None,
        required_evidence: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ActionIntent:
        if isinstance(op, str):
            try:
                op = ActionOp(op.lower())
            except ValueError:
                raise ValueError(f"Unsupported ActionOp: {op}")
        return cls(
            intent_id=f"intent_{uuid.uuid4().hex[:8]}",
            step_index=step_index,
            op=op,
            target_description=target_description.strip() if target_description else None,
            text_input=text_input,
            key_name=key_name,
            scroll_delta=scroll_delta,
            expected_outcome=expected_outcome,
            rationale=rationale,
            frame_sequence=frame_sequence,
            required_evidence=required_evidence,
            parameters=parameters or {},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_step: int = 1) -> Tuple[ActionIntent, bool]:
        """Parses intent dictionary, strictly sanitizing and flagging coordinate injection attempts.

        Returns (intent, had_coordinate_injection).
        """
        data = dict(data)
        had_coord_injection = False

        # Anti-Coordinate Injection: detect and strip coordinates from LLM
        forbidden_keys = {"x", "y", "coord", "coords", "coordinate", "coordinates", "bbox", "pixel", "pixels"}
        for k in list(data.keys()):
            if k.lower() in forbidden_keys:
                had_coord_injection = True
                del data[k]
        if "parameters" in data and isinstance(data["parameters"], dict):
            for k in list(data["parameters"].keys()):
                if k.lower() in forbidden_keys:
                    had_coord_injection = True
                    del data["parameters"][k]

        op_str = data.get("op", "observe_target")
        try:
            op = ActionOp(str(op_str).lower())
        except ValueError:
            op = ActionOp.STOP

        intent = cls(
            intent_id=data.get("intent_id", f"intent_{uuid.uuid4().hex[:8]}"),
            step_index=int(data.get("step_index", default_step)),
            op=op,
            target_description=data.get("target_description"),
            text_input=data.get("text_input"),
            key_name=data.get("key_name"),
            scroll_delta=data.get("scroll_delta"),
            expected_outcome=data.get("expected_outcome", ""),
            rationale=data.get("rationale", ""),
            frame_sequence=data.get("frame_sequence"),
            required_evidence=data.get("required_evidence"),
            parameters=data.get("parameters", {}),
        )
        return intent, had_coord_injection

    def to_dict(self) -> Dict[str, Any]:
        """Serializes intent to JSON-safe dictionary with zero coordinates."""
        return {
            "intent_id": self.intent_id,
            "step_index": self.step_index,
            "op": self.op.value,
            "target_description": self.target_description,
            "text_input": self.text_input,
            "key_name": self.key_name,
            "scroll_delta": self.scroll_delta,
            "expected_outcome": self.expected_outcome,
            "rationale": self.rationale,
            "frame_sequence": self.frame_sequence,
            "required_evidence": self.required_evidence,
            "parameters": self.parameters,
        }


@dataclasses.dataclass(frozen=True)
class System2Decision:
    """Complete reasoning cycle decision emitted by System 2."""
    thought: str
    task_status: str                         # "in_progress", "waiting_for_observation", "needs_clarification", "completed", "aborted"
    next_intent: ActionIntent
    clarification_prompt: Optional[str] = None
    alternative_intents: List[ActionIntent] = dataclasses.field(default_factory=list)
    confidence: float = 1.0


@dataclasses.dataclass(frozen=True)
class IntentValidationResult:
    """Result of checking an ActionIntent against the live System 1 observation and safety policy."""
    is_valid: bool
    reason: str
    rejection_category: Optional[RejectionCategory] = None
    action_authorized: bool = False          # Strict Fail-Closed invariant
    validated_intent: Optional[ActionIntent] = None
    grounded_target_px: Optional[Tuple[float, float]] = None
    grounded_target_norm: Optional[Tuple[float, float]] = None
    grounded_confidence: float = 0.0
    frame_sequence: int = 0
    uncertainty_flags: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "reason": self.reason,
            "rejection_category": self.rejection_category.value if self.rejection_category else None,
            "action_authorized": self.action_authorized,
            "grounded_target_px": list(self.grounded_target_px) if self.grounded_target_px else None,
            "grounded_target_norm": list(self.grounded_target_norm) if self.grounded_target_norm else None,
            "grounded_confidence": round(self.grounded_confidence, 3),
            "frame_sequence": self.frame_sequence,
            "uncertainty_flags": self.uncertainty_flags,
        }


@dataclasses.dataclass
class StepExecutionRecord:
    """Audit log record of a single turn in the observation-reasoning loop."""
    step_index: int
    timestamp: float
    state_phase: str                         # "OBSERVE", "INTERPRET", "PLAN", "VALIDATE", "PROPOSE_ACTION", "VERIFY_RESULT"
    observation_summary: Dict[str, Any]      # Bounded, privacy-preserving observation
    llm_decision: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    proposed_action: Optional[Dict[str, Any]] = None
    simulated_verification: Optional[Dict[str, Any]] = None
    step_latency_ms: float = 0.0
    provider_latency_ms: float = 0.0
    perception_latency_ms: float = 0.0
    token_usage: Optional[Dict[str, int]] = None


@dataclasses.dataclass
class LoopExecutionReport:
    """Final summary report of an observation-reasoning session."""
    task: str
    mode: str                                # "shadow" (dry-run)
    total_steps: int
    final_status: str                        # "SUCCESS", "ABORTED", "MAX_STEPS_REACHED", "PROVIDER_ERROR", "NEEDS_CLARIFICATION"
    step_history: List[StepExecutionRecord]
    total_latency_ms: float
    mean_step_latency_ms: float
    mean_provider_latency_ms: float
    rejection_count: int
    replan_count: int
    action_authorized_count: int
    summary_message: str
    total_token_usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "mode": self.mode,
            "total_steps": self.total_steps,
            "final_status": self.final_status,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "mean_step_latency_ms": round(self.mean_step_latency_ms, 2),
            "mean_provider_latency_ms": round(self.mean_provider_latency_ms, 2),
            "rejection_count": self.rejection_count,
            "replan_count": self.replan_count,
            "action_authorized_count": self.action_authorized_count,
            "summary_message": self.summary_message,
            "total_token_usage": self.total_token_usage,
            "step_history": [
                {
                    "step_index": s.step_index,
                    "timestamp": round(s.timestamp, 4),
                    "state_phase": s.state_phase,
                    "observation_summary": s.observation_summary,
                    "llm_decision": s.llm_decision,
                    "validation": s.validation,
                    "proposed_action": s.proposed_action,
                    "simulated_verification": s.simulated_verification,
                    "step_latency_ms": round(s.step_latency_ms, 2),
                    "provider_latency_ms": round(s.provider_latency_ms, 2),
                    "perception_latency_ms": round(s.perception_latency_ms, 2),
                    "token_usage": s.token_usage,
                }
                for s in self.step_history
            ],
        }

