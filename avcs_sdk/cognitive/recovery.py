"""Dynamic Obstacle Recovery for AVCS System 2.

Detects unexpected obstacles triggered by sensory predictive coding surprise
(||z_{t+1} - z_hat_{t+1}|| > tau) or state errors:
1. Modal popup dialogs (alerts, confirmations).
2. File access obstacles (locked or in-use files).
3. Code and command syntax errors.

Generates structured recovery directives (dismiss dialogs, wait for lock release,
correct syntax errors) and manages escalation policies.

Invariants:
- Pure-pixel security: zero OCR, zero DOM, zero UIA.
- Application agnosticism: zero hardcoded proprietary application names.
- Pure ASCII: 100% pure ASCII characters.
- Coordinate privacy: zero coordinate leakage into directives.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ObstacleType(str, Enum):
    """Categorization of unexpected desktop execution obstacles."""
    MODAL_POPUP = "modal_popup"
    FILE_ACCESS = "file_access"
    SYNTAX_ERROR = "syntax_error"
    UNRESPONSIVE_TARGET = "unresponsive_target"
    VISUAL_SURPRISE = "visual_surprise"
    UNKNOWN = "unknown"


@dataclass
class ObstacleEvent:
    """Record of a detected execution obstacle."""
    obstacle_type: ObstacleType
    surprise_magnitude: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_type": self.obstacle_type.value if isinstance(self.obstacle_type, ObstacleType) else str(self.obstacle_type),
            "surprise_magnitude": self.surprise_magnitude,
            "description": self.description,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class RecoveryDirective:
    """Actionable instruction for recovering from a detected obstacle."""
    action: str                       # "click", "type", "hotkey", "wait", "reformulate", "abort"
    target: Optional[str] = None      # Semantic description of recovery target
    text: Optional[str] = None        # Text payload if typing fix
    keys: Optional[List[str]] = None  # Key combination if hotkey
    reason: str = ""                  # Rationale for directive
    strategy: str = ""                # Strategy name (e.g. "dismiss_button", "escape_key")
    escalation_level: int = 1         # Current attempt escalation (1, 2, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "text": self.text,
            "keys": self.keys,
            "reason": self.reason,
            "strategy": self.strategy,
            "escalation_level": self.escalation_level,
        }


class ObstacleRecoveryManager:
    """Detects and resolves execution obstacles using sensory surprise and error signals.

    Implements adaptive recovery strategies for modal dialogs, file access contention,
    syntax errors, and unresponsive targets with progressive escalation.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.55,
        popup_surprise_threshold: float = 0.65,
        max_retries: int = 3,
    ) -> None:
        self.surprise_threshold = surprise_threshold
        self.popup_surprise_threshold = popup_surprise_threshold
        self.max_retries = max_retries
        self.attempt_counts: Dict[str, int] = {}
        self.recovery_history: List[Dict[str, Any]] = []

    def detect_obstacle(
        self,
        surprise_score: float,
        has_popup: bool = False,
        error_message: Optional[str] = None,
        screen_state: Optional[Dict[str, Any]] = None,
        ripple_verified: Optional[bool] = None,
        is_stable: Optional[bool] = None,
    ) -> Optional[ObstacleEvent]:
        """Detect if current sensory or state conditions indicate an execution obstacle."""
        err_lower = (error_message or "").lower()

        # 1. Modal popup dialog obstacle
        if has_popup or (surprise_score >= self.popup_surprise_threshold and "dialog" in err_lower):
            return ObstacleEvent(
                obstacle_type=ObstacleType.MODAL_POPUP,
                surprise_magnitude=surprise_score,
                description="Unexpected modal popup dialog interrupted active workflow",
                details={"has_popup": has_popup, "screen_state": screen_state or {}},
            )

        # 2. File access contention / locked file obstacle
        if any(w in err_lower for w in ("locked", "permission denied", "in use", "access denied", "sharing violation", "busy")):
            return ObstacleEvent(
                obstacle_type=ObstacleType.FILE_ACCESS,
                surprise_magnitude=max(surprise_score, 0.4),
                description=f"File access obstacle encountered: {error_message}",
                details={"error_message": error_message},
            )

        # 3. Syntax error / compilation failure obstacle
        if any(w in err_lower for w in ("syntaxerror", "syntax error", "traceback", "compile error", "failed to parse", "unexpected token")):
            return ObstacleEvent(
                obstacle_type=ObstacleType.SYNTAX_ERROR,
                surprise_magnitude=max(surprise_score, 0.38),
                description=f"Code or command syntax error detected: {error_message}",
                details={"error_message": error_message},
            )

        # 4. Unresponsive interaction target (action taken but zero pixel ripple registered)
        if ripple_verified is False:
            return ObstacleEvent(
                obstacle_type=ObstacleType.UNRESPONSIVE_TARGET,
                surprise_magnitude=surprise_score,
                description="Target element registered zero physical pixel ripple (disabled or unresponsive)",
                details={"ripple_verified": False},
            )

        # 5. General visual surprise mismatch without explicit error tag
        if surprise_score >= self.surprise_threshold and (is_stable is not True):
            return ObstacleEvent(
                obstacle_type=ObstacleType.VISUAL_SURPRISE,
                surprise_magnitude=surprise_score,
                description=f"High sensory prediction surprise mismatch (magnitude: {surprise_score:.3f})",
                details={"surprise_score": surprise_score},
            )

        return None

    def generate_recovery(
        self,
        obstacle: ObstacleEvent,
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryDirective:
        """Generate structured recovery directive tailored to obstacle type and escalation level."""
        otype_str = obstacle.obstacle_type.value if isinstance(obstacle.obstacle_type, ObstacleType) else str(obstacle.obstacle_type)
        current_attempts = self.attempt_counts.get(otype_str, 0) + 1
        self.attempt_counts[otype_str] = current_attempts

        if current_attempts > self.max_retries:
            return RecoveryDirective(
                action="abort",
                reason=f"Exceeded maximum recovery attempts ({self.max_retries}) for obstacle: {otype_str}",
                strategy="abort",
                escalation_level=current_attempts,
            )

        # Strategy A: Modal popup recovery
        if obstacle.obstacle_type == ObstacleType.MODAL_POPUP:
            if current_attempts == 1:
                return RecoveryDirective(
                    action="click",
                    target="modal dialog dismiss or ok button",
                    reason="Dismissing unexpected screen dialog via primary affirmative control.",
                    strategy="dismiss_button",
                    escalation_level=1,
                )
            elif current_attempts == 2:
                return RecoveryDirective(
                    action="hotkey",
                    keys=["escape"],
                    reason="Dismissing modal dialog via escape key cancellation.",
                    strategy="escape_key",
                    escalation_level=2,
                )
            else:
                return RecoveryDirective(
                    action="click",
                    target="modal dialog cancel or close button",
                    reason="Dismissing stubborn modal dialog via secondary cancel control.",
                    strategy="cancel_click",
                    escalation_level=3,
                )

        # Strategy B: File access obstacle recovery
        elif obstacle.obstacle_type == ObstacleType.FILE_ACCESS:
            if current_attempts <= 2:
                return RecoveryDirective(
                    action="wait",
                    reason="Waiting for background process to release locked file resource.",
                    strategy="wait_lock_release",
                    escalation_level=current_attempts,
                )
            else:
                return RecoveryDirective(
                    action="reformulate",
                    target="alternate file path or copy",
                    reason="File remains locked; reformulating plan to use alternate resource path.",
                    strategy="alternate_file_path",
                    escalation_level=current_attempts,
                )

        # Strategy C: Syntax error recovery
        elif obstacle.obstacle_type == ObstacleType.SYNTAX_ERROR:
            if current_attempts == 1:
                return RecoveryDirective(
                    action="type",
                    text="return True",
                    target="syntax error location",
                    reason="Typing corrective code statement to fix detected syntax error.",
                    strategy="code_syntax_fix",
                    escalation_level=1,
                )
            else:
                return RecoveryDirective(
                    action="reformulate",
                    target="code block restructuring",
                    reason="Syntax error persists; reformulating plan to rewrite surrounding block.",
                    strategy="block_reformulation",
                    escalation_level=current_attempts,
                )

        # Strategy D: Unresponsive target recovery
        elif obstacle.obstacle_type == ObstacleType.UNRESPONSIVE_TARGET:
            return RecoveryDirective(
                action="click",
                target="primary interactive control retry",
                reason="Retrying interaction on target control with secondary focus offset.",
                strategy="offset_retry",
                escalation_level=current_attempts,
            )

        # Strategy E: Visual surprise / general settlement recovery
        else:
            return RecoveryDirective(
                action="wait",
                reason="Waiting for unexpected visual motion and screen rendering to settle.",
                strategy="observe_settle",
                escalation_level=current_attempts,
            )

    def record_outcome(
        self,
        obstacle_type: ObstacleType | str,
        success: bool,
    ) -> None:
        """Record the outcome of a recovery attempt to adapt future escalation."""
        otype_str = obstacle_type.value if isinstance(obstacle_type, ObstacleType) else str(obstacle_type)
        if success:
            # Reset escalation on success
            self.attempt_counts[otype_str] = 0

        self.recovery_history.append({
            "obstacle_type": otype_str,
            "success": success,
            "timestamp": time.time(),
        })

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Return summary metrics on detected obstacles and recovery resolutions."""
        total = len(self.recovery_history)
        successes = sum(1 for h in self.recovery_history if h.get("success", False))
        return {
            "total_obstacles_handled": total,
            "successful_recoveries": successes,
            "active_attempt_counts": dict(self.attempt_counts),
        }

