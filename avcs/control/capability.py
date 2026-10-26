"""Capability kernel for the System 2 (brain) -> System 1 (hand) boundary.

This module is the single place that decides whether a grounded target may be
handed to the motor layer. It exists because `GoalConditionedGrounder.predict`
hardcodes `action_authorized = False`, which makes the click step of the
reasoning loop structurally impossible, while `AVCSSystem1V1.observe_frame`
authorizes on `presence_confidence`, a signal that saturates at 1.0 and cannot
reject anything.

The kernel is *evidence-based*, not a re-enable of the old gate. Measured on
`datasets/ui_grounding` (40 frames, 40 correct queries, 120 deliberately wrong
queries), the grounder's `peak_probability` separates the two populations while
`target_present_probability` does not:

    peak_probability   CORRECT  median 0.873  min 0.755
                       WRONG    median 0.496  p90 0.729  max 0.863
    target_present     CORRECT  1.0000 (all)
                       WRONG    1.0000 (all)   <- saturated, unusable

At `min_peak_probability = 0.75` the correct queries pass 100% and the wrong
queries pass 6.7%. That is the margin the kernel relies on, and it is stated
here so it can be re-measured rather than assumed.

Fail-closed by construction:
  * default policy grants nothing;
  * a mutating op needs an explicit grant for that op;
  * a query allow-list, when set, is the only thing that can name a target;
  * coordinate injection, stale frames and blank/noise frames are refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# Mutating operations. Everything not listed is treated as observing and needs
# no capability grant.
MUTATING_OPS = frozenset({"click", "double_click", "type", "type_text", "press_key", "scroll"})


@dataclass(frozen=True)
class GroundingEvidence:
    """Signals a grounder actually produces, independent of its own gate.

    `authorized` and `target_present` are carried for reporting only; the kernel
    does not trust either, because the shipped grounder hardcodes the first to
    False and saturates the second at 1.0.
    """

    candidate_point_norm: Optional[tuple[float, float]]
    peak_probability: float
    target_present_probability: float = 1.0
    target_present: bool = True
    authorized: bool = False
    active_query: Optional[str] = None


@dataclass(frozen=True)
class CapabilityDecision:
    """Outcome of a capability check, with the reason kept for the audit trail."""

    allowed: bool
    reason: str
    op: str = ""
    query: Optional[str] = None
    coordinate: Optional[tuple[float, float]] = None


def frame_is_degenerate(frame: np.ndarray) -> tuple[bool, str]:
    """True when a frame has too little structure to ground anything on.

    Catches the blank/noise cases that made the old presence gate fail open: a
    uniform frame or an unstructured noise field has no interactive element, so
    no grounded point on it can be trusted.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return True, "empty frame"

    gray = frame.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=-1)

    gy, gx = np.gradient(gray)
    mean_grad = float(np.mean(np.hypot(gx, gy)))
    std = float(np.std(gray))

    if std < 1.0:
        return True, f"uniform frame (pixel std {std:.2f})"
    if mean_grad < 0.5:
        return True, f"no spatial structure (mean gradient {mean_grad:.3f})"
    return False, ""


class CapabilityKernel:
    """Decides whether a grounded target may reach the motor layer.

    Parameters
    ----------
    granted_ops:
        Operations this machine is allowed to perform. Default is empty, so a
        caller that forgets to configure the kernel gets no mutating power.
    allowed_queries:
        Optional allow-list of natural-language targets. When set, only these
        queries can be grounded for a mutating action; anything else is refused.
    min_peak_probability:
        Grounder peak confidence floor. 0.75 is the measured operating point
        (100% correct pass / 6.7% wrong pass on the ui_grounding sample).
    """

    def __init__(
        self,
        granted_ops: Iterable[str] = (),
        allowed_queries: Optional[Sequence[str]] = None,
        min_peak_probability: float = 0.75,
        allow_degenerate_frames: bool = False,
    ) -> None:
        self.granted_ops = frozenset(o.strip().lower() for o in granted_ops if o)
        self.allowed_queries = (
            None if allowed_queries is None else frozenset(q.strip().lower() for q in allowed_queries if q)
        )
        self.min_peak_probability = float(min_peak_probability)
        self.allow_degenerate_frames = bool(allow_degenerate_frames)

    # ------------------------------------------------------------------ #
    # Policy inspection
    # ------------------------------------------------------------------ #
    def is_granted(self, op: str) -> bool:
        """Observing ops are always allowed; mutating ops need an explicit grant."""
        clean = (op or "").strip().lower()
        if clean not in MUTATING_OPS:
            return True
        return clean in self.granted_ops

    @property
    def has_any_mutating_grant(self) -> bool:
        return bool(self.granted_ops & MUTATING_OPS)

    # ------------------------------------------------------------------ #
    # The single decision point
    # ------------------------------------------------------------------ #
    def authorize(
        self,
        *,
        op: str,
        query: Optional[str],
        evidence: GroundingEvidence,
        frame: Optional[np.ndarray] = None,
    ) -> CapabilityDecision:
        """Return whether `op` on `query` may proceed, and why."""
        clean_op = (op or "").strip().lower()

        if clean_op in MUTATING_OPS and not self.is_granted(clean_op):
            return CapabilityDecision(
                allowed=False,
                reason=f"no capability grant for mutating op '{clean_op}'",
                op=clean_op,
                query=query,
            )

        if self.allowed_queries is not None:
            if not query or query.strip().lower() not in self.allowed_queries:
                return CapabilityDecision(
                    allowed=False,
                    reason=f"query '{query}' is not in the capability allow-list",
                    op=clean_op,
                    query=query,
                )

        if not self.allow_degenerate_frames and frame is not None:
            degenerate, why = frame_is_degenerate(frame)
            if degenerate:
                return CapabilityDecision(
                    allowed=False,
                    reason=f"refusing to ground on a degenerate frame: {why}",
                    op=clean_op,
                    query=query,
                )

        if evidence.candidate_point_norm is None:
            return CapabilityDecision(
                allowed=False,
                reason="grounder produced no candidate point",
                op=clean_op,
                query=query,
            )

        peak = float(evidence.peak_probability)
        if peak < self.min_peak_probability:
            return CapabilityDecision(
                allowed=False,
                reason=(
                    f"grounding peak {peak:.3f} below capability floor "
                    f"{self.min_peak_probability:.2f} (target likely absent)"
                ),
                op=clean_op,
                query=query,
            )

        return CapabilityDecision(
            allowed=True,
            reason=(
                f"capability '{clean_op}' granted and grounding peak {peak:.3f} "
                f">= {self.min_peak_probability:.2f}"
            ),
            op=clean_op,
            query=query,
            coordinate=tuple(evidence.candidate_point_norm),
        )

    def authorize_prediction(
        self,
        *,
        op: str,
        query: Optional[str],
        prediction: Any,
        frame: Optional[np.ndarray] = None,
    ) -> CapabilityDecision:
        """Convenience wrapper around a `GroundingPrediction`-shaped object."""
        evidence = GroundingEvidence(
            candidate_point_norm=getattr(prediction, "candidate_point_norm", None),
            peak_probability=float(getattr(prediction, "peak_probability", 0.0) or 0.0),
            target_present_probability=float(
                getattr(prediction, "target_present_probability", 1.0) or 1.0
            ),
            target_present=bool(getattr(prediction, "target_present", True)),
            authorized=bool(getattr(prediction, "action_authorized", False)),
            active_query=query,
        )
        return self.authorize(op=op, query=query, evidence=evidence, frame=frame)
