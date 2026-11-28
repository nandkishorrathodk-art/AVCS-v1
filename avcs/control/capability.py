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

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

# Mutating operations. Unknown ops are NOT silently allowed; only operations
# in OBSERVING_OPS bypass capability grants.
MUTATING_OPS = frozenset({
    "click", "left_click", "double_click", "triple_click", "right_click",
    "middle_click", "drag", "left_click_drag", "drag_start", "mouse_move",
    "type", "type_text", "press_key", "hotkey", "press_hotkey", "key", "scroll",
    "launch_app", "read_file", "list_dir",
})

OBSERVING_OPS = frozenset({
    "observe", "observe_target", "finish", "stop", "wait", "ask_user",
    "read_screen", "screenshot", "list", "stat", "scan", "abort",
})

# Mutating ops that need a *grounded point*: they aim at a specific place on
# screen, so a candidate point and a peak above the floor are the evidence that
# the target is really there. The keyboard ops below act on the focused widget
# or globally -- they have no coordinate at all -- so demanding a point for them
# made the natural "type the path, press Enter" route unreachable: the kernel
# refused with "grounder produced no candidate point" for an op that never had
# one. They still need an explicit grant; they just are not grounded.
#
# `launch_app` / `read_file` / `list_dir` are also non-grounded: they are named
# by argument (an app or a path), not located by pixels. Their containment is a
# separate check (`path_roots` / `allowed_apps`), not the grounding floor.
POINT_OPS = frozenset({
    "click", "left_click", "double_click", "triple_click", "right_click",
    "middle_click", "drag", "left_click_drag", "drag_start", "mouse_move",
})


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
    # Structured uncertainty, so a caller can decide *how* to react instead of
    # pattern-matching the reason string. `confidence` is the grounding peak the
    # decision was made on (0.0 when there was no candidate point at all), and
    # `floor` is the threshold it was compared against. A brain that sees
    # "0.72 vs floor 0.75" knows it was close and could ask; "0.10" means the
    # target is not there and asking would be pointless.
    confidence: float = 0.0
    floor: float = 0.0
    # How far the evidence is from the decision boundary, in the floor's units.
    # Negative = refused by this margin, positive = allowed by it.
    margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "op": self.op,
            "query": self.query,
            "coordinate": list(self.coordinate) if self.coordinate else None,
            "confidence": self.confidence,
            "floor": self.floor,
            "margin": self.margin,
        }


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
        allowed_apps: Optional[Iterable[str]] = None,
        path_roots: Optional[Iterable[str]] = None,
    ) -> None:
        self.granted_ops = frozenset(o.strip().lower() for o in granted_ops if o)
        self.allowed_queries = (
            None if allowed_queries is None else frozenset(q.strip().lower() for q in allowed_queries if q)
        )
        self.min_peak_probability = float(min_peak_probability)
        self.allow_degenerate_frames = bool(allow_degenerate_frames)
        # Argument containment for the non-grounded ops. Absence is denial: with
        # no allow-list a `launch_app` / `read_file` is refused, so a caller that
        # forgets to configure the kernel gets no filesystem or process reach.
        self.allowed_apps = (
            None if allowed_apps is None else frozenset(a.strip().lower() for a in allowed_apps if a)
        )
        self.path_roots = (
            None if path_roots is None else tuple(os.path.realpath(p) for p in path_roots)
        )

    # ------------------------------------------------------------------ #
    # Policy inspection
    # ------------------------------------------------------------------ #
    def is_granted(self, op: str) -> bool:
        """Observing ops need no grant; mutating and unknown ops require explicit grant."""
        clean = (op or "").strip().lower()
        if clean in OBSERVING_OPS:
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
        arguments: Optional[dict] = None,
    ) -> CapabilityDecision:
        """Return whether `op` on `query` may proceed, and why.

        ``arguments`` carries the named operands of a non-grounded op (an app
        name for ``launch_app``, a path for ``read_file`` / ``list_dir``). They
        are checked against the allow-lists here, in the same single decision
        point, so there is no second place a caller can bypass.
        """
        clean_op = (op or "").strip().lower()

        # Observing ops have nothing to actuate and nothing to ground, so they
        # are allowed without a grant, without a query allow-list and without a
        # candidate point -- exactly what `is_granted` already reports. Running
        # the grounding checks on them was a bug: it made a plain "look at the
        # screen" step fail with "grounder produced no candidate point", which
        # reads as a denied click for an action that was never a click.
        if clean_op in OBSERVING_OPS:
            return CapabilityDecision(
                allowed=True,
                reason=f"observing op '{clean_op}' needs no capability grant",
                op=clean_op,
                query=query,
                confidence=1.0,
                floor=0.0,
                margin=1.0,
            )

        if clean_op not in MUTATING_OPS and clean_op not in self.granted_ops:
            return CapabilityDecision(
                allowed=False,
                reason=f"unknown op '{clean_op}' denied (fail-closed)",
                op=clean_op,
                query=query,
                confidence=0.0,
                floor=self.min_peak_probability,
                margin=-self.min_peak_probability,
            )

        if not self.is_granted(clean_op):
            return CapabilityDecision(
                allowed=False,
                reason=f"no capability grant for mutating op '{clean_op}'",
                op=clean_op,
                query=query,
                confidence=1.0,
                floor=0.0,
                margin=1.0,
            )

        if clean_op not in POINT_OPS:
            # A keyboard op (type/type_text/press_key/scroll) has no coordinate:
            # it acts on the focused widget or globally. The grant above is the
            # whole decision. Requiring a grounded point here refused the natural
            # "type the path, press Enter" route for a reason that does not apply.
            containment = self._check_arguments(clean_op, arguments or {})
            if containment is not None:
                return containment
            return CapabilityDecision(
                allowed=True,
                reason=f"capability '{clean_op}' granted (no grounded point needed)",
                op=clean_op,
                query=query,
                confidence=1.0,
                floor=0.0,
                margin=1.0,
            )

        if self.allowed_queries is not None:
            if not query or query.strip().lower() not in self.allowed_queries:
                return CapabilityDecision(
                    allowed=False,
                    reason=f"query '{query}' is not in the capability allow-list",
                    op=clean_op,
                    query=query,
                    confidence=1.0,
                    floor=0.0,
                    margin=1.0,
                )

        raw_peak = float(evidence.peak_probability)
        safe_peak = raw_peak if math.isfinite(raw_peak) else 0.0

        if not self.allow_degenerate_frames and frame is not None:
            degenerate, why = frame_is_degenerate(frame)
            if degenerate:
                return CapabilityDecision(
                    allowed=False,
                    reason=f"refusing to ground on a degenerate frame: {why}",
                    op=clean_op,
                    query=query,
                    confidence=safe_peak,
                    floor=self.min_peak_probability,
                    margin=safe_peak - self.min_peak_probability,
                )

        # Validate that the candidate coordinate is finite and in [0, 1]
        if evidence.candidate_point_norm is not None:
            cx, cy = evidence.candidate_point_norm
            if (not math.isfinite(cx) or not math.isfinite(cy)
                    or cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0):
                return CapabilityDecision(
                    allowed=False,
                    reason=f"candidate coordinate ({cx}, {cy}) is out of bounds or non-finite",
                    op=clean_op,
                    query=query,
                    confidence=safe_peak,
                    floor=self.min_peak_probability,
                    margin=-self.min_peak_probability,
                )

        if evidence.candidate_point_norm is None:
            return CapabilityDecision(
                allowed=False,
                reason="grounder produced no candidate point",
                op=clean_op,
                query=query,
                confidence=safe_peak,
                floor=self.min_peak_probability,
                margin=safe_peak - self.min_peak_probability,
            )

        peak = float(evidence.peak_probability)
        # NaN bypass: `NaN < threshold` is False, so NaN would skip this gate.
        # Guard explicitly with math.isnan and math.isfinite; NaN/Inf is refused.
        if math.isnan(peak) or not math.isfinite(peak):
            return CapabilityDecision(
                allowed=False,
                reason=f"grounding peak is non-finite ({peak!r}), refusing",
                op=clean_op,
                query=query,
                confidence=0.0,
                floor=self.min_peak_probability,
                margin=-self.min_peak_probability,
            )
        if peak < self.min_peak_probability:
            return CapabilityDecision(
                allowed=False,
                reason=(
                    f"grounding peak {peak:.3f} below capability floor "
                    f"{self.min_peak_probability:.2f} (target likely absent)"
                ),
                op=clean_op,
                query=query,
                confidence=peak,
                floor=self.min_peak_probability,
                margin=peak - self.min_peak_probability,
            )

        if clean_op == "drag":
            # A drag has *two* ends. The source is grounded above; the destination
            # must also be a real, above-floor target, or the hand would drop the
            # item at a guessed point. Both ends are checked in the same decision
            # point so a drag can never be half-authorized.
            point2 = (arguments or {}).get("point2")
            raw_peak2 = (arguments or {}).get("peak2")
            peak2 = float(raw_peak2) if raw_peak2 is not None else 0.0
            if not point2 or len(point2) != 2:
                return CapabilityDecision(
                    allowed=False,
                    reason="drag needs a destination point (no second grounded target)",
                    op=clean_op, query=query,
                    confidence=peak, floor=self.min_peak_probability,
                    margin=peak - self.min_peak_probability,
                )
            if math.isnan(peak2) or not math.isfinite(peak2):
                return CapabilityDecision(
                    allowed=False,
                    reason=f"drag destination peak is non-finite ({peak2!r}), refusing",
                    op=clean_op, query=query,
                    confidence=0.0, floor=self.min_peak_probability,
                    margin=-self.min_peak_probability,
                )
            if peak2 < self.min_peak_probability:
                return CapabilityDecision(
                    allowed=False,
                    reason=(
                        f"drag destination peak {peak2:.3f} below capability floor "
                        f"{self.min_peak_probability:.2f} (drop target likely absent)"
                    ),
                    op=clean_op, query=query,
                    confidence=peak2, floor=self.min_peak_probability,
                    margin=peak2 - self.min_peak_probability,
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
            confidence=peak,
            floor=self.min_peak_probability,
            margin=peak - self.min_peak_probability,
        )

    # ------------------------------------------------------------------ #
    # Argument containment for non-grounded ops
    # ------------------------------------------------------------------ #
    def _check_arguments(self, op: str, arguments: dict) -> Optional[CapabilityDecision]:
        """Containment for ops named by argument rather than by pixels.

        Returns a decision when the argument is refused, ``None`` when the op has
        no argument to check or the argument passes. Absence of an allow-list is
        denial, so an unconfigured kernel cannot launch or read anything.
        """
        if op == "launch_app":
            app = str(arguments.get("app") or arguments.get("text_input") or "").strip()
            if not app:
                return self._arg_refused(op, "no app named", arguments)
            if self.allowed_apps is None:
                return self._arg_refused(
                    op, "no app allow-list configured (launch is off by default)", arguments
                )
            if app.lower() not in self.allowed_apps:
                return self._arg_refused(
                    op, f"app '{app}' is not in the allow-list {sorted(self.allowed_apps)}", arguments
                )
            return None

        if op in ("read_file", "list_dir"):
            path = str(arguments.get("path") or arguments.get("text_input") or "").strip()
            if not path:
                return self._arg_refused(op, "no path given", arguments)
            if self.path_roots is None:
                return self._arg_refused(
                    op, "no path roots configured (file access is off by default)", arguments
                )
            try:
                real = os.path.realpath(os.path.expanduser(path))
            except (OSError, ValueError) as exc:
                return self._arg_refused(op, f"unresolvable path: {exc}", arguments)
            # `realpath` collapses `..` and follows symlinks *before* the prefix
            # check, so neither traversal nor a symlink out of the root can pass.
            for root in self.path_roots:
                # On Windows, os.path.realpath may differ in case; normcase both
                # sides so the prefix check is case-insensitive on case-folding
                # filesystems.
                real_nc = os.path.normcase(real)
                root_nc = os.path.normcase(root)
                if real_nc == root_nc or real_nc.startswith(root_nc + os.sep):
                    return None
            return self._arg_refused(
                op, f"path '{real}' escapes the allowed roots {list(self.path_roots)}", arguments
            )

        return None

    def _arg_refused(self, op: str, why: str, arguments: dict) -> CapabilityDecision:
        return CapabilityDecision(
            allowed=False,
            reason=f"refusing '{op}': {why}",
            op=op,
            confidence=1.0,
            floor=0.0,
            margin=1.0,
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
        raw_peak = getattr(prediction, "peak_probability", 0.0)
        try:
            peak_val = float(raw_peak or 0.0)
        except (ValueError, TypeError):
            peak_val = 0.0
        if not math.isfinite(peak_val):
            peak_val = 0.0
        evidence = GroundingEvidence(
            candidate_point_norm=getattr(prediction, "candidate_point_norm", None),
            peak_probability=peak_val,
            target_present_probability=float(
                getattr(prediction, "target_present_probability", 1.0) or 1.0
            ),
            target_present=bool(getattr(prediction, "target_present", True)),
            authorized=bool(getattr(prediction, "action_authorized", False)),
            active_query=query,
        )
        return self.authorize(op=op, query=query, evidence=evidence, frame=frame)
