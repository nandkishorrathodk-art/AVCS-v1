"""Trajectory logging: every step the assistant took, as a reviewable dataset.

The design goal is "continuous learning", and the honest first half of that is
*recording*: you cannot learn from trajectories you did not keep. This module
writes one JSONL line per turn with the full provenance -- what was heard, what
was seen, what the gate decided, what the hand did, and whether the screen
actually changed.

What this deliberately is **not**: an online self-training loop. That was tried
in this repo and failed (rejection-sampling self-training, 3% acceptance, the
retrained model got *worse*: S 10% -> 0%). The verifier is deterministic, so it
is gameable, and a model trained against it drifts to task-blind but parseable
output. This module therefore only writes a dataset; turning it into training
data is a separate, offline, human-reviewed step.

Each line is self-contained JSON, so a corrupt or partial write costs one
trajectory, not the file. ``success`` is recorded as three separate booleans
(gate_allowed / performed / verified) rather than one aggregate, because
collapsing them is exactly how "the click was allowed" gets mistaken for "the
task worked".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    """One turn, with everything needed to review or later train on it."""

    turn: int
    heard: str
    said: str
    kind: str
    source: str
    observation: str = ""
    action: Optional[str] = None
    target: Optional[str] = None
    gate_allowed: Optional[bool] = None
    gate_reason: str = ""
    confidence: Optional[float] = None
    confidence_floor: Optional[float] = None
    performed: bool = False
    actuation: str = "none"
    detail: str = ""
    verified: Optional[bool] = None
    visual_delta: Optional[float] = None
    capabilities: Optional[Dict[str, Any]] = None
    at: float = field(default_factory=time.time)

    @property
    def outcome(self) -> str:
        """A single word for filtering, derived from the three separate booleans.

        ``noop`` (nothing was attempted) is distinct from ``refused`` (the gate
        said no) and from ``failed`` (the hand did not perform). They need
        different fixes, so they get different labels.
        """
        if self.action is None:
            return "noop"
        if not self.gate_allowed:
            return "refused"
        if not self.performed:
            return "failed"
        if self.verified is True:
            return "verified"
        if self.verified is False:
            return "no_change"
        return "performed"

    def to_line(self) -> str:
        payload = asdict(self)
        payload["outcome"] = self.outcome
        return json.dumps(payload, ensure_ascii=False)


class TrajectoryLog:
    """Append-only JSONL log of assistant turns.

    Parameters
    ----------
    path:
        File to append to. ``None`` disables logging entirely, so a caller can
        keep an assistant in-memory (tests, a privacy-sensitive run).
    """

    def __init__(self, path: Optional[os.PathLike | str] = None) -> None:
        self.path = Path(path) if path is not None else None
        self.steps: List[TrajectoryStep] = []
        self.write_error: str = ""

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(self, step: TrajectoryStep) -> None:
        """Keep the step in memory and append it to disk.

        A write failure is recorded in ``write_error`` and does not raise: a full
        disk must not take down the assistant mid-task. The step stays in
        ``steps`` so an in-process caller still sees it.
        """
        self.steps.append(step)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(step.to_line() + "\n")
        except Exception as exc:
            self.write_error = f"could not append trajectory: {exc}"

    def load(self) -> List[Dict[str, Any]]:
        """Read the log back as dicts. Corrupt lines are skipped, not fatal."""
        if self.path is None or not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def summary(self) -> Dict[str, Any]:
        """Counts by outcome, so the distribution is visible without a notebook."""
        counts: Dict[str, int] = {}
        for step in self.steps:
            counts[step.outcome] = counts.get(step.outcome, 0) + 1
        return {
            "path": str(self.path) if self.path else None,
            "total": len(self.steps),
            "by_outcome": counts,
            "write_error": self.write_error,
        }

    @staticmethod
    def default_path() -> Path:
        env = os.environ.get("AVCS_TRAJECTORY_PATH")
        if env:
            return Path(env).expanduser()
        return Path.home() / ".avcs" / "trajectories.jsonl"
