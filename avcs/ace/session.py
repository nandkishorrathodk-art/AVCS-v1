"""Stateful Session Management for the Ace-Control-Medium Architecture.

Tracks goal, multi-turn action history, visual continuity, and termination status.
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from typing import Any, Dict, List, Optional

from avcs.ace.actions import Action, ActionKind
from avcs.ace.computer import Observation
from avcs.ace.skills_vault.cache import (
    SkillCache,
    SkillCacheEntry,
    SkillLookupResult,
    UIStateAnchor,
)


@dataclasses.dataclass
class StepRecord:
    """Telemetry record for a single visual execution step."""
    step_index: int
    action: Dict[str, Any]
    capture_time_ms: float
    plan_latency_ms: float
    timestamp: float
    reason: str = ""


class AceSession:
    """Stateful runtime session tracking autonomous desktop progress across turns."""

    def __init__(self, goal: str = "", max_steps: int = 30, dry_run: bool = False) -> None:
        self.session_id: str = uuid.uuid4().hex[:8]
        self.goal: str = goal
        self.max_steps: int = max_steps
        self.dry_run: bool = dry_run
        self.step_count: int = 0
        self.is_complete: bool = False
        self.stop_reason: str = ""
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.steps: List[StepRecord] = []
        self.subgoals: List[Any] = []
        self.last_observation: Optional[Observation] = None

    def start(self, goal: str, max_steps: Optional[int] = None, dry_run: bool = False) -> None:
        """Initializes a new session run."""
        self.session_id = uuid.uuid4().hex[:8]
        self.goal = goal
        if max_steps is not None:
            self.max_steps = max_steps
        self.dry_run = dry_run
        self.step_count = 0
        self.is_complete = False
        self.stop_reason = ""
        self.start_time = time.time()
        self.end_time = None
        self.steps = []
        self.subgoals = []
        self.last_observation = None

    def _match_action(self, step_action: Dict[str, Any], target_kind: str, target_coord: Any, target_keys: Any, target_text: Any) -> bool:
        """Determines if a recorded step action matches the target parameters."""
        if step_action.get("kind") != target_kind:
            return False
        if target_coord is not None:
            step_coord = step_action.get("coordinate")
            if not step_coord or len(step_coord) < 2:
                return False
            dx = abs(step_coord[0] - target_coord[0])
            dy = abs(step_coord[1] - target_coord[1])
            if not (dx <= 5 and dy <= 5):
                return False
        if target_keys is not None:
            if step_action.get("keys") != target_keys:
                return False
        if target_text is not None:
            if step_action.get("text") != target_text:
                return False
        return True

    def get_repetition_count(self, new_action: Optional[Action] = None) -> int:
        """Counts how many consecutive recent steps match the target action (or the latest step).

        For clicks, matches kind and coordinates within a 5-pixel tolerance window.
        """
        if not self.steps and new_action is None:
            return 0

        if new_action is not None:
            target_kind = new_action.kind.value
            target_coord = new_action.coordinate
            target_keys = new_action.keys
            target_text = new_action.text
        else:
            last_act = self.steps[-1].action
            target_kind = last_act.get("kind")
            target_coord = last_act.get("coordinate")
            target_keys = last_act.get("keys")
            target_text = last_act.get("text")

        count = 0
        for step in reversed(self.steps):
            if self._match_action(step.action, target_kind, target_coord, target_keys, target_text):
                count += 1
            else:
                break
        return count

    def count_recent_matches(self, action: Action, window: int = 5) -> int:
        """Counts how many steps in the recent window match the target action."""
        if not self.steps:
            return 0
        recent = self.steps[-window:]
        target_kind = action.kind.value
        target_coord = action.coordinate
        target_keys = action.keys
        target_text = action.text

        count = 0
        for step in recent:
            if self._match_action(step.action, target_kind, target_coord, target_keys, target_text):
                count += 1
        return count

    def is_stuck(self, threshold: int = 2, new_action: Optional[Action] = None) -> bool:
        """Returns True if the session has repeated the same action at or above threshold times."""
        if self.get_repetition_count(new_action=new_action) >= threshold:
            return True
        if new_action is not None and self.count_recent_matches(new_action, window=5) >= threshold:
            return True
        return False

    def record_step(
        self,
        action: Action,
        observation: Optional[Observation] = None,
        plan_latency_ms: float = 0.0,
        is_subgoal_stop: bool = False,
    ) -> StepRecord:
        """Records an executed action and updates session termination state."""
        self.step_count += 1
        self.last_observation = observation
        capture_ms = observation.capture_time_ms if observation else 0.0

        record = StepRecord(
            step_index=self.step_count,
            action=action.to_dict(),
            capture_time_ms=capture_ms,
            plan_latency_ms=plan_latency_ms,
            timestamp=time.time(),
            reason=action.reason,
        )
        self.steps.append(record)

        # Check termination condition
        if action.kind == ActionKind.STOP and not is_subgoal_stop:
            self.finish(action.reason or "Goal completed successfully.")
        elif self.step_count >= self.max_steps:
            self.finish(f"Maximum step limit ({self.max_steps}) reached.")

        return record

    def finish(self, reason: str) -> None:
        """Marks the session as completed."""
        self.is_complete = True
        self.stop_reason = reason
        self.end_time = time.time()

    def get_history_summary(self, max_recent: int = 5) -> List[Dict[str, Any]]:
        """Returns concise list of recent action summaries for model prompt continuity."""
        recent_records = self.steps[-max_recent:] if self.steps else []
        summary = []
        for r in recent_records:
            act_dict = r.action
            item = {
                "step": r.step_index,
                "action": act_dict.get("kind"),
            }
            if "coordinate" in act_dict:
                item["coordinate"] = act_dict["coordinate"]
            if "text" in act_dict:
                item["text"] = act_dict["text"]
            if "keys" in act_dict:
                item["keys"] = act_dict["keys"]
            if "reason" in act_dict and act_dict["reason"]:
                item["reason"] = act_dict["reason"]
            summary.append(item)
        return summary

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates runtime session metrics."""
        elapsed = (self.end_time or time.time()) - self.start_time
        total_latencies = [s.plan_latency_ms for s in self.steps if s.plan_latency_ms > 0]
        avg_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "total_steps": self.step_count,
            "is_complete": self.is_complete,
            "stop_reason": self.stop_reason,
            "dry_run": self.dry_run,
            "is_stuck": self.is_stuck(threshold=2),
            "elapsed_seconds": round(elapsed, 2),
            "avg_plan_latency_ms": round(avg_latency, 2),
        }

    def get_action_sequence(self) -> List[Dict[str, Any]]:
        """Returns the list of recorded actions across all steps."""
        return [step.action for step in self.steps]

    def save_to_skill_cache(
        self,
        cache: SkillCache,
        anchors: Union[UIStateAnchor, List[UIStateAnchor]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stores the session's action trajectory into the SkillCache with verified anchors."""
        meta = {"session_id": self.session_id, "total_steps": self.step_count}
        if metadata:
            meta.update(metadata)
        return cache.save_skill(
            goal=self.goal,
            actions=self.get_action_sequence(),
            anchors=anchors,
            metadata=meta,
        )

