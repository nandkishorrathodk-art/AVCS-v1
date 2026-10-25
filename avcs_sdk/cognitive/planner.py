"""Hierarchical Goal Decomposition and Multi-Step Cognitive Planning.

System 2 prefrontal planner that transforms high-level user missions into
structured, dependency-tracked execution graphs (PlanStep, PlanGraph, ExecutionState).
Maintains topological order, tracks step prerequisites, and supports dynamic
replanning upon obstacle encounter.

Invariants:
- Pure-pixel security: zero OCR, zero DOM, zero UIA.
- Application agnosticism: zero hardcoded proprietary application names.
- Pure ASCII: zero non-ASCII characters in source and documentation.
- Coordinate privacy: zero coordinate arrays or bounding boxes in plan summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class StepStatus(str, Enum):
    """Status lifecycle of an execution step."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Individual atomic or composite step in a cognitive plan."""
    id: str
    description: str
    action_type: str                  # "inspect", "click", "type", "hotkey", "switch_app", "verify", "recover"
    target: Optional[str] = None      # Semantic description of target
    text: Optional[str] = None        # Text payload if typing
    app_role: Optional[str] = None    # Abstract role: "editor", "data_table", "terminal", "viewer"
    prerequisites: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ready(self, completed_ids: Set[str]) -> bool:
        """A step is ready when all its prerequisites have completed."""
        return self.status == StepStatus.PENDING and set(self.prerequisites).issubset(completed_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "action_type": self.action_type,
            "target": self.target,
            "text": self.text,
            "app_role": self.app_role,
            "prerequisites": list(self.prerequisites),
            "status": self.status.value if isinstance(self.status, StepStatus) else str(self.status),
            "result": self.result,
        }


@dataclass
class PlanGraph:
    """Directed acyclic execution graph representing decomposed goal hierarchy."""
    goal: str
    steps: Dict[str, PlanStep] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)

    def add_step(self, step: PlanStep, depends_on: Optional[List[str]] = None) -> None:
        """Add a step to the plan graph and update prerequisite dependencies."""
        if depends_on:
            for dep in depends_on:
                if dep not in step.prerequisites:
                    step.prerequisites.append(dep)
        self.steps[step.id] = step
        if step.id not in self.execution_order:
            self.execution_order.append(step.id)

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Retrieve step by unique identifier."""
        return self.steps.get(step_id)

    def get_ready_steps(self, completed_ids: Set[str]) -> List[PlanStep]:
        """Return all steps whose prerequisites are met and are pending."""
        ready = []
        for step_id in self.execution_order:
            step = self.steps[step_id]
            if step.is_ready(completed_ids):
                ready.append(step)
        return ready

    def is_complete(self) -> bool:
        """Check if all steps in graph have reached terminal state."""
        if not self.steps:
            return False
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": {sid: s.to_dict() for sid, s in self.steps.items()},
            "execution_order": list(self.execution_order),
            "is_complete": self.is_complete(),
        }

    def to_summary_prompt(self) -> str:
        """Generate qualitative text summary of plan progress for System 2 dialogue."""
        lines = [f"Plan Goal: {self.goal}"]
        for sid in self.execution_order:
            s = self.steps[sid]
            status_tag = f"[{s.status.value.upper()}]"
            role_tag = f" ({s.app_role})" if s.app_role else ""
            lines.append(f"  {status_tag} Step {s.id}{role_tag}: {s.description}")
        return "\n".join(lines)


@dataclass
class ExecutionState:
    """Dynamic tracking of plan execution status."""
    current_step_id: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    status: str = "idle"  # "idle", "executing", "completed", "recovering", "failed"
    history: List[Dict[str, Any]] = field(default_factory=list)


class CognitivePlanner:
    """Prefrontal cognitive planner for multi-step desktop mission execution.

    Decomposes user missions into dependency-ordered PlanGraph structures,
    manages step transitions, and adaptively reformulates sub-plans upon obstacles.
    """

    def __init__(self) -> None:
        self.active_graph: Optional[PlanGraph] = None
        self.state: ExecutionState = ExecutionState()
        self._recovery_counter: int = 0

    def decompose_goal(
        self,
        user_goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PlanGraph:
        """Decompose a high-level user goal into an executable multi-step PlanGraph."""
        goal_lower = user_goal.lower().strip()
        graph = PlanGraph(goal=user_goal)

        # 1. Code syntax inspection, correction, and verification workflow
        if any(w in goal_lower for w in ("syntax", "compile", "code", "bug", "fix error")):
            s1 = PlanStep(
                id="step_1_inspect_code",
                description="Inspect code editor foveal region to locate syntax error or anomaly",
                action_type="inspect",
                target="code syntax anomaly region",
                app_role="editor",
            )
            s2 = PlanStep(
                id="step_2_navigate_error",
                description="Position editing cursor at target error line",
                action_type="click",
                target="syntax error line",
                app_role="editor",
                prerequisites=[s1.id],
            )
            s3 = PlanStep(
                id="step_3_apply_fix",
                description="Type corrective code syntax",
                action_type="type",
                text="return True",
                app_role="editor",
                prerequisites=[s2.id],
            )
            s4 = PlanStep(
                id="step_4_verify_update",
                description="Verify physical ripple update in code editor",
                action_type="verify",
                target="updated code line",
                app_role="editor",
                prerequisites=[s3.id],
            )
            graph.add_step(s1)
            graph.add_step(s2, depends_on=[s1.id])
            graph.add_step(s3, depends_on=[s2.id])
            graph.add_step(s4, depends_on=[s3.id])

        # 2. Cross-application reference data transfer (e.g. data table to editor/terminal)
        elif any(w in goal_lower for w in ("table", "sheet", "transfer", "cross-reference", "audit data")):
            s1 = PlanStep(
                id="step_1_focus_data_table",
                description="Switch active focus to data table application",
                action_type="switch_app",
                target="data table window",
                app_role="data_table",
            )
            s2 = PlanStep(
                id="step_2_read_cell_data",
                description="Inspect and extract reference values from active data cell",
                action_type="inspect",
                target="data table target cell",
                app_role="data_table",
                prerequisites=[s1.id],
            )
            s3 = PlanStep(
                id="step_3_switch_to_editor",
                description="Switch active focus to document editor application",
                action_type="switch_app",
                target="editor window",
                app_role="editor",
                prerequisites=[s2.id],
            )
            s4 = PlanStep(
                id="step_4_insert_reference",
                description="Type transferred reference data into editor document",
                action_type="type",
                text="reference_val = 42",
                app_role="editor",
                prerequisites=[s3.id],
            )
            s5 = PlanStep(
                id="step_5_verify_document",
                description="Verify document update via sensory feedback",
                action_type="verify",
                target="editor insertion area",
                app_role="editor",
                prerequisites=[s4.id],
            )
            graph.add_step(s1)
            graph.add_step(s2, depends_on=[s1.id])
            graph.add_step(s3, depends_on=[s2.id])
            graph.add_step(s4, depends_on=[s3.id])
            graph.add_step(s5, depends_on=[s4.id])

        # 3. Terminal execution and settlement observation workflow
        elif any(w in goal_lower for w in ("terminal", "command", "run script", "execute")):
            s1 = PlanStep(
                id="step_1_focus_terminal",
                description="Switch active focus to terminal console",
                action_type="switch_app",
                target="terminal console window",
                app_role="terminal",
            )
            s2 = PlanStep(
                id="step_2_enter_command",
                description="Type execution command into prompt",
                action_type="type",
                text="python -m test",
                app_role="terminal",
                prerequisites=[s1.id],
            )
            s3 = PlanStep(
                id="step_3_execute_command",
                description="Send enter key to launch process",
                action_type="hotkey",
                target="enter",
                app_role="terminal",
                prerequisites=[s2.id],
            )
            s4 = PlanStep(
                id="step_4_wait_settle",
                description="Wait for execution output and spinner settlement",
                action_type="wait",
                app_role="terminal",
                prerequisites=[s3.id],
            )
            s5 = PlanStep(
                id="step_5_verify_output",
                description="Inspect terminal output for completion confirmation",
                action_type="inspect",
                target="terminal output region",
                app_role="terminal",
                prerequisites=[s4.id],
            )
            graph.add_step(s1)
            graph.add_step(s2, depends_on=[s1.id])
            graph.add_step(s3, depends_on=[s2.id])
            graph.add_step(s4, depends_on=[s3.id])
            graph.add_step(s5, depends_on=[s4.id])

        # 4. General multi-step desktop mission (Inspect -> Actuate -> Verify)
        else:
            s1 = PlanStep(
                id="step_1_observe_desktop",
                description=f"Inspect visual workspace context for mission: {user_goal}",
                action_type="inspect",
                target="active workspace window",
            )
            s2 = PlanStep(
                id="step_2_interact_target",
                description="Engage primary interactive control for requested goal",
                action_type="click",
                target="primary actionable control",
                prerequisites=[s1.id],
            )
            s3 = PlanStep(
                id="step_3_confirm_outcome",
                description="Verify sensory ripple feedback confirming successful interaction",
                action_type="verify",
                target="interaction region",
                prerequisites=[s2.id],
            )
            graph.add_step(s1)
            graph.add_step(s2, depends_on=[s1.id])
            graph.add_step(s3, depends_on=[s2.id])

        self.active_graph = graph
        self._sync_state()
        return graph

    def get_next_step(self) -> Optional[PlanStep]:
        """Select and return the next executable step whose prerequisites are satisfied."""
        if not self.active_graph:
            return None

        completed_set = set(self.state.completed_steps)
        ready_steps = self.active_graph.get_ready_steps(completed_set)
        if not ready_steps:
            if self.active_graph.is_complete():
                self.state.status = "completed"
                self.state.current_step_id = None
            return None

        next_step = ready_steps[0]
        next_step.status = StepStatus.IN_PROGRESS
        self.state.current_step_id = next_step.id
        self.state.status = "executing"
        self._sync_state()
        return next_step

    def mark_step_complete(self, step_id: str, result: Optional[str] = None) -> None:
        """Mark step as successfully completed and transition state."""
        if not self.active_graph:
            return

        step = self.active_graph.get_step(step_id)
        if step:
            step.status = StepStatus.COMPLETED
            step.result = result
            if step_id not in self.state.completed_steps:
                self.state.completed_steps.append(step_id)

            self.state.history.append({
                "step_id": step_id,
                "action": "completed",
                "result": result,
            })

            if self.active_graph.is_complete():
                self.state.status = "completed"
                self.state.current_step_id = None
            else:
                self.state.current_step_id = None

            self._sync_state()

    def mark_step_failed(self, step_id: str, error: Optional[str] = None) -> None:
        """Record step failure and set plan state to failed/recovering."""
        if not self.active_graph:
            return

        step = self.active_graph.get_step(step_id)
        if step:
            step.status = StepStatus.FAILED
            step.result = error
            if step_id not in self.state.failed_steps:
                self.state.failed_steps.append(step_id)

            self.state.history.append({
                "step_id": step_id,
                "action": "failed",
                "error": error,
            })
            self.state.status = "recovering"
            self._sync_state()

    def reformulate_plan(
        self,
        obstacle: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PlanGraph:
        """Dynamically reformulate active plan by injecting recovery steps before resuming blocked flow."""
        if not self.active_graph:
            return self.decompose_goal(f"Recover from obstacle: {obstacle}")

        blocked_step_id = self.state.current_step_id
        obstacle_lower = obstacle.lower()
        self._recovery_counter += 1
        recovery_id = f"recovery_{self._recovery_counter}"

        # 1. Modal popup obstacle
        if "popup" in obstacle_lower or "dialog" in obstacle_lower or "modal" in obstacle_lower:
            recovery_step = PlanStep(
                id=f"{recovery_id}_dismiss_popup",
                description=f"Dismiss unexpected modal dialog triggered by: {obstacle}",
                action_type="click",
                target="modal dialog dismiss or ok button",
                status=StepStatus.PENDING,
            )
        # 2. File locked / access obstacle
        elif "locked" in obstacle_lower or "access" in obstacle_lower or "in use" in obstacle_lower:
            recovery_step = PlanStep(
                id=f"{recovery_id}_release_file_lock",
                description=f"Wait for resource lock release and retry alternate path: {obstacle}",
                action_type="wait",
                target="file resource lock",
                status=StepStatus.PENDING,
            )
        # 3. Syntax error / compilation obstacle
        elif "syntax" in obstacle_lower or "traceback" in obstacle_lower:
            recovery_step = PlanStep(
                id=f"{recovery_id}_fix_syntax_error",
                description=f"Inspect error message and rewrite corrected code syntax: {obstacle}",
                action_type="type",
                text="pass",
                target="syntax error location",
                status=StepStatus.PENDING,
            )
        # 4. Generic obstacle recovery
        else:
            recovery_step = PlanStep(
                id=f"{recovery_id}_adapt_strategy",
                description=f"Recover state from visual anomaly: {obstacle}",
                action_type="wait",
                status=StepStatus.PENDING,
            )

        # Inject recovery step into execution order right before the blocked step
        new_order: List[str] = []
        injected = False
        for sid in self.active_graph.execution_order:
            if sid == blocked_step_id and not injected:
                if recovery_step.id not in new_order:
                    new_order.append(recovery_step.id)
                injected = True
                # Make the blocked step depend on the recovery step
                blocked_step = self.active_graph.get_step(sid)
                if blocked_step and recovery_step.id not in blocked_step.prerequisites:
                    blocked_step.prerequisites.append(recovery_step.id)
                    blocked_step.status = StepStatus.PENDING
            if sid not in new_order:
                new_order.append(sid)

        if not injected and recovery_step.id not in new_order:
            new_order.insert(0, recovery_step.id)

        self.active_graph.steps[recovery_step.id] = recovery_step
        self.active_graph.execution_order = new_order

        self.state.status = "executing"
        self.state.current_step_id = None
        self._sync_state()
        return self.active_graph

    def _sync_state(self) -> None:
        """Synchronize internal pending and completed steps lists."""
        if not self.active_graph:
            return
        pending = []
        for sid in self.active_graph.execution_order:
            s = self.active_graph.steps[sid]
            if s.status in (StepStatus.PENDING, StepStatus.READY, StepStatus.IN_PROGRESS):
                pending.append(sid)
        self.state.pending_steps = pending

