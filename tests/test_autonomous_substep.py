"""Tests for AVCS System 1 Autonomous Substep Loop.

Verifies:
1. System 2 high-level goal delegation without per-click external prompting.
2. The Look -> Ground -> Act -> Verify autonomous substep execution cycle.
3. Zero coordinate leakage in SubstepTrace and GoalExecutionResult.
4. Truthful, evidence-backed natural language summaries for System 2.
5. Immediate halting and honest reporting on unexpected modal blockers (GOAL.md Example B).
6. Fail-closed target grounding and visual focus verification.
7. Simulated application flow (Text Editor launch -> focus -> type -> close -> prompt -> restore).
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from avcs.control.controller import (
    ActionResult,
    AVCSController,
    GoalExecutionResult,
    SceneObservation,
    SubstepTrace,
)
from avcs.control.simulated_app import SimulatedAppScreen
from avcs.control.substep import AutonomousSubstepPlanner
from avcs.control.verifier import VerificationResult, VerificationStatus
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.sdk import UnvalidatedGroundingError


@pytest.fixture
def simulated_controller():
    """Creates a hermetic controller attached to a simulated interactive desktop app."""
    sim_screen = SimulatedAppScreen(width=320, height=240)
    controller = AVCSController(
        source=sim_screen,
        weights_path=None,
        device="cpu",
        target_fps=60,
        force_virtual_motor=True,
        enable_streaming=False,
    )
    sim_screen.attach_controller(controller)
    yield controller, sim_screen
    controller.close()


def test_substep_trace_and_goal_result_zero_coordinates():
    """Verify that SubstepTrace and GoalExecutionResult strictly leak ZERO coordinates."""
    trace = SubstepTrace(
        step_index=1,
        action="click",
        target_query="Play button",
        success=True,
        verified=True,
        message="Clicked and verified Play button.",
        latency_ms=12.5,
        verification_status="verified",
        observation_summary="Display is active.",
    )
    d = trace.to_dict()
    assert "x" not in d
    assert "y" not in d
    assert "target_coord" not in d
    assert "bounding_box" not in d
    assert d["step_index"] == 1
    assert d["action"] == "click"
    assert d["success"] is True

    result = GoalExecutionResult(
        goal="explore text editor",
        success=True,
        status="completed",
        summary="Explored text editor successfully.",
        substeps=[trace],
        total_latency_ms=45.2,
    )
    res_dict = result.to_dict()
    assert "x" not in res_dict
    assert "y" not in res_dict
    assert "target_coord" not in res_dict
    assert res_dict["status"] == "completed"
    assert len(res_dict["substeps"]) == 1
    assert "target_coord" not in res_dict["substeps"][0]


def test_planner_decomposes_exploration_goal():
    """Tests that AutonomousSubstepPlanner decomposes high-level exploration goals."""
    planner = AutonomousSubstepPlanner()
    dummy_obs = SimpleNamespace(
        description="Desktop is active.",
        has_active_modal=False,
        primary_category="desktop_canvas",
    )

    steps = planner.decompose_goal("explore text editor", dummy_obs)
    actions = [s.action_type for s in steps]
    assert actions == ["open_app", "focus_input", "type_text", "close_window", "dismiss_modal"]
    assert steps[0].target_query == "text editor icon"
    assert steps[1].target_query == "text area"
    assert steps[2].text_content == "Hello AVCS test"
    assert steps[3].target_query == "close button"
    assert steps[4].target_query == "Don't Save button"
    assert steps[4].is_contingent is True


def test_planner_decomposes_hindi_english_conversational_goal():
    """Tests decomposition of the realistic bilingual prompt from GOAL.md Example A."""
    planner = AutonomousSubstepPlanner()
    dummy_obs = SimpleNamespace(
        description="Desktop is active.",
        has_active_modal=False,
        primary_category="desktop_canvas",
    )

    prompt = "desktop par jo text editor dikh raha hai usko open karo, usme typing test karo, aur save kiye bina close karke batao"
    steps = planner.decompose_goal(prompt, dummy_obs)
    actions = [s.action_type for s in steps]
    assert actions == ["open_app", "focus_input", "type_text", "close_window", "dismiss_modal"]
    assert steps[0].target_query == "text editor icon"
    assert steps[4].target_query == "Don't Save button"


def test_planner_decomposes_media_player_goal():
    """Tests decomposition of media player playback goal."""
    planner = AutonomousSubstepPlanner()
    dummy_obs = SimpleNamespace(
        description="Desktop is active.",
        has_active_modal=False,
        primary_category="desktop_canvas",
    )

    steps = planner.decompose_goal(
        "open media player, check if playlist is loaded, and start the first track",
        dummy_obs,
    )
    actions = [s.action_type for s in steps]
    assert actions == ["open_app", "click", "click"]
    assert steps[0].target_query == "media player icon"
    assert steps[1].target_query == "first track"
    assert steps[2].target_query == "Play button"


def test_execute_goal_empty_raises_value_error(simulated_controller):
    ctrl, _ = simulated_controller
    with pytest.raises(ValueError, match="cannot be empty"):
        ctrl.execute_goal("")
    with pytest.raises(ValueError, match="cannot be empty"):
        ctrl.execute_goal("   ")


def test_autonomous_text_editor_exploration_flow(simulated_controller):
    """End-to-end verification of GOAL.md Example A: Text Editor Exploration Flow.

    Autonomous substeps executed:
    1. Look -> Screen is desktop
    2. Ground -> 'text editor icon' at (0.2, 0.3)
    3. Act -> Double click 'text editor icon'
    4. Verify -> Window launch verified; screen transitions to text_editor
    5. Look -> Editor open
    6. Ground -> 'text area' at (0.5, 0.5)
    7. Act -> Click into text area
    8. Verify -> Focus & blinking caret verified
    9. Act -> Type test text
    10. Verify -> Characters rendered in text area
    11. Ground -> 'close button' at (0.75, 0.22)
    12. Act -> Click close button
    13. Verify -> Save prompt dialog emerges
    14. Ground -> 'Don't Save button' at (0.58, 0.55)
    15. Act -> Click 'Don't Save button'
    16. Verify -> Dialog dismissed; desktop restored clean
    17. Final summary returned to System 2.
    """
    ctrl, sim_screen = simulated_controller
    assert sim_screen.state == SimulatedAppScreen.STATE_DESKTOP

    traces_emitted: List[SubstepTrace] = []

    res: GoalExecutionResult = ctrl.execute_goal(
        "explore text editor",
        max_substeps=8,
        substep_callback=traces_emitted.append,
    )

    # 1. Assert overall execution success
    assert res.success is True
    assert res.status == "completed"
    assert res.total_latency_ms > 0.0

    # 2. Assert all expected substeps were autonomously dispatched and verified
    assert len(res.substeps) == 5
    assert len(traces_emitted) == 5

    step1 = res.substeps[0]
    assert step1.action == "open_app"
    assert "text editor" in step1.target_query
    assert step1.success is True
    assert step1.verified is True

    step2 = res.substeps[1]
    assert step2.action == "focus_input"
    assert step2.target_query == "text area"
    assert step2.success is True
    assert step2.verified is True

    step3 = res.substeps[2]
    assert step3.action == "type_text"
    assert step3.success is True
    assert step3.verified is True

    step4 = res.substeps[3]
    assert step4.action == "close_window"
    assert step4.target_query == "close button"
    assert step4.success is True
    assert step4.verified is True

    step5 = res.substeps[4]
    assert step5.action == "dismiss_modal"
    assert step5.target_query == "Don't Save button"
    assert step5.success is True
    assert step5.verified is True

    # 3. Assert simulated environment state transitions
    assert sim_screen.state == SimulatedAppScreen.STATE_DESKTOP
    assert sim_screen.typed_text == "Hello AVCS test"
    assert sim_screen.state_history == [
        "desktop",
        "text_editor",
        "save_dialog",
        "desktop",
    ]

    # 4. Assert natural language summary generated for System 2
    assert "Text Editor launched" in res.summary
    assert "text area focused" in res.summary
    assert "typing test executed" in res.summary
    assert "close button clicked" in res.summary
    assert "confirmation prompt handled" in res.summary
    assert "Task complete." in res.summary

    # 5. Assert ZERO coordinates leaked to System 2
    d = res.to_dict()
    assert "x" not in d
    assert "y" not in d
    assert "target_coord" not in d


def test_autonomous_blocker_detection_halts_immediately(simulated_controller):
    """End-to-end verification of GOAL.md Example B: Unexpected Blocker Handling.

    When an unexpected modal popup appears (e.g. 'Session expired. Please re-authenticate.'),
    AVCS System 1 must immediately detect the obstacle, halt autonomous execution
    without clicking blind targets, set status='blocked', and return an honest
    natural language report to System 2.
    """
    ctrl, sim_screen = simulated_controller

    # Simulate an unexpected modal dialog blocker appearing before/during task
    sim_screen.trigger_blocker("Session expired. Please re-authenticate.")

    res: GoalExecutionResult = ctrl.execute_goal(
        "company portal pe jaakar latest monthly report download karo"
    )

    # 1. Assert halted with blocked status
    assert res.success is False
    assert res.status == "blocked"
    assert res.blocker_reason is not None
    assert "Session Expired" in res.blocker_reason or "session expired" in res.blocker_reason.lower()

    # 2. Assert NO blind substeps executed
    assert len(res.substeps) == 0

    # 3. Assert natural language report to System 2
    assert "Goal execution blocked" in res.summary
    assert "unexpected modal dialog" in res.summary.lower()
    assert "pending System 2 guidance" in res.summary


def test_autonomous_loop_fails_closed_on_unlocatable_target(simulated_controller):
    """Verify that if a target cannot be localized (e.g. absent or unvalidated), System 1 fails closed."""
    ctrl, sim_screen = simulated_controller

    # On clean desktop, "Unknown Hidden Button" does not exist
    res = ctrl.execute_goal("click Unknown Hidden Button")

    assert res.success is False
    assert res.status == "failed"
    assert len(res.substeps) == 1
    assert res.substeps[0].success is False
    assert res.substeps[0].verified is False
    assert "could not be localized" in res.substeps[0].message
    assert "could not be completed" in res.summary


def test_autonomous_loop_respects_max_substeps(simulated_controller):
    """Verify safety bound: loop terminates if max_substeps is reached."""
    ctrl, sim_screen = simulated_controller

    # Set max_substeps to 2 on an exploration task that normally requires 5 steps
    res = ctrl.execute_goal("explore text editor", max_substeps=2)

    assert len(res.substeps) == 2
    assert res.status == "timeout"
    assert res.success is False
    assert "timeout" in res.summary.lower() or "limit" in res.summary.lower()
