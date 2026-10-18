"""Tier 4: Real-World Scenarios E2E Test Suite.

End-to-end multi-step agent workflows:
- Scenario 1: Desktop Workspace Exploration & Scene Comprehension
- Scenario 2: Safe Form-Fill Workflow with Focus Protection
- Scenario 3: Unvalidated Grounding Safety Lockout
- Scenario 4: Negative Background Rejection in Desktop Context
- Scenario 5: Observe-Ground-Validate-Act-Verify-Describe-Replan Execution Loop
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pytest

from avcs.grounding import FovealInspection
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.sdk import HumanEyeClient, UnfocusedInputElementError
from avcs.verifier import VerificationResult, VerificationStatus
from forge.vision.cortex.motor.driver import VirtualCursorDriver
from forge.vision.cortex.sdk import UnvalidatedGroundingError


@pytest.mark.tier4
class TestTier4RealWorldScenarios:
    """E2E Real-world multi-step desktop agent workflows."""

    def test_scenario1_desktop_workspace_exploration(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Scenario 1: Agent inspects desktop workspace, builds scene tree, and captures visual diff."""
        frame1, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=401)
        state1 = avcs_client.perceive(frame=frame1)
        assert state1 is not None
        assert state1.fps > 0

        # Extract UIML scene tree
        tree = avcs_client.get_scene_tree()
        assert tree is not None
        assert hasattr(tree, "elements")
        assert len(tree.elements) > 0

        # Retrieve visual diff
        diff = avcs_client.get_visual_diff(prior_action="observe")
        assert diff is not None
        assert diff.prior_action == "observe"

        # Verify episodic memory records snapshot
        snapshot = avcs_client.get_memory_snapshot()
        assert snapshot is not None

    def test_scenario2_safe_form_fill_workflow(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Scenario 2: Agent navigates to text field, verifies active focus, and types safely."""
        # 1. Target input field
        target_point = (0.25, 0.25)
        # 2. Check focus with simulated blinking caret in the input box
        frames, ts = screen_gen.caret_frames(freq_hz=1.0, num_frames=90, fps=30.0)
        focus_state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert focus_state.is_focused
        assert focus_state.confidence >= 0.70

        # 3. Type text into input field
        driver = avcs_client.motor_driver
        assert isinstance(driver, VirtualCursorDriver)
        driver.typed_history.clear()

        avcs_client.type_text("John Doe", enforce_focus=False, target=target_point)
        assert "John Doe" in driver.typed_history

        # 4. Attempt typing into unfocused area with enforce_focus=True strictly aborts
        with pytest.raises(UnfocusedInputElementError):
            avcs_client.type_text(
                "Dangerous text",
                enforce_focus=True,
                clear_first=True,
                target=(0.02, 0.02),  # Unfocused corner
            )

    def test_scenario3_unvalidated_grounding_safety_lockout(
        self,
        avcs_client: HumanEyeClient,
    ):
        """Scenario 3: Natural language motor requests without validated weights are strictly locked out."""
        driver = avcs_client.motor_driver
        driver.click_history.clear()

        # Caller provides intent query without explicit coordinates
        with pytest.raises(UnvalidatedGroundingError) as exc_info:
            avcs_client.click("Submit Order")

        assert "validated trained semantic grounding is unavailable" in str(exc_info.value)
        # Ensure driver received zero clicks
        assert len(driver.click_history) == 0

        # Same lockout on click_and_verify
        with pytest.raises(UnvalidatedGroundingError):
            avcs_client.click_and_verify(target="Cancel Order")

    def test_scenario4_negative_background_rejection_in_live_context(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Scenario 4: Agent observes blank/empty desktop background; zero false affordance clicks emitted."""
        blank_frame = screen_gen.blank_frame(width=1280, height=720, color=(20, 20, 20))
        state = avcs_client.perceive(frame=blank_frame)

        # Zero false-positive hotspots
        assert len(state.affordance_hotspots) == 0

        # Moving objects must be empty
        assert len(state.moving_objects) == 0

        # Motion magnitude should be zero
        assert state.motion_magnitude == 0.0

    def test_scenario5_observe_ground_validate_act_verify_replan_loop(
        self,
        avcs_client: HumanEyeClient,
        screen_gen,
    ):
        """Scenario 5: Complete closed-loop execution cycle:

        Observe -> Ground -> Validate -> Act -> Verify -> Describe -> Replan.
        """
        # Step 1: OBSERVE
        frame, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=405)
        obs_state = avcs_client.perceive(frame=frame)
        assert obs_state is not None
        assert len(obs_state.affordance_hotspots) > 0

        # Step 2: GROUND
        query = "Play button"
        gaze_coords = avcs_client.look_at(query)
        assert 0.0 <= gaze_coords[0] <= 1.0
        assert 0.0 <= gaze_coords[1] <= 1.0

        # Step 3: VALIDATE
        # Natural language action must be validated before motor dispatch
        # Unvalidated string target triggers fail-closed error
        with pytest.raises(UnvalidatedGroundingError):
            avcs_client._require_coordinate_target(query)

        # Explicit coordinate target passes validation gate
        valid_coords = avcs_client._require_coordinate_target((0.3, 0.3))
        assert valid_coords == (0.3, 0.3)

        # Step 4: ACT
        move_result = avcs_client.move_to(valid_coords)
        assert move_result is not None
        assert move_result.success

        # Step 5: VERIFY
        verify_result = avcs_client.click_and_verify(target=valid_coords)
        assert isinstance(verify_result, VerificationResult)
        assert verify_result.status in list(VerificationStatus)

        # Step 6: DESCRIBE
        context_prompt = avcs_client.get_context_prompt()
        assert isinstance(context_prompt, str)

        # Step 7: REPLAN / AUDIT
        # Visual diff provides causal summary for replanning
        diff = avcs_client.get_visual_diff()
        assert diff is not None
        assert hasattr(diff, "causal_summary")
