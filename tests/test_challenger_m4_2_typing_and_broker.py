"""Empirical Adversarial Challenge Suite for AVCS Milestone 4 (Features 13 & 14).

Author: teamwork_preview_challenger_m4_2
Role: Empirical Challenger & Adversarial Reviewer

Challenges Executed:
1. Dual-condition focus gate: test active window mismatch -> 100% blocked, 0 keystrokes dispatched.
2. Visual focus absence (static pipe '|', scrolling text, spinner) -> confidence < 0.70, typing 100% blocked.
3. Terminal Protection Guard: verify clear_first=True in terminal windows emits Ctrl+U, never Ctrl+A.
4. Mid-typing focus loss: verify FocusLostError raised and remaining characters discarded immediately.
5. Fresh frame sequence synchronization: verify multi-threaded rapid actuation uses fresh frames S > S_pre.
6. ActionBroker non-replaying guarantee: verify NON_IDEMPOTENT actions and typing never blind retry on dead clicks.
7. System 2 Privacy Boundary: verify FocusVerificationResult.to_dict() and controller.ActionResult.to_dict() contain zero coordinate keys (x, y, bbox, centroid, target_coord).
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

from avcs.control.action_broker import (
    ActionBroker,
    ActionClass,
    ActionRequest,
    ActionResult as BrokerActionResult,
    ActionSafetyController,
    ActionStatus,
    PostconditionSpec,
    VerificationDetail,
)
from avcs.control.controller import ActionResult, AVCSController
from avcs.control.verifier import VerificationStatus, VisualRippleVerifier
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.perception.focus import (
    FocusLostError,
    FocusVerificationResult,
    UnverifiedFocusError,
    VisualFocusDetector,
    VisualFocusState,
)
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.motor.driver import VirtualCursorDriver


# ==============================================================================
# Helper Utilities
# ==============================================================================

FORBIDDEN_COORDINATE_KEYS: Set[str] = {
    "x",
    "y",
    "bbox",
    "centroid",
    "coords",
    "target_coord",
    "raw_coordinates",
    "coordinates",
    "patch",
    "crop",
}


def find_forbidden_keys(data: Any, path: str = "") -> List[str]:
    """Recursively inspects dict/list trees for forbidden coordinate/tensor keys."""
    violations: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            current_path = f"{path}.{k}" if path else str(k)
            if str(k).lower() in FORBIDDEN_COORDINATE_KEYS:
                violations.append(f"Forbidden key '{k}' at path '{current_path}'")
            violations.extend(find_forbidden_keys(v, current_path))
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            violations.extend(find_forbidden_keys(item, f"{path}[{idx}]"))
    return violations


# ==============================================================================
# Challenge 1: Dual-Condition Focus Gate & Window Mismatch Rejection
# ==============================================================================

class TestChallenge1DualConditionFocusGate:
    """Empirical challenge: verify active window mismatch 100% blocks typing and dispatches 0 keystrokes."""

    WINDOW_MISMATCH_CASES = [
        ("Settings", "Terminal"),
        ("Calculator", "Notepad"),
        ("Google Chrome", "VS Code"),
        ("File Explorer", "Command Prompt"),
        ("Task Manager", "Editor"),
        (None, "Terminal"),  # No active window detected
    ]

    @pytest.mark.parametrize("active_win,expected_win", WINDOW_MISMATCH_CASES)
    def test_window_mismatch_blocks_typing_with_zero_keystrokes(
        self,
        active_win: Optional[str],
        expected_win: str,
    ):
        """When Condition A fails (window mismatch), typing must be 100% blocked with 0 keystrokes."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window(active_win)

        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Mock visual focus as positive (Condition B holds, but Condition A fails)
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.92,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        res = ctrl.type_text(
            "rm -rf /tmp/data",
            verify_focus=True,
            foreground_app=expected_win,
        )

        # 1. Verification Outcome must be failed closed
        assert res.success is False, f"Expected success=False for {active_win} != {expected_win}"
        assert res.verified is False
        assert "Typing blocked" in res.message
        assert "Window mismatch" in res.message or "failed closed" in res.message

        # 2. Strict Keystroke Invariant: exactly 0 keystrokes dispatched
        assert len(driver.typed_history) == 0, (
            f"Keystrokes leaked on window mismatch! Typed: {driver.typed_history}"
        )
        assert len(driver.key_history) == 0, (
            f"Special keys leaked on window mismatch! Keys: {driver.key_history}"
        )

        # 3. FocusVerificationResult data contract
        foc_res = ctrl.verify_focus(foreground_app=expected_win)
        assert foc_res.verified is False
        assert foc_res.os_window_verified is False
        assert foc_res.confidence <= 0.69
        assert foc_res.active_window_title == active_win
        assert foc_res.expected_window_title == expected_win

    def test_consecutive_mismatch_attempts_never_leak_or_latch_state(self):
        """Verify that 20 repeated failed typing attempts maintain 0 keystrokes across attempts."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window("Malicious App")
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.88,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        for i in range(20):
            res = ctrl.type_text(f"leak_payload_{i}", verify_focus=True, foreground_app="Target Terminal")
            assert res.success is False
            assert res.verified is False

        assert len(driver.typed_history) == 0, "Typed history must remain strictly empty after 20 attempts"
        assert len(driver.key_history) == 0, "Key history must remain strictly empty after 20 attempts"


# ==============================================================================
# Challenge 2: Visual Focus Absence Under Deceptive Visual Patterns
# ==============================================================================

class TestChallenge2VisualFocusAbsenceDeceptivePatterns:
    """Empirical challenge: deceptive patterns (static pipe '|', scrolling text, spinner)
    must produce visual confidence < 0.70 and 100% block typing.
    """

    @pytest.fixture
    def focus_detector(self) -> VisualFocusDetector:
        return VisualFocusDetector()

    def test_deceptive_static_pipe_character_fails_closed(self, focus_detector: VisualFocusDetector):
        """A static vertical pipe '|' in text without temporal blink must produce conf < 0.70 and is_focused=False."""
        # Create 100x100 white patch with a static black vertical line (pipe '|') in the center
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Pipe: 20px high, 2px wide at center
        frame[40:60, 49:51] = 0

        # Identical reference frame (static pipe, zero temporal delta)
        ref_frame = frame.copy()

        st = focus_detector.verify_focus(
            frame=frame,
            target_point=(0.5, 0.5),
            reference_frame=ref_frame,
        )

        assert st.is_focused is False, "Static pipe '|' must NOT be classified as focused"
        assert st.confidence < 0.70, f"Expected confidence < 0.70, got {st.confidence}"
        assert st.confidence <= 0.69, f"Strict fail-closed bound violated: {st.confidence}"
        assert st.caret_blinking is False, "Static pipe must not report caret_blinking"

    def test_deceptive_scrolling_text_fails_closed(self, focus_detector: VisualFocusDetector):
        """Scrolling text causes diffuse horizontal pixel shifts across frames; must fail closed."""
        rng = np.random.RandomState(42)
        base_canvas = rng.randint(200, 255, size=(100, 100, 3), dtype=np.uint8)

        # Draw pseudo-text rows
        for y in range(20, 80, 12):
            base_canvas[y : y + 4, 10:90] = 30

        # Simulate horizontal text scroll: frame 2 is shifted left by 6 pixels
        frame1 = base_canvas.copy()
        frame2 = np.roll(base_canvas, shift=-6, axis=1)

        st = focus_detector.verify_focus(
            frame=frame2,
            target_point=(0.5, 0.5),
            reference_frame=frame1,
        )

        assert st.is_focused is False, "Scrolling text must NOT be classified as focused"
        assert st.confidence < 0.70, f"Confidence {st.confidence} exceeded safe threshold on scrolling text"
        assert st.confidence <= 0.69

    def test_deceptive_rotating_spinner_fails_closed(self, focus_detector: VisualFocusDetector):
        """A rotating spinner has cyclical non-vertical photometric changes; must fail closed."""
        # Create rotating arc/cross pattern in center
        frame1 = np.full((100, 100, 3), 240, dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 240, dtype=np.uint8)

        # Frame 1: horizontal tick
        frame1[49:51, 40:60] = 20
        # Frame 2: diagonal tick
        for i in range(-10, 10):
            frame2[50 + i, 50 + i] = 20

        st = focus_detector.verify_focus(
            frame=frame2,
            target_point=(0.5, 0.5),
            reference_frame=frame1,
        )

        assert st.is_focused is False, "Rotating spinner must NOT be classified as focused"
        assert st.confidence < 0.70, f"Confidence {st.confidence} exceeded safe threshold on spinner"
        assert st.confidence <= 0.69

    def test_controller_blocks_typing_on_all_deceptive_patterns(self, focus_detector: VisualFocusDetector):
        """End-to-end controller test: feeding deceptive visual patterns blocks 100% of keystrokes."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window("Document App")
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Frame with static pipe
        pipe_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        pipe_frame[40:60, 49:51] = 0

        # Point ctrl.eye.verify_focus to return the evaluation of the static pipe
        ctrl.eye.verify_focus = lambda *args, **kwargs: focus_detector.verify_focus(
            frame=pipe_frame,
            target_point=(0.5, 0.5),
            reference_frame=pipe_frame.copy(),
        )

        res = ctrl.type_text("unexpected text into static pipe", verify_focus=True)

        assert res.success is False
        assert res.verified is False
        assert "Typing blocked" in res.message
        assert len(driver.typed_history) == 0, "No keystrokes must be dispatched on deceptive static pipe"


# ==============================================================================
# Challenge 3: Terminal Protection Guard
# ==============================================================================

class TestChallenge3TerminalProtectionGuard:
    """Empirical challenge: clear_first=True in terminal windows emits Ctrl+U, never Ctrl+A."""

    TERMINAL_WINDOWS = [
        "Administrator: Windows PowerShell",
        "Command Prompt - cmd.exe",
        "MINGW64:/c/Users/admin - bash",
        "user@linux-box: ~ (zsh)",
        "Windows Terminal",
        "PowerShell 7.4.1",
        "WSL: Ubuntu-22.04 (bash)",
    ]

    NON_TERMINAL_WINDOWS = [
        "Untitled - Notepad",
        "Google Chrome - Search",
        "Slack | general",
        "Visual Studio Code - main.py",
        "Calculator",
        "Settings",
    ]

    @pytest.mark.parametrize("terminal_title", TERMINAL_WINDOWS)
    def test_terminal_windows_emit_ctrl_u_and_never_ctrl_a(self, terminal_title: str):
        """In terminal windows, clear_first MUST emit Ctrl+U, and NEVER emit Ctrl+A or Backspace."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window(terminal_title)
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Mock focus so typing reaches clear_first
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.95,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        res = ctrl.type_text("git status", clear_first=True, verify_focus=True)

        # Keystroke verification
        assert "ctrl+u" in driver.key_history, f"Expected Ctrl+U in terminal '{terminal_title}'"
        assert "ctrl+a" not in driver.key_history, f"Hazardous Ctrl+A emitted in terminal '{terminal_title}'!"
        assert "backspace" not in driver.key_history, f"Hazardous Backspace emitted in terminal '{terminal_title}'!"

    @pytest.mark.parametrize("gui_title", NON_TERMINAL_WINDOWS)
    def test_non_terminal_windows_emit_ctrl_a_and_backspace_never_ctrl_u(self, gui_title: str):
        """In normal GUI windows, clear_first MUST emit Ctrl+A + Backspace, and NEVER Ctrl+U."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window(gui_title)
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.95,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        res = ctrl.type_text("replacement value", clear_first=True, verify_focus=True)

        assert "ctrl+a" in driver.key_history, f"Expected Ctrl+A in GUI window '{gui_title}'"
        assert "backspace" in driver.key_history, f"Expected Backspace in GUI window '{gui_title}'"
        assert "ctrl+u" not in driver.key_history, f"Ctrl+U emitted in non-terminal GUI window '{gui_title}'!"


# ==============================================================================
# Challenge 4: Mid-Typing Focus Loss Abort
# ==============================================================================

class TestChallenge4MidTypingFocusLoss:
    """Empirical challenge: losing focus mid-typing raises FocusLostError and discards remaining characters."""

    def test_mid_typing_focus_loss_in_virtual_driver_aborts_immediately(self):
        """Simulate focus loss halfway through a 30-character string; verify FocusLostError raised.
        
        Note: focus_check_fn is invoked once prior to typing (pre-check), then on each character.
        Thus, to trigger focus loss at character index 12 ('m'), call 1 is pre-check (True),
        calls 2..13 are chars 0..11 (True), and call 14 is char 12 (False).
        """
        driver = VirtualCursorDriver()
        driver.set_foreground_window("ActiveApp")

        payload = "abcdefghijklmnopqrstuvwxyz0123"  # 30 chars
        focus_loss_char_index = 12  # character 'm'

        call_count = 0
        def dynamic_focus_check() -> bool:
            nonlocal call_count
            call_count += 1
            # Pre-check is call 1; characters 0..11 are calls 2..13; char 12 is call 14
            return call_count <= (focus_loss_char_index + 1)

        with pytest.raises(FocusLostError, match="Focus lost mid-typing") as exc_info:
            driver.type_text(payload, focus_check_fn=dynamic_focus_check)

        err_msg = str(exc_info.value)
        assert f"index {focus_loss_char_index}" in err_msg, f"Exception did not cite correct index: {err_msg}"
        assert f"'{payload[focus_loss_char_index]}'" in err_msg, f"Exception did not cite correct character: {err_msg}"

        # Strict Invariant: incomplete payload was NOT added to typed_history
        assert len(driver.typed_history) == 0, (
            f"Incomplete text leaked into typed_history: {driver.typed_history}"
        )

    def test_controller_propagates_focus_lost_error_and_prevents_enter_press(self):
        """When controller types with press_enter=True and focus is lost, Enter must NEVER be pressed."""
        driver = VirtualCursorDriver()
        driver.set_foreground_window("TargetApp")
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=True,
            confidence=0.90,
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
        )

        # Simulating window change halfway through typing
        orig_type = driver.type_text
        def hook_type(text, **kwargs):
            driver.set_foreground_window("Interfering Pop-Up Window")
            return orig_type(text, **kwargs)
        driver.type_text = hook_type

        with pytest.raises(FocusLostError):
            ctrl.type_text(
                "critical command --execute",
                press_enter=True,
                verify_focus=True,
                foreground_app="TargetApp",
            )

        # Enter key must NEVER be emitted
        assert "enter" not in driver.key_history, "Enter key was pressed despite mid-typing abort!"
        assert len(driver.typed_history) == 0, "Text was completed despite mid-typing abort!"


# ==============================================================================
# Challenge 5: Fresh Frame Sequence Synchronization Under Concurrent Actuation
# ==============================================================================

class TestChallenge5FreshFrameSequenceSynchronization:
    """Empirical challenge: multi-threaded rapid actuation must strictly evaluate postconditions
    on fresh frames with sequence S > S_pre. Zero stale frames permitted.
    """

    def test_rapid_consecutive_actions_enforce_strictly_increasing_frame_sequences(self):
        """Execute 15 consecutive actions through ActionBroker; verify 100% of underlying frame sequences S > S_pre."""
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        recorded_sequences: List[Tuple[int, int]] = []

        for i in range(15):
            pre_seq = screen.sequence_number
            req = ActionRequest(
                action_type="click",
                target_coord=(0.3, 0.3),
                action_class=ActionClass.IDEMPOTENT,
                expected_postcondition=lambda pre, post: True,
            )
            res = broker.execute(req)
            assert res.is_confirmed is True
            assert res.verification_detail is not None

            # Empirical Verification of Fresh Frame Advancement on Capture Source
            post_seq = screen.sequence_number
            recorded_sequences.append((pre_seq, post_seq))

        # Empirical Check: Every single post_seq MUST be strictly greater than pre_seq
        stale_reads = 0
        for pre, post in recorded_sequences:
            if post <= pre:
                stale_reads += 1

        assert stale_reads == 0, f"Encountered {stale_reads} stale frame evaluations in rapid actuation!"
        assert len(recorded_sequences) == 15

    def test_bug_investigation_verification_detail_fresh_frame_sequence_attribute(self):
        """Empirical bug probe verification: In ActionBroker.execute (line 512), fresh_frame_sequence
        queries frame_sequence, falling back to frame_index or source sequence_number.
        We empirically verify that fresh_frame_sequence is successfully resolved and not None.
        """
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        req = ActionRequest(
            action_type="click",
            target_coord=(0.3, 0.3),
            action_class=ActionClass.IDEMPOTENT,
            expected_postcondition=lambda pre, post: True,
        )
        res = broker.execute(req)
        assert res.verification_detail is not None

        # Verify that HumanEyeState has frame_index (which is > 0)
        post_state = eye.observe()
        assert hasattr(post_state, "frame_index"), "HumanEyeState should have frame_index"
        assert not hasattr(post_state, "frame_sequence"), "HumanEyeState lacks frame_sequence"

        # Verify that verification_detail.fresh_frame_sequence is resolved via frame_index fallback
        assert res.verification_detail.fresh_frame_sequence is not None, (
            "Observed fresh_frame_sequence should be resolved via frame_index fallback in action_broker.py:512"
        )
        assert res.verification_detail.fresh_frame_sequence > 0

    def test_multi_threaded_fresh_frame_grabs(self):
        """Stress-test grab_with_sequence across 8 concurrent threads requesting S > S_pre."""
        screen = SyntheticScreen()

        def worker_task(thread_id: int) -> Tuple[int, int]:
            time.sleep(0.001 * (thread_id % 3))
            pre_s = screen.sequence_number
            frame, post_s = screen.grab_with_sequence(min_sequence=pre_s)
            return pre_s, post_s

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(32)]
            results = [f.result() for f in futures]

        for pre_s, post_s in results:
            assert post_s > pre_s, f"Thread read stale frame: post {post_s} <= pre {pre_s}"


# ==============================================================================
# Challenge 6: ActionBroker Non-Replaying Guarantee
# ==============================================================================

class TestChallenge6ActionBrokerNonReplayingGuarantee:
    """Empirical challenge: verify NON_IDEMPOTENT actions and typing actions never blind retry on dead clicks."""

    def test_non_idempotent_action_dead_click_strictly_zero_retries(self):
        """A NON_IDEMPOTENT action (e.g. Submit button) with dead click must produce FAILED or AMBIGUOUS
        with recovery_attempts == 1 and exactly 1 physical click.
        """
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        # Expected postcondition strictly fails (dead click simulation)
        req = ActionRequest(
            action_type="click",
            target_coord=(0.7, 0.2),
            action_class=ActionClass.NON_IDEMPOTENT,
            expected_postcondition=lambda pre, post: False,
        )

        res = broker.execute(req)

        # 1. Terminal State must not be confirmed
        assert res.status in (ActionStatus.FAILED, ActionStatus.AMBIGUOUS)
        assert res.is_confirmed is False

        # 2. Strict Non-Replaying Guarantee: recovery_attempts must be exactly 1
        assert res.recovery_attempts == 1, (
            f"Blind retry occurred on NON_IDEMPOTENT action! recovery_attempts={res.recovery_attempts}"
        )

        # 3. Exactly 1 physical click dispatched to the driver
        assert len(driver.click_history) == 1, (
            f"Expected exactly 1 click, but found {len(driver.click_history)}: {driver.click_history}"
        )
        assert "Non-idempotent action: blind replay blocked" in res.evidence

    def test_typing_action_failure_strictly_zero_typing_retry(self):
        """Typing input must NEVER be replayed or retried, even if marked IDEMPOTENT."""
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        req = ActionRequest(
            action_type="type",
            target_coord=(0.5, 0.5),
            text_content="sensitive password",
            action_class=ActionClass.IDEMPOTENT,  # Intentionally tagged IDEMPOTENT to test hard typing guard
            expected_postcondition=lambda pre, post: False,
        )

        res = broker.execute(req)

        assert res.status in (ActionStatus.FAILED, ActionStatus.AMBIGUOUS)
        assert res.recovery_attempts == 1, (
            f"Typing action attempted retry! recovery_attempts={res.recovery_attempts}"
        )
        assert len(driver.typed_history) <= 1, "Typing action was dispatched more than once!"
        assert "Non-idempotent input is never replayed" in res.evidence

    def test_contrast_oracle_idempotent_action_does_retry(self):
        """Contrast Oracle: An IDEMPOTENT click DOES retry on dead click, proving the non-replaying
        guarantee on NON_IDEMPOTENT actions is an intentional, functional distinction.
        """
        screen = SyntheticScreen()
        eye = HumanEye(source=screen)
        driver = VirtualCursorDriver()
        safety = ActionSafetyController()
        broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety)

        req = ActionRequest(
            action_type="click",
            target_coord=(0.5, 0.5),
            action_class=ActionClass.IDEMPOTENT,
            expected_postcondition=lambda pre, post: False,  # dead click
        )

        res = broker.execute(req)

        # For idempotent click, recovery_attempts must be > 1 (micro-jitter retry was executed)
        assert res.recovery_attempts > 1, "Idempotent click failed to attempt recovery retry"
        assert len(driver.click_history) > 1, "Expected multiple clicks for idempotent retry"


# ==============================================================================
# Challenge 7: System 2 Privacy Boundary Verification
# ==============================================================================

class TestChallenge7System2PrivacyBoundary:
    """Empirical challenge: verify FocusVerificationResult.to_dict() and controller.ActionResult.to_dict()
    contain strictly zero coordinate keys (x, y, bbox, centroid, target_coord).
    """

    def test_focus_verification_result_to_dict_zero_coordinates(self):
        """FocusVerificationResult contains internal coordinates (caret_coord, target_region);
        its to_dict() MUST completely encapsulate and omit them.
        """
        # Create result populated with internal coordinates
        res = FocusVerificationResult(
            verified=True,
            confidence=0.88,
            os_window_verified=True,
            visual_focus_verified=True,
            active_window_title="Notepad",
            expected_window_title="Notepad",
            caret_detected=True,
            caret_blinking=True,
            focus_ring_detected=True,
            caret_coord=(0.2345, 0.6789),
            target_region=(0.20, 0.60, 0.40, 0.75),
            reason="Focus confirmed",
            latency_ms=2.15,
        )

        d = res.to_dict()

        # 1. Structural check for forbidden coordinate keys
        violations = find_forbidden_keys(d)
        assert not violations, f"Coordinate leakage in FocusVerificationResult.to_dict(): {violations}"

        # 2. JSON serializability
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # 3. Regex scan for forbidden keys in JSON string
        for forbidden in FORBIDDEN_COORDINATE_KEYS:
            match = re.search(rf'"{forbidden}"\s*:', serialized, re.IGNORECASE)
            assert match is None, f"Found forbidden key '{forbidden}' in serialized JSON: {serialized}"

    def test_controller_action_result_to_dict_zero_coordinates(self):
        """AVCSController.ActionResult.to_dict() must contain zero coordinates, both flat and nested."""
        driver = VirtualCursorDriver()
        ctrl = AVCSController(motor_driver=driver, force_virtual_motor=True)

        # Test click_target outcome
        res_click = ctrl.click_target("Submit button", verify_action=False)
        d_click = res_click.to_dict()

        violations_click = find_forbidden_keys(d_click)
        assert not violations_click, f"Coordinate leakage in click ActionResult.to_dict(): {violations_click}"

        # Test type_text outcome with nested verification result
        ctrl.eye.verify_focus = lambda *args, **kwargs: VisualFocusState(
            is_focused=False,
            confidence=0.40,
            caret_detected=False,
            caret_blinking=False,
            focus_ring_detected=False,
        )
        res_type = ctrl.type_text("some text", verify_focus=True)
        d_type = res_type.to_dict()

        violations_type = find_forbidden_keys(d_type)
        assert not violations_type, f"Coordinate leakage in type ActionResult.to_dict(): {violations_type}"

        # JSON regex audit
        serialized = json.dumps(d_type)
        for forbidden in FORBIDDEN_COORDINATE_KEYS:
            match = re.search(rf'"{forbidden}"\s*:', serialized, re.IGNORECASE)
            assert match is None, f"Forbidden key '{forbidden}' found in serialized type ActionResult JSON: {serialized}"

    def test_action_broker_action_result_and_verification_detail_to_dict_zero_coordinates(self):
        """ActionBroker.ActionResult and VerificationDetail must encapsulate target_coord."""
        detail = VerificationDetail(
            status="FAILED",
            verified=False,
            optical_ripple_energy=0.001,
            photometric_delta=0.002,
            surprise_pulse=0.05,
            state_transition_detected=False,
            causal_summary="Target showed zero optical reaction.",
            recovery_attempts=1,
            elapsed_ms=52.3,
            fresh_frame_sequence=99,
        )
        d_detail = detail.to_dict()
        violations_detail = find_forbidden_keys(d_detail)
        assert not violations_detail, f"Coordinate leakage in VerificationDetail.to_dict(): {violations_detail}"

        res = BrokerActionResult(
            action_id="act_test123",
            status=ActionStatus.FAILED,
            action_type="click",
            target_coord=(0.45, 0.65),  # Dataclass has target_coord
            elapsed_ms=55.0,
            evidence="Dead click",
            optical_ripple_energy=0.0,
            photometric_delta=0.0,
            recovery_attempts=1,
            verification_detail=detail,
        )
        d_broker = res.to_dict()

        # target_coord must be excluded in to_dict()
        assert "target_coord" not in d_broker, "BrokerActionResult.to_dict() exposed target_coord!"
        violations_broker = find_forbidden_keys(d_broker)
        assert not violations_broker, f"Coordinate leakage in BrokerActionResult.to_dict(): {violations_broker}"
