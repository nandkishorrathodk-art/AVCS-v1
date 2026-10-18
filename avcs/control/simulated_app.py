"""Simulated Interactive Application Flow for AVCS System 1 Testing and Verification.

Provides an interactive GUI environment with genuine pixel rendering and state transitions:
1. Desktop canvas with clickable application icons.
2. Text editor window with focusable text editing area and caret rendering.
3. Save confirmation modal prompt on window close.
4. Unexpected blocker modal dialogs (e.g. "Session expired. Please re-authenticate.").
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avcs.perception.focus import VisualFocusState
from avcs.perception.grounding import GroundingResult
from avcs.control.verifier import VerificationResult, VerificationStatus
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.sdk import UnvalidatedGroundingError


class SimulatedAppScreen(SyntheticScreen):
    """Simulates an interactive desktop application environment with pixel state transitions."""

    STATE_DESKTOP = "desktop"
    STATE_TEXT_EDITOR = "text_editor"
    STATE_SAVE_DIALOG = "save_dialog"
    STATE_BLOCKER = "blocker"

    def __init__(self, width: int = 320, height: int = 240, seed: int = 42) -> None:
        super().__init__(width=width, height=height, seed=seed)
        self.state = self.STATE_DESKTOP
        self.focused_field: Optional[str] = None
        self.typed_text: str = ""
        self.blocker_message: Optional[str] = None
        self.target_coords: Dict[str, Tuple[float, float]] = {
            "text editor icon": (0.20, 0.30),
            "media player icon": (0.20, 0.60),
            "text area": (0.50, 0.50),
            "close button": (0.75, 0.22),
            "Don't Save button": (0.58, 0.55),
            "Save button": (0.42, 0.55),
            "first track": (0.40, 0.40),
            "Play button": (0.50, 0.70),
        }
        self.state_history: List[str] = [self.state]

    def trigger_blocker(self, message: str = "Session expired. Please re-authenticate.") -> None:
        """Simulates an unexpected modal dialog blocker on screen."""
        self.state = self.STATE_BLOCKER
        self.blocker_message = message
        self.state_history.append(self.state)

    def reset_to_desktop(self) -> None:
        """Restores the clean desktop state."""
        self.state = self.STATE_DESKTOP
        self.focused_field = None
        self.typed_text = ""
        self.blocker_message = None
        self.state_history.append(self.state)

    def grab(self) -> np.ndarray:
        """Renders the pixel RGB frame for the current application state."""
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Desktop base background: pleasant dark blue-gray
        img[..., 0] = 30
        img[..., 1] = 40
        img[..., 2] = 55

        if self.state == self.STATE_DESKTOP:
            # Draw desktop icons
            # Text editor icon at (0.20, 0.30)
            ix, iy = int(0.20 * w), int(0.30 * h)
            img[iy - 8 : iy + 8, ix - 8 : ix + 8] = [200, 200, 220]

            # Media player icon at (0.20, 0.60)
            mx, my = int(0.20 * w), int(0.60 * h)
            img[my - 8 : my + 8, mx - 8 : mx + 8] = [220, 160, 80]

        elif self.state in (self.STATE_TEXT_EDITOR, self.STATE_SAVE_DIALOG):
            # Draw open text editor window
            win_top, win_bottom = int(0.18 * h), int(0.82 * h)
            win_left, win_right = int(0.25 * w), int(0.78 * w)
            img[win_top:win_bottom, win_left:win_right] = [230, 230, 235]

            # Window title bar (dark blue)
            title_bottom = int(0.26 * h)
            img[win_top:title_bottom, win_left:win_right] = [50, 75, 120]

            # Close button (red square) at (0.75, 0.22)
            cx, cy = int(0.75 * w), int(0.22 * h)
            img[cy - 4 : cy + 4, cx - 4 : cx + 4] = [220, 40, 40]

            # Central text editing area (bright white)
            ta_top, ta_bottom = int(0.32 * h), int(0.76 * h)
            ta_left, ta_right = int(0.30 * w), int(0.73 * w)
            img[ta_top:ta_bottom, ta_left:ta_right] = [255, 255, 255]

            # Caret blink rendering if focused
            if self.focused_field == "text area":
                caret_x, caret_y = int(0.33 * w), int(0.36 * h)
                img[caret_y - 6 : caret_y + 6, caret_x : caret_x + 2] = [0, 0, 0]

            # Typed text lines
            if self.typed_text:
                for idx, _ch in enumerate(self.typed_text[:20]):
                    tx = int(0.35 * w) + idx * 4
                    ty = int(0.36 * h)
                    if tx < ta_right - 4:
                        img[ty - 3 : ty + 3, tx : tx + 2] = [20, 20, 20]

            # Save confirmation modal dialog overlay
            if self.state == self.STATE_SAVE_DIALOG:
                d_top, d_bottom = int(0.40 * h), int(0.68 * h)
                d_left, d_right = int(0.35 * w), int(0.68 * w)
                img[d_top:d_bottom, d_left:d_right] = [245, 245, 245]
                # Save button at (0.42, 0.55)
                sx, sy = int(0.42 * w), int(0.55 * h)
                img[sy - 4 : sy + 4, sx - 8 : sx + 8] = [70, 130, 180]
                # Don't Save button at (0.58, 0.55)
                dsx, dsy = int(0.58 * w), int(0.55 * h)
                img[dsy - 4 : dsy + 4, dsx - 10 : dsx + 10] = [180, 70, 70]

        elif self.state == self.STATE_BLOCKER:
            # Draw alert modal dialog in center
            m_top, m_bottom = int(0.35 * h), int(0.65 * h)
            m_left, m_right = int(0.20 * w), int(0.80 * w)
            img[m_top:m_bottom, m_left:m_right] = [255, 240, 240]
            # Warning bar
            img[m_top : m_top + 8, m_left:m_right] = [220, 50, 50]

        self.tick += 1
        return img

    def get_grounding_coord(self, query: str) -> Optional[Tuple[float, float]]:
        """Returns the valid ground coordinates for the target in the current state."""
        q = query.strip().lower()
        if self.state == self.STATE_DESKTOP:
            if "text editor" in q or "editor icon" in q:
                return self.target_coords["text editor icon"]
            if "media player" in q or "player icon" in q:
                return self.target_coords["media player icon"]
            return None

        if self.state == self.STATE_TEXT_EDITOR:
            if "text area" in q or "input" in q or "editor" in q:
                return self.target_coords["text area"]
            if "close" in q:
                return self.target_coords["close button"]
            return None

        if self.state == self.STATE_SAVE_DIALOG:
            if "don't save" in q or "dont save" in q:
                return self.target_coords["Don't Save button"]
            if "save" in q:
                return self.target_coords["Save button"]
            return None

        return None

    def attach_controller(self, controller: Any) -> None:
        """Hooks AVCSController perception and motor actuation to this simulated screen."""
        screen = self

        # 1. Custom locate implementation grounded on screen state
        def custom_locate(query: str, frame: Optional[np.ndarray] = None) -> GroundingResult:
            coord = screen.get_grounding_coord(query)
            if coord is None:
                raise UnvalidatedGroundingError(
                    f"Target '{query}' could not be localized on screen in current state '{screen.state}'."
                )
            return GroundingResult(
                query=query,
                target_coord=coord,
                confidence=0.96,
                relevance_map=np.ones((4, 4), dtype=np.float32),
                snapped_to_affordance=True,
                latency_ms=1.5,
            )

        controller.locate = custom_locate

        # 2. Custom focus verification
        original_verify_focus = getattr(controller.eye, "verify_focus", None)

        def custom_verify_focus(target_point: Optional[Tuple[float, float]] = None) -> VisualFocusState:
            is_focused = screen.focused_field == "text area"
            return VisualFocusState(
                is_focused=is_focused,
                confidence=0.98 if is_focused else 0.10,
                caret_detected=is_focused,
                caret_blinking=is_focused,
                focus_ring_detected=is_focused,
                caret_coord=(0.33, 0.36) if is_focused else None,
            )

        controller.eye.verify_focus = custom_verify_focus

        # 3. Intercept motor driver clicks to trigger simulated screen state transitions
        driver = controller.motor_driver
        orig_click = driver.click
        orig_double_click = driver.double_click
        orig_type = driver.type_text

        def on_click(button: str = "left", hold_duration: float = 0.02) -> None:
            orig_click(button=button, hold_duration=hold_duration)
            px, py = driver.get_position_normalized()

            if screen.state == screen.STATE_TEXT_EDITOR:
                # Check click on text area
                if abs(px - 0.50) < 0.20 and abs(py - 0.50) < 0.20:
                    screen.focused_field = "text area"
                # Check click on close button
                elif abs(px - 0.75) < 0.08 and abs(py - 0.22) < 0.08:
                    screen.state = screen.STATE_SAVE_DIALOG
                    screen.state_history.append(screen.state)

            elif screen.state == screen.STATE_SAVE_DIALOG:
                # Check click on Don't Save button
                if abs(px - 0.58) < 0.08 and abs(py - 0.55) < 0.08:
                    screen.state = screen.STATE_DESKTOP
                    screen.focused_field = None
                    screen.state_history.append(screen.state)
                # Check click on Save button
                elif abs(px - 0.42) < 0.08 and abs(py - 0.55) < 0.08:
                    screen.state = screen.STATE_DESKTOP
                    screen.focused_field = None
                    screen.state_history.append(screen.state)

        def on_double_click(button: str = "left", interval: float = 0.08) -> None:
            orig_double_click(button=button, interval=interval)
            px, py = driver.get_position_normalized()
            if screen.state == screen.STATE_DESKTOP:
                # Check double-click on text editor icon
                if abs(px - 0.20) < 0.10 and abs(py - 0.30) < 0.10:
                    screen.state = screen.STATE_TEXT_EDITOR
                    screen.state_history.append(screen.state)

        def on_type(text: str) -> None:
            orig_type(text)
            if screen.focused_field == "text area":
                screen.typed_text += text

        driver.click = on_click
        driver.double_click = on_double_click
        driver.type_text = on_type

        # 4. Intercept verifier to report verified state transitions on valid clicks
        verifier = controller.verifier
        orig_execute_verify = verifier.execute_and_verify

        def custom_execute_and_verify(**kwargs: Any) -> VerificationResult:
            action_type = kwargs.get("action_type", "click")
            target = kwargs.get("target", (0.5, 0.5))
            prev_state = screen.state

            # Move and perform action
            if action_type == "double_click":
                driver.double_click()
            else:
                driver.click()

            new_state = screen.state
            transition = new_state != prev_state or (action_type == "click" and screen.focused_field == "text area")

            return VerificationResult(
                status=VerificationStatus.VERIFIED if transition else VerificationStatus.RETRY_MICRO_JITTER,
                action_name=action_type,
                target_coord=target,
                optical_ripple_energy=0.08 if transition else 0.0,
                surprise_pulse=0.35 if transition else 0.0,
                state_transition_detected=transition,
                causal_summary=f"Screen transitioned from {prev_state} to {new_state}" if transition else "No state transition",
                recovery_attempt=0,
                elapsed_ms=1.5,
            )

        verifier.execute_and_verify = custom_execute_and_verify

        # 5. Intercept describe_screen to return faithful SceneObservation based on screen state
        orig_describe = controller.describe_screen

        def custom_describe_screen(frame: Optional[np.ndarray] = None, *, frame_sequence: Optional[int] = None) -> Any:
            from avcs.control.controller import SceneObservation

            t0 = time.time()
            if screen.state == screen.STATE_DESKTOP:
                cat = "desktop_canvas"
                has_modal = False
                desc = "Display is active. Primary context: Desktop Canvas (stable). Visible interactive controls: Text Editor icon, Media Player icon."
            elif screen.state == screen.STATE_TEXT_EDITOR:
                cat = "editor_code"
                has_modal = False
                desc = "Display is active. Primary context: Editor Code (stable). Text editor window is open. Visible interactive controls: text area, close button."
                if screen.focused_field == "text area":
                    desc += " Caret is active and blinking in text area."
                if screen.typed_text:
                    desc += f" Content rendered: '{screen.typed_text}'."
            elif screen.state == screen.STATE_SAVE_DIALOG:
                cat = "modal_dialog"
                has_modal = True
                desc = "Display is active. Primary context: Modal Dialog (overlay). A modal alert dialog is currently open and focused: 'Do you want to save changes?'. Controls: Save button, Don't Save button."
            elif screen.state == screen.STATE_BLOCKER:
                cat = "modal_dialog"
                has_modal = True
                msg = screen.blocker_message or "Session expired. Please re-authenticate."
                desc = f"Display is active. Primary context: Modal Dialog (overlay). A modal alert dialog is currently open and focused: '{msg}'."
            else:
                cat = "unknown"
                has_modal = False
                desc = "Unknown display context."

            return SceneObservation(
                timestamp=t0,
                primary_category=cat,
                stability="stable",
                surprise_score=0.05,
                description=desc,
                visible_entities=["window", "button"],
                has_active_modal=has_modal,
                is_stable=True,
                confidence=0.95,
                uncertainty=0.05,
                frame_sequence=frame_sequence or 1,
            )

        controller.describe_screen = custom_describe_screen
