"""[3] narrator.py: AVCS Live Visual Commentator & Event-Driven Human Narrator.

Translates pixel dynamics, optical flow, active window changes, ConvGRU temporal
motion, foveal crops, and motor click ripples into Event-Driven Natural English
Commentary for System 2 LLMs.

Key Principles:
- Pure English Only: Zero non-English strings, zero regional languages.
- Application Agnostic: Strictly ZERO hardcoded application names or specific
  window titles. All perceptions are derived from pure spatial geometry.
- Event-Driven: Continuously monitors frames at 10ms ticks in RAM, but remains
  SILENT when the screen is static to prevent token burn.
- Four Core Interaction Events:
  1. Event 1 (Initial Sight / Window Focus):
     "Active window observed. Gaze focused on top-right where an interactive control is visible."
  2. Event 2 (Touch / Ripple Feedback):
     - Ripple Confirmed: "Clicked target control, pixel ripple detected (button registered successfully)!"
     - Ripple Absent: "Clicked target, but screen showed 0 pixel ripple (element unresponsive or disabled)!"
  3. Event 3 (Wait / Spinner):
     "Hold on... continuous spinner animation active in bottom region, waiting for content to settle..."
  4. Event 4 (Sudden Surprise / Popup):
     "Alert! A modal dialog popped up in the center with interactive options!"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from avcs.eye_stream import EyeFrame
from avcs.fovea_reader import FovealElement


@dataclass(frozen=True)
class LiveCommentary:
    """A single live commentary tick from AVCS to System 2 LLM."""
    narrative: str                             # Natural English commentary
    active_window: str                         # Foreground window title reported by OS
    is_stable: bool                           # True if screen has stopped changing
    has_popup: bool                           # True if unexpected modal/dialog appeared
    needs_llm_attention: bool                 # True only if LLM decision is required
    last_action_feedback: Optional[str] = None # Detailed action feedback string
    visible_elements: List[str] = None        # Pixel descriptions of detected items
    surprise_score: float = 0.0
    event_type: Optional[str] = None          # 'initial_sight', 'ripple_feedback', 'spinner_wait', 'popup_alert', 'stable'
    human_speech: Optional[str] = None        # Pure conversational English utterance

    def to_llm_prompt(self) -> str:
        """Emits an ultra-compact, token-efficient text prompt (<100 tokens)."""
        speech = self.human_speech or self.narrative
        lines = [f"[AVCS Live Eye]: \"{speech}\""]
        if self.active_window:
            lines.append(f"[Active Window]: '{self.active_window}'")
        if self.last_action_feedback and self.last_action_feedback != speech:
            lines.append(f"[Action Feedback]: {self.last_action_feedback}")
        if self.visible_elements:
            clean_controls = [
                re.sub(r"\s*at\s*\([-\+0-9.,\s]+\)", "", ctrl, flags=re.IGNORECASE) for ctrl in self.visible_elements[:4]
            ]
            lines.append(f"[Visible Controls]: {', '.join(clean_controls)}")
        return "\n".join(lines)


class LiveNarrator:
    """The Voice of AVCS: Event-Driven Natural English Visual Commentator."""

    def __init__(
        self,
        mode: str = "conversational",    # "conversational" or "technical"
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self.mode = mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.last_window_title: str = ""
        self.consecutive_stable_ticks: int = 0
        self.last_narrative: str = ""
        self.is_first_frame: bool = True
        self.last_event_emitted: Optional[str] = None

    def _get_spatial_region(self, norm_x: float, norm_y: float) -> str:
        """Maps normalized (x, y) coordinates to spatial screen regions."""
        if norm_y < 0.25:
            if norm_x > 0.65:
                return "top-right"
            if norm_x < 0.35:
                return "top-left"
            return "top bar"
        if norm_y > 0.70:
            if norm_x < 0.40:
                return "bottom-left"
            if norm_x > 0.65:
                return "bottom-right"
            return "bottom region"
        if 0.30 <= norm_x <= 0.70 and 0.25 <= norm_y <= 0.75:
            return "center"
        if norm_x < 0.35:
            return "left region"
        if norm_x > 0.65:
            return "right region"
        return "center"

    def _clean_target_name(self, action_str: Optional[str]) -> str:
        """Extracts clean generic target description without app assumptions."""
        if not action_str:
            return "target control"

        clean = str(action_str).strip()
        for prefix in ("click on ", "click ", "typed ", "type in ", "press "):
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):]
                break

        # Strip coordinate annotations like ' at (0.92, 0.04)' or ' at (-50, 100)'
        clean = re.sub(r"\s*at\s*\([-\+0-9.,\s]+\)", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*at\s*\[[-\+0-9.,\s]+\]", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*\([-\+0-9.,\s]+\)", "", clean)
        clean = re.sub(r"\s*\[[-\+0-9.,\s]+\]", "", clean)
        clean = clean.strip().strip("'\"")

        # Map internal class names to universal generic UI terms
        name_map = {
            "ui_box": "interactive control",
            "Interactive_Control": "interactive control",
            "interactive_control": "interactive control",
            "graphic": "action widget",
            "text_like": "input field",
            "input_field": "input field",
            "button": "button",
            "dialog": "dialog control",
        }
        return name_map.get(clean, clean or "interactive control")

    def narrate(
        self,
        eye_frame: EyeFrame,
        active_window: str,
        foveal_elements: Optional[List[FovealElement]] = None,
        ripple_verified: Optional[bool] = None,
        prior_action: Optional[str] = None,
        target_hint: Optional[str] = None,
    ) -> Optional[LiveCommentary]:
        """Produces event-driven English commentary based on real-time screen events."""
        event_type = None
        human_speech = None
        feedback_str = None
        needs_attention = False

        # Compute default primary gaze coordinates
        norm_x, norm_y = 0.5, 0.5
        primary_kind = "interactive control"
        if foveal_elements and len(foveal_elements) > 0:
            norm_x, norm_y = foveal_elements[0].center_norm
            primary_kind = getattr(foveal_elements[0], "category", "interactive control")
        elif eye_frame.hotspots and len(eye_frame.hotspots) > 0:
            norm_x, norm_y, primary_kind, _ = eye_frame.hotspots[0]

        region = self._get_spatial_region(norm_x, norm_y)

        # ---------------------------------------------------------------------
        # Event 2: Touch / Action Ripple Feedback
        # ---------------------------------------------------------------------
        if ripple_verified is not None and prior_action:
            event_type = "ripple_feedback"
            target_name = self._clean_target_name(target_hint if target_hint is not None else prior_action)
            needs_attention = True

            if ripple_verified:
                human_speech = f"Clicked '{target_name}', pixel ripple detected (button registered successfully)!"
                feedback_str = f"Action confirmed: Clicked '{target_name}', visual ripple verified."
            else:
                human_speech = f"Clicked '{target_name}', but screen showed 0 pixel ripple (element unresponsive or disabled)!"
                feedback_str = f"Warning: Clicked '{target_name}', zero ripple energy detected."

        # ---------------------------------------------------------------------
        # Event 4: Sudden Surprise / Modal Alert / Dialog Box Pop-up
        # ---------------------------------------------------------------------
        elif eye_frame.has_popup or eye_frame.surprise_score > 0.45:
            event_type = "popup_alert"
            needs_attention = True
            popup_region = "center" if (0.25 <= norm_x <= 0.75 and 0.25 <= norm_y <= 0.75) else region

            human_speech = f"Alert! A modal dialog popped up in the {popup_region} with interactive options!"
            feedback_str = "Alert: Unexpected modal dialog appeared with interactive options."

        # ---------------------------------------------------------------------
        # Event 3: Wait / Loading Spinner Activity
        # ---------------------------------------------------------------------
        elif not eye_frame.is_stable and (eye_frame.surprise_score > 0.08 or getattr(eye_frame, "has_spinner", False)):
            event_type = "spinner_wait"
            needs_attention = False
            spinner_region = "bottom region" if norm_y > 0.60 else region

            human_speech = f"Hold on... continuous spinner animation active in {spinner_region}, waiting for content to settle..."
            feedback_str = "Screen rendering dynamic content, holding for completion."

        # ---------------------------------------------------------------------
        # Event 1: Initial Sight (Window Switch or First Observation)
        # ---------------------------------------------------------------------
        elif self.is_first_frame or (active_window and active_window != self.last_window_title):
            event_type = "initial_sight"
            needs_attention = True
            target_name = self._clean_target_name(target_hint if target_hint is not None else primary_kind)

            # Descriptive, generic, application-agnostic framing:
            window_display = f"'{active_window}'" if active_window else "Active window"
            article = "an" if target_name and target_name[0].lower() in "aeiou" else "a"
            human_speech = f"{window_display} observed. Gaze focused on {region} where {article} {target_name} is visible."
            feedback_str = f"Foreground focus: {active_window or 'Active window'}"

            self.last_window_title = active_window
            self.is_first_frame = False

        # ---------------------------------------------------------------------
        # Event 5: Static Screen (Remain Silent)
        # ---------------------------------------------------------------------
        elif eye_frame.is_stable:
            self.consecutive_stable_ticks += 1
            if self.consecutive_stable_ticks == 1 and self.last_event_emitted != "stable":
                event_type = "stable"
                needs_attention = True
                human_speech = "Screen has settled and is ready for the next command."
                feedback_str = "Screen steady."
            else:
                # Screen static: SILENCE to prevent token spam & unnecessary calls
                return None
        else:
            self.consecutive_stable_ticks = 0
            return None

        # Build final LiveCommentary object
        self.last_event_emitted = event_type
        self.last_narrative = human_speech or "Screen active."
        element_descriptions = []
        for e in (foveal_elements or []):
            if hasattr(e, "describe"):
                desc = e.describe()
                if hasattr(e, "center_norm"):
                    reg = self._get_spatial_region(e.center_norm[0], e.center_norm[1])
                    desc = re.sub(r"\s*at\s*\([-\+0-9.,\s]+\)", f" in {reg}", desc, flags=re.IGNORECASE)
                    desc = re.sub(r"\s*at\s*\[[-\+0-9.,\s]+\]", f" in {reg}", desc, flags=re.IGNORECASE)
                element_descriptions.append(desc)
            elif isinstance(e, str):
                cleaned_str = re.sub(r"\s*at\s*\([-\+0-9.,\s]+\)", "", e, flags=re.IGNORECASE)
                cleaned_str = re.sub(r"\s*at\s*\[[-\+0-9.,\s]+\]", "", cleaned_str, flags=re.IGNORECASE)
                element_descriptions.append(cleaned_str)

        return LiveCommentary(
            narrative=human_speech if self.mode == "conversational" else (feedback_str or human_speech),
            active_window=active_window,
            is_stable=eye_frame.is_stable,
            has_popup=eye_frame.has_popup,
            needs_llm_attention=needs_attention,
            last_action_feedback=feedback_str,
            visible_elements=element_descriptions,
            surprise_score=eye_frame.surprise_score,
            event_type=event_type,
            human_speech=human_speech,
        )
