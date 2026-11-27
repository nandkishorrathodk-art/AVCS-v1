"""[3] narrator.py: AVCS Live Visual Commentator & Sensory Telemetry Streamer.

Translates pixel dynamics, optical flow, active window changes, ConvGRU temporal
motion, foveal crops, and motor click ripples into typed Sensory Telemetry and
compact Event-Driven observations for System 2 LLMs.

DEPRECATION NOTICE:
Fake conversational f-strings and natural language commentary are deprecated in
favor of typed SensoryTelemetry metrics (category, category_confidence,
optical_motion_pct, is_settled, stability_state, surprise_score, has_modal,
hotspots, recommended_action, latency_ms). Use `LiveCommentary.telemetry`,
`LiveCommentary.to_dict()`, or `LiveCommentary.to_json()`.

Key Principles:
- Pure English Only: Zero non-English strings, zero regional languages.
- Application Agnostic: Strictly ZERO hardcoded application names or specific
  window titles. All perceptions are derived from pure spatial geometry.
- Event-Driven: Continuously monitors frames at 10ms ticks in RAM, but remains
  SILENT when the screen is static to prevent token burn.
- Typed Telemetry: Exposes pure scalar metrics and typed affordances.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .eye_stream import EyeFrame
from .fovea_reader import FovealElement
from .telemetry import (
    AffordanceKind,
    RecommendedAction,
    SceneCategory,
    SensoryTelemetry,
    StabilityState,
    TelemetryHotspot,
)


@dataclass(frozen=True)
class LiveCommentary:
    """A single live commentary tick from AVCS to System 2 LLM.

    DEPRECATION NOTICE:
    Conversational text strings (narrative, human_speech) are deprecated in favor
    of pure typed SensoryTelemetry metrics exposed via the `telemetry` field.
    """
    narrative: str                             # [DEPRECATED] Natural English commentary
    active_window: str                         # Foreground window title reported by OS
    is_stable: bool                           # True if screen has stopped changing
    has_popup: bool                           # True if unexpected modal/dialog appeared
    needs_llm_attention: bool                 # True only if LLM decision is required
    last_action_feedback: Optional[str] = None # Detailed action feedback string
    visible_elements: List[str] = None        # Pixel descriptions of detected items
    surprise_score: float = 0.0
    event_type: Optional[str] = None          # 'initial_sight', 'ripple_feedback', 'spinner_wait', 'popup_alert', 'stable'
    human_speech: Optional[str] = None        # [DEPRECATED] Pure conversational English utterance
    telemetry: Optional[SensoryTelemetry] = None # Structured sensory telemetry

    def to_telemetry(self) -> SensoryTelemetry:
        """Returns the structured SensoryTelemetry representation."""
        if self.telemetry is not None:
            return self.telemetry
        return SensoryTelemetry(
            category=SceneCategory.MODAL_DIALOG if self.has_popup else SceneCategory.DESKTOP_CANVAS,
            category_confidence=0.95,
            optical_motion_pct=round(self.surprise_score * 100.0, 2),
            is_settled=self.is_stable,
            stability_state=(
                StabilityState.SURPRISE_EVENT if self.has_popup
                else (StabilityState.STABLE if self.is_stable else StabilityState.ANIMATING)
            ),
            surprise_score=self.surprise_score,
            has_modal=self.has_popup,
            recommended_action=(
                RecommendedAction.CLICK if self.needs_llm_attention
                else (RecommendedAction.WAIT if not self.is_stable else RecommendedAction.NOOP)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Outputs structured telemetry dictionary."""
        return self.to_telemetry().to_dict()

    def to_json(self, indent: Optional[int] = None) -> str:
        """Outputs structured telemetry JSON string."""
        return self.to_telemetry().to_json(indent=indent)

    def to_llm_prompt(self, structured: bool = False) -> str:
        """Emits an ultra-compact, token-efficient text prompt (<100 tokens).

        If structured is True, returns JSON string of structured sensory telemetry.
        """
        if structured:
            return self.to_json()
        speech = self.human_speech or self.narrative
        lines = [f"[AVCS Live Eye]: \"{speech}\""]
        if self.last_action_feedback and self.last_action_feedback != speech:
            lines.append(f"[Action Feedback]: {self.last_action_feedback}")
        if self.visible_elements:
            clean_controls = [
                re.sub(r"\s*at\s*\([-\+0-9.,\s]+\)", "", ctrl, flags=re.IGNORECASE) for ctrl in self.visible_elements[:4]
            ]
            lines.append(f"[Visible Controls]: {', '.join(clean_controls)}")
        return "\n".join(lines)


class LiveNarrator:
    """The Sensory Telemetry Streamer and Event-Driven Narrator for AVCS.

    DEPRECATION NOTICE:
    Conversational English f-strings are deprecated. For automated System 2 agent
    interaction, consume structured SensoryTelemetry provided via `comm.telemetry`
    or `comm.to_dict()` / `comm.to_json()`.
    """

    def __init__(
        self,
        mode: str = "conversational",    # "conversational" (legacy) or "telemetry"
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

        # Map internal class names to universal generic UI terms. Keys must
        # cover every token the eye stream emits (UI_CLASS_MAP) plus the legacy
        # aliases, or a real category renders as a raw snake_case identifier.
        name_map = {
            "ui_box": "interactive control",
            "Interactive_Control": "interactive control",
            "interactive_control": "interactive control",
            "graphic": "action widget",
            "text_like": "input field",
            "text_input": "input field",
            "input_field": "input field",
            "button": "button",
            "toggle_icon": "toggle control",
            "modal_dialog": "dialog control",
            "content_region": "content area",
            "dialog": "dialog control",
        }
        return name_map.get(clean, clean or "interactive control")

    def narrate(
        self,
        eye_frame: EyeFrame,
        active_window: str = "",
        foveal_elements: Optional[List[FovealElement]] = None,
        ripple_verified: Optional[bool] = None,
        prior_action: Optional[str] = None,
        target_hint: Optional[str] = None,
    ) -> Optional[LiveCommentary]:
        """Produces event-driven sensory telemetry and structured commentary."""
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
            norm_x, norm_y, primary_kind, *_ = eye_frame.hotspots[0]

        region = self._get_spatial_region(norm_x, norm_y)

        # ---------------------------------------------------------------------
        # Event 2: Touch / Action Ripple Feedback
        # ---------------------------------------------------------------------
        if ripple_verified is not None and prior_action:
            event_type = "ripple_feedback"
            target_name = self._clean_target_name(target_hint if target_hint is not None else prior_action)
            needs_attention = True

            # [DEPRECATED legacy conversational f-string]
            if ripple_verified:
                human_speech = f"Clicked '{target_name}', pixel ripple detected (button registered successfully)!"
                feedback_str = f"Action confirmed: Clicked '{target_name}', visual ripple verified."
            else:
                human_speech = f"Clicked '{target_name}', but screen showed 0 pixel ripple (element unresponsive or disabled)!"
                feedback_str = f"Warning: Clicked '{target_name}', zero ripple energy detected."

        # ---------------------------------------------------------------------
        # Event 4: Sudden Surprise / Modal Alert / Dialog Box Pop-up
        # ---------------------------------------------------------------------
        elif eye_frame.has_popup:
            event_type = "popup_alert"
            needs_attention = True
            popup_region = "center" if (0.25 <= norm_x <= 0.75 and 0.25 <= norm_y <= 0.75) else region

            # [DEPRECATED legacy conversational f-string]
            human_speech = f"Alert! A modal dialog popped up in the {popup_region} with interactive options!"
            feedback_str = "Alert: Unexpected modal dialog appeared with interactive options."

        # ---------------------------------------------------------------------
        # Event 3: Wait / Loading Spinner Activity
        # ---------------------------------------------------------------------
        elif not eye_frame.is_stable and (eye_frame.surprise_score > 0.08 or getattr(eye_frame, "has_spinner", False)):
            event_type = "spinner_wait"
            needs_attention = False
            spinner_region = "bottom region" if norm_y > 0.60 else region

            # [DEPRECATED legacy conversational f-string]
            human_speech = f"Hold on... continuous spinner animation active in {spinner_region}, waiting for content to settle..."
            feedback_str = "Screen rendering dynamic content, holding for completion."

        # ---------------------------------------------------------------------
        # Event 1: Initial Sight (First Pure-Pixel Observation)
        # ---------------------------------------------------------------------
        elif self.is_first_frame:
            event_type = "initial_sight"
            needs_attention = True
            target_name = self._clean_target_name(target_hint if target_hint is not None else primary_kind)

            # [DEPRECATED legacy conversational f-string]
            article = "an" if target_name and target_name[0].lower() in "aeiou" else "a"
            human_speech = f"Desktop screen observed. Gaze focused on {region} where {article} {target_name} is visible."
            feedback_str = "Pure-pixel initial observation complete."

            self.is_first_frame = False

        # ---------------------------------------------------------------------
        # Event 5: Static Screen (Remain Silent)
        # ---------------------------------------------------------------------
        elif eye_frame.is_stable:
            self.consecutive_stable_ticks += 1
            if self.consecutive_stable_ticks == 1 and self.last_event_emitted != "stable":
                event_type = "stable"
                needs_attention = True
                # [DEPRECATED legacy conversational utterance]
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
            if hasattr(e, "center_norm") and hasattr(e, "category"):
                label = self._clean_target_name(e.category)
                reg = self._get_spatial_region(e.center_norm[0], e.center_norm[1])
                element_descriptions.append(f"{label} in {reg}")
            elif hasattr(e, "describe"):
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

        # Construct pure scalar structured sensory telemetry
        tel_category = SceneCategory.DESKTOP_CANVAS
        if eye_frame.has_popup:
            tel_category = SceneCategory.MODAL_DIALOG
        elif foveal_elements and len(foveal_elements) > 0:
            try:
                tel_category = SceneCategory(getattr(foveal_elements[0], "category", "desktop_canvas"))
            except Exception:
                tel_category = SceneCategory.DESKTOP_CANVAS
        elif eye_frame.hotspots and len(eye_frame.hotspots) > 0:
            try:
                tel_category = SceneCategory(eye_frame.hotspots[0][2])
            except Exception:
                tel_category = SceneCategory.DESKTOP_CANVAS

        tel_conf = 0.95
        if foveal_elements and len(foveal_elements) > 0:
            tel_conf = float(getattr(foveal_elements[0], "confidence", 0.95))
        elif eye_frame.hotspots and len(eye_frame.hotspots) > 0:
            tel_conf = float(eye_frame.hotspots[0][3])

        if eye_frame.has_popup:
            tel_stability = StabilityState.SURPRISE_EVENT
        elif not eye_frame.is_stable:
            tel_stability = StabilityState.ANIMATING
        else:
            tel_stability = StabilityState.STABLE

        if eye_frame.has_popup or event_type in ("initial_sight", "popup_alert", "ripple_feedback"):
            tel_action = RecommendedAction.CLICK
        elif not eye_frame.is_stable:
            tel_action = RecommendedAction.WAIT
        elif eye_frame.hotspots:
            tel_action = RecommendedAction.CLICK
        else:
            tel_action = RecommendedAction.NOOP

        tel_hotspots = [TelemetryHotspot.from_tuple(h) for h in (eye_frame.hotspots or [])]

        telemetry = SensoryTelemetry(
            category=tel_category,
            category_confidence=tel_conf,
            optical_motion_pct=round(float(eye_frame.surprise_score) * 100.0, 2),
            is_settled=bool(eye_frame.is_stable),
            stability_state=tel_stability,
            surprise_score=float(eye_frame.surprise_score),
            has_modal=bool(eye_frame.has_popup),
            hotspots=tel_hotspots,
            recommended_action=tel_action,
            latency_ms=float(getattr(eye_frame, "latency_ms", 0.0)),
        )

        return LiveCommentary(
            narrative=human_speech if self.mode == "conversational" else (feedback_str or human_speech),
            active_window="",
            is_stable=eye_frame.is_stable,
            has_popup=eye_frame.has_popup,
            needs_llm_attention=needs_attention,
            last_action_feedback=feedback_str,
            visible_elements=element_descriptions,
            surprise_score=eye_frame.surprise_score,
            event_type=event_type,
            human_speech=human_speech,
            telemetry=telemetry,
        )

    def narrate_telemetry(
        self,
        eye_frame: EyeFrame,
        active_window: str = "",
        foveal_elements: Optional[List[FovealElement]] = None,
        ripple_verified: Optional[bool] = None,
        prior_action: Optional[str] = None,
        target_hint: Optional[str] = None,
    ) -> Optional[SensoryTelemetry]:
        """Direct sensory telemetry extraction without conversational prose."""
        comm = self.narrate(
            eye_frame=eye_frame,
            active_window=active_window,
            foveal_elements=foveal_elements,
            ripple_verified=ripple_verified,
            prior_action=prior_action,
            target_hint=target_hint,
        )
        return comm.to_telemetry() if comm else None
