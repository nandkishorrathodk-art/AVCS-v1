"""Pure-pixel perception adapter for the voice assistant.

The assistant's brain must never receive images or coordinates -- only short
factual text about the screen. This adapter runs the local AVCS visual cortex
(`EyeStream`, pure pixels, no OCR, no accessibility tree) and reduces its output
to a `ScreenObservation`.

Nothing here interprets text on screen. It reports *morphology*: how many
interactive elements were seen, whether the scene is settled, whether a modal
appeared. That is the System 1 -> System 2 boundary from
``docs/AVCS_ENGINEERING_RULES.md``, enforced by construction: this class has no
code path that returns a pixel, an embedding, or a coordinate to the brain.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from avcs.voice.assistant import ScreenObservation

# Human-readable phrases for the 5-class morphology vocabulary. These name the
# *kind* of control, not its text; that is what pure-pixel perception can know.
_KIND_PHRASES = {
    "interactive_control": "clickable control",
    "text_input": "text input field",
    "toggle_icon": "toggle",
    "modal_dialog": "dialog",
    "content_region": "content area",
}


class EyePerception:
    """Real screen perception via the AVCS ``EyeStream`` cortex.

    Parameters
    ----------
    eye:
        An object with ``step() -> EyeFrame`` and ``grab_screen() -> ndarray``.
        Defaults to constructing the real ``EyeStream``.
    source:
        Passed to ``EyeStream``; ``"synthetic"`` forces the deterministic buffer
        so a headless run still produces a frame.
    """

    def __init__(self, eye: Optional[object] = None, source: str = "screen") -> None:
        if eye is None:
            from avcs_sdk.eye_stream import EyeStream

            eye = EyeStream(source=source)
        self.eye = eye
        self._last_frame: Optional[np.ndarray] = None

    @property
    def weights_loaded(self) -> bool:
        return bool(getattr(self.eye, "weights_loaded", False))

    def frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        eye_frame = self.eye.step()
        self._last_frame = eye_frame.raw_frame

        hotspots = list(getattr(eye_frame, "hotspots", []) or [])
        kinds = []
        for h in hotspots[:6]:
            kind = h[2] if len(h) > 2 else "interactive_control"
            kind = str(kind)
            kinds.append(_KIND_PHRASES.get(kind, kind.replace("_", " ")))

        parts = [f"screen shows {len(hotspots)} interactive element(s)"]
        if kinds:
            parts.append("including " + ", ".join(kinds[:3]))
        if getattr(eye_frame, "has_popup", False):
            parts.append("a dialog has appeared")
        if not getattr(eye_frame, "is_stable", True):
            parts.append("the screen is still changing")

        stability = "stable" if getattr(eye_frame, "is_stable", True) else "changing"
        return ScreenObservation(
            description="; ".join(parts),
            target_present=True if hotspots else None,
            stability=stability,
            has_modal=bool(getattr(eye_frame, "has_popup", False)),
            active_window="",
            hotspot_count=len(hotspots),
        )

    def close(self) -> None:
        close = getattr(self.eye, "close", None)
        if callable(close):
            close()


class GroundingPerception:
    """Pure-pixel perception that also grounds a named target to a point.

    `EyePerception` reports morphology only. This adapter additionally runs the
    trained `GoalConditionedGrounder` on the current frame to answer "where is
    <target>?", so the capability kernel gets *real* evidence (the grounder's
    discriminating `peak_probability`) instead of a placeholder.

    Still no OCR: the grounder is query-conditioned on pixels, and the target
    text comes from the user's own speech, not from the screen.

    Measured on `datasets/ui_grounding` `test_held_out` frame 1 (shipped
    `goal_grounder_best.pt`): 8/8 named targets grounded within 0.022 normalized
    error; 3 deliberately wrong queries peaked 0.005-0.011. The kernel's default
    floor of 0.75 sits in that gap.
    """

    def __init__(
        self,
        eye: Optional[object] = None,
        grounder: Optional[object] = None,
        checkpoint: Optional[str] = None,
        source: str = "screen",
        peak_floor: float = 0.75,
    ) -> None:
        self._morphology = EyePerception(eye=eye, source=source)
        self.peak_floor = float(peak_floor)
        if grounder is None:
            from avcs.goal_grounder import load_grounding_checkpoint

            path = checkpoint or "runs/weights/goal_grounder_best.pt"
            grounder = load_grounding_checkpoint(path, device="cpu")
        self.grounder = grounder
        self._last_prediction = None

    @property
    def grounder_is_trained(self) -> bool:
        return bool(getattr(self.grounder, "is_trained", False))

    def frame(self):
        return self._morphology.frame()

    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        obs = self._morphology.observe(query)
        if query and self.grounder_is_trained and self._morphology._last_frame is not None:
            evidence = self.grounding_evidence(query)
            if evidence is not None:
                obs.target_present = bool(evidence.peak_probability >= self.peak_floor)
        return obs

    def grounding_evidence(self, query: Optional[str]):
        """Ground `query` on the current frame and return real kernel evidence.

        Runs on demand so the caller does not have to have called ``observe`` with
        the same query first. Returns ``None`` when there is nothing to ground.
        """
        if not query or not self.grounder_is_trained:
            return None
        frame = self._morphology._last_frame
        if frame is None:
            return None
        try:
            pred = self.grounder.predict(frame, query)
        except Exception:
            return None
        self._last_prediction = pred
        from avcs.control.capability import GroundingEvidence

        return GroundingEvidence(
            candidate_point_norm=pred.candidate_point_norm,
            peak_probability=float(pred.peak_probability),
            target_present=bool(pred.target_present),
            authorized=bool(getattr(pred, "action_authorized", False)),
            active_query=query,
        )

    def close(self) -> None:
        self._morphology.close()


class LocalVLAPerception:
    """Unified single-brain local VLA perception backed by ``FastVLAInferenceEngine``.

    Implements the coordinate-free ``Perception`` protocol (``observe``,
    ``frame``) plus ``grounding_evidence`` for the fail-closed capability
    kernel without modifying or breaking ``EyePerception`` / ``GroundingPerception``.
    """

    def __init__(
        self,
        frame_source: Optional[object] = None,
        engine: Optional[object] = None,
        *,
        eye: Optional[object] = None,
        source: str = "screen",
        peak_floor: float = 0.75,
        default_confidence: float = 0.85,
        session_id: str = "voice-vla",
    ) -> None:
        self._frame_source = frame_source
        self._eye = eye
        if frame_source is None and eye is None and source:
            from avcs_sdk.eye_stream import EyeStream

            self._eye = EyeStream(source=source)
        if engine is None:
            from avcs.runtime.vla_engine import get_default_fast_engine

            engine = get_default_fast_engine()
        self.engine = engine
        self.peak_floor = float(peak_floor)
        self.default_confidence = float(default_confidence)
        self.session_id = session_id
        self._last_frame: Optional[np.ndarray] = None
        self._last_eye_frame: Optional[object] = None
        self._last_result: Optional[Dict[str, Any]] = None

    def _capture(self) -> Optional[np.ndarray]:
        if self._frame_source is not None:
            src = self._frame_source
            if callable(src):
                return src()
            if hasattr(src, "capture"):
                return src.capture()
            if hasattr(src, "grab_screen"):
                return src.grab_screen()
            raise TypeError("frame_source must be callable or expose capture()/grab_screen()")
        if self._eye is not None:
            if hasattr(self._eye, "step"):
                eye_frame = self._eye.step()
                self._last_eye_frame = eye_frame
                return getattr(eye_frame, "raw_frame", None)
            if hasattr(self._eye, "grab_screen"):
                return self._eye.grab_screen()
        return None

    @property
    def capture_is_live(self) -> bool:
        if self._frame_source is not None:
            return False
        flag = getattr(self._eye, "capture_is_live", None)
        if flag is not None:
            return bool(flag)
        return self._eye is not None

    def frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        frame = self._capture()
        self._last_frame = frame
        if frame is None:
            return ScreenObservation(
                description="no screen available",
                target_present=None,
                stability="stable",
                has_modal=False,
                active_window="",
                hotspot_count=0,
            )

        eye_frame = self._last_eye_frame
        hotspots = list(getattr(eye_frame, "hotspots", []) or []) if eye_frame is not None else []
        has_modal = bool(getattr(eye_frame, "has_popup", False)) if eye_frame is not None else False
        is_stable = bool(getattr(eye_frame, "is_stable", True)) if eye_frame is not None else True

        parts = ["local VLA perception active"]
        if hotspots:
            parts.append(f"screen shows {len(hotspots)} interactive element(s)")
        if has_modal:
            parts.append("a dialog has appeared")
        if not is_stable:
            parts.append("the screen is still changing")

        target_present: Optional[bool] = True if hotspots else None
        if query:
            evidence = self.grounding_evidence(query)
            if evidence is not None:
                target_present = bool(
                    evidence.candidate_point_norm is not None
                    and math.isfinite(evidence.peak_probability)
                    and evidence.peak_probability >= self.peak_floor
                )

        return ScreenObservation(
            description="; ".join(parts),
            target_present=target_present,
            stability="stable" if is_stable else "changing",
            has_modal=has_modal,
            active_window="",
            hotspot_count=len(hotspots),
        )

    def grounding_evidence(self, query: Optional[str]):
        """Ground ``query`` via ``FastVLAInferenceEngine`` and return kernel evidence."""
        if not query or self._last_frame is None:
            return None
        frame = self._last_frame
        h, w = frame.shape[:2]
        try:
            res = self.engine.act(
                frame,
                instruction=query,
                canvas_w=int(w),
                canvas_h=int(h),
                session_id=self.session_id,
            )
        except Exception:
            return None
        if not isinstance(res, dict):
            return None
        self._last_result = res

        from avcs.control.capability import GroundingEvidence

        action = str(res.get("action") or "").strip().lower()
        coord = res.get("coordinate")
        if action in ("wait", "stop", "none") or not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return GroundingEvidence(
                candidate_point_norm=None,
                peak_probability=0.0,
                target_present=False,
                authorized=False,
                active_query=query,
            )

        try:
            px = float(coord[0])
            py = float(coord[1])
        except (TypeError, ValueError):
            return GroundingEvidence(
                candidate_point_norm=None,
                peak_probability=0.0,
                target_present=False,
                authorized=False,
                active_query=query,
            )
        if not (math.isfinite(px) and math.isfinite(py)):
            return GroundingEvidence(
                candidate_point_norm=None,
                peak_probability=0.0,
                target_present=False,
                authorized=False,
                active_query=query,
            )

        nx = max(0.0, min(1.0, px / max(1.0, float(w))))
        ny = max(0.0, min(1.0, py / max(1.0, float(h))))

        raw_conf = res.get("peak_probability", res.get("confidence"))
        if raw_conf is not None:
            try:
                peak = float(raw_conf)
            except (TypeError, ValueError):
                peak = 0.0
            if not math.isfinite(peak):
                return GroundingEvidence(
                    candidate_point_norm=None,
                    peak_probability=0.0,
                    target_present=False,
                    authorized=False,
                    active_query=query,
                )
        else:
            peak = self.default_confidence

        return GroundingEvidence(
            candidate_point_norm=(nx, ny),
            peak_probability=peak,
            target_present=peak >= self.peak_floor,
            authorized=False,
            active_query=query,
        )

    def close(self) -> None:
        close = getattr(self._eye, "close", None)
        if callable(close):
            close()
