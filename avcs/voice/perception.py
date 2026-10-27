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

from typing import Optional

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
