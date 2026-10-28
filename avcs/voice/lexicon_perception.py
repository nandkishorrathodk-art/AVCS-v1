"""Perception that grounds a *spoken word* to a point using the AVC lexicon.

The shipped pipeline had no working "click" step. `GoalConditionedGrounder`
returns `action_authorized=False` on every call and `AVCSController.locate`
raises on that, and `GroundingPerception` needs the trained goal grounder, whose
confidence gate collapses off its training generator. So the assistant's kernel
denied every click with "grounder produced no candidate point" -- the loop could
name an action but never carry it out.

This adapter closes that gap without a new model. It reuses the lexicon reader
(`avc/lexicon/`), which recognises a control *label* from pixels, and answers
"where is <word>?" by returning the centre of the box that read as that word.
The evidence it hands the kernel is real: the box exists, and the confidence is
the reader's own cosine margin against the closed vocabulary.

The vocabulary is closed and declared in `avc/lexicon/vocab.py`. A word outside
it is *refused*, not snapped to the nearest label, so an out-of-vocabulary
command fails closed exactly like an absent target.

Still no OCR: the reader compares glyph shapes against prototypes. It never
transcribes characters, and the query text comes from the user's speech, not
from the screen.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from avcs.voice.assistant import ScreenObservation

# A frame source returns the current screen as an HxWx3 uint8 array, or None
# when no frame is available. A callable keeps this adapter independent of any
# particular capture backend (mss, a test fixture, or the GUI's own buffer).
FrameSource = Callable[[], Optional[np.ndarray]]


class LexiconPerception:
    """Perception for labelled controls: reads a frame, grounds a spoken word.

    Implements the `Perception` protocol the assistant expects (`observe`,
    `frame`) plus `grounding_evidence`, which `Assistant._act` looks for. When
    that method is present the kernel gets a real candidate point instead of the
    placeholder that always denies.
    """

    def __init__(
        self,
        frame_source: Optional[object] = None,
        reader: Optional[object] = None,
        *,
        eye: Optional[object] = None,
        source: str = "screen",
        checkpoint: Optional[str] = None,
        floor: float = 0.0,
        label_floor: float = 0.0,
    ) -> None:
        self._eye = eye
        self.floor = float(floor)
        self.label_floor = float(label_floor)

        if frame_source is None and eye is None and source:
            # Attach the same EyeStream the other perceptions use, so this
            # adapter can read the real screen and not only a test fixture.
            from avcs_sdk.eye_stream import EyeStream

            self._eye = EyeStream(source=source)
        self._frame_source = frame_source

        if reader is None:
            from avc.lexicon.reader import LexiconReader

            reader = LexiconReader(checkpoint=checkpoint or "runs/weights/avc_lexicon.pt")
        self.reader = reader
        self._last_frame: Optional[np.ndarray] = None
        self._last_regions: List[object] = []

    def _capture(self) -> Optional[np.ndarray]:
        """Pull one frame from whichever source was supplied."""
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
                return getattr(eye_frame, "raw_frame", None)
            if hasattr(self._eye, "grab_screen"):
                return self._eye.grab_screen()
        return None

    # ------------------------------------------------------------------ #
    # frame handling
    # ------------------------------------------------------------------ #
    def frame(self) -> Optional[np.ndarray]:
        """The most recent frame. Cached so a pre/post pair is comparable."""
        return self._last_frame

    def grab(self) -> Optional[np.ndarray]:
        """Pull a fresh frame from the source and re-read its labels."""
        frame = self._capture()
        self._last_frame = frame
        if frame is None:
            self._last_regions = []
            return None
        # Read once per frame: the region list is reused by every query in the
        # turn, so a multi-step plan does not re-run the encoder per step.
        try:
            self._last_regions = list(self.reader.read_regions(frame, floor=self.floor))
        except Exception:
            self._last_regions = []
        return frame

    # ------------------------------------------------------------------ #
    # Perception protocol
    # ------------------------------------------------------------------ #
    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        frame = self.grab()
        if frame is None:
            return ScreenObservation(description="no screen frame available")

        labels = self._labels()
        present: Optional[bool] = None
        if query:
            hit = self._match(query)
            present = hit is not None

        description = self._describe(labels)
        return ScreenObservation(
            description=description,
            target_present=present,
            hotspot_count=len(self._last_regions),
        )

    # ------------------------------------------------------------------ #
    # kernel evidence
    # ------------------------------------------------------------------ #
    def grounding_evidence(self, query: Optional[str]):
        """Ground a spoken word to a point, or return None when it is absent.

        `None` is the honest answer for "not on screen" and for "outside the
        closed vocabulary": the kernel then denies with "no candidate point"
        rather than receiving a fabricated centre.
        """
        if not query:
            return None
        if self._last_frame is None:
            self.grab()
        if self._last_frame is None:
            return None

        hit = self._match(query)
        if hit is None:
            return None

        frame = self._last_frame
        h, w = frame.shape[:2]
        box = hit.box
        cx = ((box.x0 + box.x1) / 2.0) / max(1, w)
        cy = ((box.y0 + box.y1) / 2.0) / max(1, h)

        from avcs.control.capability import GroundingEvidence

        return GroundingEvidence(
            candidate_point_norm=(float(cx), float(cy)),
            peak_probability=float(hit.readout.confidence),
            target_present=True,
            authorized=False,
            active_query=query,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def regions(self) -> List[object]:
        """The regions read from the most recent frame (for a GUI overlay)."""
        return list(self._last_regions)

    def _labels(self) -> List[str]:
        return [r.label for r in self._last_regions]

    def _match(self, query: str):
        """Best region whose read label equals `query`, or None.

        Matching is on the *read* label, so a control is found by what it looks
        like, not by a character transcription. The query must also be inside the
        closed vocabulary: a word the model was never trained on has no prototype
        and cannot be grounded.
        """
        want = query.strip().lower()
        if not want:
            return None
        try:
            vocab = {lbl.lower() for lbl in self.reader.labels}
        except Exception:
            vocab = set()
        if vocab and want not in vocab:
            return None

        best = None
        for r in self._last_regions:
            if r.label.strip().lower() != want:
                continue
            if not r.known and r.readout.confidence < self.label_floor:
                continue
            if best is None or r.readout.confidence > best.readout.confidence:
                best = r
        return best

    @staticmethod
    def _describe(labels: Sequence[str]) -> str:
        named = sorted({lbl for lbl in labels if lbl and lbl != "?"})
        if not named:
            return "screen with no recognised controls"
        return "controls on screen: " + ", ".join(named)

    def close(self) -> None:
        closer = getattr(self._eye, "close", None)
        if callable(closer):
            closer()
