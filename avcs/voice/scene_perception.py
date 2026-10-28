"""Scene perception for the assistant: pixels in, a sentence out.

This is the "eyes" the assistant is supposed to have. It is *not* a control
detector and it does not click anything. It answers one question for the brain:
"what is on the screen right now?", as short factual text.

The chain is the repo's own pure-pixel vision stack, used end to end:

    frame -> forge.vision.reflex.ReflexPerception -> typed SceneState
          -> forge.vision.describer                  -> one plain sentence

Everything it reports is derived from luminance, edges, motion and
frame-to-frame difference. There is no OCR, no accessibility tree, and no
coordinate crosses the System 1 -> System 2 boundary: `ScreenObservation`
carries a description and coarse flags only, which is the boundary
``docs/AVCS_ENGINEERING_RULES.md`` requires.

Two deliberate properties:

- **It never names a control label.** Reading "Save"/"Cancel" is the lexicon's
  job (`avcs/voice/lexicon_perception.py`) and that path is what grounds a
  *click*. This adapter is for an assistant that *looks and talks*, so it stays
  on scene-level facts (tone, structure, motion, change, overlay) and reports no
  target. `observe(query)` accepts a query for protocol compatibility and
  ignores it: this layer has nothing to say about a named control.
- **It never authorizes.** There is no `grounding_evidence`, so the kernel can
  only ever refuse a mutating op. A talking assistant with these eyes cannot
  click, which is the intended default.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from avcs.voice.assistant import ScreenObservation

# A frame source returns the current screen as an HxWx3 uint8 array, or None.
FrameSource = Callable[[], Optional[np.ndarray]]


class ScenePerception:
    """Perception for a talking assistant: describes the screen, names no target.

    Implements the `Perception` protocol (`observe`, `frame`). It deliberately
    does **not** implement `grounding_evidence`, so an assistant wired to it can
    never authorize a mutating action -- there is no candidate point to gate on,
    and that is the honest outcome rather than a fake centre.
    """

    def __init__(
        self,
        frame_source: Optional[object] = None,
        *,
        source: str = "screen",
        rows: int = 16,
        cols: int = 24,
        describer: Optional[object] = None,
        monitor: int = 1,
    ) -> None:
        self._frame_source = frame_source
        self._source_name = source
        self._monitor = monitor
        self._capture: Optional[object] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_state: Any = None
        self._prev_state: Any = None
        self._last_desc: str = ""
        self._lock = threading.Lock()
        self._tick = 0

        from forge.vision.reflex import ReflexPerception

        self.reflex = ReflexPerception(rows=rows, cols=cols)
        if describer is None:
            from forge.vision.describer import HeuristicDescriber

            describer = HeuristicDescriber()
        self.describer = describer

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #
    def _open_capture(self):
        if self._capture is not None:
            return self._capture
        from forge.vision.capture import open_capture

        self._capture = open_capture(
            prefer_real=self._source_name != "synthetic",
            monitor=self._monitor,
            allow_synthetic_fallback=True,
        )
        return self._capture

    def grab(self) -> Optional[np.ndarray]:
        """Return the newest frame, or None when there is no screen to read."""
        if self._frame_source is not None:
            src = self._frame_source
            grab = getattr(src, "capture", None) or getattr(src, "grab", None)
            if callable(grab):
                return grab()
            if callable(src):
                return src()
            return None
        try:
            cap = self._open_capture()
            return cap.grab()
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Perception protocol
    # ------------------------------------------------------------------ #
    def frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        """Perceive one frame and describe it.

        A perception error becomes an honest description rather than an
        exception, because the brain must always have *something* to reason
        about and the page must always have something to show.
        """
        with self._lock:
            try:
                frame = self.grab()
            except Exception as exc:
                return ScreenObservation(description=f"screen unavailable ({exc})")
            if frame is None:
                return ScreenObservation(
                    description="no screen available (no frame could be captured)"
                )

            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            self._last_frame = frame

            self._tick += 1
            state = self.reflex.perceive(frame, frame_index=self._tick, timestamp=time.time())
            events = []
            try:
                from forge.vision.reflex import events_from_states

                events = events_from_states(self._prev_state, state)
            except Exception:
                events = []
            self._prev_state = state
            self._last_state = state

            try:
                description = self.describer.describe(state, None, events)
            except Exception as exc:
                description = f"screen present but could not be described ({exc})"
            self._last_desc = description

            stability = "stable" if not state.scene_changed and state.peak_motion <= 0.10 else "changing"
            return ScreenObservation(
                description=description,
                target_present=None,
                stability=stability,
                has_modal=bool(state.overlay_suspected),
                active_window="",
                hotspot_count=len(state.regions),
            )

    # ------------------------------------------------------------------ #
    # Introspection for the page
    # ------------------------------------------------------------------ #
    def state_summary(self) -> Dict[str, Any]:
        """The typed state, for a UI panel. Coordinates stay in tile units."""
        st = self._last_state
        if st is None:
            return {}
        return {
            "brightness": round(float(st.brightness), 3),
            "contrast": round(float(st.contrast), 3),
            "change_fraction": round(float(st.change_fraction), 4),
            "peak_motion": round(float(st.peak_motion), 4),
            "novelty": round(float(st.novelty), 3),
            "overlay_suspected": bool(st.overlay_suspected),
            "scene_changed": bool(st.scene_changed),
            "regions": len(st.regions),
            "luma_grid": [
                [round(float(v), 3) for v in row] for row in st.luma_grid
            ],
        }

    @property
    def description(self) -> str:
        return self._last_desc

    def close(self) -> None:
        cap = self._capture
        close = getattr(cap, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._capture = None
