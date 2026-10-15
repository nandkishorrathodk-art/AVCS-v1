"""Real-time screen vision pipeline.

Thread model:

    capture thread  ->  latest frame (1-slot mailbox)
    reflex thread   ->  perceive() every tick, ~40-80 ms budget
                        -> typed SceneState + calibrated confidence
                        -> events on change
    describe thread ->  (optional, slow) caption on change only

Only the reflex thread runs on the hot path.  Capture never blocks perception
(it drops stale frames rather than queueing them), and description never blocks
either -- it consumes the newest interesting state and is allowed to lag.  That
separation is what lets a 27 fps reflex loop and a 1 fps captioner share one
CPU without one starving the other.

Every tick is non-generative.  Text is produced only from typed state, or by an
optional VLM invoked at a deliberately low rate.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from forge.vision.capture import open_capture
from forge.vision.describer import Describer, HeuristicDescriber
from forge.vision.desktop_context import DesktopContextProvider
from forge.vision.encoder import SceneUnderstandingModel, Understanding
from forge.vision.reflex import ReflexPerception, events_from_states
from forge.vision.scene import SceneEvent, SceneState


@dataclass
class PipelineStats:
    frames: int = 0
    events: int = 0
    captions: int = 0
    reflex_latencies: list[float] = field(default_factory=list)
    tick_latencies: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def fps(self) -> float:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return self.frames / elapsed

    def _pct(self, q: float, series: list[float] | None = None) -> float:
        s = sorted(self.reflex_latencies if series is None else series)
        if not s:
            return 0.0
        idx = min(len(s) - 1, int(q * len(s)))
        return s[idx]

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "events": self.events,
            "captions": self.captions,
            "fps": round(self.fps, 2),
            "reflex_p50_ms": round(self._pct(0.50), 2),
            "reflex_p95_ms": round(self._pct(0.95), 2),
            # The number that matters for the real-time budget: reflex *plus*
            # the understanding forward pass, i.e. the whole System-1 tick.
            "tick_p50_ms": round(self._pct(0.50, self.tick_latencies), 2),
            "tick_p95_ms": round(self._pct(0.95, self.tick_latencies), 2),
        }


class LatestFrame:
    """A one-slot mailbox: producers overwrite, consumers take the newest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._seq += 1

    def take(self) -> Optional[tuple[np.ndarray, int]]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._seq


class ScreenVision:
    """The always-on visual awareness loop."""

    def __init__(
        self,
        capture: Optional[object] = None,
        describer: Optional[Describer] = None,
        understanding: Optional[SceneUnderstandingModel] = None,
        rows: int = 16,
        cols: int = 24,
        downscale: int = 4,
        describe_on_change: bool = True,
        describe_every_s: float = 2.0,
        target_fps: Optional[float] = None,
        desktop_context: Optional[DesktopContextProvider] = None,
    ) -> None:
        self.capture = capture if capture is not None else open_capture()
        self.perception = ReflexPerception(rows=rows, cols=cols, downscale=downscale)
        self.describer = describer or HeuristicDescriber()
        self.understanding = understanding
        self.describe_on_change = describe_on_change
        self.describe_every_s = describe_every_s
        self.target_fps = target_fps
        self.desktop_context = desktop_context

        self.stats = PipelineStats()
        self._latest = LatestFrame()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._desc_queue: list[tuple[SceneState, Optional[Understanding], list[SceneEvent]]] = []
        self._desc_lock = threading.Lock()
        self._last_desc = 0.0
        self._on_state: Optional[Callable[[SceneState], None]] = None
        self._on_event: Optional[Callable[[SceneEvent], None]] = None
        self._on_description: Optional[Callable[[str], None]] = None
        self._recent_events: list[SceneEvent] = []
        self.last_state: Optional[SceneState] = None
        self.last_description: str = ""
        self._analysis_lock = threading.Lock()
        self._latest_analysis: Optional[
            tuple[SceneState, Optional[Understanding], list[SceneEvent]]
        ] = None
        self._analysis_ready = threading.Event()

    # -- callbacks ----------------------------------------------------------

    def on_state(self, fn: Callable[[SceneState], None]) -> None:
        self._on_state = fn

    def on_event(self, fn: Callable[[SceneEvent], None]) -> None:
        self._on_event = fn

    def on_description(self, fn: Callable[[str], None]) -> None:
        self._on_description = fn

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "ScreenVision":
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._reflex_loop, name="reflex", daemon=True),
            threading.Thread(target=self._describe_loop, name="describe", daemon=True),
        ]
        for t in self._threads:
            t.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        close = getattr(self.capture, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "ScreenVision":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- loops --------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.capture.grab()  # type: ignore[attr-defined]
            except Exception:
                time.sleep(0.05)
                continue
            self._latest.put(frame)

    def _reflex_loop(self) -> None:
        prev: Optional[SceneState] = None
        idx = 0
        last_seq = -1
        min_period = 1.0 / self.target_fps if self.target_fps else 0.0
        last_tick = 0.0
        while not self._stop.is_set():
            item = self._latest.take()
            if item is None:
                time.sleep(0.002)
                continue
            frame, seq = item
            # Never perceive the same captured frame twice: on a fast machine
            # the reflex loop would otherwise spin on a stale buffer and report
            # a frame rate the capture never produced.
            if seq == last_seq:
                time.sleep(0.002)
                continue
            if min_period:
                wait = min_period - (time.perf_counter() - last_tick)
                if wait > 0:
                    time.sleep(wait)
            last_tick = time.perf_counter()
            last_seq = seq
            tick_start = time.perf_counter()
            state = self.perception.perceive(frame, frame_index=idx)
            idx += 1
            self.stats.frames += 1
            self.stats.reflex_latencies.append(state.latency_ms)
            if len(self.stats.reflex_latencies) > 2000:
                del self.stats.reflex_latencies[:1000]

            und: Optional[Understanding] = None
            if self.understanding is not None:
                und = self.understanding.understand(state)
            # Full System-1 tick = reflex + understanding forward pass.
            self.stats.tick_latencies.append((time.perf_counter() - tick_start) * 1000.0)
            if len(self.stats.tick_latencies) > 2000:
                del self.stats.tick_latencies[:1000]

            events = events_from_states(prev, state)
            self._recent_events = events
            for ev in events:
                self.stats.events += 1
                if self._on_event:
                    self._on_event(ev)

            self.last_state = state
            with self._analysis_lock:
                self._latest_analysis = (state, und, list(events))
                self._analysis_ready.set()
            if self._on_state:
                self._on_state(state)

            interesting = state.scene_changed or state.overlay_suspected or state.novelty > 0.35
            due = time.time() - self._last_desc >= self.describe_every_s
            if (self.describe_on_change and interesting) or due:
                with self._desc_lock:
                    self._desc_queue.append((state, und, events))
                    if len(self._desc_queue) > 1:
                        # Keep only the freshest request; description may lag.
                        del self._desc_queue[:-1]

            prev = state

    def _describe_loop(self) -> None:
        while not self._stop.is_set():
            with self._desc_lock:
                batch = self._desc_queue
                self._desc_queue = []
            if not batch:
                time.sleep(0.02)
                continue
            for state, und, events in batch:
                try:
                    text = self.describer.describe(state, und, events)
                except Exception as exc:  # pragma: no cover - backend-specific
                    text = f"(describer error: {exc})"
                self.last_description = text
                self.stats.captions += 1
                self._last_desc = time.time()
                if self._on_description:
                    self._on_description(text)

    # -- synchronous helpers (tests, single-frame use) ----------------------

    def step(self) -> SceneState:
        """Process exactly one frame synchronously and return its state."""
        frame = self.capture.grab()  # type: ignore[attr-defined]
        tick_start = time.perf_counter()
        state = self.perception.perceive(frame, frame_index=self.stats.frames)
        self.stats.frames += 1
        self.stats.reflex_latencies.append(state.latency_ms)
        und: Optional[Understanding] = None
        if self.understanding is not None:
            und = self.understanding.understand(state)
        self.stats.tick_latencies.append((time.perf_counter() - tick_start) * 1000.0)
        self.last_state = state
        self._recent_events = []
        with self._analysis_lock:
            self._latest_analysis = (state, und, [])
            self._analysis_ready.set()
        return state

    def observe(self, wait_timeout_s: float = 1.0, stale_after_s: float = 2.0) -> dict:
        """Return the latest visual evidence and optional OS context for an LLM.

        This is an observation-only API: it does not dispatch mouse or keyboard
        input. Window data is included only when a context provider is supplied.
        """
        if wait_timeout_s < 0 or stale_after_s < 0:
            raise ValueError("Observation timeouts must be non-negative.")

        with self._analysis_lock:
            analysis = self._latest_analysis

        if analysis is None:
            if any(thread.is_alive() for thread in self._threads):
                if not self._analysis_ready.wait(wait_timeout_s):
                    raise TimeoutError(
                        "ScreenVision did not produce an observation before timeout."
                    )
                with self._analysis_lock:
                    analysis = self._latest_analysis
            else:
                self.step()
                with self._analysis_lock:
                    analysis = self._latest_analysis

        if analysis is None:
            raise RuntimeError("ScreenVision has no visual observation to return.")

        state, understanding, events = analysis
        age_s = max(0.0, time.time() - state.timestamp)
        description = self.describer.describe(state, understanding, events)
        desktop = (
            self.desktop_context.snapshot()
            if self.desktop_context is not None
            else {"status": "not_configured", "source": None}
        )

        return {
            "schema_version": "avcs.desktop_observation.v1",
            "frame": {
                "sequence": state.frame_index,
                "captured_at_unix_s": state.timestamp,
                "age_ms": round(age_s * 1000.0, 1),
                "freshness": "fresh" if age_s <= stale_after_s else "stale",
                "capture_backend": type(self.capture).__name__,
            },
            "visual_perception": {
                "source": "avcs_pixels",
                "description": description,
                "description_kind": (
                    "learned_scene_prediction"
                    if understanding is not None
                    else "pixel_heuristic"
                ),
                "signals": {
                    "frame_size": [state.width, state.height],
                    "brightness": round(state.brightness, 3),
                    "contrast": round(state.contrast, 3),
                    "global_motion": round(state.global_motion, 4),
                    "peak_motion": round(state.peak_motion, 4),
                    "changed_area_fraction": round(state.change_fraction, 4),
                    "overlay_suspected": state.overlay_suspected,
                    "scene_changed": state.scene_changed,
                    "regions": [region.to_dict() for region in state.regions],
                    "understanding": understanding.to_dict() if understanding else None,
                },
                "events": [event.to_dict() for event in events],
                "limitations": [
                    "AVCS does not transcribe visible text.",
                    "Motion does not identify whether content is a game or a video.",
                    "Scene predictions are not verified facts or task-completion "
                    "evidence.",
                ],
            },
            "desktop_context": desktop,
            "interpretation_guidance": (
                "Describe only the supplied evidence. Keep OS window metadata separate "
                "from pixel-derived estimates. Qualify uncertain activity; motion "
                "alone "
                "cannot distinguish a game from a video."
            ),
            "motor_action_authorized": False,
        }
