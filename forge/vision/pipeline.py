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

import collections
import logging
import queue
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

_LOGGER = logging.getLogger(__name__)


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
        pure_rgb_only: bool = True,
    ) -> None:
        self.capture = capture if capture is not None else open_capture()
        self.perception = ReflexPerception(rows=rows, cols=cols, downscale=downscale)
        self.describer = describer or HeuristicDescriber()
        self.understanding = understanding
        self.describe_on_change = describe_on_change
        self.describe_every_s = describe_every_s
        self.target_fps = target_fps
        self.pure_rgb_only = bool(pure_rgb_only)
        self.desktop_context = None if self.pure_rgb_only else desktop_context

        self.stats = PipelineStats()
        self._latest = LatestFrame()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._callback_queue: queue.Queue = queue.Queue(maxsize=256)
        self._capture_error_count = 0
        self._desc_queue: list[
            tuple[SceneState, Optional[Understanding], list[SceneEvent], np.ndarray]
        ] = []
        self._desc_lock = threading.Lock()
        self._last_desc = 0.0
        self._description_lock = threading.Lock()
        self._last_description_frame_index: Optional[int] = None
        self._last_description_timestamp: Optional[float] = None
        self._last_description_error: Optional[str] = None
        self._on_state: Optional[Callable[[SceneState], None]] = None
        self._on_event: Optional[Callable[[SceneEvent], None]] = None
        self._on_description: Optional[Callable[[str], None]] = None
        self._on_observation: Optional[Callable[[dict], None]] = None
        self._observation_ready = threading.Event()
        self._last_observation_publish = 0.0
        self._recent_events: list[SceneEvent] = []
        self._event_history: collections.deque[SceneEvent] = collections.deque(
            maxlen=20
        )
        self.last_state: Optional[SceneState] = None
        self.last_description: str = ""
        self._last_rgb_frame: Optional[np.ndarray] = None
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

    def on_observation(self, fn: Callable[[dict], None]) -> None:
        """Subscribe to rate-limited, text-only observations on a worker thread.

        The callback receives no frames or visual embeddings. Caption text may
        still contain sensitive information and must be reviewed before sharing.
        """
        self._on_observation = fn
        self._observation_ready.set()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "ScreenVision":
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._reflex_loop, name="reflex", daemon=True),
            threading.Thread(target=self._describe_loop, name="describe", daemon=True),
            threading.Thread(target=self._observation_loop, name="observation", daemon=True),
            threading.Thread(target=self._callback_loop, name="callbacks", daemon=True),
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
                self._capture_error_count += 1
                if self._capture_error_count == 1 or self._capture_error_count % 100 == 0:
                    _LOGGER.exception(
                        "AVCS screen capture failed; retrying (failure %d)",
                        self._capture_error_count,
                    )
                time.sleep(0.05)
                continue
            self._latest.put(frame)

    def _enqueue_callback(self, callback: Callable, value: object, label: str) -> None:
        try:
            self._callback_queue.put_nowait((callback, value, label))
        except queue.Full:
            _LOGGER.error(
                "AVCS %s subscriber queue is full; dropping callback to protect reflex timing",
                label,
            )

    def _callback_loop(self) -> None:
        while not self._stop.is_set() or not self._callback_queue.empty():
            try:
                callback, value, label = self._callback_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                callback(value)
            except Exception:
                _LOGGER.exception("AVCS %s subscriber failed", label)
            finally:
                self._callback_queue.task_done()

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
                    self._enqueue_callback(self._on_event, ev, "event")

            self.last_state = state
            with self._analysis_lock:
                self._event_history.extend(events)
                self._latest_analysis = (state, und, list(events))
                self._analysis_ready.set()
            self._observation_ready.set()
            if self._on_state:
                self._enqueue_callback(self._on_state, state, "state")

            interesting = state.scene_changed or state.overlay_suspected or state.novelty > 0.35
            due = time.time() - self._last_desc >= self.describe_every_s
            if (self.describe_on_change and interesting) or due:
                with self._desc_lock:
                    self._desc_queue.append((state, und, events, frame.copy()))
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
            for state, und, events, frame in batch:
                try:
                    describe_frame = getattr(self.describer, "describe_frame", None)
                    if callable(describe_frame):
                        text = describe_frame(frame, state, und, events)
                    else:
                        text = self.describer.describe(state, und, events)
                except Exception as exc:  # pragma: no cover - backend-specific
                    with self._description_lock:
                        self._last_description_error = str(exc)
                        self._last_description_frame_index = state.frame_index
                        self._last_description_timestamp = state.timestamp
                    self._last_desc = time.time()
                    continue
                with self._description_lock:
                    self.last_description = text
                    self._last_description_frame_index = state.frame_index
                    self._last_description_timestamp = state.timestamp
                    self._last_description_error = None
                self.stats.captions += 1
                self._last_desc = time.time()
                self._observation_ready.set()
                if self._on_description:
                    self._enqueue_callback(
                        self._on_description, text, "description"
                    )

    def _observation_loop(self) -> None:
        min_interval_s = 0.5
        while not self._stop.is_set():
            self._observation_ready.wait(timeout=0.1)
            self._observation_ready.clear()
            if self._stop.is_set() or self._on_observation is None:
                continue

            wait_s = min_interval_s - (
                time.monotonic() - self._last_observation_publish
            )
            if wait_s > 0 and self._stop.wait(wait_s):
                break

            callback = self._on_observation
            if callback is None:
                continue
            try:
                observation = self.observe(wait_timeout_s=0.0)
                callback(observation)
            except Exception:
                _LOGGER.exception("AVCS observation subscriber failed")
            self._last_observation_publish = time.monotonic()

    # -- synchronous helpers (tests, single-frame use) ----------------------

    def step(self) -> SceneState:
        """Process exactly one frame synchronously and return its state."""
        frame = self.capture.grab()  # type: ignore[attr-defined]
        self._last_rgb_frame = frame.copy()
        previous_state = self.last_state
        tick_start = time.perf_counter()
        state = self.perception.perceive(frame, frame_index=self.stats.frames)
        self.stats.frames += 1
        self.stats.reflex_latencies.append(state.latency_ms)
        und: Optional[Understanding] = None
        if self.understanding is not None:
            und = self.understanding.understand(state)
        self.stats.tick_latencies.append((time.perf_counter() - tick_start) * 1000.0)
        events = events_from_states(previous_state, state)
        self._recent_events = events
        self.last_state = state
        with self._analysis_lock:
            self._event_history.extend(events)
            self._latest_analysis = (state, und, list(events))
            self._analysis_ready.set()
        return state

    def _semantic_caption(
        self,
        state: SceneState,
        understanding: Optional[Understanding],
        events: list[SceneEvent],
    ) -> dict:
        """Return local VLM output with frame provenance and explicit uncertainty."""
        if getattr(self.describer, "name", None) != "vlm":
            return {
                "text": None,
                "source": None,
                "frame_sequence": None,
                "status": "disabled",
                "verified": False,
            }

        with self._description_lock:
            description_frame_index = self._last_description_frame_index
            description_timestamp = self._last_description_timestamp
            description_error = self._last_description_error
            caption = self.last_description
        if (
            not any(thread.is_alive() for thread in self._threads)
            and description_frame_index != state.frame_index
            and self._last_rgb_frame is not None
        ):
            try:
                caption = self.describer.describe_frame(
                    self._last_rgb_frame.copy(), state, understanding, events
                )
            except Exception as exc:
                description_error = str(exc)
                description_frame_index = state.frame_index
                description_timestamp = state.timestamp
                with self._description_lock:
                    self._last_description_error = description_error
                    self._last_description_frame_index = description_frame_index
                    self._last_description_timestamp = description_timestamp
                self._last_desc = time.time()
            else:
                description_frame_index = state.frame_index
                description_timestamp = state.timestamp
                description_error = None
                with self._description_lock:
                    self.last_description = caption
                    self._last_description_frame_index = description_frame_index
                    self._last_description_timestamp = description_timestamp
                    self._last_description_error = None
                self.stats.captions += 1
                self._last_desc = time.time()

        if description_error and description_frame_index == state.frame_index:
            status = "error"
            caption = None
        elif description_frame_index is None:
            status = "pending"
            caption = None
        elif description_frame_index == state.frame_index:
            status = "ready"
        else:
            status = "stale"

        return {
            "text": caption,
            "source": "local_vision_model",
            "frame_sequence": description_frame_index,
            "age_ms": (
                round(max(0.0, time.time() - description_timestamp) * 1000.0, 1)
                if description_timestamp is not None
                else None
            ),
            "status": status,
            "verified": False,
            "error": description_error if status == "error" else None,
        }

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
        with self._analysis_lock:
            recent_events = [event.to_dict() for event in self._event_history]
        age_s = max(0.0, time.time() - state.timestamp)
        description = self.describer.describe(state, understanding, events)
        desktop = (
            self.desktop_context.snapshot()
            if (not getattr(self, "pure_rgb_only", True) and self.desktop_context is not None)
            else {"status": "not_configured", "source": None}
        )

        return {
            "schema_version": "avcs.desktop_observation.v1",
            "payload_policy": "text_only_no_frame_or_embedding",
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
                "recent_events": recent_events,
                "local_semantic_caption": self._semantic_caption(
                    state, understanding, events
                ),
                "limitations": [
                    (
                        "The reflex path does not transcribe visible text; an optional "
                        "local VLM caption may imperfectly report it."
                    ),
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
