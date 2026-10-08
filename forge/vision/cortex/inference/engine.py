"""Real-Time Inference Engine for AVCS.

Provides an always-on visual awareness loop running on a dedicated thread,
feeding continuous screen state updates at native display rates.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

from forge.vision.capture import open_capture
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState


class AVCSInferenceEngine:
    """Always-on real-time perception engine."""

    def __init__(
        self,
        cortex: Optional[ArtificialVisualCortex] = None,
        source: Optional[object] = None,
        device: Optional[str] = None,
        fps_target: int = 30,
        on_state: Optional[Callable[[CortexState], None]] = None,
        on_surprise: Optional[Callable[[CortexState], None]] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cortex = (cortex or ArtificialVisualCortex()).to(self.device).eval()
        self.source = source or open_capture()
        self.fps_target = fps_target
        self.interval = 1.0 / max(fps_target, 1)

        self.on_state = on_state
        self.on_surprise = on_surprise

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_state: Optional[CortexState] = None

    @property
    def latest_state(self) -> Optional[CortexState]:
        with self._lock:
            return self._latest_state

    def step(self, frame: Optional[np.ndarray] = None) -> CortexState:
        """Executes a single perception step."""
        if frame is None:
            frame = self.source.grab()

        state = self.cortex.perceive(frame, device=self.device)

        with self._lock:
            self._latest_state = state

        if self.on_state:
            self.on_state(state)

        if self.on_surprise and (state.surprise_score > 0.5 or state.stability_state == "surprise_event"):
            self.on_surprise(state)

        return state

    def start(self) -> None:
        """Starts asynchronous perception thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops perception thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                self.step()
            except Exception as e:
                print(f"[AVCS] Error in inference loop: {e}")

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

