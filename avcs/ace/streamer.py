"""Background Real-Time Screen Streamer for AVCS Autopilot.

Maintains a continuous lock-free in-memory visual frame stream at 30-60 FPS.
Decouples visual capture from LLM inference latency so observations are instant (<0.1ms).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avcs.ace.computer import Observation, WindowsComputer

logger = logging.getLogger("avcs.ace.streamer")


class ScreenStreamer:
    """Continuous background desktop stream grabber."""

    def __init__(
        self,
        computer: WindowsComputer,
        target_fps: float = 30.0,
        image_format: str = "jpeg",
    ) -> None:
        self.computer = computer
        self.target_fps = max(1.0, min(120.0, target_fps))
        self.image_format = image_format
        self._interval = 1.0 / self.target_fps

        self._latest_obs: Optional[Observation] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps_measured = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self, startup_timeout: float = 1.5) -> None:
        """Starts the background continuous screen capture thread."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="ScreenStreamerWorker")
        self._thread.start()
        # Wait up to startup_timeout (default 1.5s) for the first frame to populate
        t_start = time.time()
        while self.get_latest_observation() is None and (time.time() - t_start) < startup_timeout:
            time.sleep(0.005)
        logger.info(f"ScreenStreamer live: continuous capture at {self.target_fps:.0f} FPS ({self.image_format})")

    def stop(self) -> None:
        """Stops the streaming thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        logger.info("ScreenStreamer stopped.")

    def get_latest_observation(self) -> Optional[Observation]:
        """Returns the most recent in-memory observation instantaneously (<0.1ms)."""
        with self._lock:
            return self._latest_obs

    def _worker_loop(self) -> None:
        """Continuous background capture loop."""
        last_time = 0.0
        fps_timer = time.time()
        frames_in_second = 0

        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - last_time

            if elapsed >= self._interval:
                last_time = now
                try:
                    obs = self.computer.observe(format=self.image_format)
                    with self._lock:
                        self._latest_obs = obs
                    self._frame_count += 1
                    frames_in_second += 1
                except Exception as e:
                    logger.debug(f"Streaming frame grab error: {e}")

                if now - fps_timer >= 1.0:
                    self._fps_measured = frames_in_second / (now - fps_timer)
                    frames_in_second = 0
                    fps_timer = now

            # Yield CPU briefly to maintain target FPS without 100% core spin
            sleep_time = self._interval - (time.time() - last_time)
            if sleep_time > 0.002:
                time.sleep(sleep_time)
            else:
                time.sleep(0.001)

    def __enter__(self) -> ScreenStreamer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

