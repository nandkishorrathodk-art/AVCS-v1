"""Capture and replay adapters kept separate from the neural model."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np


class NpzSequenceReplaySource:
    """Paced replay from one validated recorded sequence file."""

    def __init__(
        self,
        sequence_path: str | Path,
        *,
        playback_speed: float = 1.0,
        realtime: bool = True,
    ) -> None:
        if playback_speed <= 0:
            raise ValueError("playback_speed must be positive.")
        self.path = Path(sequence_path)
        self.playback_speed = playback_speed
        self.realtime = realtime
        self._arrays = None
        self._index = 0
        self._start: float | None = None
        self._timestamps: np.ndarray | None = None

    def open(self) -> NpzSequenceReplaySource:
        self.close()
        self._arrays = np.load(self.path, allow_pickle=False)
        frames = self._arrays.get("frames")
        timestamps = self._arrays.get("timestamps")
        if (
            frames is None or frames.ndim != 4 or frames.shape[-1] != 3
            or frames.dtype != np.uint8
            or timestamps is None or timestamps.shape != (frames.shape[0],)
        ):
            self.close()
            raise ValueError("replay file must contain uint8 RGB frames and timestamps.")
        if not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
            self.close()
            raise ValueError("replay timestamps must be finite and strictly increasing.")
        self._timestamps = timestamps
        self._index = 0
        self._start = time.monotonic()
        return self

    def grab(self) -> np.ndarray:
        if self._arrays is None or self._timestamps is None or self._start is None:
            raise RuntimeError("replay source is not open.")
        if self._index >= self._timestamps.shape[0]:
            raise EOFError("recording replay is complete.")
        if self.realtime:
            target = self._start + (
                float(self._timestamps[self._index] - self._timestamps[0])
                / self.playback_speed
            )
            time.sleep(max(0.0, target - time.monotonic()))
        frame = self._arrays["frames"][self._index].copy()
        self._index += 1
        return frame

    def close(self) -> None:
        if self._arrays is not None:
            self._arrays.close()
        self._arrays = None
        self._timestamps = None
        self._start = None
        self._index = 0


class WindowsDesktopCaptureSource:
    """Opt-in local Windows desktop capture; captured pixels are not persisted."""

    def __init__(self, *, confirm_screen_capture: bool = False) -> None:
        if not confirm_screen_capture:
            raise PermissionError(
                "Live desktop capture requires confirm_screen_capture=True after "
                "the operator closes private windows."
            )
        self._adapter = None

    def open(self) -> WindowsDesktopCaptureSource:
        import sys

        if sys.platform != "win32":
            raise OSError("Windows GDI desktop capture is only available on Windows.")
        from avcs.runtime.capture_adapter import Win32GDICaptureAdapter

        self._adapter = Win32GDICaptureAdapter().open()
        return self

    def grab(self) -> np.ndarray:
        if self._adapter is None:
            raise RuntimeError("desktop capture source is not open.")
        return self._adapter.grab().copy()

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
        self._adapter = None


class SyntheticCaptureSource:
    """Headless synthetic desktop frame generator for testing and CI.

    Generates deterministic in-memory uint8 RGB frames (e.g. moving shapes)
    without requiring physical display or OS capture adapters. Zero disk persistence.
    """

    def __init__(
        self,
        *,
        width: int = 128,
        height: int = 128,
        fps: float = 30.0,
        total_frames: int | None = None,
    ) -> None:
        if min(width, height) < 32:
            raise ValueError("width and height must each be at least 32.")
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self._index = 0
        self._is_open = False

    def open(self) -> SyntheticCaptureSource:
        self._is_open = True
        self._index = 0
        return self

    def grab(self) -> np.ndarray:
        if not self._is_open:
            raise RuntimeError("synthetic capture source is not open.")
        if self.total_frames is not None and self._index >= self.total_frames:
            raise EOFError("synthetic stream has reached total_frames limit.")

        # Generate procedural RGB frame in RAM only (zero disk persistence)
        t = self._index
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = np.linspace(20, 60, self.width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(40, 80, self.height, dtype=np.uint8)[:, None]
        frame[:, :, 2] = 100
        # Moving box pattern
        box_x = int((t * 4) % max(1, self.width - 20))
        box_y = int((t * 2) % max(1, self.height - 20))
        frame[box_y : box_y + 20, box_x : box_x + 20, 0] = 255
        frame[box_y : box_y + 20, box_x : box_x + 20, 1] = 220
        frame[box_y : box_y + 20, box_x : box_x + 20, 2] = 50

        self._index += 1
        return frame

    def close(self) -> None:
        self._is_open = False
        self._index = 0
