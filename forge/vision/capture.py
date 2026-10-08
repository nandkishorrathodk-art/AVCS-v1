"""Raw screen capture sources.

Two sources, one interface:

* :class:`ScreenCapture` grabs the real desktop with ``mss`` (fast, ~5-15 ms at
  1080p) and returns a raw ``(H, W, 3)`` uint8 array.
* :class:`SyntheticScreen` renders a scripted "desktop" in memory so the whole
  pipeline can be developed and tested on a headless box with no display.

Both hand the reflex layer nothing but pixels -- there is no other channel.
"""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np


class CaptureError(RuntimeError):
    """Raised when a real screen cannot be grabbed (no display, permissions)."""


class ScreenCapture:
    """Real desktop capture via ``mss``."""

    def __init__(self, monitor: int = 1) -> None:
        self.monitor = monitor
        self._sct = None

    def open(self) -> "ScreenCapture":
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise CaptureError("mss is not installed; `pip install mss`") from exc
        try:
            # mss renamed the factory: `MSS` is current, `mss` is deprecated.
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()
            # Touch the monitor list so a missing display fails here, not later.
            _ = self._sct.monitors[self.monitor]
        except Exception as exc:  # pragma: no cover - depends on host display
            raise CaptureError(f"cannot open display: {exc}") from exc
        return self

    def grab(self) -> np.ndarray:
        if self._sct is None:
            self.open()
        assert self._sct is not None
        shot = self._sct.grab(self._sct.monitors[self.monitor])
        return np.asarray(shot)[:, :, :3].copy()

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            finally:
                self._sct = None

    def __enter__(self) -> "ScreenCapture":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def available() -> bool:
        """True if a real screen can actually be grabbed on this host."""
        try:
            with ScreenCapture() as cap:
                cap.grab()
            return True
        except Exception:
            return False


class SyntheticScreen:
    """A scripted desktop rendered to pixels, for headless development.

    It draws a background, a couple of "windows", and a cursor that moves each
    tick -- enough structure for layout, motion, change and overlay logic to be
    exercised without a display.
    """

    def __init__(self, width: int = 640, height: int = 360, seed: int = 7) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.tick = 0

    def grab(self) -> np.ndarray:
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(24, 72, w, dtype=np.uint8)[None, :]
        img[..., 1] = np.linspace(24, 56, h, dtype=np.uint8)[:, None]
        img[..., 2] = 48

        # Two static "windows".
        img[int(h * 0.15):int(h * 0.55), int(w * 0.10):int(w * 0.45)] = 205
        img[int(h * 0.30):int(h * 0.85), int(w * 0.52):int(w * 0.92)] = 120

        # A sliding "window" large enough for tile-level motion to register.
        wx = int(w * 0.05 + (w * 0.35) * (0.5 + 0.5 * np.sin(self.tick * 0.25)))
        img[int(h * 0.60):int(h * 0.85), wx:wx + int(w * 0.18)] = 235

        # A moving cursor (the thing motion detection must find).
        cx = int(w * 0.5 + 0.35 * w * np.sin(self.tick * 0.35))
        cy = int(h * 0.5 + 0.3 * h * np.cos(self.tick * 0.35))
        r = 6
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        img[y0:y1, x0:x1] = 255

        # A blinking overlay rectangle in the corner on odd ticks.
        if self.tick % 2 == 1:
            img[0:24, w - 120:w] = 250

        self.tick += 1
        return img

    def stream(self, count: int, interval: float = 0.0) -> Iterator[np.ndarray]:
        for _ in range(count):
            yield self.grab()
            if interval:
                time.sleep(interval)


class FrameSource:
    """Wrap any user-supplied frame provider (webcam, video, remote stream).

    ``provider`` must return an ``(H, W, 3)`` uint8 array per call.  This is the
    seam for non-screen sources without the pipeline knowing anything about
    them.
    """

    def __init__(self, provider) -> None:
        self.provider = provider

    def grab(self) -> np.ndarray:
        frame = self.provider()
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8, copy=False)

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def open_capture(prefer_real: bool = True, monitor: int = 1) -> object:
    """Return a real capture if a display exists, else a synthetic one."""
    if prefer_real and ScreenCapture.available():
        return ScreenCapture(monitor=monitor).open()
    return SyntheticScreen()
