"""Unit Tests for Win32FastCapture and ThreadedScreenCapture."""
from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from avcs.capture_adapter import DisplayGeometry, Win32GDICaptureAdapter
from forge.vision.capture import (
    ScreenCapture,
    SyntheticScreen,
    ThreadedScreenCapture,
    Win32FastCapture,
    open_capture,
)


def test_win32_fast_capture_on_available_display():
    """Verify Win32FastCapture returns valid RGB uint8 frames if display is available."""
    if not Win32FastCapture.available():
        pytest.skip("Win32FastCapture not available on this platform/session")

    cap = Win32FastCapture()
    with cap:
        frame = cap.grab()
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.shape[0] > 0 and frame.shape[1] > 0


def test_win32_fast_capture_with_target_size():
    """Verify hardware StretchBlt resizing to explicit target dimensions."""
    if not Win32FastCapture.available():
        pytest.skip("Win32FastCapture not available on this platform/session")

    cap = Win32FastCapture(target_size=(320, 180))
    with cap:
        frame = cap.grab()
        assert frame.shape == (180, 320, 3)
        assert frame.dtype == np.uint8


def test_threaded_screen_capture_with_synthetic():
    """Verify ThreadedScreenCapture runs asynchronously and provides non-blocking grabs."""
    syn = SyntheticScreen(width=160, height=120)
    threaded = ThreadedScreenCapture(source=syn, target_fps=100.0)

    with threaded:
        # Initial grab should be available immediately
        f1 = threaded.grab()
        assert f1.shape == (120, 160, 3)

        # Allow worker thread to produce several frames
        time.sleep(0.08)
        seq1 = threaded.sequence_number
        assert seq1 > 0

        # Grab should be non-blocking (<1ms)
        t0 = time.perf_counter()
        f2 = threaded.grab()
        grab_time = (time.perf_counter() - t0) * 1000.0
        assert grab_time < 5.0
        assert f2.shape == (120, 160, 3)


def test_open_capture_threaded_flag():
    """Verify open_capture correctly wraps sources in ThreadedScreenCapture when requested."""
    cap = open_capture(prefer_real=False, threaded=True)
    assert isinstance(cap, ThreadedScreenCapture)
    with cap:
        frame = cap.grab()
        assert frame.ndim == 3 and frame.shape[2] == 3


def test_gdi_region_capture_copies_only_requested_pixels():
    class Gdi:
        calls = []

        def BitBlt(self, *args):
            self.calls.append(args)
            return 1

        def GdiFlush(self):
            pass

    adapter = Win32GDICaptureAdapter.__new__(Win32GDICaptureAdapter)
    adapter._is_open = True
    adapter._geometry = DisplayGeometry(100, 50, 5, 4, 1.0, True)
    adapter._raw_np = np.zeros((4, 5, 4), dtype=np.uint8)
    adapter._gdi32 = Gdi()
    adapter._h_mem_dc = 1
    adapter._h_screen_dc = 2
    adapter._raw_np[0, 0] = (3, 2, 1, 255)

    frame = adapter.grab_region((101, 51, 104, 53))

    assert frame.shape == (2, 3, 3)
    assert frame[0, 0].tolist() == [1, 2, 3]
    assert adapter._gdi32.calls == [(1, 0, 0, 3, 2, 2, 101, 51, 0x00CC0020)]


def test_gdi_region_capture_rejects_out_of_bounds_rectangle():
    adapter = Win32GDICaptureAdapter.__new__(Win32GDICaptureAdapter)
    adapter._is_open = True
    adapter._geometry = DisplayGeometry(100, 50, 5, 4, 1.0, True)
    adapter._raw_np = np.zeros((4, 5, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="outside"):
        adapter.grab_region((99, 50, 102, 52))
