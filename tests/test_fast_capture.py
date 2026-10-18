"""Unit Tests for Win32FastCapture and ThreadedScreenCapture."""
from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from avcs.capture_adapter import DisplayGeometry, Win32GDICaptureAdapter
from forge.vision.capture import (
    CaptureError,
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


def test_open_capture_can_fail_closed_without_synthetic_fallback(monkeypatch):
    import forge.vision.capture as capture_module

    monkeypatch.setattr(capture_module.Win32FastCapture, "available", staticmethod(lambda: False))
    monkeypatch.setattr(capture_module.ScreenCapture, "available", staticmethod(lambda: True))

    def fail_open(self):
        raise CaptureError("display unavailable")

    monkeypatch.setattr(capture_module.ScreenCapture, "open", fail_open)
    with pytest.raises(CaptureError, match="display unavailable"):
        open_capture(allow_synthetic_fallback=False)

    assert isinstance(open_capture(allow_synthetic_fallback=True), SyntheticScreen)


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


def test_win32_fast_capture_sequence_monotonicity():
    """Verify grab_with_sequence is strictly monotonic with no stale or backwards frames."""
    if not Win32FastCapture.available():
        pytest.skip("Win32FastCapture not available on this platform/session")

    cap = Win32FastCapture()
    with cap:
        last_seq = 0
        for expected_seq in range(1, 11):
            frame, seq = cap.grab_with_sequence()
            assert seq > last_seq, f"Sequence must strictly increase: {seq} <= {last_seq}"
            assert seq == expected_seq
            assert cap.sequence_number == seq
            assert frame.ndim == 3 and frame.shape[2] == 3
            last_seq = seq


def test_win32_fast_capture_gdi_handles_no_leak():
    """Verify Win32FastCapture releases all GDI and desktop handles with zero leaks."""
    if sys.platform != "win32" or not Win32FastCapture.available():
        pytest.skip("Requires live Windows GDI desktop")

    import ctypes
    user32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32
    h_proc = k32.GetCurrentProcess()

    # Warmup
    with Win32FastCapture() as cap:
        _ = cap.grab()

    gdi_before = user32.GetGuiResources(h_proc, 0)
    for _ in range(25):
        cap = Win32FastCapture()
        cap.open()
        _ = cap.grab()
        cap.close()

    gdi_after = user32.GetGuiResources(h_proc, 0)
    assert gdi_after <= gdi_before, f"GDI handle leak detected: before={gdi_before}, after={gdi_after}"


def test_win32_fast_capture_out_buffer_optimization():
    """Verify grab(out=...) writes directly into preallocated buffer without allocation."""
    if not Win32FastCapture.available():
        pytest.skip("Win32FastCapture not available on this platform/session")

    with Win32FastCapture(target_size=(320, 180)) as cap:
        out_buf = np.zeros((180, 320, 3), dtype=np.uint8)
        frame = cap.grab(out=out_buf)
        assert frame is out_buf
        assert out_buf.shape == (180, 320, 3)
        assert out_buf.dtype == np.uint8


def test_threaded_capture_condition_wait_monotonicity():
    """Verify grab_with_sequence(min_sequence=...) prevents stale frames and guarantees monotonicity."""
    syn = SyntheticScreen(width=160, height=120)
    threaded = ThreadedScreenCapture(source=syn, target_fps=80.0)

    with threaded:
        f1, seq1 = threaded.grab_with_sequence()
        assert seq1 >= 1

        # Waiting for next frame strictly greater than seq1
        f2, seq2 = threaded.grab_with_sequence(min_sequence=seq1, timeout=1.0)
        assert seq2 > seq1, f"Expected monotonic sequence advance: {seq2} <= {seq1}"
        assert not np.array_equal(f1, f2)

        # Immediate grab without min_sequence returns current snapshot without regressing
        f3, seq3 = threaded.grab_with_sequence()
        assert seq3 >= seq2
