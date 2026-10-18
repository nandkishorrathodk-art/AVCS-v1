"""Run the trained model continuously against an explicitly approved live screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from avcs.streaming_v1.data import sha256_file
from avcs.streaming_v1.runtime import StreamingObserver
from avcs.streaming_v1.sources import WindowsDesktopCaptureSource
from avcs.streaming_v1.training import load_streaming_checkpoint


def _process_memory_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def run_live_observation(
    checkpoint: str | Path,
    *,
    duration_seconds: float,
    target_hz: float,
    confirm_screen_capture: bool,
    input_size: int = 128,
) -> dict[str, Any]:
    if not confirm_screen_capture:
        raise PermissionError(
            "Live capture is opt-in. Close private windows and pass "
            "confirm_screen_capture=True before screen pixels are read."
        )
    if duration_seconds <= 0 or target_hz <= 0:
        raise ValueError("duration_seconds and target_hz must be positive.")
    model, _ = load_streaming_checkpoint(checkpoint, device="cpu")
    observer = StreamingObserver(model, input_size=(input_size, input_size))
    source = WindowsDesktopCaptureSource(confirm_screen_capture=True)
    process_start = time.process_time()
    wall_start = time.monotonic()
    observer.start(source, target_hz=target_hz)
    try:
        time.sleep(duration_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
    wall_elapsed = time.monotonic() - wall_start
    cpu_elapsed = time.process_time() - process_start
    updates = observer.updates
    model_latencies = [item.model_latency_ms for item in updates]
    e2e_latencies = [item.capture_to_update_ms for item in updates]
    intervals = np.diff([item.timestamp for item in updates]) if len(updates) > 1 else np.array([])
    report = {
        "schema": "avcs.streaming_v1.live_benchmark",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint)),
        "duration_seconds": wall_elapsed,
        "target_capture_hz": target_hz,
        "captured_frames": observer.captured_count,
        "model_updates": observer.processed_count,
        "capture_rate_hz": observer.captured_count / wall_elapsed,
        "model_update_rate_hz": observer.processed_count / wall_elapsed,
        "configured_capture_rate_hz": target_hz,
        "queue_replaced_frames": observer.dropped_count,
        "duplicate_frames": observer.duplicate_count,
        "out_of_order_frames": observer.out_of_order_count,
        "stale_frames": observer.stale_count,
        "capture_recoveries": observer.capture_recovery_count,
        "capture_to_update_latency_ms": {
            "p50": float(np.percentile(e2e_latencies, 50)) if e2e_latencies else None,
            "p95": float(np.percentile(e2e_latencies, 95)) if e2e_latencies else None,
        },
        "model_only_latency_ms": {
            "p50": float(np.percentile(model_latencies, 50)) if model_latencies else None,
            "p95": float(np.percentile(model_latencies, 95)) if model_latencies else None,
            "fps": 1000.0 / statistics.fmean(model_latencies) if model_latencies else None,
        },
        "observed_input_interval_ms": {
            "p50": float(np.percentile(intervals, 50) * 1000.0) if intervals.size else None,
            "p95": float(np.percentile(intervals, 95) * 1000.0) if intervals.size else None,
        },
        "process_cpu_percent_of_one_core": 100.0 * cpu_elapsed / max(wall_elapsed, 1e-9),
        "working_set_bytes": _process_memory_bytes(),
        "state_updates_retained": len(updates),
        "last_error": observer.last_error,
        "raw_frames_persisted": False,
        "warning": "One local screen session is an operational benchmark, not independent generalization evidence.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--confirm-screen-capture", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--input-size", type=int, default=128)
    args = parser.parse_args()
    report = run_live_observation(
        args.checkpoint,
        duration_seconds=args.duration_seconds,
        target_hz=args.target_hz,
        confirm_screen_capture=args.confirm_screen_capture,
        input_size=args.input_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
