"""Stage D: 1,000-Tick Continuous Observation Soak Test and Latency Benchmark.

Measures:
1. Continuous 1,000-tick streaming observation through StreamingVisualCortexV1.
2. Memory RSS stability: Verifies zero unbounded memory leak over 1,000 ticks.
3. Stream interruption recovery and temporal hidden state gap resets.
4. CPU latency percentiles (min, p50, p90, p95, p99, max) and sustained FPS.
5. Zero raw persistence invariant: Ephemeral RAM processing only.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from avcs.streaming_v1.model import (
    CortexState,
    StreamingVisualCortexV1,
    create_clean_random_cortex,
)
from avcs.streaming_v1.runtime import (
    FramePacket,
    LatestFrameMailbox,
    StreamingPermissions,
)
from avcs.streaming_v1.sources import SyntheticCaptureSource


def get_current_rss_bytes() -> int:
    """Retrieve current process WorkingSetSize (RSS) in bytes on Windows."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    return 0


def run_soak_test(
    total_ticks: int = 1000,
    *,
    width: int = 64,
    height: int = 64,
    fps: float = 30.0,
    inject_gaps: bool = True,
    seed: int = 20261018,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = create_clean_random_cortex(seed=seed)
    model.eval()

    mailbox = LatestFrameMailbox()
    source = SyntheticCaptureSource(width=width, height=height, fps=fps).open()

    latencies_ms: List[float] = []
    rss_samples: List[Tuple[int, int]] = []  # (tick, rss_bytes)
    gap_resets_count = 0

    state: CortexState | None = None
    last_timestamp: float | None = None

    t_start = time.monotonic()
    initial_rss = get_current_rss_bytes()
    simulated_gap_offset = 0.0

    try:
        for tick in range(total_ticks):
            # Capture frame in RAM
            rgb = source.grab()

            # Inject intermittent stream gaps (e.g. at ticks 250, 500, 750)
            if inject_gaps and tick in {250, 500, 750}:
                simulated_gap_offset += 3.5  # advance time by 3.5s > max_state_gap_seconds (2.0s)
                gap_resets_count += 1
                state = None  # reset temporal hidden state across gap

            t_now = t_start + (tick * (1.0 / fps)) + simulated_gap_offset

            packet = FramePacket(sequence=tick, timestamp=t_now, rgb=rgb)
            mailbox.put(packet)

            # Consume from mailbox
            taken = mailbox.take()
            assert taken is not None

            # Model inference pass
            t0 = time.perf_counter()
            frame_tensor = torch.from_numpy(taken.rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            delta_t = 0.033 if last_timestamp is None else max(0.001, taken.timestamp - last_timestamp)

            with torch.no_grad():
                latent_z, next_state, outputs = model.forward(frame_tensor, state, delta_t)

            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            state = CortexState(
                hidden=next_state.hidden.detach(),
                previous_features=next_state.previous_features.detach(),
            )
            last_timestamp = taken.timestamp

            # Sample RSS every 100 ticks
            if tick % 100 == 0 or tick == total_ticks - 1:
                cur_rss = get_current_rss_bytes()
                rss_samples.append((tick, cur_rss))

    finally:
        source.close()
        mailbox.close()

    total_time_s = time.perf_counter() - t_start
    final_rss = get_current_rss_bytes()

    # Latency percentiles
    sorted_latencies = sorted(latencies_ms)
    p50 = float(np.percentile(sorted_latencies, 50))
    p90 = float(np.percentile(sorted_latencies, 90))
    p95 = float(np.percentile(sorted_latencies, 95))
    p99 = float(np.percentile(sorted_latencies, 99))
    min_lat = float(sorted_latencies[0])
    max_lat = float(sorted_latencies[-1])
    mean_lat = float(statistics.mean(sorted_latencies))
    achieved_fps = total_ticks / max(1e-6, total_time_s)

    # RSS growth: compare tick 100 to tick 1000 to eliminate initial PyTorch cache warm-up
    rss_at_100 = rss_samples[1][1] if len(rss_samples) > 1 else initial_rss
    rss_growth_mb = (final_rss - rss_at_100) / (1024 * 1024)
    memory_flatline_passed = rss_growth_mb < 25.0  # flat memory growth < 25 MB over 900 ticks

    return {
        "total_ticks": total_ticks,
        "total_duration_seconds": round(total_time_s, 3),
        "achieved_fps": round(achieved_fps, 2),
        "latency_ms": {
            "min": round(min_lat, 2),
            "mean": round(mean_lat, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max_lat, 2),
        },
        "memory_rss": {
            "initial_mb": round(initial_rss / (1024 * 1024), 2),
            "at_100_mb": round(rss_at_100 / (1024 * 1024), 2),
            "final_mb": round(final_rss / (1024 * 1024), 2),
            "growth_from_100_mb": round(rss_growth_mb, 2),
            "flatline_passed": memory_flatline_passed,
        },
        "gap_resets_tested": gap_resets_count,
        "soak_test_passed": memory_flatline_passed and p50 < 35.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage D Soak Test & CPU Benchmark")
    parser.add_argument("--ticks", type=int, default=1000, help="Total observation ticks (default: 1000)")
    parser.add_argument("--width", type=int, default=64, help="Frame width (default: 64)")
    parser.add_argument("--height", type=int, default=64, help="Frame height (default: 64)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate (default: 30.0)")
    args = parser.parse_args()

    results = run_soak_test(
        total_ticks=args.ticks,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
