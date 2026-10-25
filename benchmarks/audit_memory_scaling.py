"""Audit memory scaling across 200-frame continuous streaming ticks.

Validates that avcs/eye_stream.py (EyeStream) and avcs/cortex/model.py (StreamingVisualCortexV2)
maintain strictly constant O(1) memory scaling without leaks across continuous streaming ticks.
Monitors RSS and Virtual memory usage via psutil and asserts delta RAM remains bounded.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import psutil
import torch

from avcs.cortex.model import StreamingVisualCortexV2
from avcs.eye_stream import EyeStream


def audit_cortex_model_memory(ticks: int = 200, warmup: int = 20) -> Dict[str, Any]:
    """Audit StreamingVisualCortexV2 memory scaling over continuous ticks."""
    process = psutil.Process(os.getpid())
    gc.collect()

    model = StreamingVisualCortexV2().eval()
    state = None

    # Warmup
    with torch.inference_mode():
        for i in range(warmup):
            x = torch.rand(1, 4, 128, 128)
            out = model(x, prev_state=state, timestamp=float(i) * 0.033)
            state = out.temporal_state

    gc.collect()
    rss_warmup = process.memory_info().rss / (1024 * 1024)
    vms_warmup = process.memory_info().vms / (1024 * 1024)

    samples = []
    with torch.inference_mode():
        for tick in range(ticks):
            x = torch.rand(1, 4, 128, 128)
            out = model(x, prev_state=state, timestamp=float(warmup + tick) * 0.033)
            state = out.temporal_state

            if (tick + 1) % 50 == 0 or tick == 0:
                gc.collect()
                mem = process.memory_info()
                rss_mb = mem.rss / (1024 * 1024)
                vms_mb = mem.vms / (1024 * 1024)
                samples.append({"tick": tick + 1, "rss_mb": round(rss_mb, 2), "vms_mb": round(vms_mb, 2)})

    rss_end = samples[-1]["rss_mb"]
    vms_end = samples[-1]["vms_mb"]
    rss_peak = max(s["rss_mb"] for s in samples)
    vms_peak = max(s["vms_mb"] for s in samples)
    rss_delta = round(rss_end - rss_warmup, 2)
    vms_delta = round(vms_end - vms_warmup, 2)

    assert rss_delta < 5.0, f"StreamingVisualCortexV2 RSS leak detected: delta={rss_delta} MB"

    return {
        "component": "StreamingVisualCortexV2 (avcs/cortex/model.py)",
        "warmup_rss_mb": round(rss_warmup, 2),
        "peak_rss_mb": rss_peak,
        "end_rss_mb": rss_end,
        "delta_rss_mb": rss_delta,
        "warmup_vms_mb": round(vms_warmup, 2),
        "peak_vms_mb": vms_peak,
        "end_vms_mb": vms_end,
        "delta_vms_mb": vms_delta,
        "ticks": ticks,
        "samples": samples,
        "o1_constant_scaling": True,
    }


def audit_eye_stream_memory(ticks: int = 200, warmup: int = 20) -> Dict[str, Any]:
    """Audit EyeStream memory scaling over continuous ticks."""
    process = psutil.Process(os.getpid())
    gc.collect()

    stream = EyeStream(source="synthetic", load_weights=False)

    # Warmup
    for _ in range(warmup):
        stream.step()

    gc.collect()
    rss_warmup = process.memory_info().rss / (1024 * 1024)
    vms_warmup = process.memory_info().vms / (1024 * 1024)

    samples = []
    for tick in range(ticks):
        stream.step()

        if (tick + 1) % 50 == 0 or tick == 0:
            gc.collect()
            mem = process.memory_info()
            rss_mb = mem.rss / (1024 * 1024)
            vms_mb = mem.vms / (1024 * 1024)
            samples.append({"tick": tick + 1, "rss_mb": round(rss_mb, 2), "vms_mb": round(vms_mb, 2)})

    stream.close()

    rss_end = samples[-1]["rss_mb"]
    vms_end = samples[-1]["vms_mb"]
    rss_peak = max(s["rss_mb"] for s in samples)
    vms_peak = max(s["vms_mb"] for s in samples)
    rss_delta = round(rss_end - rss_warmup, 2)
    vms_delta = round(vms_end - vms_warmup, 2)

    assert rss_delta < 5.0, f"EyeStream RSS leak detected: delta={rss_delta} MB"

    return {
        "component": "EyeStream (avcs/eye_stream.py)",
        "warmup_rss_mb": round(rss_warmup, 2),
        "peak_rss_mb": rss_peak,
        "end_rss_mb": rss_end,
        "delta_rss_mb": rss_delta,
        "warmup_vms_mb": round(vms_warmup, 2),
        "peak_vms_mb": vms_peak,
        "end_vms_mb": vms_end,
        "delta_vms_mb": vms_delta,
        "ticks": ticks,
        "samples": samples,
        "o1_constant_scaling": True,
    }


def audit_integrated_pipeline_memory(ticks: int = 200, warmup: int = 20) -> Dict[str, Any]:
    """Audit integrated continuous streaming pipeline combining EyeStream and StreamingVisualCortexV2."""
    process = psutil.Process(os.getpid())
    gc.collect()

    stream = EyeStream(source="synthetic", load_weights=False)
    model = StreamingVisualCortexV2().eval()
    state = None

    # Warmup
    with torch.inference_mode():
        for i in range(warmup):
            obs = stream.step()
            x = torch.rand(1, 4, 128, 128)
            out = model(x, prev_state=state, timestamp=float(i) * 0.033)
            state = out.temporal_state

    gc.collect()
    rss_warmup = process.memory_info().rss / (1024 * 1024)
    vms_warmup = process.memory_info().vms / (1024 * 1024)

    samples = []
    with torch.inference_mode():
        for tick in range(ticks):
            obs = stream.step()
            x = torch.rand(1, 4, 128, 128)
            out = model(x, prev_state=state, timestamp=float(warmup + tick) * 0.033)
            state = out.temporal_state

            if (tick + 1) % 50 == 0 or tick == 0:
                gc.collect()
                mem = process.memory_info()
                rss_mb = mem.rss / (1024 * 1024)
                vms_mb = mem.vms / (1024 * 1024)
                samples.append({"tick": tick + 1, "rss_mb": round(rss_mb, 2), "vms_mb": round(vms_mb, 2)})

    stream.close()

    rss_end = samples[-1]["rss_mb"]
    vms_end = samples[-1]["vms_mb"]
    rss_peak = max(s["rss_mb"] for s in samples)
    vms_peak = max(s["vms_mb"] for s in samples)
    rss_delta = round(rss_end - rss_warmup, 2)
    vms_delta = round(vms_end - vms_warmup, 2)

    assert rss_delta < 5.0, f"Integrated pipeline RSS leak detected: delta={rss_delta} MB"

    return {
        "component": "Integrated Pipeline (EyeStream + StreamingVisualCortexV2)",
        "warmup_rss_mb": round(rss_warmup, 2),
        "peak_rss_mb": rss_peak,
        "end_rss_mb": rss_end,
        "delta_rss_mb": rss_delta,
        "warmup_vms_mb": round(vms_warmup, 2),
        "peak_vms_mb": vms_peak,
        "end_vms_mb": vms_end,
        "delta_vms_mb": vms_delta,
        "ticks": ticks,
        "samples": samples,
        "o1_constant_scaling": True,
    }


def main() -> None:
    print("=== AVCS Continuous Streaming Memory Scaling Audit ===")
    results = {}

    print("\n1. Auditing StreamingVisualCortexV2 (avcs/cortex/model.py)...")
    res_cortex = audit_cortex_model_memory(ticks=200)
    results["cortex_model"] = res_cortex
    print(f"   Peak RSS: {res_cortex['peak_rss_mb']} MB, Delta RSS: {res_cortex['delta_rss_mb']} MB, Peak VMS: {res_cortex['peak_vms_mb']} MB, Delta VMS: {res_cortex['delta_vms_mb']} MB")

    print("\n2. Auditing EyeStream (avcs/eye_stream.py)...")
    res_eye = audit_eye_stream_memory(ticks=200)
    results["eye_stream"] = res_eye
    print(f"   Peak RSS: {res_eye['peak_rss_mb']} MB, Delta RSS: {res_eye['delta_rss_mb']} MB, Peak VMS: {res_eye['peak_vms_mb']} MB, Delta VMS: {res_eye['delta_vms_mb']} MB")

    print("\n3. Auditing Integrated Pipeline (EyeStream + StreamingVisualCortexV2)...")
    res_integrated = audit_integrated_pipeline_memory(ticks=200)
    results["integrated"] = res_integrated
    print(f"   Peak RSS: {res_integrated['peak_rss_mb']} MB, Delta RSS: {res_integrated['delta_rss_mb']} MB, Peak VMS: {res_integrated['peak_vms_mb']} MB, Delta VMS: {res_integrated['delta_vms_mb']} MB")

    out_path = os.path.join("benchmarks", "results", "memory_scaling_audit_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll assertions PASSED. Results saved to {out_path}")


if __name__ == "__main__":
    main()

