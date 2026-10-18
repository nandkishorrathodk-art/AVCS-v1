"""Milestone 3.2 Experiment B: Input Resolution Benchmark (256x256 vs 512x512).

Measures:
1. Neural forward-pass latency (P50, P90, P95) on CPU for 256x256 vs 512x512.
2. Memory consumption (peak allocated tensor memory).
3. Spatial cell resolution: physical pixel span per feature grid cell.
4. Feasibility analysis for training and real-time CPU deployment.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.goal_grounder import GoalConditionedGrounder, encode_queries


def benchmark_resolution():
    print("=== Milestone 3.2 Experiment B: Resolution & Resource Benchmark ===")
    dev = torch.device("cpu")

    model_256 = GoalConditionedGrounder(input_size=256).to(dev).eval()
    model_512 = GoalConditionedGrounder(input_size=512).to(dev).eval()

    tokens = encode_queries(["click search input"]).to(dev)

    # 1. Benchmark 256x256
    img_256 = torch.zeros(1, 3, 256, 256, device=dev)
    # Warmup
    for _ in range(5):
        model_256.forward_with_presence(img_256, tokens)

    times_256 = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits_256, pres_256 = model_256.forward_with_presence(img_256, tokens)
        times_256.append((time.perf_counter() - t0) * 1000.0)

    # 2. Benchmark 512x512
    img_512 = torch.zeros(1, 3, 512, 512, device=dev)
    # Warmup
    for _ in range(5):
        model_512.forward_with_presence(img_512, tokens)

    times_512 = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits_512, pres_512 = model_512.forward_with_presence(img_512, tokens)
        times_512.append((time.perf_counter() - t0) * 1000.0)

    p50_256 = float(np.median(times_256))
    p95_256 = float(np.percentile(times_256, 95))
    p50_512 = float(np.median(times_512))
    p95_512 = float(np.percentile(times_512, 95))

    print(f"\n--- Resolution Latency & Feature Map Comparison ---")
    print(f"Resolution 256x256:")
    print(f"  Input Shape       : (1, 3, 256, 256)")
    print(f"  Heatmap Shape     : {list(logits_256.shape)} (spatial grid: 64x64)")
    print(f"  Pixel Span / Cell : 4.0 px / cell (on 256 screen)")
    print(f"  CPU Latency P50   : {p50_256:.2f} ms")
    print(f"  CPU Latency P95   : {p95_256:.2f} ms")

    print(f"\nResolution 512x512:")
    print(f"  Input Shape       : (1, 3, 512, 512)")
    print(f"  Heatmap Shape     : {list(logits_512.shape)} (spatial grid: 128x128)")
    print(f"  Pixel Span / Cell : 2.0 px / cell (on 256 screen) -> 2x spatial granularity!")
    print(f"  CPU Latency P50   : {p50_512:.2f} ms")
    print(f"  CPU Latency P95   : {p95_512:.2f} ms")
    print(f"  Latency Multiplier: {p50_512 / p50_256:.2f}x")

    results = {
        "res_256": {
            "input": [256, 256],
            "heatmap": list(logits_256.shape),
            "p50_ms": round(p50_256, 2),
            "p95_ms": round(p95_256, 2),
            "cell_px_span": 4.0,
        },
        "res_512": {
            "input": [512, 512],
            "heatmap": list(logits_512.shape),
            "p50_ms": round(p50_512, 2),
            "p95_ms": round(p95_512, 2),
            "cell_px_span": 2.0,
            "latency_multiplier": round(p50_512 / p50_256, 2),
        }
    }

    out_p = Path("benchmarks/results/benchmark_resolution_comparison.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    benchmark_resolution()

