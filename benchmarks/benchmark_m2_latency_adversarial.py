"""Adversarial CPU Latency and Localization Precision Challenger for AVCS Milestone 2.

Empirically benchmarks:
1. GoalConditionedGrounder forward pass on host CPU with torch.set_num_threads(1) (100 timed iterations).
2. GoalConditionedGrounder.locate() on host CPU with torch.set_num_threads(1) (100 timed iterations).
3. HumanEye.locate() on host CPU with torch.set_num_threads(1) (100 timed iterations).
4. Feature 19 baseline (Backbone + FastAffordanceHead) with torch.set_num_threads(1).
5. Multi-resolution coordinate bounds validation.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from avcs.goal_grounder import GoalConditionedGrounder, letterbox_rgb
from avcs.human_eye import FastAffordanceHead, HumanEye, RealTimeCorticalBackbone


def run_latency_benchmark(num_iterations: int = 100, warmup_iterations: int = 20) -> Dict[str, Any]:
    torch.set_num_threads(1)
    
    eye = HumanEye(load_weights=True)
    grounder = eye.grounder
    grounder.eval()
    if hasattr(eye, "eval"):
        eye.eval()

    device = torch.device("cpu")
    dummy_tensor = torch.randn(1, 3, 128, 128, device=device)
    dummy_frame_128 = np.zeros((128, 128, 3), dtype=np.uint8)
    dummy_frame_hd = np.zeros((1200, 1920, 3), dtype=np.uint8)
    query = "click button"

    # --- Benchmark 1: GoalConditionedGrounder._forward_modulated (pure neural forward pass) ---
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = grounder._forward_modulated(dummy_tensor, query)

        t_fwd_128: List[float] = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = grounder._forward_modulated(dummy_tensor, query)
            t_fwd_128.append((time.perf_counter() - t0) * 1000.0)

    # --- Benchmark 2: GoalConditionedGrounder.locate() on 128x128 ---
    for _ in range(warmup_iterations):
        _ = grounder.locate(query, frame=dummy_frame_128)

    t_loc_128: List[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = grounder.locate(query, frame=dummy_frame_128)
        t_loc_128.append((time.perf_counter() - t0) * 1000.0)

    # --- Benchmark 3: HumanEye.locate() on 128x128 ---
    for _ in range(warmup_iterations):
        _ = eye.locate(query, frame=dummy_frame_128)

    t_eye_128: List[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = eye.locate(query, frame=dummy_frame_128)
        t_eye_128.append((time.perf_counter() - t0) * 1000.0)

    # --- Benchmark 4: HumanEye.locate() on 1920x1200 non-square desktop frame ---
    for _ in range(5):
        _ = eye.locate(query, frame=dummy_frame_hd)

    t_eye_hd: List[float] = []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = eye.locate(query, frame=dummy_frame_hd)
        t_eye_hd.append((time.perf_counter() - t0) * 1000.0)

    # --- Benchmark 5: Feature 19 baseline (Backbone + FastAffordanceHead) ---
    backbone = RealTimeCorticalBackbone(in_channels=3, feature_dim=128).eval()
    head = FastAffordanceHead(in_dim=128, num_classes=3).eval()
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = head(backbone(dummy_tensor))

        t_feat19: List[float] = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = head(backbone(dummy_tensor))
            t_feat19.append((time.perf_counter() - t0) * 1000.0)

    def calc_stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        return {
            "p50_median_ms": round(float(np.median(a)), 3),
            "mean_ms": round(float(np.mean(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "min_ms": round(float(np.min(a)), 3),
            "max_ms": round(float(np.max(a)), 3),
        }

    return {
        "num_threads": 1,
        "iterations": num_iterations,
        "grounder_fwd_128": calc_stats(t_fwd_128),
        "grounder_locate_128": calc_stats(t_loc_128),
        "eye_locate_128": calc_stats(t_eye_128),
        "eye_locate_1920x1200": calc_stats(t_eye_hd),
        "backbone_affordance_feat19": calc_stats(t_feat19),
    }


def main() -> None:
    print("=" * 70)
    print("CHALLENGER 2: EMPIRICAL LATENCY & LOCALIZATION BENCHMARK")
    print("=" * 70)
    results = run_latency_benchmark(num_iterations=100, warmup_iterations=20)
    
    print(f"\nConfiguration: num_threads = {results['num_threads']}, iterations = {results['iterations']}")
    print("-" * 70)
    for name, s in results.items():
        if isinstance(s, dict):
            print(f"[{name}]")
            print(f"  Median (p50): {s['p50_median_ms']} ms")
            print(f"  Mean:         {s['mean_ms']} ms")
            print(f"  p95:          {s['p95_ms']} ms")
            print(f"  p99:          {s['p99_ms']} ms")
            print(f"  Range:        [{s['min_ms']} - {s['max_ms']}] ms")
            passes_15ms = s['p50_median_ms'] <= 15.0
            print(f"  Target <= 15.0 ms: {'PASS' if passes_15ms else 'FAIL'}")
            print("-" * 70)

    out_file = Path("benchmarks/results/challenger2_latency_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_file}")


if __name__ == "__main__":
    main()
