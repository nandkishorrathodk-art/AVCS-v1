"""Milestone 3.3 Section D: Latency Discrepancy Forensic Audit.

Investigates and reproduces the exact difference between:
- 71.93 ms coarse P50 reported in Milestone 3.1
- 124.20 ms coarse P50 reported in Milestone 3.2 Experiment B

Tests:
1. Impact of PyTorch intra-op threads: torch.set_num_threads(1) vs torch.set_num_threads(2) vs torch.set_num_threads(4).
2. Impact of checkpoint weights (trained candidate_m3_highres_v2.pt vs random initialization).
3. Impact of method boundary: predict_point() (full tokenization + subpixel extraction) vs forward_with_presence() vs raw forward_pyramid().
4. Warmup effects (5 iterations vs 50 iterations cache warmth).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.goal_grounder import GoalConditionedGrounder, load_grounding_checkpoint, encode_queries


def audit_latency():
    print("=== AVCS Latency Discrepancy Forensic Audit ===")
    dev = torch.device("cpu")
    ckpt_path = "runs/weights/candidates/candidate_m3_highres_v2.pt"

    print(f"Loading checkpoint: {ckpt_path}")
    trained_model = load_grounding_checkpoint(ckpt_path, device=dev).eval()
    random_model = GoalConditionedGrounder(input_size=256).to(dev).eval()

    query = "click search input"
    tokens = encode_queries([query]).to(dev)
    img_t = torch.zeros(1, 3, 256, 256, device=dev)

    configurations = [
        # (name, model, thread_count, use_predict_point, warmup_n)
        ("Trained V2, 1 Thread, predict_point, 50 Warmup", trained_model, 1, True, 50),
        ("Trained V2, 1 Thread, forward_with_presence, 50 Warmup", trained_model, 1, False, 50),
        ("Trained V2, 2 Threads, predict_point, 50 Warmup", trained_model, 2, True, 50),
        ("Trained V2, 4 Threads, predict_point, 50 Warmup", trained_model, 4, True, 50),
        ("Trained V2, 4 Threads, forward_with_presence, 5 Warmup (Exp B Setup)", trained_model, 4, False, 5),
        ("Random Init, 4 Threads, forward_with_presence, 5 Warmup", random_model, 4, False, 5),
    ]

    results = []

    print(f"\n{'Configuration':58s} | {'P50 (ms)':10s} | {'P95 (ms)':10s} | {'Min (ms)':10s}")
    print("-" * 95)

    for name, model, threads, use_predict, warmup_n in configurations:
        torch.set_num_threads(threads)

        # Warmup
        for _ in range(warmup_n):
            if use_predict:
                model.predict_point(img_t, query)
            else:
                with torch.inference_mode():
                    model.forward_with_presence(img_t, tokens)

        # Timed runs
        timings = []
        for _ in range(50):
            t0 = time.perf_counter()
            if use_predict:
                _ = model.predict_point(img_t, query)
            else:
                with torch.inference_mode():
                    _ = model.forward_with_presence(img_t, tokens)
            timings.append((time.perf_counter() - t0) * 1000.0)

        p50 = float(np.median(timings))
        p95 = float(np.percentile(timings, 95))
        min_t = float(np.min(timings))

        print(f"{name:58s} | {p50:8.2f} ms | {p95:8.2f} ms | {min_t:8.2f} ms")
        results.append({
            "configuration": name,
            "threads": threads,
            "use_predict_point": use_predict,
            "warmup_iterations": warmup_n,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(min_t, 2),
        })

    out_p = Path("benchmarks/forensics/latency_discrepancy_audit.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved discrepancy forensic data to {out_p}")


if __name__ == "__main__":
    audit_latency()

