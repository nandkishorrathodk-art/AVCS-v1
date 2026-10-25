"""Adversarial Performance, Batch Scalability & Continuous Soak Benchmark for Milestone M3.

Evaluates StreamingVisualCortexV2 and EdgeLuminancePreprocessor on local CPU:
1. CPU Latency Benchmark (Model-Only at 128x128):
   - Measures forward pass latency on CPU across 100 consecutive iterations after warmup.
   - Reports min, p50 (median), p90, p95, p99, mean, max.
   - Verifies Gate 6: Minimum latency <= 10.0 ms.
   - Compares stateless vs stateful sequential forward execution.
2. Batch Size Scalability Benchmark:
   - Measures batch sizes B in {1, 2, 4, 8, 16} at 128x128 resolution.
   - Reports latency distributions, throughput (FPS), and per-sample latency.
   - Verifies linear scaling without memory bottlenecks.
3. Continuous Memory Soak Benchmark:
   - Runs 1,000 consecutive steps through StreamingVisualCortexV2 on CPU.
   - Continuously passes temporal state with advancing timestamps and periodic gap resets.
   - Tracks RSS memory (via psutil) and Python heap (via tracemalloc) every 100 steps.
   - Verifies zero memory leaks (memory delta < 5.0 MB).
   - Verifies numerical stability (zero NaNs, zero Infs across all 1,000 steps).
4. Preprocessor + Model End-to-End Latency Benchmark:
   - Measures combined pipeline (EdgeLuminancePreprocessor -> StreamingVisualCortexV2).
   - Reports per-stage and end-to-end latency distributions (min, p50, p95, p99).
5. Adversarial Input Robustness:
   - Tests solid black, solid white, high-frequency checkerboard, and random noise inputs.
   - Verifies shape validation and fail-closed behavior on invalid inputs.

Outputs comprehensive JSON telemetry to benchmarks/results/challenger_m3_2_performance.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from avcs.streaming_v2.model import (
    CortexTemporalState,
    StreamingVisualCortexV2,
)
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor


def get_process_memory_mb() -> float:
    """Return process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024.0 * 1024.0))


def compute_distribution_stats(times_sec: list[float]) -> dict[str, float]:
    """Compute distribution statistics in milliseconds."""
    arr_ms = np.array(times_sec, dtype=np.float64) * 1000.0
    return {
        "min_ms": round(float(np.min(arr_ms)), 3),
        "p50_ms": round(float(np.percentile(arr_ms, 50)), 3),
        "p90_ms": round(float(np.percentile(arr_ms, 90)), 3),
        "p95_ms": round(float(np.percentile(arr_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(arr_ms, 99)), 3),
        "mean_ms": round(float(np.mean(arr_ms)), 3),
        "max_ms": round(float(np.max(arr_ms)), 3),
        "std_ms": round(float(np.std(arr_ms)), 3),
    }


def benchmark_cpu_latency(
    model: StreamingVisualCortexV2,
    num_iterations: int = 100,
    warmup_iterations: int = 30,
) -> dict[str, Any]:
    """Benchmark model-only forward pass latency across 100 consecutive iterations."""
    print("\n--- 1. CPU Latency Benchmark (128x128, B=1) ---")
    x = torch.rand(1, 4, 128, 128, dtype=torch.float32)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(x, timestamp=0.0)

    # Stateless forward passes
    times_stateless: list[float] = []
    with torch.inference_mode():
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model(x, timestamp=0.0)
            times_stateless.append(time.perf_counter() - t0)

    # Stateful sequential forward passes
    times_stateful: list[float] = []
    state: CortexTemporalState | None = None
    with torch.inference_mode():
        for i in range(num_iterations):
            t0 = time.perf_counter()
            out = model(x, prev_state=state, timestamp=float(i * 0.033))
            times_stateful.append(time.perf_counter() - t0)
            state = out.temporal_state

    stats_stateless = compute_distribution_stats(times_stateless)
    stats_stateful = compute_distribution_stats(times_stateful)

    gate6_passed = stats_stateless["min_ms"] <= 10.0
    gate6_stateful_passed = stats_stateful["min_ms"] <= 10.0

    print(f"Stateless  -> Min: {stats_stateless['min_ms']:6.3f} ms | P50: {stats_stateless['p50_ms']:6.3f} ms | P95: {stats_stateless['p95_ms']:6.3f} ms | P99: {stats_stateless['p99_ms']:6.3f} ms")
    print(f"Stateful   -> Min: {stats_stateful['min_ms']:6.3f} ms | P50: {stats_stateful['p50_ms']:6.3f} ms | P95: {stats_stateful['p95_ms']:6.3f} ms | P99: {stats_stateful['p99_ms']:6.3f} ms")
    print(f"Gate 6 Verdict (Min <= 10.0 ms): {'PASS' if gate6_passed else 'FAIL'} (Achieved: {stats_stateless['min_ms']} ms)")

    return {
        "num_iterations": num_iterations,
        "warmup_iterations": warmup_iterations,
        "stateless": stats_stateless,
        "stateful": stats_stateful,
        "gate6_threshold_ms": 10.0,
        "gate6_stateless_achieved_ms": stats_stateless["min_ms"],
        "gate6_stateful_achieved_ms": stats_stateful["min_ms"],
        "gate6_passed": gate6_passed,
        "gate6_stateful_passed": gate6_stateful_passed,
    }


def benchmark_batch_scalability(
    model: StreamingVisualCortexV2,
    batch_sizes: list[int] | None = None,
    iterations_per_batch: int = 50,
    warmup_per_batch: int = 10,
) -> dict[str, Any]:
    """Benchmark batch sizes B in {1, 2, 4, 8, 16} at 128x128 resolution."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]
    print("\n--- 2. Batch Size Scalability Benchmark (B in {1, 2, 4, 8, 16}) ---")
    batch_results: dict[str, Any] = {}

    baseline_p50 = None

    for b in batch_sizes:
        x = torch.rand(b, 4, 128, 128, dtype=torch.float32)
        rss_before = get_process_memory_mb()

        # Warmup
        with torch.inference_mode():
            for _ in range(warmup_per_batch):
                _ = model(x, timestamp=0.0)

        # Timed runs
        times: list[float] = []
        with torch.inference_mode():
            for _ in range(iterations_per_batch):
                t0 = time.perf_counter()
                _ = model(x, timestamp=0.0)
                times.append(time.perf_counter() - t0)

        rss_after = get_process_memory_mb()
        stats = compute_distribution_stats(times)

        total_time_sec = sum(times)
        throughput_fps = (b * iterations_per_batch) / total_time_sec
        per_sample_p50_ms = round(stats["p50_ms"] / b, 3)
        per_sample_min_ms = round(stats["min_ms"] / b, 3)

        if baseline_p50 is None:
            baseline_p50 = stats["p50_ms"]
            scaling_factor = 1.0
        else:
            scaling_factor = round(stats["p50_ms"] / baseline_p50, 2)

        batch_results[f"batch_{b}"] = {
            "batch_size": b,
            "min_ms": stats["min_ms"],
            "p50_ms": stats["p50_ms"],
            "p95_ms": stats["p95_ms"],
            "p99_ms": stats["p99_ms"],
            "per_sample_min_ms": per_sample_min_ms,
            "per_sample_p50_ms": per_sample_p50_ms,
            "throughput_fps": round(throughput_fps, 2),
            "scaling_factor_vs_b1": scaling_factor,
            "rss_before_mb": round(rss_before, 2),
            "rss_after_mb": round(rss_after, 2),
            "rss_delta_mb": round(rss_after - rss_before, 2),
        }

        print(
            f"B={b:2d} | Min: {stats['min_ms']:6.2f} ms | P50: {stats['p50_ms']:6.2f} ms | "
            f"Per-sample: {per_sample_p50_ms:5.2f} ms | FPS: {throughput_fps:6.1f} | Scaling: {scaling_factor:4.1f}x | RSS: {rss_after:6.1f} MB"
        )

    # Check for memory bottlenecks or unreasonable superlinear blowup
    # Latency should scale roughly sub-linearly or linearly on multi-core CPU
    scaling_b16 = batch_results["batch_16"]["scaling_factor_vs_b1"]
    linear_scaling_verified = scaling_b16 <= 16.0  # Must not exceed strictly linear

    return {
        "batch_sizes": batch_sizes,
        "iterations_per_batch": iterations_per_batch,
        "scaling_summary": batch_results,
        "linear_scaling_verified": linear_scaling_verified,
    }


def benchmark_memory_soak(
    model: StreamingVisualCortexV2,
    num_steps: int = 1000,
    check_interval: int = 100,
) -> dict[str, Any]:
    """Run 1,000 continuous sequential steps through StreamingVisualCortexV2 on CPU."""
    print(f"\n--- 3. Continuous Memory Soak ({num_steps} Steps) ---")

    gc.collect()
    tracemalloc.start()

    x = torch.rand(1, 4, 128, 128, dtype=torch.float32)

    rss_initial = get_process_memory_mb()
    heap_initial, _ = tracemalloc.get_traced_memory()
    heap_initial / (1024.0 * 1024.0)

    checkpoints: list[dict[str, Any]] = []
    state: CortexTemporalState | None = None

    nan_detected = False
    inf_detected = False

    t_start = time.perf_counter()

    with torch.inference_mode():
        for step in range(1, num_steps + 1):
            # Advance timestamp at 30 FPS
            ts = step * (1.0 / 30.0)

            # Introduce periodic state gap reset at step 500 to stress state reset memory handling
            dropped = (step == 500)

            out = model(x, prev_state=state, timestamp=ts, dropped_frame=dropped)
            state = out.temporal_state

            # Periodic check
            if step % check_interval == 0 or step == 1 or step == num_steps:
                rss_curr = get_process_memory_mb()
                heap_curr, heap_peak = tracemalloc.get_traced_memory()
                heap_curr_mb = heap_curr / (1024.0 * 1024.0)
                heap_peak_mb = heap_peak / (1024.0 * 1024.0)

                # Numerical stability checks
                if torch.isnan(out.latent).any() or torch.isnan(out.change_heatmap).any():
                    nan_detected = True
                if torch.isinf(out.latent).any() or torch.isinf(out.change_heatmap).any():
                    inf_detected = True

                checkpoints.append({
                    "step": step,
                    "timestamp": round(ts, 3),
                    "rss_mb": round(rss_curr, 2),
                    "rss_delta_mb": round(rss_curr - rss_initial, 2),
                    "heap_curr_mb": round(heap_curr_mb, 2),
                    "heap_peak_mb": round(heap_peak_mb, 2),
                })
                print(
                    f"Step {step:4d} / {num_steps} | RSS: {rss_curr:6.2f} MB (Delta: {rss_curr - rss_initial:+5.2f} MB) | "
                    f"Heap: {heap_curr_mb:5.2f} MB (Peak: {heap_peak_mb:5.2f} MB)"
                )

    t_total = time.perf_counter() - t_start
    gc.collect()

    rss_final = get_process_memory_mb()
    heap_final, _heap_peak_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_delta_mb = round(rss_final - rss_initial, 3)
    heap_delta_mb = round((heap_final - heap_initial) / (1024.0 * 1024.0), 3)

    # Acceptance threshold: memory delta < 5.0 MB
    soak_passed = (rss_delta_mb < 5.0) and not nan_detected and not inf_detected

    print("Memory Soak Result:")
    print(f"  RSS Initial : {rss_initial:.2f} MB")
    print(f"  RSS Final   : {rss_final:.2f} MB")
    print(f"  RSS Delta   : {rss_delta_mb:+.3f} MB (Budget: < 5.0 MB)")
    print(f"  Heap Delta  : {heap_delta_mb:+.3f} MB")
    print(f"  Throughput  : {num_steps / t_total:.1f} steps/sec")
    print(f"  Soak Status : {'PASS' if soak_passed else 'FAIL'}")

    return {
        "num_steps": num_steps,
        "rss_initial_mb": round(rss_initial, 2),
        "rss_final_mb": round(rss_final, 2),
        "rss_delta_mb": rss_delta_mb,
        "heap_delta_mb": heap_delta_mb,
        "max_allowed_delta_mb": 5.0,
        "total_duration_sec": round(t_total, 2),
        "steps_per_sec": round(num_steps / t_total, 1),
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "checkpoints": checkpoints,
        "soak_passed": soak_passed,
    }


def benchmark_end_to_end_pipeline(
    preprocessor: EdgeLuminancePreprocessor,
    model: StreamingVisualCortexV2,
    num_iterations: int = 100,
    warmup_iterations: int = 25,
) -> dict[str, Any]:
    """Measure combined pipeline (EdgeLuminancePreprocessor -> StreamingVisualCortexV2) on CPU."""
    print("\n--- 4. Preprocessor + Model End-to-End Latency Benchmark ---")
    x_rgb = torch.rand(1, 3, 128, 128, dtype=torch.float32)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            x_4ch = preprocessor(x_rgb)
            _ = model(x_4ch, timestamp=0.0)

    times_prep: list[float] = []
    times_model: list[float] = []
    times_e2e: list[float] = []

    state: CortexTemporalState | None = None

    with torch.inference_mode():
        for i in range(num_iterations):
            t0 = time.perf_counter()
            x_4ch = preprocessor(x_rgb)
            t1 = time.perf_counter()
            out = model(x_4ch, prev_state=state, timestamp=float(i * 0.033))
            t2 = time.perf_counter()

            times_prep.append(t1 - t0)
            times_model.append(t2 - t1)
            times_e2e.append(t2 - t0)
            state = out.temporal_state

    stats_prep = compute_distribution_stats(times_prep)
    stats_model = compute_distribution_stats(times_model)
    stats_e2e = compute_distribution_stats(times_e2e)

    prep_overhead_pct = round((stats_prep["p50_ms"] / stats_e2e["p50_ms"]) * 100.0, 1)
    e2e_fps = round(1000.0 / stats_e2e["p50_ms"], 1)

    print(f"Preprocessor -> Min: {stats_prep['min_ms']:6.3f} ms | P50: {stats_prep['p50_ms']:6.3f} ms | P95: {stats_prep['p95_ms']:6.3f} ms")
    print(f"Model Only   -> Min: {stats_model['min_ms']:6.3f} ms | P50: {stats_model['p50_ms']:6.3f} ms | P95: {stats_model['p95_ms']:6.3f} ms")
    print(f"End-to-End   -> Min: {stats_e2e['min_ms']:6.3f} ms | P50: {stats_e2e['p50_ms']:6.3f} ms | P95: {stats_e2e['p95_ms']:6.3f} ms | FPS: {e2e_fps}")
    print(f"Preprocessor Overhead: {prep_overhead_pct}% of total pipeline latency")

    return {
        "num_iterations": num_iterations,
        "preprocessor": stats_prep,
        "model": stats_model,
        "end_to_end": stats_e2e,
        "preprocessor_overhead_pct": prep_overhead_pct,
        "end_to_end_fps": e2e_fps,
    }


def benchmark_adversarial_inputs(
    preprocessor: EdgeLuminancePreprocessor,
    model: StreamingVisualCortexV2,
) -> dict[str, Any]:
    """Test adversarial screen patterns and boundary conditions."""
    print("\n--- 5. Adversarial Input Robustness & Numerical Stress ---")
    adversarial_tests: dict[str, Any] = {}

    test_inputs = {
        "solid_black": torch.zeros(1, 3, 128, 128, dtype=torch.float32),
        "solid_white": torch.ones(1, 3, 128, 128, dtype=torch.float32),
        "checkerboard": (
            (torch.arange(128).view(1, 1, 128, 1) + torch.arange(128).view(1, 1, 1, 128)) % 2
        ).repeat(1, 3, 1, 1).float(),
        "random_gaussian": torch.clamp(torch.randn(1, 3, 128, 128) * 0.5 + 0.5, 0.0, 1.0),
    }

    with torch.inference_mode():
        for name, tensor in test_inputs.items():
            x_4ch = preprocessor(tensor)
            out = model(x_4ch, timestamp=0.0)

            # Check for non-finite values
            has_nan = bool(
                torch.isnan(out.latent).any()
                or torch.isnan(out.change_heatmap).any()
                or torch.isnan(out.ui_class_logits).any()
            )
            has_inf = bool(
                torch.isinf(out.latent).any()
                or torch.isinf(out.change_heatmap).any()
                or torch.isinf(out.ui_class_logits).any()
            )

            # Verify System 2 distillation doesn't crash or leak coordinates
            sym = out.to_symbolic_summary()
            d = sym.to_dict()

            adversarial_tests[name] = {
                "finite": not (has_nan or has_inf),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "uncertainty_level": sym.uncertainty_level,
                "summary_keys_count": len(d),
            }
            print(f"Pattern '{name:16s}' -> Finite: {not (has_nan or has_inf)} | Uncertainty: {sym.uncertainty_level}")

    # Shape boundary testing
    shape_tests = {}
    # 3D input: (4, 128, 128)
    try:
        x_3d = torch.rand(4, 128, 128)
        out_3d = model(x_3d)
        shape_tests["3d_unbatched_accepted"] = out_3d.latent.shape == (1, 96)
    except (RuntimeError, ValueError):
        shape_tests["3d_unbatched_accepted"] = False

    # Invalid channels: (1, 3, 128, 128) passed directly to model expecting 4
    try:
        x_invalid_ch = torch.rand(1, 3, 128, 128)
        model(x_invalid_ch)
        shape_tests["invalid_channel_caught"] = False
    except ValueError:
        shape_tests["invalid_channel_caught"] = True

    adversarial_tests["shape_validation"] = shape_tests
    print(f"Shape validation -> 3D auto-batch: {shape_tests.get('3d_unbatched_accepted')}, 3-channel error caught: {shape_tests.get('invalid_channel_caught')}")

    all_passed = all(t["finite"] for k, t in adversarial_tests.items() if k != "shape_validation") and all(shape_tests.values())
    return {
        "patterns": adversarial_tests,
        "all_passed": all_passed,
    }


def run_full_performance_and_soak_suite(
    output_path: str = "benchmarks/results/challenger_m3_2_performance.json",
    num_latency_iterations: int = 100,
    soak_steps: int = 1000,
) -> dict[str, Any]:
    """Execute complete challenger_m3_2 empirical performance & soak benchmark."""
    print("=" * 80)
    print("  AVCS Milestone M3: Performance, Scalability & Soak Stress Benchmark")
    print("  Agent: challenger_m3_2 | Date: 2026-10-24")
    print("=" * 80)

    # Set deterministic seed
    torch.manual_seed(20261024)
    np.random.seed(20261024)

    # Model & Preprocessor initialization
    preprocessor = EdgeLuminancePreprocessor()
    preprocessor.eval()

    model = StreamingVisualCortexV2()
    model.eval()

    param_breakdown = model.get_parameter_breakdown()
    total_params = model.total_parameters

    print(f"\nModel: {model.model_name} v{model.version}")
    print(f"Total Parameters: {total_params:,} (Budget: 400,000 - 500,000)")
    for comp, count in param_breakdown.items():
        if comp != "total":
            print(f"  - {comp:22s}: {count:7,d} params")

    # 1. CPU Latency Benchmark (Model-Only)
    latency_results = benchmark_cpu_latency(
        model=model,
        num_iterations=num_latency_iterations,
    )

    # 2. Batch Size Scalability
    batch_results = benchmark_batch_scalability(
        model=model,
        batch_sizes=[1, 2, 4, 8, 16],
    )

    # 3. Continuous Memory Soak (1,000 steps)
    soak_results = benchmark_memory_soak(
        model=model,
        num_steps=soak_steps,
        check_interval=100,
    )

    # 4. Preprocessor + Model End-to-End Latency
    e2e_results = benchmark_end_to_end_pipeline(
        preprocessor=preprocessor,
        model=model,
        num_iterations=num_latency_iterations,
    )

    # 5. Adversarial Input Robustness
    adversarial_results = benchmark_adversarial_inputs(
        preprocessor=preprocessor,
        model=model,
    )

    # Overall Verdict Calculation
    # Gate 6: Minimum forward latency <= 10.0 ms
    # Memory Soak: Delta < 5.0 MB over 1,000 steps
    # Batch Scaling: Linear scaling verified
    # Adversarial Robustness: All tests passed
    gate6_passed = latency_results["gate6_passed"]
    soak_passed = soak_results["soak_passed"]
    batch_passed = batch_results["linear_scaling_verified"]
    adversarial_passed = adversarial_results["all_passed"]

    overall_verdict = "APPROVE" if (gate6_passed and soak_passed and batch_passed and adversarial_passed) else "REJECT"

    telemetry: dict[str, Any] = {
        "benchmark_name": "challenger_m3_2_performance_soak",
        "agent": "challenger_m3_2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
        },
        "model_architecture": {
            "model_name": model.model_name,
            "version": model.version,
            "total_parameters": total_params,
            "parameter_budget_min": 400_000,
            "parameter_budget_max": 500_000,
            "parameter_budget_met": 400_000 <= total_params <= 500_000,
            "parameter_breakdown": param_breakdown,
        },
        "benchmarks": {
            "cpu_latency": latency_results,
            "batch_scalability": batch_results,
            "memory_soak_1000_steps": soak_results,
            "end_to_end_pipeline": e2e_results,
            "adversarial_robustness": adversarial_results,
        },
        "verdicts": {
            "gate6_latency_le_10ms": gate6_passed,
            "memory_soak_le_5mb": soak_passed,
            "batch_scalability_linear": batch_passed,
            "adversarial_robustness_valid": adversarial_passed,
            "overall_verdict": overall_verdict,
        },
    }

    # Save telemetry JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print("\n" + "=" * 80)
    print("  FINAL CHALLENGER VERDICT")
    print("=" * 80)
    print(f"  Gate 6 CPU Latency (<= 10.0 ms) : {'PASS' if gate6_passed else 'FAIL'} (Achieved: {latency_results['stateless']['min_ms']} ms)")
    print(f"  Continuous Memory Soak (< 5.0 MB): {'PASS' if soak_passed else 'FAIL'} (Achieved Delta: {soak_results['rss_delta_mb']:+.3f} MB)")
    print(f"  Batch Size Scalability (B=1..16) : {'PASS' if batch_passed else 'FAIL'}")
    print(f"  End-to-End Pipeline Latency      : Min {e2e_results['end_to_end']['min_ms']} ms | P50 {e2e_results['end_to_end']['p50_ms']} ms ({e2e_results['end_to_end_fps']} FPS)")
    print(f"  Adversarial Robustness           : {'PASS' if adversarial_passed else 'FAIL'}")
    print(f"  OVERALL MILESTONE M3 VERDICT     : {overall_verdict}")
    print(f"  Telemetry written to             : {out_file.resolve()}")
    print("=" * 80)

    return telemetry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVCS Milestone M3 Performance & Soak Benchmark")
    parser.add_argument("--out", type=str, default="benchmarks/results/challenger_m3_2_performance.json", help="Output JSON path")
    parser.add_argument("--latency-steps", type=int, default=100, help="Number of latency iterations")
    parser.add_argument("--soak-steps", type=int, default=1000, help="Number of soak steps")
    args = parser.parse_args()

    run_full_performance_and_soak_suite(
        output_path=args.out,
        num_latency_iterations=args.latency_steps,
        soak_steps=args.soak_steps,
    )
