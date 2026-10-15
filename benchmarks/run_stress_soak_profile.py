"""AVCS Foundation Model: Single-Thread CPU Performance, Memory Leak & Thermal Soak Profiler.

Investigates:
1. 500-iteration continuous perception loop (`HumanEye.observe()`) on 1 CPU thread.
2. Latency percentiles (min, p50, p90, p95, p99, max, mean, std) and true achieved wall-clock FPS.
3. Memory consumption (RSS/working set, VMS, OS handles) and circular reference / memory leak audit.
4. Isolated sub-component breakdown (preprocessing, motion optical flow, backbone, affordance head, foveal inspection, ventral, memory diff) and bottleneck identification.
"""
from __future__ import annotations

import collections
import ctypes
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.grounding import FovealAcuityEngine, FovealInspection
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.benchmark_real_desktop import RealisticDesktopRenderer


def linear_regression_slope(xs: List[float], ys: List[float]) -> float:
    """Calculates ordinary least squares slope: dy/dx."""
    n = len(xs)
    if n < 2:
        return 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0
    return float((n * sum_xy - sum_x * sum_y) / denom)


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


class OSResourceTracker:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.is_windows = platform.system() == "Windows"
        self._user32 = None
        self._current_process_handle = None
        if self.is_windows:
            try:
                kernel32 = ctypes.windll.kernel32
                self._user32 = ctypes.windll.user32
                self._current_process_handle = kernel32.GetCurrentProcess()
            except Exception:
                pass

    def sample(self) -> Dict[str, Any]:
        mem = self.process.memory_info()
        rss_mb = mem.rss / (1024.0 * 1024.0)
        vms_mb = mem.vms / (1024.0 * 1024.0)

        handles = 0
        try:
            handles = self.process.num_handles()
        except Exception:
            pass

        gdi_objects = 0
        user_objects = 0
        if self.is_windows and self._user32 and self._current_process_handle:
            try:
                gdi_objects = int(self._user32.GetGuiResources(self._current_process_handle, 0))
                user_objects = int(self._user32.GetGuiResources(self._current_process_handle, 1))
            except Exception:
                pass

        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "handles": handles,
            "gdi_objects": gdi_objects,
            "user_objects": user_objects,
        }


def get_gc_type_counts() -> Dict[str, int]:
    counts: Dict[str, int] = collections.defaultdict(int)
    for obj in gc.get_objects():
        tname = type(obj).__name__
        if tname in {
            "HumanEyeState", "Tensor", "ndarray", "UIMLElement", "UIMLTree",
            "VisualDiff", "FovealInspection", "dict", "list", "tuple", "set"
        }:
            counts[tname] += 1
    return dict(counts)


def run_stress_test(
    num_iterations: int = 500,
    warmup_iterations: int = 30,
) -> Dict[str, Any]:
    print("=" * 80)
    print("   AVCS FOUNDATION MODEL: SINGLE-THREAD CPU SOAK & PERFORMANCE TEST   ")
    print("=" * 80)

    # 1. Enforce strict single-thread CPU execution
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    actual_threads = torch.get_num_threads()
    device = torch.device("cpu")
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    tracker = OSResourceTracker()

    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor()}")
    print(f"Cores: Physical={cpu_count_physical}, Logical={cpu_count_logical}")
    print(f"PyTorch Thread Count: {actual_threads} (Strict Single-Thread Mode)")
    print(f"Initial Memory: RSS={tracker.sample()['rss_mb']:.2f} MB")
    print("-" * 80)

    # Instantiate HumanEye with Synthetic streaming capture source
    # SyntheticScreen provides continuous active frames (moving window, moving cursor, blinking overlay)
    screen = SyntheticScreen()
    eye = HumanEye(source=screen, device="cpu", load_weights=True)
    eye.eval()

    print(f"AVCS HumanEye Initialized:")
    print(f"  Weights Loaded: {eye.weights_loaded}")
    print(f"  Input Resolution: {eye.input_size}")
    print(f"  Affordance Classes: {eye.affordance_head.num_classes}")
    print("-" * 80)

    # Warmup
    print(f"[*] Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        eye.observe()
    gc.collect()

    initial_os_metrics = tracker.sample()
    initial_gc_objects = len(gc.get_objects())
    initial_type_counts = get_gc_type_counts()

    print(f"[*] Baseline Post-Warmup: RSS={initial_os_metrics['rss_mb']:.2f} MB, GC objects={initial_gc_objects}")
    print(f"[*] Starting sustained {num_iterations}-iteration perception loop...")

    iteration_latencies_ms: List[float] = []
    rss_history_mb: List[float] = []
    vms_history_mb: List[float] = []
    iteration_indices: List[float] = []
    gc_object_history: List[int] = []

    # Epoch trackers (5 epochs of 100 iterations each)
    epoch_size = num_iterations // 5
    epoch_latencies: Dict[int, List[float]] = collections.defaultdict(list)

    wall_start = time.perf_counter()

    for it in range(1, num_iterations + 1):
        t0 = time.perf_counter()
        state = eye.observe()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        iteration_latencies_ms.append(latency_ms)
        iteration_indices.append(float(it))

        epoch_idx = min(4, (it - 1) // epoch_size)
        epoch_latencies[epoch_idx].append(latency_ms)

        # Sample resource metrics every 10 iterations and at the end
        if it % 10 == 0 or it == num_iterations:
            metrics = tracker.sample()
            rss_history_mb.append(metrics["rss_mb"])
            vms_history_mb.append(metrics["vms_mb"])
            gc_object_history.append(len(gc.get_objects()))

        if it % 100 == 0:
            cur_rss = tracker.sample()["rss_mb"]
            cur_p50 = float(np.percentile(iteration_latencies_ms[-100:], 50))
            print(f"  -> Iteration {it:3d}/{num_iterations}: Latency p50={cur_p50:.2f}ms, RSS={cur_rss:.2f} MB")

    wall_end = time.perf_counter()
    total_wall_time_s = wall_end - wall_start
    true_sustained_fps = num_iterations / total_wall_time_s

    # Final soak metrics
    final_os_metrics = tracker.sample()
    final_gc_objects_pre_collect = len(gc.get_objects())
    uncollected_garbage_count = len(gc.garbage)

    # Perform explicit garbage collection
    collected_objects = gc.collect()
    final_os_metrics_post_collect = tracker.sample()
    final_gc_objects_post_collect = len(gc.get_objects())
    final_type_counts = get_gc_type_counts()

    # Latency percentiles
    latency_stats = calculate_percentiles(iteration_latencies_ms)

    # Memory regression slope across iterations
    # Map rss_history_mb to corresponding iterations
    rss_sample_iterations = [float(i) for i in range(10, num_iterations + 1, 10)]
    if len(rss_sample_iterations) < len(rss_history_mb):
        rss_sample_iterations.append(float(num_iterations))
    rss_slope_mb_per_iter = linear_regression_slope(rss_sample_iterations, rss_history_mb)
    latency_slope_ms_per_iter = linear_regression_slope(iteration_indices, iteration_latencies_ms)

    # Epoch analysis for thermal soak / drift
    epoch_summaries = []
    for e_idx in range(5):
        e_lats = epoch_latencies[e_idx]
        e_stats = calculate_percentiles(e_lats)
        start_it = e_idx * epoch_size + 1
        end_it = (e_idx + 1) * epoch_size
        epoch_summaries.append({
            "epoch": e_idx + 1,
            "iterations": f"{start_it}-{end_it}",
            "mean_ms": round(e_stats["mean"], 2),
            "p50_ms": round(e_stats["p50"], 2),
            "p90_ms": round(e_stats["p90"], 2),
            "p95_ms": round(e_stats["p95"], 2),
            "min_ms": round(e_stats["min"], 2),
            "max_ms": round(e_stats["max"], 2),
            "fps": round(1000.0 / e_stats["p50"], 1),
        })

    # Type count deltas
    type_deltas = {}
    all_keys = set(initial_type_counts.keys()) | set(final_type_counts.keys())
    for k in sorted(all_keys):
        type_deltas[k] = {
            "initial": initial_type_counts.get(k, 0),
            "final": final_type_counts.get(k, 0),
            "delta": final_type_counts.get(k, 0) - initial_type_counts.get(k, 0),
        }

    # Bounded collections audit on HumanEye instance
    bounded_collections_audit = {
        "_latencies_len": len(eye._latencies),
        "_latencies_maxlen": eye._latencies.maxlen,
        "memory_diff_history_len": len(eye.memory.diff_history),
        "memory_diff_history_maxlen": eye.memory.diff_history.maxlen,
        "memory_registry_len": len(eye.memory.registry.registry),
        "memory_lifecycles_len": len(eye.memory.registry.lifecycles),
        "memory_occlusion_ttl_len": len(eye.memory.registry.occlusion_ttl),
    }

    # -------------------------------------------------------------------------
    # Step 4: Component-by-Component Isolated Latency Profiling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   AVCS SUB-COMPONENT BREAKDOWN & BOTTLENECK PROFILING (1 CPU THREAD)  ")
    print("=" * 80)

    # Use realistic desktop screen (VS Code 1080p) & synthetic frame for comprehensive testing
    renderer = RealisticDesktopRenderer(width=1920, height=1080)
    sample_frame_1080p = renderer.render_vscode().frame
    sample_frame_synthetic = screen.grab()

    num_profile_iters = 200
    profile_warmup = 30

    subcomponent_benchmarks = {}

    # A. Preprocessing: 1080p uint8 -> 128x128 normalized float32 tensor
    def bench_preprocessing_1080p():
        H, W = sample_frame_1080p.shape[:2]
        sy = max(1, H // eye.input_size[0])
        sx = max(1, W // eye.input_size[1])
        sub = sample_frame_1080p[::sy, ::sx][:eye.input_size[0], :eye.input_size[1]]
        t_in = torch.from_numpy(np.ascontiguousarray(sub)).float().div_(255.0)
        t_in = t_in.permute(2, 0, 1).unsqueeze(0).to(device)
        return t_in

    def bench_preprocessing_synthetic():
        H, W = sample_frame_synthetic.shape[:2]
        sy = max(1, H // eye.input_size[0])
        sx = max(1, W // eye.input_size[1])
        sub = sample_frame_synthetic[::sy, ::sx][:eye.input_size[0], :eye.input_size[1]]
        t_in = torch.from_numpy(np.ascontiguousarray(sub)).float().div_(255.0)
        t_in = t_in.permute(2, 0, 1).unsqueeze(0).to(device)
        return t_in

    # Prepare sample tensor & features for isolated downstream benchmarks
    t_input = bench_preprocessing_synthetic()
    with torch.inference_mode():
        feat_sample = eye.backbone(t_input)

    # B. Motion Optical Flow (Magnocellular 16x24)
    motion_eng = eye.motion_engine
    def bench_motion_flow():
        return motion_eng.process(sample_frame_synthetic)

    # C. Cortical Backbone Forward
    def bench_backbone():
        with torch.inference_mode():
            return eye.backbone(t_input)

    # D. Affordance Head Forward & Hotspot Extraction
    def bench_affordance_head():
        with torch.inference_mode():
            return eye.affordance_head.find_hotspots(
                feat_sample,
                action_type="all",
                top_k=5,
                threshold=0.35,
                margin=0.05,
            )

    # E. Foveal Acuity Inspection (Zoom crop, contrast, FovealAcuityHead forward, bbox regression)
    fovea_eng = eye.fovea_engine
    target_coord = (0.5, 0.5)
    def bench_foveal_inspection():
        with torch.inference_mode():
            return fovea_eng.inspect(sample_frame_synthetic, target_coord)

    # F. Ventral Semantic Stream
    def bench_ventral():
        with torch.inference_mode():
            return eye.ventral(feat_sample)

    # G. Semantic Scene Tree Diff & Episodic Memory Step
    affordances_sample = eye.affordance_head.find_hotspots(feat_sample, top_k=5)
    mem_mgr = eye.memory
    def bench_memory_step():
        return mem_mgr.step(
            affordances=affordances_sample,
            category="editor_code",
            category_confidence=0.95,
            foveal_inspection=None,
            moving_objects=[],
            surprise_score=0.02,
            prior_action="observe",
            window_title="VS Code",
        )

    # H. Complete End-to-End Observe Tick
    def bench_full_observe():
        return eye.observe()

    runners = [
        ("preprocessing_1080p", "Screen Preprocessing (1080p -> 128x128)", bench_preprocessing_1080p),
        ("preprocessing_synthetic", "Screen Preprocessing (Synthetic 640x360 -> 128x128)", bench_preprocessing_synthetic),
        ("motion_optical_flow", "Magnocellular Motion Flow (16x24 optical grid)", bench_motion_flow),
        ("backbone_forward", "Cortical Feature Backbone (ConvBNAct + Inverted Residuals)", bench_backbone),
        ("affordance_head", "Fast Affordance Head (NMS & Hotspot Extraction)", bench_affordance_head),
        ("foveal_inspection", "Foveal Acuity Inspection (Unscaled Crop + Acuity Head + BBox)", bench_foveal_inspection),
        ("ventral_stream", "Ventral Stream (Scene Category & Visual Latent)", bench_ventral),
        ("memory_diff_step", "Episodic Memory Diff Step (UIML Tree + Hungarian Re-ID + Diff)", bench_memory_step),
        ("full_observe_tick", "Full HumanEye.observe() End-to-End Tick", bench_full_observe),
    ]

    for key, name, fn in runners:
        # Warmup
        for _ in range(profile_warmup):
            fn()
        gc.disable()
        times = []
        for _ in range(num_profile_iters):
            t_s = time.perf_counter()
            fn()
            t_e = time.perf_counter()
            times.append((t_e - t_s) * 1000.0)
        gc.enable()

        p_stats = calculate_percentiles(times)
        subcomponent_benchmarks[key] = {
            "name": name,
            "min_ms": round(p_stats["min"], 3),
            "p50_ms": round(p_stats["p50"], 3),
            "p90_ms": round(p_stats["p90"], 3),
            "p95_ms": round(p_stats["p95"], 3),
            "p99_ms": round(p_stats["p99"], 3),
            "max_ms": round(p_stats["max"], 3),
            "mean_ms": round(p_stats["mean"], 3),
            "std_ms": round(p_stats["std"], 3),
            "fps": round(1000.0 / p_stats["p50"], 1) if p_stats["p50"] > 0 else 0.0,
        }
        print(f"  {name:62s} : p50={p_stats['p50']:6.2f}ms | p95={p_stats['p95']:6.2f}ms | FPS={1000.0/p_stats['p50']:6.1f}")

    # Calculate sub-component percentage breakdown of full observe tick
    full_observe_p50 = subcomponent_benchmarks["full_observe_tick"]["p50_ms"]
    breakdown_percentages = {}
    for comp_key in [
        "preprocessing_synthetic", "motion_optical_flow", "backbone_forward",
        "affordance_head", "ventral_stream", "memory_diff_step"
    ]:
        comp_p50 = subcomponent_benchmarks[comp_key]["p50_ms"]
        pct = (comp_p50 / full_observe_p50) * 100.0 if full_observe_p50 > 0 else 0.0
        breakdown_percentages[comp_key] = round(pct, 1)

    # Determine bottleneck
    ranked_components = sorted(
        [
            ("Preprocessing (Synthetic)", subcomponent_benchmarks["preprocessing_synthetic"]["p50_ms"]),
            ("Motion Optical Flow", subcomponent_benchmarks["motion_optical_flow"]["p50_ms"]),
            ("Cortical Backbone Forward", subcomponent_benchmarks["backbone_forward"]["p50_ms"]),
            ("Affordance Head", subcomponent_benchmarks["affordance_head"]["p50_ms"]),
            ("Ventral Stream", subcomponent_benchmarks["ventral_stream"]["p50_ms"]),
            ("Memory Diff Step", subcomponent_benchmarks["memory_diff_step"]["p50_ms"]),
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    primary_bottleneck = ranked_components[0]
    secondary_bottleneck = ranked_components[1]

    # Package complete results
    report_data = {
        "metadata": {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "processor": platform.processor(),
            "physical_cores": cpu_count_physical,
            "logical_cores": cpu_count_logical,
            "pytorch_version": torch.__version__,
            "pytorch_threads": actual_threads,
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "iterations": num_iterations,
            "warmup_iterations": warmup_iterations,
        },
        "soak_test_summary": {
            "total_iterations": num_iterations,
            "total_wall_time_s": round(total_wall_time_s, 2),
            "true_sustained_fps": round(true_sustained_fps, 2),
            "latency_percentiles_ms": {k: round(v, 2) for k, v in latency_stats.items()},
            "latency_slope_ms_per_iter": round(latency_slope_ms_per_iter, 5),
            "thermal_drift_detected": abs(latency_slope_ms_per_iter) > 0.05,
            "epoch_stability": epoch_summaries,
        },
        "memory_leak_audit": {
            "initial_rss_mb": round(initial_os_metrics["rss_mb"], 2),
            "final_rss_mb_pre_collect": round(final_os_metrics["rss_mb"], 2),
            "final_rss_mb_post_collect": round(final_os_metrics_post_collect["rss_mb"], 2),
            "peak_rss_mb": round(max(rss_history_mb), 2),
            "delta_rss_mb": round(final_os_metrics_post_collect["rss_mb"] - initial_os_metrics["rss_mb"], 2),
            "rss_slope_mb_per_iter": round(rss_slope_mb_per_iter, 6),
            "initial_vms_mb": round(initial_os_metrics["vms_mb"], 2),
            "final_vms_mb": round(final_os_metrics_post_collect["vms_mb"], 2),
            "initial_os_handles": initial_os_metrics["handles"],
            "final_os_handles": final_os_metrics["handles"],
            "delta_os_handles": final_os_metrics["handles"] - initial_os_metrics["handles"],
            "initial_gdi_objects": initial_os_metrics["gdi_objects"],
            "final_gdi_objects": final_os_metrics["gdi_objects"],
            "initial_user_objects": initial_os_metrics["user_objects"],
            "final_user_objects": final_os_metrics["user_objects"],
            "initial_gc_objects": initial_gc_objects,
            "final_gc_objects_pre_collect": final_gc_objects_pre_collect,
            "final_gc_objects_post_collect": final_gc_objects_post_collect,
            "uncollectable_garbage_cycles": uncollected_garbage_count,
            "type_distribution_deltas": type_deltas,
            "bounded_collections_audit": bounded_collections_audit,
            "leak_detected": (
                rss_slope_mb_per_iter > 0.05
                or (final_os_metrics_post_collect["rss_mb"] - initial_os_metrics["rss_mb"]) > 50.0
                or uncollected_garbage_count > 0
                or (final_gc_objects_post_collect - initial_gc_objects) > 500
            ),
        },
        "subcomponent_profiling": {
            "subcomponents": subcomponent_benchmarks,
            "breakdown_percentages": breakdown_percentages,
            "primary_bottleneck": {
                "name": primary_bottleneck[0],
                "p50_ms": primary_bottleneck[1],
                "percentage_of_tick": round((primary_bottleneck[1] / full_observe_p50) * 100.0, 1),
            },
            "secondary_bottleneck": {
                "name": secondary_bottleneck[0],
                "p50_ms": secondary_bottleneck[1],
                "percentage_of_tick": round((secondary_bottleneck[1] / full_observe_p50) * 100.0, 1),
            },
            "foveal_inspection_standalone": {
                "p50_ms": subcomponent_benchmarks["foveal_inspection"]["p50_ms"],
                "p95_ms": subcomponent_benchmarks["foveal_inspection"]["p95_ms"],
                "fps": subcomponent_benchmarks["foveal_inspection"]["fps"],
            },
        },
    }

    # Save JSON report
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "single_thread_soak_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[+] Detailed benchmark JSON exported to: {json_path}")

    return report_data


if __name__ == "__main__":
    run_stress_test(num_iterations=500, warmup_iterations=30)
