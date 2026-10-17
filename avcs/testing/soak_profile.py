"""CPU Latency & Memory Soak Profiler for AVCS System 1.

Profiles AVCS System 1 on 1 CPU thread over 200 continuous perception iterations:
1. Measures latency percentiles: min, p50, p90, p95, p99, and achieved FPS.
2. Monitors process memory consumption (RSS and Working Set) and OS handles across iterations.
3. Profiles latency breakdown across capture, motion optical flow, backbone inference, and describer.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import collections
import gc
import json
import os
import platform
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import torch

from avcs.controller import AVCSController, SceneObservation
from forge.vision.cortex.benchmark_real_desktop import RealisticDesktopRenderer


def format_percentiles(arr: np.ndarray) -> Dict[str, float]:
    """Computes standard latency percentiles and summary statistics in ms."""
    if len(arr) == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": round(float(np.min(arr)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr)), 3),
    }


def instrument_controller(controller: AVCSController):
    """Hooks controller components to capture latency breakdowns per perception tick."""
    latencies = {
        "capture": [],
        "motion_flow": [],
        "backbone": [],
        "eye_observe": [],
        "describer": [],
        "total": [],
    }

    current = {}

    # 1. Hook capture
    source = controller.source
    orig_grab_seq = getattr(source, "grab_with_sequence", None)
    orig_grab = getattr(source, "grab", None)

    if callable(orig_grab_seq):
        def hooked_grab_seq(*args, **kwargs):
            t0 = time.perf_counter()
            ret = orig_grab_seq(*args, **kwargs)
            current["capture"] = (time.perf_counter() - t0) * 1000.0
            return ret
        source.grab_with_sequence = hooked_grab_seq
    elif callable(orig_grab):
        def hooked_grab(*args, **kwargs):
            t0 = time.perf_counter()
            ret = orig_grab(*args, **kwargs)
            current["capture"] = (time.perf_counter() - t0) * 1000.0
            return ret
        source.grab = hooked_grab

    # 2. Hook motion optical flow
    orig_motion = controller.eye.motion_engine.process
    def hooked_motion(frame):
        t0 = time.perf_counter()
        ret = orig_motion(frame)
        current["motion_flow"] = (time.perf_counter() - t0) * 1000.0
        return ret
    controller.eye.motion_engine.process = hooked_motion

    # 3. Hook backbone inference
    orig_backbone = controller.eye.backbone.forward
    def hooked_backbone(x):
        t0 = time.perf_counter()
        ret = orig_backbone(x)
        current["backbone"] = (time.perf_counter() - t0) * 1000.0
        return ret
    controller.eye.backbone.forward = hooked_backbone

    # 4. Hook eye.observe
    orig_observe = controller.eye.observe
    def hooked_observe(*args, **kwargs):
        current["backbone"] = 0.0  # Reset in case static bypass occurs
        t0 = time.perf_counter()
        ret = orig_observe(*args, **kwargs)
        current["eye_observe"] = (time.perf_counter() - t0) * 1000.0
        return ret
    controller.eye.observe = hooked_observe

    # 5. Hook describe_screen
    orig_describe = controller.describe_screen
    def hooked_describe(*args, **kwargs):
        current.clear()
        current["capture"] = 0.0
        current["motion_flow"] = 0.0
        current["backbone"] = 0.0
        current["eye_observe"] = 0.0

        t0 = time.perf_counter()
        obs = orig_describe(*args, **kwargs)
        t_total = (time.perf_counter() - t0) * 1000.0

        cap = current.get("capture", 0.0)
        eye_obs = current.get("eye_observe", 0.0)
        describer = max(0.0, t_total - cap - eye_obs)

        latencies["capture"].append(cap)
        latencies["motion_flow"].append(current.get("motion_flow", 0.0))
        latencies["backbone"].append(current.get("backbone", 0.0))
        latencies["eye_observe"].append(eye_obs)
        latencies["describer"].append(describer)
        latencies["total"].append(t_total)

        return obs
    controller.describe_screen = hooked_describe

    return latencies


def run_profiling_session(
    name: str,
    dynamic_stimuli: bool = False,
    num_iterations: int = 250,
    warmup_iterations: int = 20,
) -> Dict[str, Any]:
    print(f"\n{'='*78}")
    print(f" Starting Soak Profiling Session: {name}")
    print(f" Iterations: {num_iterations} | 1 CPU Thread | Dynamic Stimuli: {dynamic_stimuli}")
    print(f"{'='*78}")

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    proc = psutil.Process()
    gc.collect()

    controller = AVCSController()
    backend_name = controller.capture_backend
    print(f"Capture Backend: {backend_name}")
    print(f"Model Weights Loaded: {getattr(controller.eye, 'weights_loaded', False)}")

    latencies = instrument_controller(controller)

    # Pre-render dynamic frames if needed to simulate live UI interactions
    dynamic_frames = []
    if dynamic_stimuli:
        renderer = RealisticDesktopRenderer(width=1280, height=720)
        apps = [
            renderer.render_vscode().frame,
            renderer.render_chrome().frame,
            renderer.render_excel().frame,
            renderer.render_notion().frame,
            renderer.render_terminal().frame,
        ]
        # Generate 250 dynamic frames with varying simulated cursor motion & UI updates
        for i in range(num_iterations + warmup_iterations):
            base_frame = apps[i % len(apps)].copy()
            # Simulate a moving cursor/highlight to induce optical flow and prevent static screen lock
            cx = int(640 + 300 * np.sin(i * 0.2))
            cy = int(360 + 180 * np.cos(i * 0.2))
            base_frame[max(0, cy-5):min(720, cy+5), max(0, cx-5):min(1280, cx+5)] = 255
            dynamic_frames.append(base_frame)

    # Warmup
    print(f"Running {warmup_iterations} warmup iterations...")
    for i in range(warmup_iterations):
        f = dynamic_frames[i] if dynamic_stimuli else None
        controller.describe_screen(frame=f)

    # Clear warmup measurements
    for k in latencies:
        latencies[k].clear()
    gc.collect()

    # Memory Tracking Snapshots
    memory_snapshots = []

    def record_memory(iteration: int, label: str):
        mem_info = proc.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        wset_mb = getattr(mem_info, "wset", mem_info.rss) / (1024 * 1024)
        try:
            handles = proc.num_handles()
        except Exception:
            handles = -1
        snap = {
            "iteration": iteration,
            "label": label,
            "rss_mb": round(rss_mb, 2),
            "wset_mb": round(wset_mb, 2),
            "handles": handles,
            "timestamp": time.perf_counter(),
        }
        memory_snapshots.append(snap)
        return snap

    init_snap = record_memory(0, "Baseline Start")
    print(f"Initial Memory: RSS={init_snap['rss_mb']} MB, WorkingSet={init_snap['wset_mb']} MB, Handles={init_snap['handles']}")

    checkpoint_interval = 25
    t_start_total = time.perf_counter()

    for i in range(1, num_iterations + 1):
        f = dynamic_frames[warmup_iterations + i - 1] if dynamic_stimuli else None
        controller.describe_screen(frame=f)

        if i % checkpoint_interval == 0 or i == num_iterations:
            snap = record_memory(i, f"Checkpoint Iter {i}")
            p50_curr = np.median(latencies["total"])
            print(f"  [Iter {i:3d}/{num_iterations}] RSS={snap['rss_mb']:.1f} MB, WSet={snap['wset_mb']:.1f} MB, Handles={snap['handles']}, Tick p50={p50_curr:.2f} ms")

    t_end_total = time.perf_counter()
    total_elapsed_sec = t_end_total - t_start_total

    # Cleanup
    controller.source.close() if hasattr(controller.source, "close") else None
    gc.collect()

    final_snap = record_memory(num_iterations, "Post-Soak Finished")

    # Compute Statistics
    totals = np.array(latencies["total"])
    caps = np.array(latencies["capture"])
    flows = np.array(latencies["motion_flow"])
    backs = np.array(latencies["backbone"])
    descs = np.array(latencies["describer"])
    eyes = np.array(latencies["eye_observe"])

    sustained_fps = num_iterations / total_elapsed_sec
    p50_fps = 1000.0 / np.percentile(totals, 50) if np.percentile(totals, 50) > 0 else 0.0

    # Memory drift analysis (linear regression on RSS and handles)
    iters_arr = np.array([s["iteration"] for s in memory_snapshots])
    rss_arr = np.array([s["rss_mb"] for s in memory_snapshots])
    handles_arr = np.array([s["handles"] for s in memory_snapshots])

    rss_slope_per_iter = float(np.polyfit(iters_arr, rss_arr, 1)[0]) if len(iters_arr) > 1 else 0.0
    handles_delta = int(final_snap["handles"] - init_snap["handles"])
    rss_delta = round(float(final_snap["rss_mb"] - init_snap["rss_mb"]), 2)
    wset_delta = round(float(final_snap["wset_mb"] - init_snap["wset_mb"]), 2)

    result = {
        "session_name": name,
        "capture_backend": backend_name,
        "iterations": num_iterations,
        "cpu_threads": 1,
        "dynamic_stimuli": dynamic_stimuli,
        "total_elapsed_sec": round(total_elapsed_sec, 3),
        "sustained_fps": round(sustained_fps, 2),
        "instantaneous_p50_fps": round(p50_fps, 2),
        "latencies_ms": {
            "total_perception_tick": format_percentiles(totals),
            "capture": format_percentiles(caps),
            "motion_optical_flow": format_percentiles(flows),
            "backbone_inference": format_percentiles(backs),
            "eye_observe_full": format_percentiles(eyes),
            "describer": format_percentiles(descs),
        },
        "memory_telemetry": {
            "initial_rss_mb": init_snap["rss_mb"],
            "final_rss_mb": final_snap["rss_mb"],
            "peak_rss_mb": round(float(np.max(rss_arr)), 2),
            "delta_rss_mb": rss_delta,
            "initial_working_set_mb": init_snap["wset_mb"],
            "final_working_set_mb": final_snap["wset_mb"],
            "peak_working_set_mb": round(float(np.max([s["wset_mb"] for s in memory_snapshots])), 2),
            "delta_working_set_mb": wset_delta,
            "rss_drift_mb_per_100_iters": round(rss_slope_per_iter * 100.0, 3),
            "initial_handles": init_snap["handles"],
            "final_handles": final_snap["handles"],
            "peak_handles": int(np.max(handles_arr)),
            "handles_delta": handles_delta,
            "snapshots": memory_snapshots,
        },
    }

    return result


def main():
    print("=" * 78)
    print("   AVCS SYSTEM 1: CPU LATENCY & MEMORY SOAK PROFILER (AGENT 15)")
    print("=" * 78)

    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Physical / Logical Cores: {psutil.cpu_count(logical=False)} / {psutil.cpu_count(logical=True)}")
    print(f"Total System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Single-Thread CPU Enforcement: torch.set_num_threads(1)")

    # Suite 1: Live Desktop Perception Soak (Win32FastCapture, 200 iterations)
    res_live = run_profiling_session(
        name="Live Desktop Capture (Win32FastCapture)",
        dynamic_stimuli=False,
        num_iterations=200,
        warmup_iterations=20,
    )

    # Suite 2: Continuous Dynamic Desktop Perception Soak (Realistic 5-App GUI Diversity, 200 iterations)
    res_dynamic = run_profiling_session(
        name="Continuous Dynamic Desktop (5-App Realistic GUI Diversity)",
        dynamic_stimuli=True,
        num_iterations=200,
        warmup_iterations=20,
    )

    out_dir = Path("runs/profiling")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agent15_cpu_soak_report.json"

    combined = {
        "timestamp": time.time(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "torch_version": torch.__version__,
        },
        "live_desktop_soak": res_live,
        "dynamic_desktop_soak": res_dynamic,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nProfiler benchmark JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
