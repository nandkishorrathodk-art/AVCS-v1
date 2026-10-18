"""Empirical Adversarial Challenge Suite for AVCS Milestone 4 (Features 11 & 12).

Adversarial Stress Harness executing:
1. 5th-Order Minimum-Jerk Polynomial & Derivatives Stress:
   - Initial/final velocity & acceleration vanishing (identically 0.0)
   - Velocity symmetry: |v(0.5 - tau) - v(0.5 + tau)| < 1e-6 analytically and numerically
   - Flash & Hogan peak velocity bound: vmax = 1.875 * D / T (+/- 1%)
2. Boundary Distances & Numerical Stability:
   - D -> 0, zero-distance moves (start == target)
   - Max diagonal distance D -> sqrt(2) ((0, 0) -> (1, 1))
   - Extreme target widths W in [1e-4, 1.0]
   - Fitts' Law duration clamping strictly in [0.15, 0.95] s
   - Zero NaNs, zero division-by-zero, zero infs
3. Discrete AR(2) Resonator Frequency Response:
   - 10,000 frequency steps
   - Sample rates: 60, 100, 120, 240 Hz
   - Verify empirical resonant peak |H(e^jw)| is strictly 10.50 +/- 0.15 Hz
4. Terminal Landing Precision Stress:
   - 10,000 random trajectories with enable_micro_tremor=True
   - Verify terminal landing offset jitter is strictly (0.0, 0.0) on 100% of runs
   - Verify final coordinate matches target within 1e-5
5. Micro-Tremor Toggle Verification:
   - enable_micro_tremor=False produces strictly zero jitter on 100% of waypoints
   - Instance vs per-call toggle equivalence
6. CPU Latency & Throughput Benchmark:
   - Single-thread CPU latency <= 0.15 ms per trajectory (>6,000 traj/s)
   - Benchmarked across 10,000 runs
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from avcs.control.servoing import (
    NeuromuscularServoingController,
    NeuromuscularTrajectoryPoint,
)
from forge.vision.cortex.motor.driver import VirtualCursorDriver


def challenge_1_polynomial_and_derivatives() -> Dict[str, Any]:
    """Challenge 1: Verify 5th-order minimum-jerk polynomial derivatives, symmetry, and vmax."""
    print("--- Running Challenge 1: 5th-Order Minimum-Jerk Polynomial & Derivatives ---")
    driver = VirtualCursorDriver()
    controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=120, enable_micro_tremor=False)

    # 1.1 Analytical formula check on grid of tau in [0, 1]
    tau_vals = np.linspace(0.0, 1.0, 10001)
    # s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    poly = 10.0 * (tau_vals ** 3) - 15.0 * (tau_vals ** 4) + 6.0 * (tau_vals ** 5)
    # s'(tau) = 30*tau^2 - 60*tau^3 + 30*tau^4 = 30*tau^2 * (1 - tau)^2
    d_poly = 30.0 * (tau_vals ** 2) - 60.0 * (tau_vals ** 3) + 30.0 * (tau_vals ** 4)
    # s''(tau) = 60*tau - 180*tau^2 + 120*tau^3 = 60*tau*(1 - tau)*(1 - 2*tau)
    dd_poly = 60.0 * tau_vals - 180.0 * (tau_vals ** 2) + 120.0 * (tau_vals ** 3)

    # Boundary conditions
    s0, s1 = poly[0], poly[-1]
    ds0, ds1 = d_poly[0], d_poly[-1]
    dds0, dds1 = dd_poly[0], dd_poly[-1]

    assert s0 == 0.0 and s1 == 1.0, f"Polynomial boundary failure: s(0)={s0}, s(1)={s1}"
    assert ds0 == 0.0 and ds1 == 0.0, f"Velocity boundary failure: s'(0)={ds0}, s'(1)={ds1}"
    assert dds0 == 0.0 and dds1 == 0.0, f"Acceleration boundary failure: s''(0)={dds0}, s''(1)={dds1}"

    # 1.2 Velocity Symmetry test: for all tau in [0, 0.5], |s'(0.5 - tau) - s'(0.5 + tau)| < 1e-6
    delta_tau = np.linspace(0.0, 0.5, 5001)
    tau_left = 0.5 - delta_tau
    tau_right = 0.5 + delta_tau
    d_left = 30.0 * (tau_left ** 2) - 60.0 * (tau_left ** 3) + 30.0 * (tau_left ** 4)
    d_right = 30.0 * (tau_right ** 2) - 60.0 * (tau_right ** 3) + 30.0 * (tau_right ** 4)
    max_symmetry_error = float(np.max(np.abs(d_left - d_right)))
    assert max_symmetry_error < 1e-6, f"Velocity asymmetry detected: max error = {max_symmetry_error}"

    # 1.3 Flash & Hogan analytical peak speed: vmax = 1.875 * (D / T)
    # Test across 100 random combinations of D in [0.01, 1.4] and T in [0.15, 0.95]
    peak_v_errors = []
    for _ in range(100):
        start = (random.uniform(0.1, 0.4), random.uniform(0.1, 0.4))
        target = (random.uniform(0.6, 0.9), random.uniform(0.6, 0.9))
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        dur = random.uniform(0.2, 0.9)
        # Use high tick rate (e.g. 1000 Hz) to sample continuous velocity peak accurately
        c_high = NeuromuscularServoingController(driver=driver, tick_rate_hz=1000, enable_micro_tremor=False)
        pts = c_high.generate_minimum_jerk_profile(start, target, duration_s=dur)
        speeds = [math.hypot(p.vx, p.vy) for p in pts]
        measured_vmax = max(speeds)
        theoretical_vmax = 1.875 * (dist / dur)
        rel_err = abs(measured_vmax - theoretical_vmax) / theoretical_vmax
        peak_v_errors.append(rel_err)

    max_v_rel_err = float(np.max(peak_v_errors))
    mean_v_rel_err = float(np.mean(peak_v_errors))
    # Relative error must be strictly within 1% (0.01)
    assert max_v_rel_err < 0.01, f"Peak velocity error exceeded 1%: max rel error = {max_v_rel_err:.4%}"

    # 1.4 Test trajectory waypoints directly from controller
    test_pts = controller.generate_minimum_jerk_profile((0.2, 0.3), (0.7, 0.8), duration_s=0.5)
    start_v = (test_pts[0].vx, test_pts[0].vy)
    end_v = (test_pts[-1].vx, test_pts[-1].vy)
    start_a = test_pts[0].acceleration
    end_a = test_pts[-1].acceleration

    assert start_v == (0.0, 0.0) and end_v == (0.0, 0.0), f"Trajectory boundary v non-zero: start={start_v}, end={end_v}"
    assert start_a == 0.0 and end_a == 0.0, f"Trajectory boundary a non-zero: start={start_a}, end={end_a}"

    result = {
        "status": "PASS",
        "max_symmetry_error": max_symmetry_error,
        "mean_v_rel_error": mean_v_rel_err,
        "max_v_rel_error": max_v_rel_err,
        "boundary_derivatives_zero": True,
    }
    print(f"  Challenge 1 PASS: max_symmetry_err={max_symmetry_error:.2e}, max_v_rel_err={max_v_rel_err:.4%}")
    return result


def challenge_2_boundary_distances_and_stability() -> Dict[str, Any]:
    """Challenge 2: Stress test boundary distances (D=0, D->sqrt(2)), target widths, and clamping."""
    print("--- Running Challenge 2: Boundary Distances & Numerical Stability ---")
    driver = VirtualCursorDriver()
    controller = NeuromuscularServoingController(driver=driver, tick_rate_hz=120, enable_micro_tremor=True)

    test_cases = [
        # Zero distance (start == target)
        ((0.0, 0.0), (0.0, 0.0), "Zero-distance origin"),
        ((0.5, 0.5), (0.5, 0.5), "Zero-distance center"),
        ((1.0, 1.0), (1.0, 1.0), "Zero-distance max"),
        # Max diagonal distance
        ((0.0, 0.0), (1.0, 1.0), "Max diagonal full span D=sqrt(2)"),
        ((1.0, 0.0), (0.0, 1.0), "Max anti-diagonal full span D=sqrt(2)"),
        # Tiny steps
        ((0.5, 0.5), (0.5 + 1e-6, 0.5 + 1e-6), "Micro step 1e-6"),
        ((0.5, 0.5), (0.5 + 1e-4, 0.5), "Step 1e-4"),
    ]

    # Extreme target widths
    target_widths = [1e-6, 1e-4, 1e-3, 0.04, 0.5, 1.0, 10.0]

    fitts_durations = []
    nan_count = 0
    inf_count = 0

    for start, target, desc in test_cases:
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        for w in target_widths:
            dur = controller.calculate_fitts_duration(dist, target_width=w)
            fitts_durations.append(dur)
            assert 0.15 <= dur <= 0.95, f"Duration {dur} outside [0.15, 0.95] for dist={dist}, w={w}"

            # Generate full trajectory and inspect every point
            pts = controller.generate_minimum_jerk_profile(start, target, duration_s=dur)
            for pt in pts:
                for val in [pt.x, pt.y, pt.vx, pt.vy, pt.acceleration, pt.jitter_offset[0], pt.jitter_offset[1]]:
                    if math.isnan(val):
                        nan_count += 1
                    if math.isinf(val):
                        inf_count += 1

            # Check generate_trajectory contract
            traj = controller.generate_trajectory(start, target, duration_s=dur)
            assert len(traj) >= 5, f"Trajectory too short: {len(traj)}"
            assert math.isclose(traj[0][0], start[0], abs_tol=1e-5)
            assert math.isclose(traj[0][1], start[1], abs_tol=1e-5)
            assert math.isclose(traj[-1][0], target[0], abs_tol=1e-5)
            assert math.isclose(traj[-1][1], target[1], abs_tol=1e-5)

    assert nan_count == 0, f"Found {nan_count} NaNs in boundary stress test!"
    assert inf_count == 0, f"Found {inf_count} Infs in boundary stress test!"

    # Test random extreme aspect ratios
    for _ in range(500):
        x0, y0 = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
        xf, yf = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
        w = random.choice([1e-4, 1e-3, 0.01, 0.1, 1.0])
        dist = math.hypot(xf - x0, yf - y0)
        dur = controller.calculate_fitts_duration(dist, target_width=w)
        assert 0.15 <= dur <= 0.95

    result = {
        "status": "PASS",
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min_duration": min(fitts_durations),
        "max_duration": max(fitts_durations),
        "total_boundary_combinations": len(test_cases) * len(target_widths),
    }
    print(f"  Challenge 2 PASS: 0 NaNs, 0 Infs, durations clamped strictly [{min(fitts_durations):.3f}, {max(fitts_durations):.3f}] s")
    return result


def challenge_3_discrete_ar2_frequency_response() -> Dict[str, Any]:
    """Challenge 3: Discrete AR(2) frequency response |H(e^jw)| across 10,000 steps."""
    print("--- Running Challenge 3: Discrete AR(2) Frequency Response ---")
    sample_rates = [60.0, 100.0, 120.0, 240.0]
    target_freq = 10.50
    r = 0.82
    num_steps = 10000

    rates_results = {}
    all_within_tolerance = True

    for fs in sample_rates:
        w_target = 2.0 * math.pi * (target_freq / fs)
        # All-pole peak correction formula from avcs/control/servoing.py
        cos_w0 = min(1.0, max(-1.0, (2.0 * r / (1.0 + r ** 2)) * math.cos(w_target)))
        w0 = math.acos(cos_w0)
        a1 = 2.0 * r * math.cos(w0)
        a2 = -(r ** 2)

        # 10,000 discrete frequency steps across [5.0, 20.0] Hz
        freqs = np.linspace(5.0, 20.0, num_steps)
        w = 2.0 * np.pi * freqs / fs
        z = np.exp(1j * w)
        # H(z) = 1 / (1 - a1*z^-1 - a2*z^-2)
        H = 1.0 / (1.0 - a1 * (z ** -1) - a2 * (z ** -2))
        mag = np.abs(H)

        peak_idx = int(np.argmax(mag))
        peak_freq = float(freqs[peak_idx])
        peak_mag = float(mag[peak_idx])
        err = abs(peak_freq - target_freq)

        passed = err <= 0.15
        if not passed:
            all_within_tolerance = False

        rates_results[str(int(fs))] = {
            "fs_hz": fs,
            "peak_freq_hz": peak_freq,
            "peak_magnitude": peak_mag,
            "error_hz": err,
            "within_tolerance": passed,
        }
        print(f"  fs={fs:5.1f} Hz -> peak={peak_freq:.4f} Hz (error={err:.4f} Hz) -> {'PASS' if passed else 'FAIL'}")

    assert all_within_tolerance, f"AR(2) frequency response peak shifted beyond +/- 0.15 Hz tolerance: {rates_results}"

    result = {
        "status": "PASS",
        "target_frequency_hz": target_freq,
        "sample_rates": rates_results,
        "frequency_steps": num_steps,
    }
    return result


def challenge_4_landing_precision_10k_runs() -> Dict[str, Any]:
    """Challenge 4: Run 10,000 random trajectories with enable_micro_tremor=True; verify terminal offset (0.0, 0.0)."""
    print("--- Running Challenge 4: Landing Precision Stress (10,000 Trajectories) ---")
    driver = VirtualCursorDriver()
    controller = NeuromuscularServoingController(
        driver=driver,
        tick_rate_hz=120,
        enable_micro_tremor=True,
        micro_tremor_std=0.001,
    )

    num_runs = 10000
    zero_offset_count = 0
    exact_landing_count = 0
    max_landing_err = 0.0

    t0 = time.perf_counter()
    rng = random.Random(42)

    for i in range(num_runs):
        start = (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95))
        target = (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95))
        dur = rng.uniform(0.15, 0.95)

        pts = controller.generate_minimum_jerk_profile(start, target, duration_s=dur, enable_micro_tremor=True)
        # Check initial jitter offset
        init_jx, init_jy = pts[0].jitter_offset
        # Check terminal jitter offset
        term_jx, term_jy = pts[-1].jitter_offset

        if term_jx == 0.0 and term_jy == 0.0:
            zero_offset_count += 1

        # Check landing coordinate vs target
        final_x, final_y = pts[-1].x, pts[-1].y
        err = math.hypot(final_x - target[0], final_y - target[1])
        if err > max_landing_err:
            max_landing_err = err

        if err < 1e-5:
            exact_landing_count += 1

    elapsed_s = time.perf_counter() - t0
    zero_offset_pct = (zero_offset_count / num_runs) * 100.0
    exact_landing_pct = (exact_landing_count / num_runs) * 100.0

    print(f"  10,000 runs completed in {elapsed_s:.2f} s ({num_runs / elapsed_s:.0f} traj/s)")
    print(f"  Zero terminal offset rate: {zero_offset_pct:.2f}% ({zero_offset_count}/{num_runs})")
    print(f"  Exact landing rate: {exact_landing_pct:.2f}% ({exact_landing_count}/{num_runs}), max error: {max_landing_err:.2e}")

    assert zero_offset_count == num_runs, f"Terminal jitter offset non-zero on {num_runs - zero_offset_count} runs!"
    assert exact_landing_count == num_runs, f"Landing coordinate deviated on {num_runs - exact_landing_count} runs!"

    result = {
        "status": "PASS",
        "total_runs": num_runs,
        "zero_offset_count": zero_offset_count,
        "zero_offset_pct": zero_offset_pct,
        "exact_landing_count": exact_landing_count,
        "max_landing_error": max_landing_err,
        "elapsed_seconds": elapsed_s,
    }
    return result


def challenge_5_micro_tremor_toggle() -> Dict[str, Any]:
    """Challenge 5: Verify enable_micro_tremor=False produces strictly zero jitter across all waypoints."""
    print("--- Running Challenge 5: Micro-Tremor Toggle Verification ---")
    driver = VirtualCursorDriver()

    # 5.1 Test instance-level toggle
    controller_disabled = NeuromuscularServoingController(
        driver=driver,
        tick_rate_hz=120,
        enable_micro_tremor=False,
    )

    total_waypoints_checked = 0
    non_zero_jitters = 0

    rng = random.Random(123)
    for _ in range(500):
        start = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        target = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        dur = rng.uniform(0.2, 0.8)

        pts = controller_disabled.generate_minimum_jerk_profile(start, target, duration_s=dur)
        for pt in pts:
            total_waypoints_checked += 1
            if pt.jitter_offset != (0.0, 0.0):
                non_zero_jitters += 1

    assert non_zero_jitters == 0, f"Found {non_zero_jitters} non-zero jitters with enable_micro_tremor=False!"

    # 5.2 Test per-call override on an enabled controller
    controller_enabled = NeuromuscularServoingController(
        driver=driver,
        tick_rate_hz=120,
        enable_micro_tremor=True,
    )

    per_call_override_waypoints = 0
    per_call_non_zero_jitters = 0

    for _ in range(500):
        start = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        target = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        dur = rng.uniform(0.2, 0.8)

        # Override per call
        pts = controller_enabled.generate_minimum_jerk_profile(
            start, target, duration_s=dur, enable_micro_tremor=False
        )
        for pt in pts:
            per_call_override_waypoints += 1
            if pt.jitter_offset != (0.0, 0.0):
                per_call_non_zero_jitters += 1

    assert per_call_non_zero_jitters == 0, f"Found {per_call_non_zero_jitters} non-zero jitters on per-call override!"

    # 5.3 Test generate_trajectory with explicit enable_micro_tremor=False
    clean_traj_explicit_1 = controller_disabled.generate_trajectory((0.2, 0.2), (0.8, 0.8), duration_s=0.5, enable_micro_tremor=False)
    clean_traj_explicit_2 = controller_enabled.generate_trajectory((0.2, 0.2), (0.8, 0.8), duration_s=0.5, enable_micro_tremor=False)
    assert clean_traj_explicit_1 == clean_traj_explicit_2, "generate_trajectory outputs diverged with explicit enable_micro_tremor=False!"

    # 5.4 Check whether generate_trajectory() without arguments inherits instance setting enable_micro_tremor=False
    clean_traj_default = controller_disabled.generate_trajectory((0.2, 0.2), (0.8, 0.8), duration_s=0.5)
    instance_setting_inherited = (clean_traj_default == clean_traj_explicit_1)

    result = {
        "status": "PASS",
        "instance_waypoints_checked": total_waypoints_checked,
        "instance_non_zero_jitters": non_zero_jitters,
        "per_call_waypoints_checked": per_call_override_waypoints,
        "per_call_non_zero_jitters": per_call_non_zero_jitters,
        "explicit_disable_equal": True,
        "instance_setting_inherited_by_generate_trajectory": instance_setting_inherited,
    }
    print(f"  Challenge 5: explicit enable_micro_tremor=False yields zero jitter (PASS).")
    print(f"  Challenge 5 observation: instance setting inherited without argument? {instance_setting_inherited}")
    return result



def challenge_6_cpu_latency_benchmark() -> Dict[str, Any]:
    """Challenge 6: Benchmark single-thread CPU latency (verify <= 0.15 ms per trajectory, >6,000 traj/s)."""
    print("--- Running Challenge 6: Single-Thread CPU Latency Benchmark ---")
    driver = VirtualCursorDriver()
    controller = NeuromuscularServoingController(
        driver=driver,
        tick_rate_hz=120,
        enable_micro_tremor=True,
    )

    # Warmup
    for _ in range(500):
        controller.generate_trajectory((0.1, 0.1), (0.9, 0.9))

    # Benchmark 10,000 iterations
    num_iters = 10000
    latencies_us = []
    rng = random.Random(999)

    # Pre-generate start/target pairs to isolate trajectory generation time
    pairs = [
        ((rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)),
         (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)))
        for _ in range(num_iters)
    ]

    t_start = time.perf_counter()
    for start, target in pairs:
        t0 = time.perf_counter()
        controller.generate_trajectory(start, target)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)
    total_elapsed_s = time.perf_counter() - t_start

    arr = np.array(latencies_us)
    p50_us = float(np.percentile(arr, 50))
    p95_us = float(np.percentile(arr, 95))
    p99_us = float(np.percentile(arr, 99))
    mean_us = float(np.mean(arr))
    throughput_trajs_per_sec = num_iters / total_elapsed_s

    mean_ms = mean_us / 1000.0
    p50_ms = p50_us / 1000.0
    p95_ms = p95_us / 1000.0

    print(f"  Benchmark Results across {num_iters} trajectories:")
    print(f"  - Mean Latency:   {mean_ms:.4f} ms ({mean_us:.1f} us)")
    print(f"  - Median (p50):   {p50_ms:.4f} ms ({p50_us:.1f} us)")
    print(f"  - 95th percentile:{p95_ms:.4f} ms ({p95_us:.1f} us)")
    print(f"  - 99th percentile:{p99_us / 1000.0:.4f} ms ({p99_us:.1f} us)")
    print(f"  - Throughput:     {throughput_trajs_per_sec:,.0f} trajectories/sec")

    # Verification threshold: <= 0.15 ms (150 us), >6,000 traj/s
    latency_pass = (mean_ms <= 0.15) and (p50_ms <= 0.15) and (throughput_trajs_per_sec >= 6000.0)

    result = {
        "status": "PASS" if latency_pass else "FAIL",
        "iterations": num_iters,
        "mean_latency_ms": mean_ms,
        "median_p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_us / 1000.0,
        "throughput_traj_per_sec": throughput_trajs_per_sec,
        "target_ms": 0.15,
        "target_throughput": 6000.0,
        "pass_criteria": "mean <= 0.15ms and p50 <= 0.15ms and throughput >= 6000 traj/s",
        "verdict": "PASS" if latency_pass else "FAIL",
    }
    if not latency_pass:
        print(f"  Challenge 6 FAIL: Mean {mean_ms:.4f} ms > 0.15 ms target; Throughput {throughput_trajs_per_sec:,.0f} < 6,000 target.")
    else:
        print(f"  Challenge 6 PASS: Throughput {throughput_trajs_per_sec:,.0f} traj/s meets >6,000 target.")
    return result


def main():
    print("================================================================================")
    print("STARTING EMPIRICAL ADVERSARIAL CHALLENGE: AVCS FEATURES 11 & 12")
    print("================================================================================")
    all_results = {}

    t_all_start = time.perf_counter()
    all_results["challenge_1"] = challenge_1_polynomial_and_derivatives()
    all_results["challenge_2"] = challenge_2_boundary_distances_and_stability()
    all_results["challenge_3"] = challenge_3_discrete_ar2_frequency_response()
    all_results["challenge_4"] = challenge_4_landing_precision_10k_runs()
    all_results["challenge_5"] = challenge_5_micro_tremor_toggle()
    all_results["challenge_6"] = challenge_6_cpu_latency_benchmark()
    total_time = time.perf_counter() - t_all_start

    print("================================================================================")
    print(f"ALL 6 CHALLENGES COMPLETED IN {total_time:.2f} s")
    any_failed = any(v.get("status") == "FAIL" for v in all_results.values())
    if any_failed:
        print("OVERALL VERDICT: REQUEST_CHANGES (Defects found)")
    else:
        print("OVERALL VERDICT: APPROVE")
    print("================================================================================")

    # Save results to benchmarks/results/adversarial_challenge_m4.json
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adversarial_challenge_m4.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved empirical results to {out_path}")


if __name__ == "__main__":
    main()

