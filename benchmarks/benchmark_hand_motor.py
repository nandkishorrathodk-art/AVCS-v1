"""Benchmark and mathematical validation of Flash & Hogan (1985) trajectory generation.

Audits:
1. Mathematical rigor of Flash & Hogan 5th-order polynomial jerk minimization.
2. Boundary conditions: position, velocity, acceleration, jerk.
3. Micro-tremor modeling: frequency, phase, initial jump, physiological fidelity.
4. Latency and throughput benchmarks across short, medium, and long trajectories.
"""
from __future__ import annotations

import math
import time
import numpy as np
import json

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from avcs_sdk.hand_motor import HandMotor
from avcs.control.servoing import NeuromuscularServoingController


def audit_mathematical_derivation():
    print("=" * 70)
    print("FLASH & HOGAN (1985) 5TH-ORDER MINIMUM-JERK POLYNOMIAL AUDIT")
    print("=" * 70)

    # Cost functional: J = 0.5 * int_0^1 (d^3 s / dtau^3)^2 dtau
    # Euler-Lagrange: d^6 s / dtau^6 = 0 => s(tau) = a0 + a1*tau + ... + a5*tau^5
    # BCs: s(0)=0, s'(0)=0, s''(0)=0 => a0=a1=a2=0
    # BCs: s(1)=1, s'(1)=0, s''(1)=0
    # Linear system:
    # [ 1  1  1 ] [a3]   [1]
    # [ 3  4  5 ] [a4] = [0]
    # [ 6 12 20 ] [a5]   [0]
    M = np.array([
        [1.0, 1.0, 1.0],
        [3.0, 4.0, 5.0],
        [6.0, 12.0, 20.0],
    ])
    b = np.array([1.0, 0.0, 0.0])
    coeffs = np.linalg.solve(M, b)
    a3, a4, a5 = coeffs
    print(f"Solved coefficients [a3, a4, a5]: [{a3:.4f}, {a4:.4f}, {a5:.4f}]")
    assert np.allclose(coeffs, [10.0, -15.0, 6.0]), "Coefficients do not match Flash & Hogan!"
    print("Verification: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 is mathematically EXACT.")

    # Boundary conditions check
    def s(t): return 10*t**3 - 15*t**4 + 6*t**5
    def ds(t): return 30*t**2 - 60*t**3 + 30*t**4
    def d2s(t): return 60*t - 180*t**2 + 120*t**3
    def d3s(t): return 60 - 360*t + 360*t**2

    print("\nBoundary Conditions Verification:")
    print(f"  s(0)   = {s(0.0):.6f} (target: 0.0)")
    print(f"  s'(0)  = {ds(0.0):.6f} (target: 0.0)")
    print(f"  s''(0) = {d2s(0.0):.6f} (target: 0.0)")
    print(f"  s(1)   = {s(1.0):.6f} (target: 1.0)")
    print(f"  s'(1)  = {ds(1.0):.6f} (target: 0.0)")
    print(f"  s''(1) = {d2s(1.0):.6f} (target: 0.0)")
    print(f"  Peak velocity at tau=0.5: s'(0.5) = {ds(0.5):.4f} (theoretical: 1.8750)")
    print(f"  Zero acceleration at tau=0.5: s''(0.5) = {d2s(0.5):.4f}")
    print(f"  Initial jerk s'''(0) = {d3s(0.0):.4f}, Final jerk s'''(1) = {d3s(1.0):.4f}")

    # Integrated squared jerk
    taus = np.linspace(0, 1, 100000)
    dtau = taus[1] - taus[0]
    jerk_vals = d3s(taus)
    integrated_jerk_sq = np.sum(jerk_vals**2) * dtau
    print(f"  Integrated squared jerk: {integrated_jerk_sq:.2f} (theoretical: 720.00)")


def audit_micro_tremor():
    print("\n" + "=" * 70)
    print("MICRO-TREMOR MODELING AUDIT")
    print("=" * 70)

    # In avcs_sdk/hand_motor.py:
    # tremor_x = math.sin(tau * 18.0) * 0.8 * (1.0 - tau)
    # tremor_y = math.cos(tau * 15.0) * 0.8 * (1.0 - tau)

    tau_0_x = math.sin(0.0 * 18.0) * 0.8 * (1.0 - 0.0)
    tau_0_y = math.cos(0.0 * 15.0) * 0.8 * (1.0 - 0.0)
    tau_1_x = math.sin(1.0 * 18.0) * 0.8 * (1.0 - 1.0)
    tau_1_y = math.cos(1.0 * 15.0) * 0.8 * (1.0 - 1.0)

    print(f"HandMotor micro-tremor at tau=0: x={tau_0_x:.4f} px, y={tau_0_y:.4f} px")
    print(f"HandMotor micro-tremor at tau=1: x={tau_1_x:.4f} px, y={tau_1_y:.4f} px")
    print("  -> Anomaly detected at tau=0: tremor_y = +0.8 px!")
    print("     Because cos(0) = 1.0, cur_y at tau=0 evaluates to round(y0 + 0.8) = y0 + 1.")
    print("     The initial waypoint has a 1-pixel discontinuity from start_pt!")

    # Check actual output of generate_trajectory at tau=0
    hm = HandMotor()
    traj = hm.generate_trajectory((100, 100), (500, 500), duration_s=0.35)
    print(f"  Start point requested: (100, 100)")
    print(f"  First generated waypoint: {traj[0]}")
    print(f"  Last generated waypoint: {traj[-1]}")
    if traj[0] != (100, 100):
        print(f"  [DISCONTINUITY CONFIRMED] traj[0] is {traj[0]}, jumping by {traj[0][0]-100, traj[0][1]-100} px!")
    else:
        print("  traj[0] matched start point.")

    # Frequency analysis
    # omega_x = 18.0 rad / tau_span
    # If duration T varies from 0.18s to 0.65s:
    print("\nFrequency vs Duration Analysis (HandMotor):")
    for dur in [0.18, 0.35, 0.65]:
        fx = 18.0 / (2 * math.pi * dur)
        fy = 15.0 / (2 * math.pi * dur)
        print(f"  Duration {dur:.2f}s: f_x = {fx:.2f} Hz, f_y = {fy:.2f} Hz")
    print("  -> Physiological Note: Human physiological tremor has a fixed physical resonance")
    print("     at 8-12 Hz (~10.5 Hz) driven by inferior-olive neural pacemakers, NOT a fixed cycle count per movement.")

    # Compare with NeuromuscularServoingController
    nc = NeuromuscularServoingController()
    print("\nNeuromuscularServoingController (avcs/control/servoing.py) Tremor Design:")
    print(f"  Tuned central frequency: {nc.micro_tremor_freq_hz} Hz (tick_rate: {nc.tick_rate_hz} Hz)")
    print(f"  Filter: AR(2) / 2nd-order IIR resonator with radius r=0.82, a1={nc._a1:.4f}, a2={nc._a2:.4f}")
    print(f"  Envelope: scaled_sin_table = std_scale * sin(pi * tau), strictly zero at tau=0 and tau=1")
    print(f"  Drive: Gaussian white noise pool (16,384 samples)")


def benchmark_latencies():
    print("\n" + "=" * 70)
    print("LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 70)

    hm = HandMotor()
    nc = NeuromuscularServoingController()

    cases = [
        ("Short (dist=50px)", (100, 100), (135, 135), 0.20, (0.1, 0.1), (0.135, 0.135)),
        ("Medium (dist=300px)", (100, 100), (312, 312), 0.35, (0.1, 0.1), (0.312, 0.312)),
        ("Long (dist=1000px)", (100, 100), (807, 807), 0.55, (0.1, 0.1), (0.807, 0.807)),
    ]

    iterations = 5000
    results = {}

    for label, start_px, end_px, dur_s, start_norm, end_norm in cases:
        # Warmup
        for _ in range(200):
            hm.generate_trajectory(start_px, end_px, duration_s=dur_s)
            nc.generate_trajectory(start_norm, end_norm, duration_s=dur_s, enable_micro_tremor=True)
            nc.generate_trajectory(start_norm, end_norm, duration_s=dur_s, enable_micro_tremor=False)

        # 1. HandMotor
        times_hm = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            pts = hm.generate_trajectory(start_px, end_px, duration_s=dur_s)
            t1 = time.perf_counter_ns()
            times_hm.append((t1 - t0) / 1000.0) # microseconds

        num_pts_hm = len(pts)

        # 2. NeuromuscularServoingController (with tremor)
        times_nc_tremor = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            pts_nc = nc.generate_trajectory(start_norm, end_norm, duration_s=dur_s, enable_micro_tremor=True)
            t1 = time.perf_counter_ns()
            times_nc_tremor.append((t1 - t0) / 1000.0)

        num_pts_nc = len(pts_nc)

        # 3. NeuromuscularServoingController (clean, without tremor)
        times_nc_clean = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            pts_nc_c = nc.generate_trajectory(start_norm, end_norm, duration_s=dur_s, enable_micro_tremor=False)
            t1 = time.perf_counter_ns()
            times_nc_clean.append((t1 - t0) / 1000.0)

        def stats(arr):
            arr = np.array(arr)
            return {
                "min_us": float(np.min(arr)),
                "p50_us": float(np.percentile(arr, 50)),
                "mean_us": float(np.mean(arr)),
                "p95_us": float(np.percentile(arr, 95)),
                "p99_us": float(np.percentile(arr, 99)),
                "max_us": float(np.max(arr)),
            }

        st_hm = stats(times_hm)
        st_nc_t = stats(times_nc_tremor)
        st_nc_c = stats(times_nc_clean)

        results[label] = {
            "HandMotor (avcs_sdk)": {
                "steps": num_pts_hm,
                "stats": st_hm,
                "throughput_trajectories_per_sec": int(1e6 / st_hm["mean_us"]),
                "throughput_waypoints_per_sec": int(num_pts_hm * (1e6 / st_hm["mean_us"])),
            },
            "Neuromuscular (avcs.control, tremor=True)": {
                "steps": num_pts_nc,
                "stats": st_nc_t,
                "throughput_trajectories_per_sec": int(1e6 / st_nc_t["mean_us"]),
                "throughput_waypoints_per_sec": int(num_pts_nc * (1e6 / st_nc_t["mean_us"])),
            },
            "Neuromuscular (avcs.control, tremor=False)": {
                "steps": num_pts_nc,
                "stats": st_nc_c,
                "throughput_trajectories_per_sec": int(1e6 / st_nc_c["mean_us"]),
                "throughput_waypoints_per_sec": int(num_pts_nc * (1e6 / st_nc_c["mean_us"])),
            },
        }

        print(f"\n--- {label} ---")
        print(f"HandMotor ({num_pts_hm} waypoints):")
        print(f"  Min: {st_hm['min_us']:.2f} µs | p50: {st_hm['p50_us']:.2f} µs | Mean: {st_hm['mean_us']:.2f} µs | p95: {st_hm['p95_us']:.2f} µs | p99: {st_hm['p99_us']:.2f} µs")
        print(f"  Throughput: {results[label]['HandMotor (avcs_sdk)']['throughput_trajectories_per_sec']:,} trajs/s ({results[label]['HandMotor (avcs_sdk)']['throughput_waypoints_per_sec']:,} pts/s)")
        print(f"Neuromuscular (tremor=True, {num_pts_nc} waypoints):")
        print(f"  Min: {st_nc_t['min_us']:.2f} µs | p50: {st_nc_t['p50_us']:.2f} µs | Mean: {st_nc_t['mean_us']:.2f} µs | p95: {st_nc_t['p95_us']:.2f} µs | p99: {st_nc_t['p99_us']:.2f} µs")
        print(f"  Throughput: {results[label]['Neuromuscular (avcs.control, tremor=True)']['throughput_trajectories_per_sec']:,} trajs/s")
        print(f"Neuromuscular (tremor=False, {num_pts_nc} waypoints):")
        print(f"  Min: {st_nc_c['min_us']:.2f} µs | p50: {st_nc_c['p50_us']:.2f} µs | Mean: {st_nc_c['mean_us']:.2f} µs | p95: {st_nc_c['p95_us']:.2f} µs | p99: {st_nc_c['p99_us']:.2f} µs")
        print(f"  Throughput: {results[label]['Neuromuscular (avcs.control, tremor=False)']['throughput_trajectories_per_sec']:,} trajs/s")

    with open("benchmarks/results_hand_motor_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark results to benchmarks/results_hand_motor_audit.json")


if __name__ == "__main__":
    audit_mathematical_derivation()
    audit_micro_tremor()
    benchmark_latencies()
