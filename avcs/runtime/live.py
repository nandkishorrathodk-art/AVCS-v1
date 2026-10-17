"""AVCS Real-Time Desktop Stream: Integrated Biological Eyes + Hands Engine.

Provides an end-to-end, real-time streaming environment combining:
1. Biological Eyes: Continuous 30-60 FPS visual perception, retinal foveation,
   optical motion tracking, and dorsal affordance localization.
2. Neuromuscular Hands: Closed-loop minimum-jerk physical cursor steering,
   Fitts' Law velocity profiles, and optical ripple post-click verification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time
from typing import Optional, Tuple

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import avcs
from avcs.gui import launch_gui
from avcs.control.servoing import NeuromuscularServoingController
from avcs.control.verifier import VisualRippleVerifier
from forge.vision.cortex.motor import get_default_driver


def run_eyes_and_hands_stream(
    seconds: float = 10.0,
    fps: int = 60,
    capture_mode: str = "desktop",
    auto_servo: bool = False,
    virtual_only: bool = False,
) -> None:
    """Streams live desktop perception with continuous Eyes telemetry and Hand coordination."""
    print("=" * 78)
    print("    AVCS // REAL-TIME DESKTOP STREAM: BIOLOGICAL EYES + HANDS ENGINE    ")
    print("=" * 78)
    print(f"[*] Initializing Biological Vision Stream (Source: {capture_mode}, Target: {fps} FPS)...")

    client = avcs.eye(source=capture_mode, fps=fps, force_virtual_motor=virtual_only)
    state = client.perceive()
    print(f"[*] Human Eye Online. Initial Gaze Fixation: ({state.gaze_fixation[0]:.3f}, {state.gaze_fixation[1]:.3f})")
    print(f"[*] Motor Servoing Driver: {'Virtual Driver' if virtual_only else 'Native Win32 Driver'}")
    print("=" * 78)
    print("STREAMING LIVE DESKTOP FRAMES (Press Ctrl+C to stop)...")

    start_t = time.perf_counter()
    tick_count = 0
    last_servo_time = 0.0

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    try:
        while True:
            if seconds > 0 and (time.perf_counter() - start_t) >= seconds:
                break

            t0 = time.perf_counter()
            state = client.perceive()
            tick_count += 1

            moving_str = f"{len(state.moving_objects)} obj" if state.moving_objects else "static"
            aff_count = len(state.affordance_hotspots) if state.affordance_hotspots else 0

            # Real-time streaming status line
            sys.stdout.write(
                f"\r[TICK #{state.frame_index:04d}] "
                f"FPS: {state.fps:4.1f} | "
                f"Latency: {state.latency_ms:4.1f}ms | "
                f"Gaze: ({state.gaze_fixation[0]:.2f}, {state.gaze_fixation[1]:.2f}) | "
                f"Affordances: {aff_count:2d} | "
                f"Motion: {state.motion_magnitude:.3f} ({moving_str}) | "
                f"Stability: {state.stability:<8}"
            )
            sys.stdout.flush()

            # Auto-servo demonstration: smoothly steer hand to top affordance every 3 seconds
            if auto_servo and (time.perf_counter() - last_servo_time) >= 3.0:
                last_servo_time = time.perf_counter()
                if state.affordance_hotspots:
                    best = max(state.affordance_hotspots, key=lambda a: a[3])
                    target = (best[0], best[1])
                    sys.stdout.write(f"\n  [HAND] Neuromuscular servoing to '{best[2]}' at ({target[0]:.2f}, {target[1]:.2f})...\n")
                    res = client.move_to(target=target, realtime=True)
                    sys.stdout.write(f"  [HAND] Docked in {res.duration_ms:.0f}ms (Fitts ID: {res.fitts_index_of_difficulty:.2f} bits)\n")

            elapsed = time.perf_counter() - t0
            sleep_dt = max(0.0, (1.0 / fps) - elapsed)
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    except KeyboardInterrupt:
        print("\n\nStream paused by operator.")
    finally:
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass

    total_time = max(0.001, time.perf_counter() - start_t)
    effective_fps = tick_count / total_time
    usage = client.get_usage()
    receipt = client.generate_receipt()
    client.close()

    print("\n" + "-" * 78)
    print("STREAM SESSION SUMMARY:")
    print(f"  * Runtime                 : {total_time:.2f} seconds")
    print(f"  * Total Processed Frames  : {tick_count} frames")
    print(f"  * Sustained Vision FPS    : {effective_fps:.1f} FPS")
    print(f"  * Median Latency          : {client.eye.p50_latency_ms:.2f} ms")
    print(f"  * Cryptographic Receipt   : {receipt.receipt_id}")
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVCS Live Desktop Stream - Real-Time Biological Eyes + Hands Engine"
    )
    parser.add_argument("--gui", action="store_true", help="Launch native Desktop HUD workstation")
    parser.add_argument("--seconds", type=float, default=0.0, help="Stream duration in seconds (0 for indefinite)")
    parser.add_argument("--fps", type=int, default=30, help="Perception stream frame rate (default: 30)")
    parser.add_argument("--auto-servo", action="store_true", help="Periodically steer hand to detected affordances")
    parser.add_argument("--virtual", action="store_true", help="Force virtual cursor motor driver")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic desktop frames instead of live monitor")
    args = parser.parse_args()

    if args.gui:
        print("Launching AVCS Real-Time Desktop Cortex HUD...")
        launch_gui()
    else:
        capture_mode = "synthetic" if args.synthetic else "desktop"
        run_eyes_and_hands_stream(
            seconds=args.seconds,
            fps=args.fps,
            capture_mode=capture_mode,
            auto_servo=args.auto_servo,
            virtual_only=args.virtual,
        )


if __name__ == "__main__":
    main()
