"""Explicit-consent live desktop capture and temporal-model training command."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from avcs.temporal_cortex import TemporalVisualCortex
from avcs.training.train_temporal import (
    FramePairBuffer,
    evaluate_temporal_cortex,
    train_temporal_cortex,
)


def collect_live_pairs(
    buffer: FramePairBuffer,
    *,
    duration_seconds: float,
    target_hz: float,
    pair_stride: int = 1,
    progress_interval: float = 2.0,
) -> dict[str, float | int]:
    """Capture desktop samples and retain strided resized frame pairs in RAM."""
    if sys.platform != "win32":
        raise OSError("Live GDI temporal capture is supported only on Windows.")
    if duration_seconds <= 0 or target_hz <= 0 or progress_interval <= 0 or pair_stride < 1:
        raise ValueError(
            "duration_seconds, target_hz, and progress_interval must be positive; "
            "pair_stride must be at least one."
        )

    from avcs.capture_adapter import Win32GDICaptureAdapter

    capture = Win32GDICaptureAdapter().open()
    interval = 1.0 / target_hz
    started = time.perf_counter()
    finish = started + duration_seconds
    deadline = started
    pairs = 0
    capture_calls = 0
    sampled_frames = 0
    capture_latencies: list[float] = []
    next_progress = started + progress_interval
    try:
        anchor = capture.grab()
        capture_calls += 1
        sampled_frames += 1
        while time.perf_counter() < finish and pairs < buffer.capacity:
            wait_seconds = deadline - time.perf_counter()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            if time.perf_counter() >= finish:
                break
            capture_started = time.perf_counter()
            current = capture.grab()
            captured_at = time.perf_counter()
            capture_latencies.append((captured_at - capture_started) * 1000.0)
            capture_calls += 1
            sampled_frames += 1
            if (sampled_frames - 1) % pair_stride == 0:
                buffer.append(anchor, current)
                pairs += 1
                anchor = current
            deadline += interval
            if deadline < captured_at:
                deadline = captured_at + interval
            if captured_at >= next_progress:
                elapsed = captured_at - started
                print(
                    f"Live capture active: {pairs} frame pairs; "
                    f"{pairs / max(elapsed, 1e-9):.1f} pairs/s; "
                    "frames remain in RAM only.",
                    flush=True,
                )
                next_progress = captured_at + progress_interval
    finally:
        capture.close()

    elapsed = time.perf_counter() - started
    if pairs < 2:
        raise RuntimeError(
            f"Captured only {pairs} frame pair(s); at least two are required to train."
        )
    return {
        "duration_seconds": elapsed,
        "target_hz": target_hz,
        "pairs": pairs,
        "capture_calls": capture_calls,
        "sampled_frames": sampled_frames,
        "pair_stride": pair_stride,
        "nominal_pair_interval_ms": 1000.0 * pair_stride / target_hz,
        "achieved_pair_hz": pairs / max(elapsed, 1e-9),
        "capture_p50_ms": float(np.percentile(capture_latencies, 50)),
        "capture_p95_ms": float(np.percentile(capture_latencies, 95)),
        "frame_width": buffer.frame_size[0],
        "frame_height": buffer.frame_size[1],
    }


def run_live_training(args: argparse.Namespace) -> dict[str, object]:
    """Collect explicitly authorized live pixels, train, and save metadata only."""
    if not args.confirm_live_capture:
        raise ValueError(
            "Live capture is opt-in. Pass --confirm-live-capture after closing "
            "private windows and reviewing the capture scope."
        )
    if not args.checkpoint:
        raise ValueError("Provide --checkpoint explicitly to save trained weights.")
    if not 1.0 <= args.duration_seconds <= 120.0:
        raise ValueError("duration_seconds must be between 1 and 120.")
    if not 1.0 <= args.target_hz <= 30.0:
        raise ValueError("target_hz must be between 1 and 30.")
    if not 1 <= args.pair_stride <= 10:
        raise ValueError("pair_stride must be between 1 and 10.")
    if not 2 <= args.capacity <= 200:
        raise ValueError("capacity must be between 2 and 200 frame pairs.")
    if not 1 <= args.epochs <= 100:
        raise ValueError("epochs must be between 1 and 100.")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two.")
    if not math.isfinite(args.change_threshold) or not 0.0 <= args.change_threshold <= 1.0:
        raise ValueError("change_threshold must be between 0 and 1.")
    if not math.isfinite(args.change_weight) or args.change_weight < 0:
        raise ValueError("change_weight must be non-negative.")

    buffer = FramePairBuffer(
        capacity=args.capacity,
        frame_size=(args.frame_size, args.frame_size),
    )
    estimated_pairs = min(
        args.capacity,
        int(args.duration_seconds * args.target_hz / args.pair_stride),
    )
    print(
        "LIVE DESKTOP CAPTURE: full screen is sampled in memory, aspect-preserving "
        "letterboxed into a "
        f"bounded {args.frame_size}x{args.frame_size} buffer (max {args.capacity} pairs). "
        f"Pair stride: {args.pair_stride}; change weighting: {args.change_weight:g}. "
        "No raw frames are written. Close private windows first; Ctrl+C stops capture.",
        flush=True,
    )
    capture_metrics = collect_live_pairs(
        buffer,
        duration_seconds=args.duration_seconds,
        target_hz=args.target_hz,
        pair_stride=args.pair_stride,
    )
    print(
        f"Capture complete: {len(buffer)} pairs "
        f"(target was approximately {estimated_pairs}); evaluating random initialization.",
        flush=True,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = TemporalVisualCortex(
            feature_dim=args.feature_dim,
            latent_dim=args.latent_dim,
        )
    initial_metrics = evaluate_temporal_cortex(
        model,
        buffer,
        batch_size=args.batch_size,
        device=args.device,
        change_threshold=args.change_threshold,
        change_weight=args.change_weight,
    )
    history = train_temporal_cortex(
        model,
        buffer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        model_init_seed=args.seed,
        change_threshold=args.change_threshold,
        change_weight=args.change_weight,
        checkpoint_path=Path(args.checkpoint).expanduser(),
        data_provenance={
            "source": "live_desktop_gdi",
            **capture_metrics,
            "duration_requested_seconds": args.duration_seconds,
            "pair_stride": args.pair_stride,
            "nominal_pair_interval_ms": 1000.0 * args.pair_stride / args.target_hz,
            "evaluation_scope": "same_training_buffer_only",
        },
    )
    trained_metrics = evaluate_temporal_cortex(
        model,
        buffer,
        batch_size=args.batch_size,
        device=args.device,
        change_threshold=args.change_threshold,
        change_weight=args.change_weight,
    )
    checkpoint = Path(args.checkpoint).expanduser()
    report: dict[str, object] = {
        "capture": capture_metrics,
        "initial_random_weights_on_training_buffer": initial_metrics,
        "trained_weights_on_same_training_buffer": trained_metrics,
        "training_history": history,
        "checkpoint": str(checkpoint),
        "held_out_evaluation": "not_run",
        "raw_frames_saved": False,
        "claim": "experimental training run; no generalization or task success established",
    }
    print(f"Training-buffer evaluation: before={initial_metrics}; after={trained_metrics}")
    print(f"Checkpoint saved: {checkpoint}")
    print("No held-out evaluation was run; this does not establish model readiness.")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live-capture", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--change-threshold", type=float, default=0.03)
    parser.add_argument("--change-weight", type=float, default=0.0)
    parser.add_argument("--feature-dim", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_live_training(args)
    except KeyboardInterrupt:
        raise SystemExit("Live temporal capture interrupted; no checkpoint was saved.")


if __name__ == "__main__":
    main()
