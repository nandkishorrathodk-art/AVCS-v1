"""Explicit-consent held-out stream evaluation for a temporal cortex checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from avcs.training.collect_temporal import collect_live_pairs
from avcs.training.train_temporal import (
    FramePairBuffer,
    compare_temporal_cortex_on_held_out_stream,
)


def run_held_out_evaluation(args: argparse.Namespace) -> dict[str, object]:
    if not args.confirm_live_capture:
        raise ValueError(
            "Held-out live capture is opt-in. Pass --confirm-live-capture after "
            "closing private windows and confirming this is a distinct Stream B."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Stream A checkpoint does not exist: {args.checkpoint}")
    if not 1.0 <= args.duration_seconds <= 120.0:
        raise ValueError("duration_seconds must be between 1 and 120.")
    if not 1.0 <= args.target_hz <= 30.0:
        raise ValueError("target_hz must be between 1 and 30.")
    if not 1 <= args.pair_stride <= 10:
        raise ValueError("pair_stride must be between 1 and 10.")
    if not 2 <= args.capacity <= 200:
        raise ValueError("capacity must be between 2 and 200 frame pairs.")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two.")
    if args.change_threshold is not None and (
        not math.isfinite(args.change_threshold)
        or not 0.0 <= args.change_threshold <= 1.0
    ):
        raise ValueError("change_threshold must be between 0 and 1.")
    if args.change_weight is not None and (
        not math.isfinite(args.change_weight) or args.change_weight < 0.0
    ):
        raise ValueError("change_weight must be non-negative.")

    buffer = FramePairBuffer(
        capacity=args.capacity,
        frame_size=(args.frame_size, args.frame_size),
    )
    print(
        "HELD-OUT STREAM B: capture a distinct, non-sensitive scene from Stream A. "
        "Frames remain in a bounded RAM buffer and are not saved.",
        flush=True,
    )
    capture_metrics = collect_live_pairs(
        buffer,
        duration_seconds=args.duration_seconds,
        target_hz=args.target_hz,
        pair_stride=args.pair_stride,
    )
    comparison = compare_temporal_cortex_on_held_out_stream(
        args.checkpoint,
        buffer,
        batch_size=args.batch_size,
        device=args.device,
        change_threshold=args.change_threshold,
        change_weight=args.change_weight,
    )
    report: dict[str, object] = {
        "operator_assigned_regime": args.regime,
        "stream_b_capture": capture_metrics,
        **comparison,
    }
    if args.report:
        report["report_path"] = str(args.report)
    rendered_report = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = args.report.with_name(f".{args.report.name}.tmp")
        try:
            temporary_report.write_text(rendered_report + "\n", encoding="utf-8")
            temporary_report.replace(args.report)
        except Exception:
            temporary_report.unlink(missing_ok=True)
            raise
    print(rendered_report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live-capture", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--regime",
        choices=("static_micro_motion", "structured_scroll", "dynamic_scene"),
        required=True,
        help="Operator annotation only; the evaluator does not detect scene type.",
    )
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--change-threshold", type=float)
    parser.add_argument("--change-weight", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON output path for metrics only; no frames are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_held_out_evaluation(args)
    except KeyboardInterrupt:
        raise SystemExit("Held-out capture interrupted; evaluation was not completed.")


if __name__ == "__main__":
    main()
