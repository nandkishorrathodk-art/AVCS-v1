"""Compare full-frame and changed-region inputs from one live desktop stream."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from avcs.training.train_temporal import (
    FramePairBuffer,
    evaluate_temporal_cortex,
    letterbox_rgb_frame,
    load_temporal_cortex_checkpoint,
)


def crop_changed_region(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    pixel_threshold: int,
    padding_fraction: float,
    minimum_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[int, int, int, int]] | None:
    """Return matched crops, fractions, and an ephemeral (left, top, right, bottom) box."""
    if (
        previous.dtype != np.uint8
        or current.dtype != np.uint8
        or previous.ndim != 3
        or current.ndim != 3
        or previous.shape != current.shape
        or previous.shape[2] != 3
    ):
        raise ValueError("Frames must be identically shaped uint8 RGB arrays.")
    height, width, _ = previous.shape
    if height < minimum_size or width < minimum_size:
        raise ValueError("Source frames must be at least minimum_size in each dimension.")
    if not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be in [0, 255].")
    if not math.isfinite(padding_fraction) or not 0.0 <= padding_fraction <= 2.0:
        raise ValueError("padding_fraction must be in [0, 2].")

    difference = np.abs(previous.astype(np.int16) - current.astype(np.int16)).mean(axis=2)
    changed = difference > pixel_threshold
    changed_count = int(np.count_nonzero(changed))
    if changed_count == 0:
        return None

    ys, xs = np.nonzero(changed)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    box_width, box_height = right - left, bottom - top
    pad_x = math.ceil(box_width * padding_fraction)
    pad_y = math.ceil(box_height * padding_fraction)
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(height, bottom + pad_y)

    if right - left < minimum_size:
        center_x = (left + right) // 2
        left = max(0, min(width - minimum_size, center_x - minimum_size // 2))
        right = left + minimum_size
    if bottom - top < minimum_size:
        center_y = (top + bottom) // 2
        top = max(0, min(height - minimum_size, center_y - minimum_size // 2))
        bottom = top + minimum_size

    roi_area = (right - left) * (bottom - top)
    return (
        previous[top:bottom, left:right].copy(),
        current[top:bottom, left:right].copy(),
        changed_count / (height * width),
        roi_area / (height * width),
        (left, top, right, bottom),
    )


def crop_static_control_region(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    active_bounds: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
    pixel_threshold: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return an equal-sized crop outside the changed bbox, if one fits."""
    height, width, _ = previous.shape
    crop_height, crop_width = crop_shape
    if not 0 < crop_height <= height or not 0 < crop_width <= width:
        raise ValueError("Control crop dimensions must fit within the source frame.")

    active_left, active_top, active_right, active_bottom = active_bounds
    max_left = width - crop_width
    max_top = height - crop_height
    stride = max(1, min(crop_width, crop_height) // 4)
    left_candidates = set(range(0, max_left + 1, stride)) | {max_left}
    top_candidates = set(range(0, max_top + 1, stride)) | {max_top}

    difference = np.abs(previous.astype(np.int16) - current.astype(np.int16)).mean(axis=2)
    changed = difference > pixel_threshold
    candidates: list[tuple[float, int, int]] = []
    active_center_x = (active_left + active_right) / 2
    active_center_y = (active_top + active_bottom) / 2
    for top in top_candidates:
        for left in left_candidates:
            right = left + crop_width
            bottom = top + crop_height
            overlaps_active = (
                left < active_right
                and right > active_left
                and top < active_bottom
                and bottom > active_top
            )
            if overlaps_active or np.any(changed[top:bottom, left:right]):
                continue
            control_center_x = left + crop_width / 2
            control_center_y = top + crop_height / 2
            distance_squared = (
                (control_center_x - active_center_x) ** 2
                + (control_center_y - active_center_y) ** 2
            )
            candidates.append((distance_squared, top, left))

    if not candidates:
        return None
    _, top, left = max(candidates)
    bottom = top + crop_height
    right = left + crop_width
    changed_fraction = float(np.mean(changed[top:bottom, left:right]))
    return (
        previous[top:bottom, left:right].copy(),
        current[top:bottom, left:right].copy(),
        changed_fraction,
    )


def crop_fixed_changed_region(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    patch_size: int,
    pixel_threshold: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    tuple[int, int, int, int],
] | None:
    """Crop a fixed square centered on changed-pixel centroid in model space."""
    if (
        previous.dtype != np.uint8
        or current.dtype != np.uint8
        or previous.ndim != 3
        or current.ndim != 3
        or previous.shape != current.shape
        or previous.shape[2] != 3
    ):
        raise ValueError("Frames must be identically shaped uint8 RGB arrays.")
    height, width, _ = previous.shape
    if not 1 <= patch_size <= min(height, width):
        raise ValueError("patch_size must fit within the input frame.")
    if not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be in [0, 255].")

    difference = np.abs(previous.astype(np.int16) - current.astype(np.int16)).mean(axis=2)
    changed = difference > pixel_threshold
    ys, xs = np.nonzero(changed)
    if len(xs) == 0:
        return None

    center_x = round(float(xs.mean()))
    center_y = round(float(ys.mean()))
    left = max(0, min(width - patch_size, center_x - patch_size // 2))
    top = max(0, min(height - patch_size, center_y - patch_size // 2))
    right = left + patch_size
    bottom = top + patch_size
    active_changed_fraction = float(np.mean(changed[top:bottom, left:right]))
    return (
        previous[top:bottom, left:right].copy(),
        current[top:bottom, left:right].copy(),
        float(len(xs) / (height * width)),
        active_changed_fraction,
        (left, top, right, bottom),
    )


def run_region_comparison(args: argparse.Namespace) -> dict[str, object]:
    if not args.confirm_live_capture:
        raise ValueError(
            "Live capture is opt-in. Pass --confirm-live-capture after closing "
            "private windows and confirming the desktop is non-sensitive."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Temporal checkpoint does not exist: {args.checkpoint}")
    if sys.platform != "win32":
        raise OSError("Live GDI region comparison is supported only on Windows.")
    if not 1.0 <= args.duration_seconds <= 120.0:
        raise ValueError("duration_seconds must be between 1 and 120.")
    if not 1.0 <= args.target_hz <= 30.0:
        raise ValueError("target_hz must be between 1 and 30.")
    if not 2 <= args.capacity <= 200:
        raise ValueError("capacity must be between 2 and 200 frame pairs.")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two.")
    if not 0 <= args.pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be between 0 and 255.")
    if not 32 <= args.fixed_patch_size <= args.frame_size:
        raise ValueError("fixed_patch_size must be between 32 and frame_size.")

    model, checkpoint = load_temporal_cortex_checkpoint(args.checkpoint, device=args.device)
    training_config = checkpoint["training_config"]
    threshold = (
        float(training_config.get("change_threshold", 0.03))
        if args.change_threshold is None
        else args.change_threshold
    )
    weight = (
        float(training_config.get("change_weight", 0.0))
        if args.change_weight is None
        else args.change_weight
    )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("change_threshold must be between 0 and 1.")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("change_weight must be non-negative.")

    full_all = FramePairBuffer(capacity=args.capacity, frame_size=(args.frame_size,) * 2)
    full_paired = FramePairBuffer(capacity=args.capacity, frame_size=(args.frame_size,) * 2)
    cropped = FramePairBuffer(capacity=args.capacity, frame_size=(args.frame_size,) * 2)
    control_cropped = FramePairBuffer(capacity=args.capacity, frame_size=(args.frame_size,) * 2)
    changed_fractions: list[float] = []
    roi_area_fractions: list[float] = []
    active_patch_changed_fractions: list[float] = []
    control_changed_fractions: list[float] = []
    capture_latencies: list[float] = []
    changed_without_control = 0
    interval = 1.0 / args.target_hz
    started = time.perf_counter()
    finish = started + args.duration_seconds
    deadline = started
    capture_calls = 0

    from avcs.capture_adapter import Win32GDICaptureAdapter

    capture = Win32GDICaptureAdapter().open()
    try:
        previous = capture.grab()
        capture_calls += 1
        while time.perf_counter() < finish and len(full_all) < args.capacity:
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
            previous_model = letterbox_rgb_frame(
                previous,
                (args.frame_size, args.frame_size),
            )
            current_model = letterbox_rgb_frame(
                current,
                (args.frame_size, args.frame_size),
            )
            full_all.append(previous_model, current_model)
            region = crop_fixed_changed_region(
                previous_model,
                current_model,
                patch_size=args.fixed_patch_size,
                pixel_threshold=args.pixel_threshold,
            )
            if region is not None:
                (
                    previous_crop,
                    current_crop,
                    changed_fraction,
                    active_patch_changed_fraction,
                    active_bounds,
                ) = region
                control = crop_static_control_region(
                    previous_model,
                    current_model,
                    active_bounds=active_bounds,
                    crop_shape=previous_crop.shape[:2],
                    pixel_threshold=args.pixel_threshold,
                )
                if control is None:
                    changed_without_control += 1
                else:
                    previous_control, current_control, control_changed_fraction = control
                    full_paired.append(previous_model, current_model)
                    cropped.append(previous_crop, current_crop)
                    control_cropped.append(previous_control, current_control)
                    changed_fractions.append(changed_fraction)
                    roi_area_fractions.append(args.fixed_patch_size**2 / (args.frame_size**2))
                    active_patch_changed_fractions.append(active_patch_changed_fraction)
                    control_changed_fractions.append(control_changed_fraction)
            previous = current
            deadline += interval
            if deadline < captured_at:
                deadline = captured_at + interval
    finally:
        capture.close()

    elapsed = time.perf_counter() - started
    if len(full_all) < 2:
        raise RuntimeError(
            f"Captured only {len(full_all)} pairs; at least two are required."
        )
    if len(cropped) < 2:
        raise RuntimeError(
            f"Only {len(cropped)} pair(s) had both a changed region and a "
            "non-overlapping static control crop; at least two are required."
        )

    full_metrics = evaluate_temporal_cortex(
        model,
        full_paired,
        batch_size=args.batch_size,
        device=args.device,
        change_threshold=threshold,
        change_weight=weight,
    )
    crop_metrics = evaluate_temporal_cortex(
        model,
        cropped,
        batch_size=args.batch_size,
        device=args.device,
        change_threshold=threshold,
        change_weight=weight,
    )
    control_metrics = evaluate_temporal_cortex(
        model,
        control_cropped,
        batch_size=args.batch_size,
        device=args.device,
        change_threshold=threshold,
        change_weight=weight,
    )

    component_names = (
        "total",
        "prediction",
        "variance",
        "decorrelation",
        "photometric",
        "motion_smoothness",
        "latent_std_mean",
        "latent_std_min",
        "latent_covariance_offdiagonal_rms",
        "predicted_motion_magnitude_norm_mean",
        "predicted_motion_magnitude_norm_p95",
        "surprise_mean",
        "surprise_per_pair_mean_p95",
    )
    session_component_differences = {
        name: {
            "active_roi": crop_metrics[name],
            "static_control": control_metrics[name],
            "active_minus_control": crop_metrics[name] - control_metrics[name],
            "full_frame": full_metrics[name],
            "active_minus_full": crop_metrics[name] - full_metrics[name],
        }
        for name in component_names
    }
    session_id = uuid.uuid4().hex
    report: dict[str, object] = {
        "session_summary": {
            "session_id": session_id,
            "evidence_unit": "single_capture_session",
            "paired_triplets": len(cropped),
            "between_session_uncertainty": None,
            "components": session_component_differences,
        },
        "checkpoint": str(args.checkpoint),
        "capture": {
            "duration_seconds": elapsed,
            "requested_duration_seconds": args.duration_seconds,
            "target_hz": args.target_hz,
            "capture_calls": capture_calls,
            "capture_p50_ms": float(np.percentile(capture_latencies, 50)),
            "capture_p95_ms": float(np.percentile(capture_latencies, 95)),
            "captured_pairs": len(full_all),
            "changed_pairs": len(cropped) + changed_without_control,
            "paired_triplets": len(cropped),
            "changed_pairs_without_static_control": changed_without_control,
            "unchanged_pairs": len(full_all) - len(cropped) - changed_without_control,
            "achieved_pair_hz": len(full_all) / max(elapsed, 1e-9),
            "frame_size": [args.frame_size, args.frame_size],
        },
        "region_detection": {
            "raw_rgb_mean_abs_difference_threshold": args.pixel_threshold,
            "fixed_patch_size": args.fixed_patch_size,
            "changed_pixel_fraction_mean": float(np.mean(changed_fractions)),
            "changed_pixel_fraction_p95": float(np.percentile(changed_fractions, 95)),
            "roi_area_fraction_mean": float(np.mean(roi_area_fractions)),
            "roi_area_fraction_p95": float(np.percentile(roi_area_fractions, 95)),
            "control_changed_pixel_fraction_mean": float(np.mean(control_changed_fractions)),
            "control_changed_pixel_fraction_max": float(np.max(control_changed_fractions)),
            "active_patch_changed_pixel_fraction_mean": float(
                np.mean(active_patch_changed_fractions)
            ),
            "control_crop_area_matches_active_crop": True,
            "control_excludes_active_region": True,
            "coordinates_saved": False,
        },
        "paired_full_frame": full_metrics,
        "changed_region_crop": crop_metrics,
        "matched_control_crop": control_metrics,
        "active_minus_control_same_pairs": {
            key: crop_metrics[key] - control_metrics[key]
            for key in component_names
        },
        "active_minus_full_same_pairs": {
            key: crop_metrics[key] - full_metrics[key]
            for key in component_names
        },
        "evaluation_loss_config": {
            "change_threshold": threshold,
            "change_weight": weight,
        },
        "raw_frames_saved": False,
        "interpretation": (
            "All three arms use the same checkpoint and exact paired triplets. "
            "The static control has the active crop's dimensions and is selected "
            "outside the changed-region bounding box; its pixels are checked "
            "against the same threshold. Active-versus-control differences "
            "still reflect spatial content and context as well as crop geometry. "
            "Metrics describe input-view sensitivity only, not proof of better "
            "grounding, motion understanding, or task performance."
        ),
    }
    if args.report:
        report["report_path"] = str(args.report)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.tmp")
        try:
            temporary.write_text(rendered + "\n", encoding="utf-8")
            temporary.replace(args.report)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live-capture", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pixel-threshold", type=int, default=8)
    parser.add_argument("--fixed-patch-size", type=int, default=32)
    parser.add_argument("--change-threshold", type=float)
    parser.add_argument("--change-weight", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_region_comparison(args)
    except KeyboardInterrupt:
        raise SystemExit("Region comparison interrupted; no report was written.")


if __name__ == "__main__":
    main()
