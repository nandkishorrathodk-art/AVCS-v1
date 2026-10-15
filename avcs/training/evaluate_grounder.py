"""Evaluate AVCS grounding on the session-disjoint held-out test split."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.goal_grounder import load_grounding_checkpoint
from avcs.training.demonstrations import (
    DemonstrationClickDataset,
    DemoRecord,
    discover_demonstrations,
    fingerprint_sessions,
    split_sessions,
)
from avcs.training.train_grounder import _decode_canvas_points


def _point_metrics(
    predictions: list[tuple[float, float]],
    records: list[DemoRecord],
) -> dict[str, float]:
    if len(predictions) != len(records) or not records:
        raise ValueError("Point metrics require one prediction per demonstration")

    normalized_errors: list[float] = []
    pixel_errors: list[float] = []
    diagonal_errors: list[float] = []
    for prediction, record in zip(predictions, records):
        dx_norm = prediction[0] - record.point_norm[0]
        dy_norm = prediction[1] - record.point_norm[1]
        normalized_errors.append((dx_norm * dx_norm + dy_norm * dy_norm) ** 0.5)
        with Image.open(record.frame_path) as image:
            width, height = image.size
        dx_pixels = dx_norm * width
        dy_pixels = dy_norm * height
        pixel_error = (dx_pixels * dx_pixels + dy_pixels * dy_pixels) ** 0.5
        pixel_errors.append(pixel_error)
        diagonal_errors.append(pixel_error / (width * width + height * height) ** 0.5)

    sample_count = len(records)
    return {
        "mean_normalized_point_distance": sum(normalized_errors) / sample_count,
        "mean_pixel_point_distance": sum(pixel_errors) / sample_count,
        "pck_1_percent_diagonal": sum(error <= 0.01 for error in diagonal_errors)
        / sample_count,
        "pck_2_percent_diagonal": sum(error <= 0.02 for error in diagonal_errors)
        / sample_count,
        "pck_5_percent_diagonal": sum(error <= 0.05 for error in diagonal_errors)
        / sample_count,
    }


def evaluate_grounder(
    *,
    data_dir: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,  # noqa: UP045
    batch_size: int = 16,
    device_name: Optional[str] = None,  # noqa: UP045
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("grounder checkpoint must be a metadata dictionary")
    split_seed = checkpoint.get("split_seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("grounder checkpoint has no valid session split seed")
    sessions = discover_demonstrations(data_dir)
    train_records, validation_records, test_records = split_sessions(
        sessions, seed=split_seed
    )
    if (
        len(train_records) != checkpoint.get("training_samples")
        or len(validation_records) != checkpoint.get("validation_samples")
        or len(test_records) != checkpoint.get("held_out_test_samples")
    ):
        raise ValueError(
            "Demonstration dataset no longer matches the checkpoint's recorded "
            "session-disjoint split."
        )
    expected_fingerprints = {
        "training_session_fingerprint": fingerprint_sessions(data_dir, train_records),
        "validation_session_fingerprint": fingerprint_sessions(data_dir, validation_records),
        "held_out_test_session_fingerprint": fingerprint_sessions(data_dir, test_records),
    }
    if any(checkpoint.get(key) != value for key, value in expected_fingerprints.items()):
        raise ValueError(
            "Demonstration data changed since this checkpoint was trained; "
            "held-out evaluation is invalid."
        )

    model = load_grounding_checkpoint(checkpoint_path)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)
    input_size = model.input_size
    test_dataset = DemonstrationClickDataset(test_records, input_size=input_size)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors_norm: list[float] = []
    errors_pixels: list[float] = []
    relative_diagonal_errors: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["query_tokens"].to(device))
            predicted_canvas = _decode_canvas_points(logits).cpu()
            letterbox = batch["letterbox"]
            original_x = (predicted_canvas[:, 0] - letterbox[:, 0]) / letterbox[:, 2]
            original_y = (predicted_canvas[:, 1] - letterbox[:, 1]) / letterbox[:, 3]
            predicted_original = torch.stack((original_x, original_y), dim=1)
            target_original = batch["point_norm"]
            error = predicted_original - target_original
            errors_norm.extend(torch.linalg.vector_norm(error, dim=1).tolist())
            for idx, _point in enumerate(batch["point_norm"]):
                frame_index = len(errors_pixels)
                frame_path = test_records[frame_index].frame_path
                with Image.open(frame_path) as image:
                    width, height = image.size
                dx = float(error[idx, 0]) * width
                dy = float(error[idx, 1]) * height
                pixel_distance = (dx * dx + dy * dy) ** 0.5
                errors_pixels.append(pixel_distance)
                relative_diagonal_errors.append(
                    pixel_distance / (width * width + height * height) ** 0.5
                )

    if len(errors_norm) != len(test_records) or not errors_norm:
        raise RuntimeError("Held-out test evaluation did not cover all test samples")
    training_points = [record.point_norm for record in train_records]
    mean_training_point = (
        sum(point[0] for point in training_points) / len(training_points),
        sum(point[1] for point in training_points) / len(training_points),
    )
    center_baseline = _point_metrics(
        [(0.5, 0.5)] * len(test_records), test_records
    )
    training_mean_baseline = _point_metrics(
        [mean_training_point] * len(test_records), test_records
    )
    report = {
        "dataset_provenance": "recorded_ui_click_demonstrations",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "recording_sessions": len(sessions),
        "held_out_test_samples": len(test_records),
        "session_split_seed": split_seed,
        "mean_normalized_point_distance": sum(errors_norm) / len(errors_norm),
        "median_normalized_point_distance": statistics.median(errors_norm),
        "mean_pixel_point_distance": sum(errors_pixels) / len(errors_pixels),
        "pck_1_percent_diagonal": sum(value <= 0.01 for value in relative_diagonal_errors)
        / len(relative_diagonal_errors),
        "pck_2_percent_diagonal": sum(value <= 0.02 for value in relative_diagonal_errors)
        / len(relative_diagonal_errors),
        "pck_5_percent_diagonal": sum(value <= 0.05 for value in relative_diagonal_errors)
        / len(relative_diagonal_errors),
        "baselines": {
            "screen_center": center_baseline,
            "training_mean_click": training_mean_baseline,
        },
        "model_beats_training_mean_baseline": (
            sum(errors_norm) / len(errors_norm)
            < training_mean_baseline["mean_normalized_point_distance"]
        ),
        "target_box_hit_rate": "not_measurable; recorder stores click points, not target boxes",
        "llm_or_desktop_actions_used": False,
    }
    if output_path:
        destination = Path(output_path).expanduser()
        if destination.exists():
            raise FileExistsError(f"Evaluation report already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AVCS grounder on its held-out demo sessions."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:  # noqa: UP045
    args = build_parser().parse_args(argv)
    report = evaluate_grounder(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
