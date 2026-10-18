"""Audit saved validation-selected models on a held-out sequence split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch

from avcs.streaming_v1.data import SequenceDataset, sha256_file, validate_manifest
from avcs.streaming_v1.training import load_streaming_checkpoint


def audit_checkpoint_predictions(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    *,
    split: str = "test",
    change_threshold: float,
) -> dict[str, Any]:
    """Summarize pixel-head outputs without persisting frames or prediction maps."""
    if not np.isfinite(change_threshold) or not 0.0 < change_threshold < 1.0:
        raise ValueError("change_threshold must be finite and strictly between zero and one.")
    model, _ = load_streaming_checkpoint(checkpoint_path, device="cpu")
    dataset = SequenceDataset(dataset_dir, split)
    recording_reports: list[dict[str, Any]] = []
    all_positive_scores: list[float] = []
    all_negative_scores: list[float] = []
    total_tp = total_fp = total_fn = total_labelled = 0
    for index in range(len(dataset)):
        item = dataset[index]
        frames = item["frames"]
        state = None
        transition_reports: list[dict[str, Any]] = []
        totals = {"tp": 0, "fp": 0, "fn": 0, "predicted_positive_pixels": 0}
        labelled_transitions = 0
        for frame_index in range(frames.shape[0]):
            with torch.inference_mode():
                output, state = model.forward_step(
                    frames[frame_index : frame_index + 1],
                    state,
                    timestamp_delta=(
                        0.0
                        if frame_index == 0
                        else float(item["timestamps"][frame_index] - item["timestamps"][frame_index - 1])
                    ),
                )
            if (
                frame_index == 0
                or item["annotations"].get("change_masks") != "available"
            ):
                continue
            truth = item["change_masks"][frame_index - 1].bool()
            scores = torch.sigmoid(output["change_logits"][0, 0]).cpu()
            predicted = scores >= change_threshold
            tp = int((predicted & truth).sum())
            fp = int((predicted & ~truth).sum())
            fn = int((~predicted & truth).sum())
            predicted_count = int(predicted.sum())
            positive_scores = scores[truth].numpy()
            negative_scores = scores[~truth].numpy()
            if positive_scores.size:
                all_positive_scores.extend(positive_scores.tolist())
            if negative_scores.size:
                all_negative_scores.extend(negative_scores.tolist())
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["predicted_positive_pixels"] += predicted_count
            labelled_transitions += 1
            total_labelled += 1
            transition_reports.append({
                "transition_index": frame_index - 1,
                "timestamp": float(item["timestamps"][frame_index]),
                "pixel_label": "exact_rgb_difference",
                "ground_truth_changed_pixels": int(truth.sum()),
                "predicted_changed_pixels": predicted_count,
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "mean_score_on_changed_pixels": (
                    float(positive_scores.mean()) if positive_scores.size else None
                ),
                "mean_score_on_unchanged_pixels": (
                    float(negative_scores.mean()) if negative_scores.size else None
                ),
            })
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        recording_reports.append({
            "recording_id": item["recording_id"],
            "source_id": item["source_id"],
            "frames": int(frames.shape[0]),
            "pixel_change_annotation_status": item["annotations"].get(
                "change_masks", "not_collected"
            ),
            "labelled_transitions": labelled_transitions,
            "true_positive_pixels": tp if labelled_transitions else None,
            "false_positive_pixels": fp if labelled_transitions else None,
            "false_negative_pixels": fn if labelled_transitions else None,
            "precision": precision if labelled_transitions else None,
            "recall": recall if labelled_transitions else None,
            "f1": (
                2.0 * precision * recall / max(1e-12, precision + recall)
                if labelled_transitions
                else None
            ),
            "mean_predicted_positive_fraction": (
                totals["predicted_positive_pixels"]
                / (labelled_transitions * frames.shape[-2] * frames.shape[-1])
                if labelled_transitions
                else None
            ),
            "transitions": transition_reports,
        })
        if labelled_transitions:
            total_tp += tp
            total_fp += fp
            total_fn += fn

    has_labels = total_labelled > 0
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    return {
        "schema": "avcs.streaming_v1.2.heldout_prediction_audit",
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "dataset_manifest_sha256": validate_manifest(dataset_dir)["manifest_sha256"],
        "split": split,
        "selection_policy": "change_threshold supplied from validation-only selection; test split not used for selection",
        "change_threshold": change_threshold,
        "pixel_label_semantics": "exact RGB inequality between adjacent frames; not semantic scene change",
        "raw_frames_or_prediction_maps_persisted": False,
        "labelled_transitions": total_labelled,
        "aggregate_pixel_metrics": {
            "precision": precision if has_labels else None,
            "recall": recall if has_labels else None,
            "f1": (
                2.0 * precision * recall / max(1e-12, precision + recall)
                if has_labels
                else None
            ),
            "true_positive_pixels": total_tp if has_labels else None,
            "false_positive_pixels": total_fp if has_labels else None,
            "false_negative_pixels": total_fn if has_labels else None,
        },
        "score_means": {
            "changed_pixels": (
                statistics.fmean(all_positive_scores) if all_positive_scores else None
            ),
            "unchanged_pixels": (
                statistics.fmean(all_negative_scores) if all_negative_scores else None
            ),
        },
        "recordings": recording_reports,
        "warning": (
            "False positives are defined only against exact pixel-difference masks. "
            "No semantic event labels or human-reviewed error classes are available "
            "for this corpus."
        ),
    }


def audit_comparison(
    comparison_path: str | Path,
    dataset_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    comparison_file = Path(comparison_path).resolve()
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    expected_manifest = validate_manifest(dataset_dir)["manifest_sha256"]
    if comparison.get("dataset_manifest_sha256") != expected_manifest:
        raise ValueError("comparison report manifest hash does not match the supplied dataset.")
    reports: dict[str, Any] = {}
    for name, result in comparison.get("variants", {}).items():
        checkpoint = comparison_file.parent / name / "candidate.pt"
        if not checkpoint.is_file():
            continue
        if result.get("checkpoint_sha256") is None:
            raise ValueError(f"{name}: comparison report lacks a checkpoint hash.")
        report = audit_checkpoint_predictions(
            checkpoint,
            dataset_dir,
            split=split,
            change_threshold=float(result["selected_threshold"]),
        )
        if report["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint does not match comparison report.")
        reports[name] = report
    if not reports:
        raise FileNotFoundError("no variant checkpoints found beside the comparison report.")
    return {
        "schema": "avcs.streaming_v1.2.comparison_prediction_audit",
        "dataset_manifest_sha256": expected_manifest,
        "split": split,
        "variants": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_comparison(args.comparison, args.dataset, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "variants": {
            name: variant["aggregate_pixel_metrics"]
            for name, variant in report["variants"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
