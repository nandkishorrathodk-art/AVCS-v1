"""Cross-generator grounding benchmark.

Evaluates one grounder checkpoint on two different frame generators and reports
the gap between them. A checkpoint that scores well at home and near chance
abroad has memorised its generator, not the task; see AGENTS.md
("Cross-generator grounding").

The ui_grounding split is scored by the official runner
(benchmarks/run_ui_grounding_benchmark.py) so the home number is directly
comparable to the published one. The clean-room split is scored with the same
hit rule, applied through the same model inference path:
    hit = (inside ground-truth box OR within 15 px) AND confidence >= 0.35.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch

from avcs.dataset.cleanroom_v1.dataset import CleanroomSystem1Dataset
from benchmarks.run_ui_grounding_benchmark import UIGroundingBenchmarkRunner

HIT_TOLERANCE_PX = 15.0
CONFIDENCE_THRESHOLD = 0.35
CLEANROOM_IMAGE_SIZE = 256
DEFAULT_CHECKPOINT = "runs/weights/goal_grounder_best.pt"
DEFAULT_UI_MANIFEST = "datasets/ui_grounding/manifest.json"
DEFAULT_CLEANROOM_DIR = "datasets/cleanroom_system1_v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_cleanroom(
    model: Any,
    corpus_dir: str,
    split: str = "test_heldout",
    negative_split: str = "test_negative",
) -> Dict[str, Any]:
    """Score the clean-room split with the official hit rule.

    The shipped corpus fails its own manifest hash check, so samples load with
    verification off. That is recorded in the report rather than hidden.
    """
    dataset = CleanroomSystem1Dataset(corpus_dir, split, validate_hashes=False)
    hits = 0
    hits_ungated = 0
    positives = 0
    errors: List[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if float(sample["target_present"]) <= 0.5:
            continue
        positives += 1
        pred = model.predict_point(sample["image"].float(), str(sample["query"]))
        pred_x, pred_y = pred["point"]
        confidence = float(pred["confidence"])
        gt_x, gt_y = (float(value) for value in sample["target_point"])
        x1, y1, x2, y2 = (float(value) for value in sample["target_box"])
        error_px = math.hypot(
            (pred_x - gt_x) * CLEANROOM_IMAGE_SIZE,
            (pred_y - gt_y) * CLEANROOM_IMAGE_SIZE,
        )
        errors.append(error_px)
        in_box = x1 <= pred_x <= x2 and y1 <= pred_y <= y2
        in_radius = error_px <= HIT_TOLERANCE_PX
        if in_box or in_radius:
            hits_ungated += 1
            if confidence >= CONFIDENCE_THRESHOLD:
                hits += 1

    rejections = 0
    negatives = 0
    try:
        negative_dataset = CleanroomSystem1Dataset(
            corpus_dir, negative_split, validate_hashes=False
        )
        for index in range(len(negative_dataset)):
            sample = negative_dataset[index]
            negatives += 1
            pred = model.predict_point(sample["image"].float(), str(sample["query"]))
            if float(pred["confidence"]) < CONFIDENCE_THRESHOLD:
                rejections += 1
    except (FileNotFoundError, ValueError):
        negatives = 0

    return {
        "positives": positives,
        "hits": hits,
        "top1_accuracy_pct": round(100.0 * hits / positives, 2) if positives else 0.0,
        "hits_ungated": hits_ungated,
        "top1_accuracy_pct_ungated": (
            round(100.0 * hits_ungated / positives, 2) if positives else 0.0
        ),
        "median_error_px": round(float(np.median(errors)), 2) if errors else 0.0,
        "negatives": negatives,
        "rejection_rate_pct": (
            round(100.0 * rejections / negatives, 2) if negatives else 0.0
        ),
    }


def run(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    ui_manifest: str = DEFAULT_UI_MANIFEST,
    cleanroom_dir: str = DEFAULT_CLEANROOM_DIR,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    import avcs

    device = torch.device(device_name)
    model = avcs.load_grounding_checkpoint(checkpoint, device=device)
    if not model.is_trained:
        raise ValueError(
            f"{checkpoint} does not carry training provenance; refusing to report "
            "accuracy for an untrained checkpoint."
        )

    runner = UIGroundingBenchmarkRunner(
        manifest_path=ui_manifest,
        device=device_name,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    ui_results = runner.evaluate_model(model, split="test_held_out").to_dict()
    cleanroom = _evaluate_cleanroom(model, cleanroom_dir)

    ui_accuracy = float(ui_results.get("top_1_accuracy_pct", 0.0))
    return {
        "benchmark": "cross_generator_grounding",
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256_file(Path(checkpoint)),
        "checkpoint_architecture": type(model).__module__,
        "is_trained": model.is_trained,
        "device": str(device),
        "hit_rule": "(inside gt box or within 15 px) and confidence >= 0.35",
        "ui_grounding": ui_results,
        "cleanroom_system1_v1": cleanroom,
        "ui_grounding_top1_pct": ui_accuracy,
        "cross_generator_gap_pct": round(ui_accuracy - cleanroom["top1_accuracy_pct"], 2),
        "cleanroom_manifest_hashes_verified": False,
        "cleanroom_manifest_note": (
            "The shipped corpus fails its own manifest hash check, so samples are "
            "loaded with verification off. This measures the checkpoint, not the corpus."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-generator grounding benchmark")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ui-manifest", default=DEFAULT_UI_MANIFEST)
    parser.add_argument("--cleanroom-dir", default=DEFAULT_CLEANROOM_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = run(
        checkpoint=args.checkpoint,
        ui_manifest=args.ui_manifest,
        cleanroom_dir=args.cleanroom_dir,
        device_name=args.device,
    )
    ui = report["ui_grounding"]
    cleanroom = report["cleanroom_system1_v1"]
    print("=== Cross-Generator Grounding Benchmark ===")
    print(f"Checkpoint: {report['checkpoint']} ({report['checkpoint_architecture']})")
    print(
        f"ui_grounding        : {ui['top_1_accuracy_pct']:6.2f}% "
        f"({ui['top_1_hits']}/{ui['total_positive_samples']}), "
        f"median {ui['median_centroid_error_px']} px"
    )
    print(
        f"cleanroom_system1_v1: {cleanroom['top1_accuracy_pct']:6.2f}% "
        f"({cleanroom['hits']}/{cleanroom['positives']}) gated | "
        f"{cleanroom['top1_accuracy_pct_ungated']:6.2f}% "
        f"({cleanroom['hits_ungated']}/{cleanroom['positives']}) ungated, "
        f"median {cleanroom['median_error_px']} px"
    )
    print(f"Cross-generator gap : {report['cross_generator_gap_pct']:.2f} points")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
