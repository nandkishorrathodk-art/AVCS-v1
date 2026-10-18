"""Rights-gated held-out evaluation for clean-lineage System 1 checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from avcs.dataset.cleanroom_v1.dataset import CleanroomSystem1Dataset
from avcs.dataset.cleanroom_v1.generator import _sha256, validate_dataset
from avcs.perception.system1_v1 import AVCSSystem1V1
from avcs.perception.word_recognizer import ID_TO_WORD
from avcs.training.train_system1_clean_v1 import (
    DEFAULT_ATTESTATION,
    DEFAULT_APPROVAL,
    DEFAULT_MANIFEST,
    BASELINE_CHECKPOINT,
    RightsGateError,
    collate_cleanroom_batch,
    validate_rights_attestation,
)

FROZEN_BASELINE_SHA256 = "c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4"


def _peak_resident_memory_bytes() -> tuple[int, str]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        process = get_current_process()
        ok = get_process_memory_info(
            process, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize), "Windows peak working set"
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if os.name == "darwin":
        return peak, "macOS peak resident set"
    return peak * 1024, "POSIX peak resident set (ru_maxrss KiB)"


def _iou_yxyx(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])
    intersection = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _normalized_model_box(region: Any) -> tuple[float, float, float, float]:
    y1, x1, y2, x2 = region.bbox_norm
    return float(y1), float(x1), float(y2), float(x2)


def _match_region_boxes(predicted: list[Any], truth: list[dict[str, Any]], iou_threshold: float = 0.5) -> int:
    candidates = sorted(
        (
            (_iou_yxyx(_normalized_model_box(pred), tuple(truth_item["bbox_xyxy_norm"][i] for i in (1, 0, 3, 2))), pi, gi)
            for pi, pred in enumerate(predicted)
            for gi, truth_item in enumerate(truth)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches = 0
    for iou, pred_index, truth_index in candidates:
        if iou < iou_threshold:
            break
        if pred_index in used_pred or truth_index in used_truth:
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
        matches += 1
    return matches


def evaluate_clean_candidate(
    checkpoint_path: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    attestation_path: str | Path = DEFAULT_ATTESTATION,
    output_path: str | Path | None = None,
    *,
    approval_path: str | Path | None = None,
    trust_config_path: str | Path | None = None,
    expected_trust_config_sha256: str | None = None,
    frozen_baseline_reference: bool = False,
) -> dict[str, Any]:
    """Evaluate held-out positive and negative splits without using validation data."""
    checkpoint = Path(checkpoint_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint is missing: {checkpoint}")
    manifest_validation = validate_dataset(manifest_file.parent)
    rights = validate_rights_attestation(
        attestation_path,
        manifest_validation["manifest_sha256"],
        approval_path=approval_path,
        trust_config_path=trust_config_path,
        expected_trust_config_sha256=expected_trust_config_sha256,
    )
    sidecar_path = checkpoint.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Checkpoint sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checkpoint_digest = _sha256(checkpoint)
    if sidecar.get("sha256") != checkpoint_digest:
        raise ValueError("Checkpoint SHA-256 does not match its sidecar.")
    metadata = sidecar.get("training_metadata", {})
    if frozen_baseline_reference:
        if checkpoint != BASELINE_CHECKPOINT.resolve():
            raise RightsGateError("Reference mode is restricted to the frozen baseline checkpoint path.")
        if checkpoint_digest != FROZEN_BASELINE_SHA256:
            raise RightsGateError("Frozen baseline checkpoint hash differs from the recorded reference hash.")
    else:
        if metadata.get("lineage") != "clean_random_initialization_v1":
            raise RightsGateError("Refusing to evaluate a checkpoint without clean random-initialization lineage.")
        if metadata.get("dataset", {}).get("manifest_sha256") != manifest_validation["manifest_sha256"]:
            raise RightsGateError("Checkpoint training manifest does not match the approved evaluation corpus.")
        if metadata.get("rights_attestation", {}).get("approval_sha256") != rights["approval_sha256"]:
            raise RightsGateError("Current rights attestation differs from the attestation recorded at training.")
        if (
            metadata.get("rights_attestation", {}).get("trusted_config_sha256")
            != rights["trusted_config_sha256"]
        ):
            raise RightsGateError("Current trusted-key configuration differs from the configuration recorded at training.")
    model = AVCSSystem1V1.load_from_checkpoint(str(checkpoint), device="cpu")

    heldout = CleanroomSystem1Dataset(manifest_file.parent, "test_heldout", validate_hashes=False)
    negative = CleanroomSystem1Dataset(manifest_file.parent, "test_negative", validate_hashes=False)
    heldout_loader = DataLoader(
        heldout,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cleanroom_batch,
    )
    negative_loader = DataLoader(
        negative,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cleanroom_batch,
    )

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    query_latencies: list[float] = []
    query_results: list[dict[str, Any]] = []
    word_correct = word_total = 0
    unknown_rejected = unknown_total = 0
    region_tp = region_pred = region_gt = 0
    region_frames_seen: set[str] = set()
    word_crop_count = 0
    warmup_count = 0
    start_all = time.perf_counter()
    try:
        model.eval()
        all_loaders = (heldout_loader, negative_loader)
        for loader in all_loaders:
            for batch in loader:
                frame_id = batch["frame_id"][0]
                split = batch["split"][0]
                frame_np = (batch["image"][0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                query = batch["query"][0]
                started = time.perf_counter()
                observation = model.observe_frame(frame_np, active_query=query)
                latency = time.perf_counter() - started
                if warmup_count < 2:
                    warmup_count += 1
                else:
                    query_latencies.append(latency)

                target_present = bool(batch["target_present"][0].item())
                predicted_present = observation.target_presence == "PRESENT"
                point_error_px = None
                point = observation.selected_target_norm
                if point is None and observation.target_candidates:
                    point = observation.target_candidates[0].coord_norm
                point_in_box = None
                if target_present and point is not None:
                    truth_x, truth_y = batch["target_point"][0].tolist()
                    pred_x, pred_y = point
                    point_error_px = math.hypot(
                        (pred_x - truth_x) * frame_np.shape[1],
                        (pred_y - truth_y) * frame_np.shape[0],
                    )
                    x1, y1, x2, y2 = batch["target_box"][0].tolist()
                    point_in_box = x1 <= pred_x <= x2 and y1 <= pred_y <= y2

                query_results.append({
                    "frame_id": frame_id,
                    "split": split,
                    "query": query,
                    "target_present": target_present,
                    "predicted_presence": observation.target_presence,
                    "presence_confidence": observation.presence_confidence,
                    "presence_correct": predicted_present == target_present,
                    "target_point_error_px": point_error_px,
                    "predicted_point_in_target_box": point_in_box,
                    "ambiguous": observation.is_ambiguous,
                    "uncertainty_flags": observation.uncertainty_flags,
                })

                if frame_id not in region_frames_seen:
                    region_frames_seen.add(frame_id)
                    manifest_record = next(
                        row for row in heldout.manifest["files"] + negative.manifest["files"]
                        if row["frame_id"] == frame_id
                    )
                    annotation_path = heldout.corpus_dir / manifest_record["annotation"]["relative_path"]
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                    truth_regions = annotation["regions"]
                    region_tp += _match_region_boxes(observation.detected_regions, truth_regions)
                    region_pred += len(observation.detected_regions)
                    region_gt += len(truth_regions)

                with torch.inference_mode():
                    word_crop_count += int(batch["word_crops"].shape[1])
                    word_logits = model.forward_word_recognition(batch["word_crops"][0])
                    word_prediction = word_logits.argmax(dim=1)
                    labels = batch["word_ids"][0]
                    known_mask = labels.gt(0)
                    word_correct += int((word_prediction[known_mask] == labels[known_mask]).sum().item())
                    word_total += int(known_mask.sum().item())
                    unknown_index = int(batch["unknown_word_index"][0].item())
                    unknown_total += 1
                    unknown_rejected += int(word_prediction[unknown_index].item() == 0)
    finally:
        torch.set_num_threads(old_threads)
    elapsed = time.perf_counter() - start_all

    positives = [item for item in query_results if item["target_present"]]
    negatives = [item for item in query_results if not item["target_present"]]
    point_errors = [item["target_point_error_px"] for item in positives if item["target_point_error_px"] is not None]
    memory_bytes, memory_method = _peak_resident_memory_bytes()
    report = {
        "schema": "avcs.system1.cleanroom.evaluation.v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "evaluation_mode": "frozen_baseline_reference" if frozen_baseline_reference else "clean_candidate",
        "dataset_manifest_sha256": manifest_validation["manifest_sha256"],
        "rights_attestation_sha256": rights["attestation_record_sha256"],
        "rights_approval_sha256": rights["approval_sha256"],
        "trusted_config_sha256": rights["trusted_config_sha256"],
        "device": "cpu",
        "evaluation_splits": ["test_heldout", "test_negative"],
        "validation_split_used": False,
        "training_split_used_for_metric_calculation": False,
        "sample_counts": {
            "heldout_queries": sum(item["split"] == "test_heldout" for item in query_results),
            "negative_queries": sum(item["split"] == "test_negative" for item in query_results),
            "positive_target_queries": len(positives),
            "absent_target_queries": len(negatives),
            "unique_frames": len(region_frames_seen),
            "region_ground_truth_count": region_gt,
            "word_crop_count": word_crop_count,
            "known_word_crop_count": word_total,
            "unknown_word_crop_count": unknown_total,
        },
        "metrics": {
            "interactive_region_precision_iou_0_5": region_tp / region_pred if region_pred else 0.0,
            "interactive_region_recall_iou_0_5": region_tp / region_gt if region_gt else 0.0,
            "query_presence_accuracy": sum(item["presence_correct"] for item in query_results) / len(query_results),
            "positive_point_in_box_success": (
                sum(item["predicted_point_in_target_box"] is True for item in positives)
                / sum(item["predicted_point_in_target_box"] is not None for item in positives)
                if any(item["predicted_point_in_target_box"] is not None for item in positives) else None
            ),
            "positive_point_prediction_coverage": (
                sum(item["predicted_point_in_target_box"] is not None for item in positives) / len(positives)
                if positives else None
            ),
            "median_localization_error_px": statistics.median(point_errors) if point_errors else None,
            "p95_localization_error_px": float(np.percentile(point_errors, 95)) if point_errors else None,
            "known_word_accuracy": word_correct / word_total if word_total else None,
            "unknown_word_rejection_rate": unknown_rejected / unknown_total if unknown_total else None,
            "absent_target_rejection_rate": (
                sum(item["predicted_presence"] == "ABSENT" for item in negatives) / len(negatives)
                if negatives else None
            ),
            "blank_screen_rejection_rate": (
                sum(
                    item["predicted_presence"] == "ABSENT"
                    for item in query_results
                    if item["split"] == "test_negative"
                )
                / sum(item["split"] == "test_negative" for item in query_results)
                if any(item["split"] == "test_negative" for item in query_results) else None
            ),
        },
        "performance": {
            "inference_latency_warmup_excluded_samples": len(query_latencies),
            "warmup_inference_count": warmup_count,
            "median_cpu_inference_latency_ms": float(np.percentile(query_latencies, 50) * 1000)
            if query_latencies else None,
            "p95_cpu_inference_latency_ms": float(np.percentile(query_latencies, 95) * 1000)
            if query_latencies else None,
            "cpu_queries_per_second_excluding_warmup": (
                len(query_latencies) / sum(query_latencies) if query_latencies and sum(query_latencies) > 0 else None
            ),
            "evaluation_wall_time_seconds": round(elapsed, 6),
            "peak_resident_memory_bytes": memory_bytes,
            "peak_resident_memory_method": memory_method,
        },
        "synthetic_data_warning": "These are synthetic-corpus measurements, not real-desktop accuracy claims.",
        "queries": query_results,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--trust-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--frozen-baseline-reference",
        action="store_true",
        help="Evaluate only the hash-pinned frozen baseline with this same protocol.",
    )
    args = parser.parse_args()
    report = evaluate_clean_candidate(
        args.checkpoint,
        manifest_path=args.manifest,
        attestation_path=args.attestation,
        output_path=args.output,
        approval_path=args.approval,
        trust_config_path=args.trust_config,
        expected_trust_config_sha256=os.environ.get("AVCS_RIGHTS_TRUST_CONFIG_SHA256"),
        frozen_baseline_reference=args.frozen_baseline_reference,
    )
    summary = {key: value for key, value in report.items() if key != "queries"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
