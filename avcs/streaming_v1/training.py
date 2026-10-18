"""Rights-gated clean-lineage sequence training and held-out evaluation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from avcs.streaming_v1.data import SequenceDataset, sha256_file, validate_manifest
from avcs.streaming_v1.model import StreamingVisualCortexV1


def _approval(manifest_sha256: str, approval_path: Path, trust_path: Path) -> dict[str, Any]:
    trust_pin = os.environ.get("AVCS_RIGHTS_TRUST_CONFIG_SHA256")
    if not trust_pin:
        raise RuntimeError(
            "No process-level AVCS_RIGHTS_TRUST_CONFIG_SHA256 pin is set; "
            "new temporal data requires its own manifest-bound signed approval."
        )
    from avcs.training.cleanroom_rights import validate_signed_rights_approval

    return validate_signed_rights_approval(
        approval_path,
        manifest_sha256,
        trust_path,
        trust_pin,
    )


def _require_annotations(
    dataset_dir: str | Path,
    splits: set[str],
    required: set[str],
) -> None:
    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["recordings"]:
        if record["split"] not in splits:
            continue
        unavailable = sorted(
            name
            for name in required
            if record["annotations"].get(name) != "available"
        )
        if unavailable:
            raise ValueError(
                f"{record['recording_id']}: current pipeline requires available "
                f"annotations {unavailable}; missing or ambiguous labels are not treated as negatives."
            )


def temporal_losses(
    model: StreamingVisualCortexV1,
    batch: dict[str, Any],
    *,
    change_positive_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    if not math.isfinite(change_positive_weight) or change_positive_weight <= 0:
        raise ValueError("change_positive_weight must be positive and finite.")
    frames = batch["frames"]
    outputs = model.forward_sequence(frames, batch.get("timestamps"))
    if frames.shape[1] < 3:
        raise ValueError("Training sequence must have at least three frames.")
    predicted = outputs["predicted_next_latent"][:, :-1]
    next_latent = outputs["latent"][:, 1:].detach()
    prediction = F.smooth_l1_loss(predicted, next_latent)

    change_truth = batch["change_masks"].float().unsqueeze(2)
    change_logits = outputs["change_logits"][:, 1:]
    change_values = F.binary_cross_entropy_with_logits(
        change_logits,
        change_truth,
        pos_weight=change_logits.new_tensor(change_positive_weight),
        reduction="none",
    )
    annotation_rows = batch.get("annotation_valid")
    if annotation_rows is None:
        change_valid = change_logits.new_ones((change_logits.shape[0],))
        motion_valid = change_valid
    else:
        change_valid = change_logits.new_tensor([
            bool(row.get("change_masks", False)) for row in annotation_rows
        ])
        motion_valid = change_logits.new_tensor([
            bool(
                row.get("motion_grid_displacement", False)
                and row.get("motion_mask", False)
            )
            for row in annotation_rows
        ])
    change_weights = change_valid.view(-1, 1, 1, 1, 1)
    change = (change_values * change_weights).sum() / (
        change_weights.sum() * change_values[0].numel()
    ).clamp_min(1.0)

    predicted_motion = outputs["motion_grid_displacement"][:, 1:]
    flow_truth = batch["motion_grid_displacement"].float().permute(0, 1, 4, 2, 3)
    flow_truth = F.interpolate(
        flow_truth.flatten(0, 1),
        size=predicted_motion.shape[-2:],
        mode="area",
    ).view_as(predicted_motion)
    flow_mask = batch["motion_mask"].float().unsqueeze(2)
    flow_mask = F.interpolate(
        flow_mask.flatten(0, 1),
        size=predicted_motion.shape[-2:],
        mode="area",
    ).view_as(outputs["motion_grid_displacement"][:, 1:, :1])
    flow_valid_mask = flow_mask * motion_valid.view(-1, 1, 1, 1, 1)
    flow_error = (predicted_motion - flow_truth).abs()
    motion = (flow_error * flow_valid_mask).sum() / (
        flow_valid_mask.sum() * 2.0
    ).clamp_min(1.0)

    latents = outputs["latent"][:, 1:].flatten(0, 1)
    centered = latents - latents.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(0.5 - std).square().mean()
    total = prediction + 0.5 * change + 2.0 * motion + 0.05 * variance
    return {
        "total": total,
        "future_feature_prediction": prediction,
        "change_bce": change,
        "motion_endpoint_l1": motion,
        "representation_variance": variance,
    }


def _collate_sequences(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = {
        "frames",
        "timestamps",
        "change_masks",
        "motion_grid_displacement",
        "motion_mask",
    }
    result: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key in tensor_keys:
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def train_streaming_v1(
    dataset_dir: str | Path,
    output_checkpoint: str | Path,
    *,
    approval_path: str | Path,
    trust_config_path: str | Path,
    epochs: int = 12,
    batch_size: int = 2,
    seed: int = 731,
    learning_rate: float = 1e-3,
    patience: int = 3,
    resume: bool = False,
    model_config: dict[str, int | float | str | bool] | None = None,
    change_positive_weight: float = 1.0,
) -> dict[str, Any]:
    """Train on recording-level train/validation splits after signed approval."""
    if (
        epochs < 1
        or batch_size < 2
        or patience < 1
        or learning_rate <= 0
        or seed < 0
        or not math.isfinite(change_positive_weight)
        or change_positive_weight <= 0
    ):
        raise ValueError("invalid epochs, batch_size, patience, learning_rate, or seed.")
    root = Path(dataset_dir).resolve()
    validation = validate_manifest(root)
    _require_annotations(
        root,
        {"train", "validation"},
        {"change_masks", "motion_grid_displacement", "motion_mask"},
    )
    rights = _approval(
        validation["manifest_sha256"],
        Path(approval_path),
        Path(trust_config_path),
    )
    if rights.get("permissions", {}).get("internal_training") != "GRANTED":
        raise RuntimeError("signed approval does not grant internal training for this manifest.")
    checkpoint_path = Path(output_checkpoint).resolve()
    resume_path = checkpoint_path.with_suffix(".resume.pt")
    report_path = checkpoint_path.with_suffix(".training.json")
    sidecar_path = checkpoint_path.with_suffix(".json")
    if checkpoint_path.exists() or sidecar_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing training artifact for: {checkpoint_path}")
    if resume and not resume_path.is_file():
        raise FileNotFoundError(f"resume was requested but no recovery checkpoint exists: {resume_path}")
    if not resume and resume_path.exists():
        raise FileExistsError(f"interrupted-run checkpoint exists; pass resume=True: {resume_path}")
    config = {
        "feature_dim": 24,
        "hidden_dim": 32,
        "latent_dim": 48,
        "max_state_gap_seconds": 2.0,
        **(model_config or {}),
    }
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = StreamingVisualCortexV1(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    training_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "seed": seed,
        "change_positive_weight": change_positive_weight,
    }
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_loss = math.inf
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    stale_epochs = 0
    if resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (
            saved.get("format") != "avcs.streaming_v1.resume"
            or saved.get("manifest_sha256") != validation["manifest_sha256"]
            or saved.get("model_config") != config
            or saved.get("seed") != seed
            or saved.get("training_config") != training_config
        ):
            raise ValueError("resume checkpoint is incompatible with this run/data/config.")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = saved["history"]
        start_epoch = saved["epoch"]
        best_loss = saved["best_loss"]
        best_epoch = saved["best_epoch"]
        best_state = saved["best_state"]
        stale_epochs = saved["stale_epochs"]
        generator.set_state(saved["dataloader_generator_state"])

    train_data = SequenceDataset(root, "train")
    validation_data = SequenceDataset(root, "validation")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_sequences,
        generator=generator,
    )
    val_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_sequences,
    )

    def run_epoch(loader: DataLoader, training: bool) -> dict[str, float]:
        model.train(training)
        sums: dict[str, float] = {}
        count = 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for batch in loader:
                batch = {
                    key: value
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                if training:
                    optimizer.zero_grad(set_to_none=True)
                losses = temporal_losses(
                    model,
                    batch,
                    change_positive_weight=change_positive_weight,
                )
                if training:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                for name, value in losses.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach())
                count += 1
        if not count:
            raise RuntimeError("no sequence batches were produced.")
        return {key: value / count for key, value in sums.items()}

    started = time.perf_counter()
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            train_metrics = run_epoch(train_loader, True)
            val_metrics = run_epoch(val_loader, False)
            history.append({
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            })
            if val_metrics["total"] < best_loss:
                best_loss = val_metrics["total"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            _atomic_torch_save(
                {
                    "format": "avcs.streaming_v1.resume",
                    "manifest_sha256": validation["manifest_sha256"],
                    "model_config": config,
                    "training_config": training_config,
                    "seed": seed,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "stale_epochs": stale_epochs,
                    "dataloader_generator_state": generator.get_state(),
                    "history": history,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                resume_path,
            )
            if stale_epochs >= patience:
                break
    except KeyboardInterrupt:
        raise
    duration = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    checkpoint = {
        "format": "avcs.streaming_v1.checkpoint",
        "format_version": 1,
        "architecture": StreamingVisualCortexV1.model_name,
        "model_config": config,
        "model_state_dict": model.state_dict(),
        "lineage": {
            "initialization": "random PyTorch module initialization",
            "seed": seed,
            "pretrained_weights_loaded": False,
            "source_checkpoint_hashes": [],
        },
        "dataset_manifest_sha256": validation["manifest_sha256"],
        "rights_approval": rights,
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "change_positive_weight": change_positive_weight,
            "device": "cpu",
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "history": history,
            "elapsed_seconds": duration,
        },
    }
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
    _atomic_torch_save(checkpoint, checkpoint_path)
    digest = sha256_file(checkpoint_path)
    sidecar = {
        "format": checkpoint["format"],
        "checkpoint_sha256": digest,
        "dataset_manifest_sha256": validation["manifest_sha256"],
        "architecture": StreamingVisualCortexV1.model_name,
        "lineage": checkpoint["lineage"],
        "training": checkpoint["training"],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": "avcs.streaming_v1.training_report",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "sidecar": str(sidecar_path),
        "resume_checkpoint": str(resume_path),
        "dataset_manifest_sha256": validation["manifest_sha256"],
        "dataset_recordings": validation["recording_counts"],
        "lineage": checkpoint["lineage"],
        "training": checkpoint["training"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resume_path.unlink(missing_ok=True)
    return report


def load_streaming_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[StreamingVisualCortexV1, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if sidecar.get("checkpoint_sha256") != digest:
        raise ValueError("streaming checkpoint digest does not match sidecar.")
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        payload.get("format") != "avcs.streaming_v1.checkpoint"
        or payload.get("format_version") != 1
        or payload.get("architecture") != StreamingVisualCortexV1.model_name
    ):
        raise ValueError("unsupported streaming checkpoint format.")
    model = StreamingVisualCortexV1(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def evaluate_streaming_checkpoint(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    *,
    split: str = "test",
    device: str = "cpu",
    change_threshold: float = 0.5,
    report_path: str | Path | None = None,
    include_baselines: bool = True,
    _model_override: StreamingVisualCortexV1 | None = None,
    _reset_recurrence: bool = False,
) -> dict[str, Any]:
    """Evaluate source-held-out change, flow, prediction and continuity tasks."""
    if not math.isfinite(change_threshold) or not 0.0 < change_threshold < 1.0:
        raise ValueError("change_threshold must be finite and strictly between zero and one.")
    model, payload = (
        load_streaming_checkpoint(checkpoint_path, device=device)
        if _model_override is None
        else (_model_override.to(device).eval(), torch.load(
            checkpoint_path, map_location=device, weights_only=True
        ))
    )
    dataset = SequenceDataset(dataset_dir, split)
    records: list[dict[str, Any]] = []
    model_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    all_change_tp = all_change_fp = all_change_fn = 0
    stable_cosines: list[float] = []
    semantic_stable_cosines: list[float] = []
    semantic_event_counts = {
        name: {"positive": 0, "negative": 0, "unknown": 0}
        for name in (
            "stable",
            "pixel_noise_only",
            "motion",
            "appearance",
            "disappearance",
            "occlusion",
            "scene_transition",
            "unexpected_event",
        )
    }
    def bootstrap_interval(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"recordings": 0, "low_95": None, "high_95": None}
        if len(values) == 1:
            return {"recordings": 1, "low_95": values[0], "high_95": values[0]}
        seed_value = int(validate_manifest(dataset_dir)["manifest_sha256"][:8], 16)
        rng = np.random.default_rng(seed_value)
        samples = np.asarray(values, dtype=np.float64)
        means = np.asarray(
            [
                rng.choice(samples, size=samples.size, replace=True).mean()
                for _ in range(2000)
            ]
        )
        return {
            "recordings": len(values),
            "low_95": float(np.percentile(means, 2.5)),
            "high_95": float(np.percentile(means, 97.5)),
        }

    def bootstrap_change_f1_interval(
        recording_rows: list[dict[str, Any]],
    ) -> dict[str, float | int | None]:
        if not recording_rows:
            return {"recordings": 0, "low_95": None, "high_95": None}
        if len(recording_rows) == 1:
            row = recording_rows[0]
            score = 2 * row["change_tp"] / max(
                1, 2 * row["change_tp"] + row["change_fp"] + row["change_fn"]
            )
            return {"recordings": 1, "low_95": score, "high_95": score}
        seed_value = int(validate_manifest(dataset_dir)["manifest_sha256"][:8], 16)
        rng = np.random.default_rng(seed_value)
        tp = np.asarray([row["change_tp"] for row in recording_rows], dtype=np.float64)
        fp = np.asarray([row["change_fp"] for row in recording_rows], dtype=np.float64)
        fn = np.asarray([row["change_fn"] for row in recording_rows], dtype=np.float64)
        choices = rng.integers(0, len(recording_rows), size=(2000, len(recording_rows)))
        sampled_tp = tp[choices].sum(axis=1)
        sampled_fp = fp[choices].sum(axis=1)
        sampled_fn = fn[choices].sum(axis=1)
        scores = 2 * sampled_tp / np.maximum(
            1, 2 * sampled_tp + sampled_fp + sampled_fn
        )
        return {
            "recordings": len(recording_rows),
            "low_95": float(np.percentile(scores, 2.5)),
            "high_95": float(np.percentile(scores, 97.5)),
        }

    for index in range(len(dataset)):
        item = dataset[index]
        state = None
        sequence_latencies: list[float] = []
        change_tp = change_fp = change_fn = 0
        change_labelled_transitions = 0
        semantic_labelled_transitions = 0
        sequence_flow: list[float] = []
        sequence_prediction: list[float] = []
        sequence_stable_cosine: list[float] = []
        frames = item["frames"].unsqueeze(0).to(device)
        with torch.inference_mode():
            for t in range(frames.shape[1]):
                started = time.perf_counter()
                output, next_state = model.forward_step(
                    frames[:, t],
                    state,
                    timestamp_delta=0.0 if t == 0 else float(
                        item["timestamps"][t] - item["timestamps"][t - 1]
                    ),
                )
                model_ms = (time.perf_counter() - started) * 1000.0
                model_latencies.append(model_ms)
                sequence_latencies.append(model_ms)
                if t:
                    has_change_labels = (
                        item["annotations"].get("change_masks") == "available"
                    )
                    truth_change = None
                    if has_change_labels:
                        truth_change = item["change_masks"][t - 1].to(device).bool()
                        predicted_change = (
                            torch.sigmoid(output["change_logits"][0, 0]) >= change_threshold
                        )
                        tp = int((predicted_change & truth_change).sum())
                        fp = int((predicted_change & ~truth_change).sum())
                        fn = int((~predicted_change & truth_change).sum())
                        change_tp += tp
                        change_fp += fp
                        change_fn += fn
                        change_labelled_transitions += 1
                    if item["annotations"].get("motion_mask") == "available":
                        truth_flow = item["motion_grid_displacement"][t - 1].to(device)
                        truth_mask = item["motion_mask"][t - 1].to(device).bool()
                        predicted_flow = F.interpolate(
                            output["motion_grid_displacement"],
                            size=truth_mask.shape,
                            mode="bilinear",
                            align_corners=False,
                        )[0].permute(1, 2, 0)
                        if truth_mask.any():
                            sequence_flow.extend(
                                torch.linalg.vector_norm(
                                    predicted_flow[truth_mask] - truth_flow[truth_mask],
                                    dim=1,
                                ).cpu().tolist()
                            )
                    target_latent = output["latent"].detach()
                    if t > 1:
                        sequence_prediction.append(
                            float(F.mse_loss(previous_output["predicted_next_latent"], target_latent))
                        )
                        if truth_change is not None and not truth_change.any():
                            a = previous_output["latent"].flatten()
                            b = target_latent.flatten()
                            sequence_stable_cosine.append(float(F.cosine_similarity(a, b, dim=0)))
                    if item["annotations"].get("semantic_event_labels") == "available":
                        labels = item["semantic_event_labels"][t - 1].to(device)
                        semantic_labelled_transitions += 1
                        for index, name in enumerate(semantic_event_counts):
                            value = int(labels[index])
                            key = "unknown" if value == -1 else "positive" if value == 1 else "negative"
                            semantic_event_counts[name][key] += 1
                        if int(labels[0]) == 1:
                            a = previous_output["latent"].flatten()
                            b = output["latent"].flatten()
                            semantic_stable_cosines.append(
                                float(F.cosine_similarity(a, b, dim=0))
                            )
                    end_to_end_latencies.append((time.perf_counter() - started) * 1000.0)
                previous_output = output
                state = None if _reset_recurrence else next_state
        all_change_tp += change_tp
        all_change_fp += change_fp
        all_change_fn += change_fn
        stable_cosines.extend(sequence_stable_cosine)
        records.append({
            "recording_id": item["recording_id"],
            "source_id": item["source_id"],
            "frames": int(frames.shape[1]),
            "change_labelled_transitions": change_labelled_transitions,
            "change_tp": change_tp if change_labelled_transitions else None,
            "change_fp": change_fp if change_labelled_transitions else None,
            "change_fn": change_fn if change_labelled_transitions else None,
            "change_f1": (
                2.0 * (change_tp / max(1, change_tp + change_fp))
                * (change_tp / max(1, change_tp + change_fn))
                / max(
                    1e-12,
                    (change_tp / max(1, change_tp + change_fp))
                    + (change_tp / max(1, change_tp + change_fn)),
                ) if change_labelled_transitions else None
            ),
            "flow_epe_mean": statistics.fmean(sequence_flow) if sequence_flow else None,
            "future_feature_mse": statistics.fmean(sequence_prediction) if sequence_prediction else None,
            "semantic_labelled_transitions": semantic_labelled_transitions,
            "stable_latent_cosine_mean": (
                statistics.fmean(sequence_stable_cosine) if sequence_stable_cosine else None
            ),
        })
    precision = all_change_tp / max(1, all_change_tp + all_change_fp)
    recall = all_change_tp / max(1, all_change_tp + all_change_fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    change_records = [record for record in records if record["change_labelled_transitions"]]
    has_pixel_change_labels = bool(change_records)
    report = {
        "schema": "avcs.streaming_v1.evaluation_report",
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "dataset_manifest_sha256": validate_manifest(dataset_dir)["manifest_sha256"],
        "split": split,
        "device": device,
        "change_threshold": change_threshold,
        "recording_count": len(records),
        "frame_count": sum(record["frames"] for record in records),
        "change_metrics": {
            "pixel_precision": precision if has_pixel_change_labels else None,
            "pixel_recall": recall if has_pixel_change_labels else None,
            "pixel_f1": f1 if has_pixel_change_labels else None,
            "true_positive_pixels": all_change_tp if has_pixel_change_labels else None,
            "false_positive_pixels": all_change_fp if has_pixel_change_labels else None,
            "false_negative_pixels": all_change_fn if has_pixel_change_labels else None,
            "labelled_recordings": len(change_records),
            "unlabelled_recordings": len(records) - len(change_records),
        },
        "semantic_event_capability": {
            "status": "not_evaluated_no_semantic_event_head",
            "label_counts": semantic_event_counts,
            "labelled_transitions": sum(
                int(record["semantic_labelled_transitions"]) for record in records
            ),
        },
        "motion_endpoint_error_grid_units_mean": (
            statistics.fmean(
                float(item["flow_epe_mean"])
                for item in records
                if item["flow_epe_mean"] is not None
            )
            if any(item["flow_epe_mean"] is not None for item in records)
            else None
        ),
        "future_feature_prediction_mse_mean": (
            statistics.fmean(
                float(item["future_feature_mse"])
                for item in records
                if item["future_feature_mse"] is not None
            )
            if any(item["future_feature_mse"] is not None for item in records)
            else None
        ),
        "stable_pair_latent_cosine_mean": (
            statistics.fmean(stable_cosines) if stable_cosines else None
        ),
        "semantic_stable_pair_latent_cosine_mean": (
            statistics.fmean(semantic_stable_cosines)
            if semantic_stable_cosines
            else None
        ),
        "recording_bootstrap_95pct_intervals": {
            "change_pixel_f1": bootstrap_change_f1_interval(change_records),
            "motion_epe_grid_units": bootstrap_interval(
                [float(item["flow_epe_mean"]) for item in records if item["flow_epe_mean"] is not None]
            ),
            "future_feature_mse": bootstrap_interval(
                [
                    float(item["future_feature_mse"])
                    for item in records
                    if item["future_feature_mse"] is not None
                ]
            ),
        },
        "model_latency_ms": {
            "samples": len(model_latencies),
            "p50": float(np.percentile(model_latencies, 50)) if model_latencies else None,
            "p95": float(np.percentile(model_latencies, 95)) if model_latencies else None,
            "inference_fps": 1000.0 / statistics.fmean(model_latencies) if model_latencies else None,
        },
        "test_recordings": records,
        "trained_lineage": payload["lineage"],
        "warning": (
            "Pixel-change labels measure exact RGB differences, not semantic events. "
            "Unlabeled transitions are excluded rather than treated as negatives; "
            "procedural sequences do not establish real-world visual generalization."
        ),
    }
    if include_baselines:
        seed = payload.get("lineage", {}).get("seed")
        if not isinstance(seed, int):
            raise ValueError("checkpoint lineage lacks the initialization seed for baseline comparison.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            random_model = StreamingVisualCortexV1(**payload["model_config"])
        random_report = evaluate_streaming_checkpoint(
            checkpoint_path,
            dataset_dir,
            split=split,
            device=device,
            change_threshold=change_threshold,
            include_baselines=False,
            _model_override=random_model,
        )
        no_recurrence_report = evaluate_streaming_checkpoint(
            checkpoint_path,
            dataset_dir,
            split=split,
            device=device,
            change_threshold=change_threshold,
            include_baselines=False,
            _model_override=model,
            _reset_recurrence=True,
        )
        report["matched_random_initialization"] = {
            key: random_report[key]
            for key in (
                "change_metrics",
                "motion_endpoint_error_grid_units_mean",
                "future_feature_prediction_mse_mean",
                "stable_pair_latent_cosine_mean",
                "model_latency_ms",
            )
        }
        report["trained_minus_random"] = {
            "change_pixel_f1": (
                report["change_metrics"]["pixel_f1"]
                - random_report["change_metrics"]["pixel_f1"]
                if report["change_metrics"]["pixel_f1"] is not None
                and random_report["change_metrics"]["pixel_f1"] is not None
                else None
            ),
            "motion_endpoint_error_grid_units_mean": (
                report["motion_endpoint_error_grid_units_mean"]
                - random_report["motion_endpoint_error_grid_units_mean"]
                if report["motion_endpoint_error_grid_units_mean"] is not None
                and random_report["motion_endpoint_error_grid_units_mean"] is not None
                else None
            ),
            "future_feature_prediction_mse_mean": (
                report["future_feature_prediction_mse_mean"]
                - random_report["future_feature_prediction_mse_mean"]
                if report["future_feature_prediction_mse_mean"] is not None
                and random_report["future_feature_prediction_mse_mean"] is not None
                else None
            ),
        }
        report["no_recurrent_state_ablation"] = {
            key: no_recurrence_report[key]
            for key in (
                "change_metrics",
                "motion_endpoint_error_grid_units_mean",
                "future_feature_prediction_mse_mean",
                "stable_pair_latent_cosine_mean",
                "model_latency_ms",
            )
        }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite evaluation report: {destination}")
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
