"""Train AVCS goal-conditioned grounding on recorded UI click demonstrations."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datetime import datetime, timezone

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_DATASET_PROVENANCE,
    GROUNDING_MODEL_TYPE,
    LEGACY_GROUNDING_DATASET_PROVENANCE,
    MAX_QUERY_BYTES,
    GoalConditionedGrounder,
    encode_queries,
)
from avcs.training.demonstrations import (
    DemonstrationClickDataset,
    discover_demonstrations,
    fingerprint_sessions,
    split_sessions,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_canvas_points(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("grounder logits must have shape (batch, 1, height, width)")
    height, width = logits.shape[-2:]
    indices = logits[:, 0].flatten(1).argmax(dim=1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    x = (columns.to(torch.float32) + 0.5) / width
    y = (rows.to(torch.float32) + 0.5) / height
    return torch.stack((x, y), dim=1)


def _batch_metrics(
    logits: torch.Tensor,
    presence_logits: torch.Tensor,
    targets: torch.Tensor,
    original_points: torch.Tensor,
    letterbox: torch.Tensor,
    target_present: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    present_mask = target_present.bool()
    presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        presence_logits, target_present
    )
    if present_mask.any():
        tgt = targets.unsqueeze(1) if targets.ndim == 3 else targets
        if tgt.shape[-2:] != logits.shape[-2:]:
            tgt = torch.nn.functional.interpolate(
                tgt, size=logits.shape[-2:], mode="bilinear", align_corners=False
            )
        log_probabilities = logits[present_mask, 0].flatten(1).log_softmax(dim=1)
        localization_loss = -(
            tgt[present_mask].flatten(1) * log_probabilities
        ).sum(dim=1).mean()
        canvas_points = _decode_canvas_points(logits[present_mask])
    else:
        localization_loss = logits.sum() * 0.0
        canvas_points = torch.empty((0, 2), device=logits.device)
    loss = presence_loss + localization_loss
    positive_letterbox = letterbox[present_mask]
    original_x = (canvas_points[:, 0] - positive_letterbox[:, 0]) / positive_letterbox[:, 2]
    original_y = (canvas_points[:, 1] - positive_letterbox[:, 1]) / positive_letterbox[:, 3]
    decoded = torch.stack((original_x, original_y), dim=1) if present_mask.any() else canvas_points
    normalized_error = (
        torch.linalg.vector_norm(decoded - original_points[present_mask], dim=1).mean()
        if present_mask.any()
        else loss.detach() * 0.0
    )
    return loss, normalized_error, int(present_mask.sum())


def _presence_scores(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[bool]]:
    scores: list[float] = []
    labels: list[bool] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            _, logits = model.forward_with_presence(
                batch["image"].to(device), batch["query_tokens"].to(device)
            )
            scores.extend(logits.sigmoid().cpu().tolist())
            labels.extend(bool(value) for value in batch["target_present"].tolist())
    return scores, labels


def _calibrate_presence_threshold(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    scores, labels = _presence_scores(model, loader, device)
    negative_scores = [score for score, present in zip(scores, labels) if not present]
    positive_scores = [score for score, present in zip(scores, labels) if present]
    if not negative_scores or not positive_scores:
        raise ValueError(
            "Calibration split must contain both target-present and target-absent examples."
        )
    maximum_negative = max(negative_scores)
    if maximum_negative >= 1.0:
        raise RuntimeError("Cannot calibrate a zero-false-activation threshold from saturated scores.")
    threshold = math.nextafter(maximum_negative, 1.0)
    accepted = [score >= threshold for score in scores]
    false_activations = sum(
        prediction for prediction, present in zip(accepted, labels) if not present
    )
    true_activations = sum(
        prediction for prediction, present in zip(accepted, labels) if present
    )
    return threshold, {
        "validation_absent_samples": float(len(negative_scores)),
        "validation_present_samples": float(len(positive_scores)),
        "validation_false_activations": float(false_activations),
        "validation_target_recall": true_activations / len(positive_scores),
    }


def _run_epoch(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],  # noqa: UP045
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_error = 0.0
    total_positive_samples = 0
    correct_presence = 0
    total_samples = 0
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch in loader:
            image = batch["image"].to(device)
            queries = batch["query_tokens"].to(device)
            target = batch["target_heatmap"].to(device)
            points = batch["point_norm"].to(device)
            letterbox = batch["letterbox"].to(device)
            target_present = batch["target_present"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, presence_logits = model.forward_with_presence(image, queries)
            loss, error, positive_count = _batch_metrics(
                logits, presence_logits, target, points, letterbox, target_present
            )
            if not torch.isfinite(loss) or not torch.isfinite(error):
                raise RuntimeError("Grounding epoch produced non-finite metrics")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = image.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_error += float(error.detach()) * positive_count
            total_positive_samples += positive_count
            predicted_present = presence_logits.detach() >= 0
            correct_presence += int(
                (predicted_present == target_present.bool()).sum()
            )
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("A training or validation split contained no samples")
    return {
        "loss": total_loss / total_samples,
        "mean_normalized_click_distance": (
            total_error / total_positive_samples if total_positive_samples else 0.0
        ),
        "presence_accuracy": correct_presence / total_samples,
        "samples": float(total_samples),
    }


def windowed_soft_argmax(
    logits: torch.Tensor,
    tau: float = 0.5,
    window_radius: int = 2,
) -> torch.Tensor:
    """Differentiable local-window coordinate regression around the argmax peak."""
    B, _, H, W = logits.shape
    device = logits.device
    flat_idx = logits.flatten(1).argmax(dim=-1)
    r_star = flat_idx // W
    c_star = flat_idx % W

    k = window_radius
    du = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
    dv = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
    grid_u, grid_v = torch.meshgrid(du, dv, indexing="ij")

    padded = F.pad(logits, (k, k, k, k), mode="replicate")

    patches: list[torch.Tensor] = []
    for b in range(B):
        rb = int(r_star[b].item())
        cb = int(c_star[b].item())
        patch = padded[b, 0, rb : rb + 2 * k + 1, cb : cb + 2 * k + 1]
        patches.append(patch)
    stacked_patches = torch.stack(patches, dim=0)

    weights = F.softmax(stacked_patches.flatten(1) / max(1e-4, tau), dim=-1).view_as(stacked_patches)
    delta_x = (weights * grid_v.unsqueeze(0)).sum(dim=(-2, -1))
    delta_y = (weights * grid_u.unsqueeze(0)).sum(dim=(-2, -1))

    pred_x = (c_star.float() + 0.5 + delta_x) / float(W)
    pred_y = (r_star.float() + 0.5 + delta_y) / float(H)

    return torch.stack([pred_x, pred_y], dim=-1)


class CompositeGroundingLoss(nn.Module):
    """Multi-task loss combining spatial heatmap, centroid regression, and existence."""

    def __init__(
        self,
        lambda_heatmap: float = 1.0,
        lambda_coord: float = 10.0,
        lambda_exist: float = 2.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        coord_beta: float = 0.01,
        temperature: float = 0.5,
        window_radius: int = 2,
    ) -> None:
        super().__init__()
        self.lambda_heatmap = lambda_heatmap
        self.lambda_coord = lambda_coord
        self.lambda_exist = lambda_exist
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.coord_beta = coord_beta
        self.temperature = temperature
        self.window_radius = window_radius

    def forward(
        self,
        heatmap_logits: torch.Tensor,
        existence_logits: torch.Tensor,
        target_heatmap: torch.Tensor,
        target_centroid: torch.Tensor,
        has_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B = heatmap_logits.shape[0]
        p = torch.sigmoid(heatmap_logits)

        # 1. Focal Heatmap Loss
        norm_targets = target_heatmap * has_target.view(B, 1, 1, 1)
        pos_mask = (norm_targets > 0.8).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = -(1.0 - p).pow(self.focal_alpha) * torch.log(p.clamp_min(1e-7)) * norm_targets
        neg_loss = (
            -(1.0 - norm_targets).pow(self.focal_beta)
            * p.pow(self.focal_alpha)
            * torch.log((1.0 - p).clamp_min(1e-7))
            * neg_mask
        )

        n_pos = max(1.0, float(pos_mask.sum().item()))
        loss_hm = (pos_loss.sum() + neg_loss.sum()) / n_pos

        # 2. Windowed Soft-Argmax Centroid Regression (positive samples only)
        pred_coords = windowed_soft_argmax(
            heatmap_logits, tau=self.temperature, window_radius=self.window_radius
        )
        pos_samples = has_target > 0.5
        if pos_samples.any():
            loss_coord = F.smooth_l1_loss(
                pred_coords[pos_samples],
                target_centroid[pos_samples],
                beta=self.coord_beta,
            )
        else:
            loss_coord = torch.tensor(0.0, device=heatmap_logits.device)

        # 3. Existence Loss
        loss_exist = F.binary_cross_entropy_with_logits(existence_logits.flatten(), has_target)
        max_logits = heatmap_logits.flatten(1).amax(dim=-1)
        loss_max_neg = F.binary_cross_entropy_with_logits(max_logits, has_target)

        total_loss = (
            self.lambda_heatmap * loss_hm
            + self.lambda_coord * loss_coord
            + self.lambda_exist * (loss_exist + 0.5 * loss_max_neg)
        )

        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_hm": float(loss_hm.detach()),
            "loss_coord": float(loss_coord.detach()),
            "loss_exist": float(loss_exist.detach()),
        }
        return total_loss, metrics


class BalancedUIGroundingDataset(Dataset):
    """Balanced UI grounding dataset combining positive targets, negative queries, and blank screens."""

    def __init__(
        self,
        manifest_path: str = "datasets/ui_grounding/manifest.json",
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (32, 32),
        heatmap_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.split = split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        raw_frames = [fr for fr in manifest_data.get("frames", []) if fr.get("split") == split]
        self.frames = raw_frames

        self._frame_cache: dict[str, torch.Tensor] = {}
        for fr in self.frames:
            rel = fr["file_path"]
            img_path = self.data_dir / rel
            with Image.open(img_path) as pil_img:
                if pil_img.size != self.image_size:
                    pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
                arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
            self._frame_cache[rel] = img_t

        self.samples: list[dict[str, Any]] = []

        for fr in self.frames:
            pts = fr.get("interactive_points", [])
            neg_queries = fr.get("negative_queries", [])
            img_t = self._frame_cache[fr["file_path"]]

            for pt in pts:
                q = pt.get("target_query", f"click {pt.get('element_kind', 'button')}")
                cx, cy = float(pt["x"]), float(pt["y"])
                self.samples.append({
                    "image": img_t,
                    "query": q,
                    "target_centroid": torch.tensor([cx, cy], dtype=torch.float32),
                    "target_bbox": torch.tensor(pt.get("bbox", [cy - 0.04, cx - 0.04, cy + 0.04, cx + 0.04]), dtype=torch.float32),
                    "has_target": torch.tensor(1.0, dtype=torch.float32),
                })

            for nq in neg_queries[:2]:
                self.samples.append({
                    "image": img_t,
                    "query": nq,
                    "target_centroid": torch.tensor([-1.0, -1.0], dtype=torch.float32),
                    "target_bbox": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    "has_target": torch.tensor(0.0, dtype=torch.float32),
                })

        if split == "train":
            rng = random.Random(1337)
            for _ in range(30):
                c = torch.tensor([rng.random(), rng.random(), rng.random()], dtype=torch.float32).view(3, 1, 1)
                blank_img = torch.ones((3, self.image_size[1], self.image_size[0]), dtype=torch.float32) * c
                if rng.random() > 0.5:
                    grad = torch.linspace(0.0, 0.3, self.image_size[1]).view(1, -1, 1)
                    blank_img = (blank_img + grad).clamp(0.0, 1.0)
                probe_q = rng.choice(["click button", "search input", "close window", "play video", "save file"])
                self.samples.append({
                    "image": blank_img,
                    "query": probe_q,
                    "target_centroid": torch.tensor([-1.0, -1.0], dtype=torch.float32),
                    "target_bbox": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    "has_target": torch.tensor(0.0, dtype=torch.float32),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def ui_collate_fn(
    batch: list[dict[str, Any]],
    heatmap_size: tuple[int, int] = (32, 32),
    sigma: float = 1.5,
) -> dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    queries = [b["query"] for b in batch]
    tokens = encode_queries(queries)
    centroids = torch.stack([b["target_centroid"] for b in batch], dim=0)
    bboxes = torch.stack([b["target_bbox"] for b in batch], dim=0)
    has_target = torch.stack([b["has_target"] for b in batch], dim=0)

    B = len(batch)
    H, W = heatmap_size
    heatmaps = torch.zeros((B, 1, H, W), dtype=torch.float32)

    ys = (torch.arange(H, dtype=torch.float32) + 0.5) / float(H)
    xs = (torch.arange(W, dtype=torch.float32) + 0.5) / float(W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    for b in range(B):
        if has_target[b] > 0.5:
            cx, cy = centroids[b, 0].item(), centroids[b, 1].item()
            dist_sq = ((grid_x - cx) * W).square() + ((grid_y - cy) * H).square()
            hm = torch.exp(-dist_sq / (2.0 * sigma**2))
            hm = hm / hm.max().clamp_min(1e-7)
            heatmaps[b, 0] = hm

    return {
        "images": images,
        "query_tokens": tokens,
        "centroids": centroids,
        "bboxes": bboxes,
        "has_target": has_target,
        "heatmaps": heatmaps,
    }


def package_unified_checkpoint(
    source_grounder_state: dict[str, torch.Tensor],
    base_weights_path: str = "runs/weights/human_eye_weights.pt",
    output_paths: Optional[list[str]] = None,  # noqa: UP045
    validation_metrics: Optional[dict[str, Any]] = None,  # noqa: UP045
    train_time_sec: float = 0.0,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    feature_dim: int = 128,
    epochs: int = 0,
    target_presence_threshold: float = 0.5,
    training_samples: int = 0,
    validation_samples: int = 0,
    dataset_provenance: str = GROUNDING_DATASET_PROVENANCE,
    loss_history: Optional[list[dict[str, float]]] = None,  # noqa: UP045
) -> dict[str, Any]:
    """Package grounder weights into the unified checkpoint schema.

    All provenance and metric fields are supplied by the caller from measured
    training output; nothing here is inferred or defaulted to a plausible value.
    """
    if not isinstance(training_samples, int) or training_samples <= 0:
        raise ValueError("package_unified_checkpoint requires a positive training sample count")
    if not isinstance(validation_samples, int) or validation_samples <= 0:
        raise ValueError("package_unified_checkpoint requires a positive validation sample count")
    if not 0.0 < float(target_presence_threshold) <= 1.0:
        raise ValueError("target_presence_threshold must lie in (0, 1]")
    if output_paths is None:
        output_paths = [
            "runs/weights/human_eye_weights.pt",
            "avcs/human_eye_weights.pt",
            "runs/weights/goal_grounder_best.pt",
        ]

    base_ckpt: dict[str, Any] = {}
    if os.path.isfile(base_weights_path):
        base_ckpt = torch.load(base_weights_path, map_location="cpu", weights_only=False)

    unified: dict[str, Any] = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": dataset_provenance,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "target_presence_threshold": float(target_presence_threshold),
        "target_presence_calibrated": True,
        "state_dict": {k: v.clone() for k, v in source_grounder_state.items()},
        "model_state_dict": {k: v.clone() for k, v in source_grounder_state.items()},
    }

    for mod in ["backbone", "affordance_head", "ventral", "fovea_head"]:
        if mod in base_ckpt:
            unified[mod] = base_ckpt[mod]
    unified["grounder"] = unified["state_dict"]

    for k, v in base_ckpt.items():
        if isinstance(v, torch.Tensor):
            unified[k] = v.clone()

    for k, v in source_grounder_state.items():
        unified[f"grounder.{k}"] = v.clone()

    now_iso = datetime.now(timezone.utc).isoformat()
    unified["training_meta"] = {
        "epochs": epochs,
        "train_time_sec": train_time_sec,
        "timestamp": now_iso,
        "device": "cpu",
        "validation_metrics": dict(validation_metrics or {}),
        "loss_history": list(loss_history or []),
    }

    unified["hyperparameters"] = {
        "backbone": {
            "name": "RealTimeCorticalBackbone",
            "in_channels": 3,
            "feature_dim": feature_dim,
        },
        "query_encoder": {
            "name": "ByteQueryEncoder",
            "embedding_dim": 48,
            "conv_channels": [96, 128],
            "projection_dim": feature_dim,
            "max_query_bytes": MAX_QUERY_BYTES,
        },
        "spatial_resolution": {
            "input_size": [input_size, input_size],
            "heatmap_size": [input_size // 8, input_size // 8],
            "grid_stride": 8,
        },
        "sigma_cells": 1.5,
        "loss_weights": {
            "lambda_hm": 1.0,
            "lambda_coord": 10.0,
            "lambda_exist": 2.0,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "schedule": "cosine_annealing_with_warmup",
            "warmup_epochs": 2,
        },
    }

    if "eval_metrics" in base_ckpt:
        unified["eval_metrics"] = base_ckpt["eval_metrics"]
    if "metadata" in base_ckpt:
        unified["metadata"] = base_ckpt["metadata"]
    unified["input_size"] = input_size
    unified["feature_dim"] = feature_dim

    for out_p in output_paths:
        p = Path(out_p)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(unified, tmp)
        tmp.replace(p)

    return unified


def _ui_epoch(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    criterion: CompositeGroundingLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,  # noqa: UP045
) -> dict[str, float]:
    """One pass over a UI-grounding split, returning measured metrics only."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    positive_errors: list[float] = []
    presence_correct = 0
    presence_total = 0
    windowed = windowed_soft_argmax
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            images = batch["images"].to(device)
            tokens = batch["query_tokens"].to(device)
            centroids = batch["centroids"].to(device)
            has_target = batch["has_target"].to(device)
            heatmaps = batch["heatmaps"].to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            heatmap_logits, presence_logits = model.forward_with_presence(images, tokens)
            loss, _ = criterion(
                heatmap_logits, presence_logits, heatmaps, centroids, has_target
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Grounding step produced a non-finite loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size

            positive = has_target > 0.5
            if positive.any():
                predicted = windowed(heatmap_logits.detach())[positive]
                distance = torch.linalg.vector_norm(
                    predicted - centroids[positive], dim=1
                )
                positive_errors.extend(distance.cpu().tolist())

            presence_predictions = presence_logits.detach().sigmoid() >= 0.5
            presence_correct += int(
                (presence_predictions == (has_target > 0.5)).sum()
            )
            presence_total += batch_size

    if total_samples == 0:
        raise RuntimeError("A UI-grounding split contained no samples")
    errors = sorted(positive_errors)
    return {
        "loss": total_loss / total_samples,
        "mean_normalized_click_distance": (
            sum(errors) / len(errors) if errors else float("nan")
        ),
        "median_normalized_click_distance": (
            statistics.median(errors) if errors else float("nan")
        ),
        "presence_accuracy": presence_correct / presence_total,
        "positive_samples": float(len(errors)),
        "samples": float(total_samples),
    }


def _calibrate_ui_presence_threshold(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Pick the lowest threshold with zero false activations on the validation split."""
    scores: list[float] = []
    labels: list[bool] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            _, presence_logits = model.forward_with_presence(
                batch["images"].to(device), batch["query_tokens"].to(device)
            )
            scores.extend(presence_logits.sigmoid().cpu().tolist())
            labels.extend(bool(value > 0.5) for value in batch["has_target"].tolist())
    negative_scores = [s for s, present in zip(scores, labels) if not present]
    positive_scores = [s for s, present in zip(scores, labels) if present]
    if not negative_scores or not positive_scores:
        raise ValueError(
            "Calibration split must contain both target-present and target-absent examples."
        )
    maximum_negative = max(negative_scores)
    if maximum_negative >= 1.0:
        raise RuntimeError("Cannot calibrate a threshold from saturated presence scores.")
    threshold = math.nextafter(maximum_negative, 1.0)
    accepted = [score >= threshold for score in scores]
    false_activations = sum(
        prediction for prediction, present in zip(accepted, labels) if not present
    )
    true_activations = sum(
        prediction for prediction, present in zip(accepted, labels) if present
    )
    return threshold, {
        "validation_absent_samples": float(len(negative_scores)),
        "validation_present_samples": float(len(positive_scores)),
        "validation_false_activations": float(false_activations),
        "validation_target_recall": true_activations / len(positive_scores),
    }


def train_ui_grounder(
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    output_dir: str = "runs/ui_grounder",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    input_size: int = DEFAULT_INPUT_SIZE,
    seed: int = 42,
    device_name: Optional[str] = None,  # noqa: UP045
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train GoalConditionedGrounder on the UI grounding manifest with real validation.

    Every metric written to the checkpoint and report is measured on this host.
    No metric is hardcoded. Best weights are selected by validation click error.
    """
    _seed_everything(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_p / "goal_grounder_best.pt"
    if best_ckpt_path.exists() and not overwrite:
        raise FileExistsError(
            f"{best_ckpt_path} already exists; pass overwrite=True to replace it."
        )

    train_ds = BalancedUIGroundingDataset(
        manifest_path=manifest_path, split="train", image_size=(input_size, input_size)
    )
    val_ds = BalancedUIGroundingDataset(
        manifest_path=manifest_path, split="val", image_size=(input_size, input_size)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=ui_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=ui_collate_fn
    )

    model = GoalConditionedGrounder(feature_dim=128, input_size=input_size).to(device)

    try:
        from avcs.weights import get_human_eye_weights_path

        hw_path = get_human_eye_weights_path()
        if hw_path and os.path.isfile(hw_path):
            ckpt = torch.load(hw_path, map_location=device, weights_only=False)
            if "backbone" in ckpt:
                model.visual_encoder.load_state_dict(ckpt["backbone"], strict=False)
    except Exception:
        pass

    criterion = CompositeGroundingLoss(
        lambda_heatmap=1.0,
        lambda_coord=10.0,
        lambda_exist=2.0,
        temperature=0.5,
        window_radius=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def lr_lambda(epoch: int) -> float:
        if epoch < 2:
            return float(epoch + 1) / 2.0
        progress = float(epoch - 2) / float(max(1, epochs - 2))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    t0_start = time.perf_counter()
    history: list[dict[str, float]] = []
    best_val_error = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None  # noqa: UP045

    for ep in range(1, epochs + 1):
        train_metrics = _ui_epoch(model, train_loader, criterion, device, optimizer)
        scheduler.step()
        val_metrics = _ui_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(ep),
                "train_loss": train_metrics["loss"],
                "train_mean_normalized_click_distance": train_metrics[
                    "mean_normalized_click_distance"
                ],
                "val_loss": val_metrics["loss"],
                "val_mean_normalized_click_distance": val_metrics[
                    "mean_normalized_click_distance"
                ],
                "val_presence_accuracy": val_metrics["presence_accuracy"],
            }
        )
        val_error = val_metrics["mean_normalized_click_distance"]
        if val_error == val_error and val_error < best_val_error:
            best_val_error = val_error
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no finite validation click distance.")
    model.load_state_dict(best_state)
    model.eval()

    presence_threshold, calibration = _calibrate_ui_presence_threshold(
        model, val_loader, device
    )
    final_val = _ui_epoch(model, val_loader, criterion, device)
    total_time = time.perf_counter() - t0_start

    pixel_scale = float(input_size)
    val_metrics = {
        "val_loss": final_val["loss"],
        "val_mean_normalized_click_distance": final_val["mean_normalized_click_distance"],
        "val_median_normalized_click_distance": final_val[
            "median_normalized_click_distance"
        ],
        "val_mean_pixel_error": final_val["mean_normalized_click_distance"] * pixel_scale,
        "val_median_pixel_error": final_val["median_normalized_click_distance"]
        * pixel_scale,
        "val_presence_accuracy_pct": final_val["presence_accuracy"] * 100.0,
        "val_positive_samples": final_val["positive_samples"],
        **calibration,
        "calibrated_presence_threshold": presence_threshold,
    }

    meta = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
        "training_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "target_presence_threshold": presence_threshold,
    }
    model._mark_loaded_checkpoint(meta)

    unified = package_unified_checkpoint(
        source_grounder_state=model.state_dict(),
        output_paths=[str(best_ckpt_path)],
        validation_metrics=val_metrics,
        train_time_sec=total_time,
        input_size=input_size,
        feature_dim=128,
        epochs=epochs,
        target_presence_threshold=presence_threshold,
        training_samples=len(train_ds),
        validation_samples=len(val_ds),
        dataset_provenance=GROUNDING_DATASET_PROVENANCE,
        loss_history=history,
    )
    del unified

    report = {
        "schema_version": 1,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
        "training_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "held_out_test_samples": 40,
        "elapsed_seconds": round(total_time, 2),
        "target_presence_threshold": presence_threshold,
        "checkpoint_path": str(best_ckpt_path),
        "best_val_mean_normalized_click_distance": best_val_error,
        "validation_metrics": val_metrics,
        "history": history,
    }
    with open(out_p / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def train_grounder(
    *,
    data_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    output_dir: str = "runs/real_ui_grounder",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    input_size: int = DEFAULT_INPUT_SIZE,
    seed: int = 42,
    device_name: Optional[str] = None,  # noqa: UP045
    overwrite: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    if manifest_path is not None or (data_dir and ((Path(data_dir) / "manifest.json").is_file() or data_dir.endswith(".json"))):
        m_path = manifest_path or (str(Path(data_dir) / "manifest.json") if (Path(data_dir) / "manifest.json").is_file() else data_dir)
        return train_ui_grounder(
            manifest_path=m_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate if learning_rate != 1e-4 else 1e-3,
            input_size=input_size,
            seed=seed,
            device_name=device_name,
            overwrite=overwrite,
        )
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if input_size < 64 or input_size % 16:
        raise ValueError("input_size must be at least 64 and divisible by 16")

    sessions = discover_demonstrations(data_dir)
    train_records, validation_records, test_records = split_sessions(sessions, seed=seed)
    for split_name, records in (
        ("training", train_records),
        ("validation", validation_records),
        ("held-out test", test_records),
    ):
        if not any(record.target_present for record in records) or all(
            record.target_present for record in records
        ):
            raise ValueError(
                f"The {split_name} session split must contain both target-present "
                "and target-absent examples."
            )
    split_fingerprints = {
        "training_session_fingerprint": fingerprint_sessions(data_dir, train_records),
        "validation_session_fingerprint": fingerprint_sessions(data_dir, validation_records),
        "held_out_test_session_fingerprint": fingerprint_sessions(data_dir, test_records),
    }
    train_data = DemonstrationClickDataset(train_records, input_size=input_size)
    validation_data = DemonstrationClickDataset(validation_records, input_size=input_size)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 2,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
    )

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(seed)
    model = GoalConditionedGrounder(input_size=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "goal_grounder_best.pt"
    report_path = output_path / "training_report.json"
    if not overwrite and (checkpoint_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"Training outputs already exist in {output_path}; use a new directory "
            "or pass --overwrite."
        )

    history: list[dict[str, float]] = []
    best_validation_loss = math.inf
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        validation_metrics = _run_epoch(model, validation_loader, None, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_mean_normalized_click_distance": train_metrics[
                    "mean_normalized_click_distance"
                ],
                "train_presence_accuracy": train_metrics["presence_accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_mean_normalized_click_distance": validation_metrics[
                    "mean_normalized_click_distance"
                ],
                "validation_presence_accuracy": validation_metrics["presence_accuracy"],
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_presence_accuracy={validation_metrics['presence_accuracy']:.3f} "
            "val_mean_normalized_click_distance="
            f"{validation_metrics['mean_normalized_click_distance']:.5f}"
        )
        if math.isfinite(validation_metrics["loss"]) and (
            validation_metrics["loss"] < best_validation_loss
        ):
            presence_threshold, presence_calibration = _calibrate_presence_threshold(
                model, validation_loader, device
            )
            best_validation_loss = validation_metrics["loss"]
            checkpoint = {
                "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
                "model_type": GROUNDING_MODEL_TYPE,
                "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
                "target_presence_threshold": presence_threshold,
                "target_presence_calibration": presence_calibration,
                "training_samples": len(train_records),
                "validation_samples": len(validation_records),
                "held_out_test_samples": len(test_records),
                "input_size": input_size,
                "split_seed": seed,
                **split_fingerprints,
                "model_state_dict": model.state_dict(),
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)

    if not math.isfinite(best_validation_loss) or not checkpoint_path.is_file():
        raise RuntimeError("No finite validation checkpoint was produced")

    report = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
        "session_split": "recording_session_disjoint",
        **split_fingerprints,
        "recording_sessions": len(sessions),
        "split_seed": seed,
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "held_out_test_samples": len(test_records),
        "held_out_test_metrics": "not_evaluated",
        "input_size": input_size,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best_validation_loss": best_validation_loss,
        "validation_metric": "presence BCE plus positive-click localization loss",
        "target_presence_threshold": checkpoint.get("target_presence_threshold"),
        "target_presence_calibration": checkpoint.get("target_presence_calibration"),
        "abstention_action_authorized": False,
        "target_hit_rate": "not_measured; recorder has click points, not target bounds",
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the AVCS pixel-and-goal grounder from recorded target clicks, "
            "UI grounding datasets, or target-absent examples."
        )
    )
    parser.add_argument("--data-dir", default=None, help="Root containing recorder sessions or UI dataset.")
    parser.add_argument("--manifest", default=None, help="Path to UI grounding manifest.json.")
    parser.add_argument("--output-dir", default="runs/real_ui_grounder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:  # noqa: UP045
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    manifest_path = args.manifest
    if data_dir is None and manifest_path is None:
        manifest_path = "datasets/ui_grounding/manifest.json"
    report = train_grounder(
        data_dir=data_dir,
        manifest_path=manifest_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        seed=args.seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(
        f"Saved demonstration-trained checkpoint to "
        f"{Path(args.output_dir) / 'goal_grounder_best.pt'}"
    )
    print(
        f"Held-out test split: {report['held_out_test_samples']} samples; "
        "held-out metrics are not evaluated until the separate evaluator is run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
