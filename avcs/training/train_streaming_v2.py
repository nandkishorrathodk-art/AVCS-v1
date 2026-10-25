"""Multi-Task Training Pipeline & Staging for StreamingVisualCortexV2.

Milestone M4 Implementation:
  - Multi-Task Loss Formulation:
      * Head 1 Loss: BCE loss on spatial change heatmap (128x128) + BCE loss on spinner activity logit.
      * Head 2 Loss: Anchor-free UI region detection on 16x16 grid:
          - Objectness BCEWithLogits loss.
          - 5-class generic UI morphology CrossEntropy loss on positive grid cells.
          - Bounding box SmoothL1 loss on box offsets (dx, dy, dw, dh) on positive grid cells.
      * Head 3 Loss: BCEWithLogits loss on post-action visual response logit.
      * Future Latent Prediction Loss: Normalized MSE loss between actual z_{t+1} and predicted z_hat_{t+1}.
      * Total joint loss: weighted sum of all loss components with balanced loss coefficients.
  - Optimizer & Scheduler:
      * AdamW optimizer with weight decay 1e-4, learning rate 1e-3, cosine / step LR scheduling.
  - Dataset & Augmentation Integration:
      * Arm D2 balanced DataLoader (50% UI Grounding + 50% Cleanroom with 50% photometric inversion).
      * Procedural temporal sequence generator (Head 1 and latent prediction).
      * Procedural pre/post action pair generator (Head 3).
      * EdgeLuminancePreprocessor 4-channel input (I_lum, G_x, G_y, M).
  - Dual-Target Execution:
      * Standard 4-core CPU execution (default: --device cpu, batch size 16, 250 steps).
      * Optional GPU execution (--device cuda).
  - Staging Checkpoint Serialization:
      * Serializes model weights to runs/weights/staging/candidate_streaming_v2.pt.
      * Writes sidecar metadata to runs/weights/staging/candidate_streaming_v2.json.
      * Strict baseline integrity: protects baseline weights against modification.
  - Pure-Pixel Boundary: Zero OCR, zero DOM, zero UIA dependencies.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from avcs.dataset.unified.temporal_sampler import (
    sample_action_pair_batch,
    sample_temporal_batch,
)
from avcs.streaming_v2.model import (
    CortexTemporalState,
    CortexV2Output,
    StreamingVisualCortexV2,
)
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.training.multi_generator_dataset import (
    create_arm_d2_dataloader,
    generate_hard_negative_screen,
)

# ==============================================================================
# Baseline Checkpoint Integrity Hashes (Must NEVER be modified or overwritten)
# ==============================================================================
BASELINE_WEIGHT_HASHES: Dict[str, str] = {
    "runs/weights/goal_grounder_best.pt": (
        "a6b4aaf3461f2a78b8a1776925bf6a1454d886cde9d6e54dd30b9164bd8259f4"
    ),
    "runs/weights/candidates/candidate_m3_word_recognizer_v1.pt": (
        "2a63534dde9545d0f33d0632534349935e0f190646e2ecfd273caecfa5065639"
    ),
    "runs/weights/candidates/avcs_streaming_cortex_v1.pt": (
        "418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d"
    ),
}


def verify_baseline_weights_untouched(base_dir: Union[str, Path] = ".") -> Dict[str, str]:
    """Verify that all frozen baseline weights exist and match their expected SHA-256 digests.

    Raises:
        FileNotFoundError: If a baseline weight file is missing.
        AssertionError: If a baseline weight file has been modified or corrupted.

    Returns:
        Dict mapping filepath to computed SHA-256 digest.
    """
    base_path = Path(base_dir)
    verified_hashes: Dict[str, str] = {}

    for rel_path, expected_hash in BASELINE_WEIGHT_HASHES.items():
        full_path = base_path / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Baseline weight file not found: {full_path}")

        data = full_path.read_bytes()
        computed_hash = hashlib.sha256(data).hexdigest()
        if computed_hash != expected_hash:
            raise AssertionError(
                f"INTEGRITY VIOLATION: Baseline weight file {rel_path} was modified!\n"
                f"  Expected: {expected_hash}\n"
                f"  Computed: {computed_hash}"
            )
        verified_hashes[rel_path] = computed_hash

    return verified_hashes


# ==============================================================================
# Multi-Task Loss Data Structures & Functions
# ==============================================================================


@dataclass
class MultiTaskLossResult:
    """Detailed loss metrics for a single multi-task training step."""

    total_loss: torch.Tensor
    loss_head1: torch.Tensor
    loss_head2: torch.Tensor
    loss_head3: torch.Tensor
    loss_latent: torch.Tensor

    # Sub-component metrics
    loss_change: torch.Tensor
    loss_spinner: torch.Tensor
    loss_obj: torch.Tensor
    loss_cls: torch.Tensor
    loss_box: torch.Tensor

    def to_metrics_dict(self) -> Dict[str, float]:
        """Convert all loss tensors to scalar float metrics."""
        return {
            "total_loss": float(self.total_loss.item()),
            "loss_head1": float(self.loss_head1.item()),
            "loss_head2": float(self.loss_head2.item()),
            "loss_head3": float(self.loss_head3.item()),
            "loss_latent": float(self.loss_latent.item()),
            "loss_change": float(self.loss_change.item()),
            "loss_spinner": float(self.loss_spinner.item()),
            "loss_obj": float(self.loss_obj.item()),
            "loss_cls": float(self.loss_cls.item()),
            "loss_box": float(self.loss_box.item()),
        }


def build_head2_targets(
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    box_mask: torch.Tensor,
    grid_size: int = 16,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert padded bounding box annotations into 16x16 anchor-free grid targets.

    For each image in batch:
      - `target_obj` (B, 1, 16, 16): 1.0 at cells where an object centroid falls, 0.0 elsewhere.
      - `target_cls` (B, 16, 16): class ID (0..4) at positive cells, -1 for background.
      - `target_box` (B, 4, 16, 16): (dx, dy, dw, dh) offsets at positive cells.
         where:
           stride = 1.0 / grid_size
           cell_cx = (gx + 0.5) * stride
           cell_cy = (gy + 0.5) * stride
           dx = (obj_cx - cell_cx) / stride  (in range [-0.5, 0.5])
           dy = (obj_cy - cell_cy) / stride  (in range [-0.5, 0.5])
           dw = obj_w (normalized box width in [0.0, 1.0])
           dh = obj_h (normalized box height in [0.0, 1.0])
      - `pos_mask` (B, 16, 16): boolean mask of positive grid cells.

    Args:
        target_boxes: (B, max_objs, 4) in normalized [x1, y1, x2, y2].
        target_classes: (B, max_objs) int64 class IDs in 0..4.
        box_mask: (B, max_objs) bool mask indicating valid objects.
        grid_size: Spatial grid resolution (default: 16).
        device: Target torch device.

    Returns:
        (target_obj, target_cls, target_box, pos_mask)
    """
    b_size = target_boxes.shape[0]
    dev = device or target_boxes.device

    target_obj = torch.zeros((b_size, 1, grid_size, grid_size), dtype=torch.float32, device=dev)
    target_cls = torch.full((b_size, grid_size, grid_size), -1, dtype=torch.int64, device=dev)
    target_box = torch.zeros((b_size, 4, grid_size, grid_size), dtype=torch.float32, device=dev)
    pos_mask = torch.zeros((b_size, grid_size, grid_size), dtype=torch.bool, device=dev)

    stride = 1.0 / float(grid_size)

    for b in range(b_size):
        valid_indices = torch.where(box_mask[b])[0]
        for idx in valid_indices:
            x1, y1, x2, y2 = target_boxes[b, idx]
            cid = target_classes[b, idx].item()
            if cid < 0 or cid > 4:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = torch.clamp(x2 - x1, min=1e-4, max=1.0)
            bh = torch.clamp(y2 - y1, min=1e-4, max=1.0)

            gx = int(torch.clamp(cx * grid_size, 0, grid_size - 1).item())
            gy = int(torch.clamp(cy * grid_size, 0, grid_size - 1).item())

            cell_cx = (gx + 0.5) * stride
            cell_cy = (gy + 0.5) * stride

            dx = (cx - cell_cx) / stride
            dy = (cy - cell_cy) / stride

            target_obj[b, 0, gy, gx] = 1.0
            target_cls[b, gy, gx] = int(cid)
            target_box[b, 0, gy, gx] = dx
            target_box[b, 1, gy, gx] = dy
            target_box[b, 2, gy, gx] = bw
            target_box[b, 3, gy, gx] = bh
            pos_mask[b, gy, gx] = True

    return target_obj, target_cls, target_box, pos_mask


def compute_head2_loss(
    ui_objectness_logits: torch.Tensor,
    ui_class_logits: torch.Tensor,
    ui_box_regression: torch.Tensor,
    target_obj: torch.Tensor,
    target_cls: torch.Tensor,
    target_box: torch.Tensor,
    pos_mask: torch.Tensor,
    box_loss_type: str = "smooth_l1",
    hard_neg_boost: float = 10.0,
    margin_threshold: float = 0.25,
    pos_weight: float = 1.0,
    neg_frame_multiplier: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute anchor-free Head 2 detection losses with adaptive hard-negative weighting.

    Args:
        ui_objectness_logits: (B, 1, 16, 16) raw logits.
        ui_class_logits: (B, 5, 16, 16) raw logits for 5 classes.
        ui_box_regression: (B, 4, 16, 16) predicted offsets (dx, dy, dw, dh).
        target_obj: (B, 1, 16, 16) objectness targets in {0.0, 1.0}.
        target_cls: (B, 16, 16) class targets in 0..4 (-1 for background).
        target_box: (B, 4, 16, 16) box regression targets.
        pos_mask: (B, 16, 16) boolean mask of positive grid cells.
        box_loss_type: "smooth_l1" or "l1".
        hard_neg_boost: Additive weight boost on background cells with p >= margin_threshold (default: 10.0).
        margin_threshold: Probability threshold above which background cells receive hard_neg_boost (default: 0.25).
        pos_weight: Loss weight on positive cells (default: 1.0).
        neg_frame_multiplier: Loss multiplier on frames with zero positive objects (default: 2.0).

    Returns:
        (loss_head2, loss_obj, loss_cls, loss_box)
    """
    dev = ui_objectness_logits.device

    # 1. Base BCE without reduction
    bce = F.binary_cross_entropy_with_logits(ui_objectness_logits, target_obj, reduction="none")
    weights = torch.ones_like(bce)

    pos_mask_4d = pos_mask.unsqueeze(1)
    if pos_weight != 1.0:
        weights[pos_mask_4d] = pos_weight

    # Adaptive Hard-Negative boost: punish any negative cell with p >= margin_threshold
    p_obj = torch.sigmoid(ui_objectness_logits)
    is_hard_neg = (~pos_mask_4d) & (p_obj.detach() >= margin_threshold)
    weights[is_hard_neg] += hard_neg_boost

    # Apply negative-frame multiplier if image has zero positive objects
    has_pos_per_image = pos_mask.any(dim=-1).any(dim=-1)
    is_neg_image = ~has_pos_per_image
    weights[is_neg_image] *= neg_frame_multiplier

    loss_obj = (weights * bce).mean()

    # 2. 5-Class Morphology Loss on positive cells
    if pos_mask.any():
        # Permute (B, 5, 16, 16) -> (B, 16, 16, 5) then slice positive cells
        cls_preds = ui_class_logits.permute(0, 2, 3, 1)[pos_mask]  # (N_pos, 5)
        cls_targets = target_cls[pos_mask]                          # (N_pos,)
        loss_cls = F.cross_entropy(cls_preds, cls_targets)

        # 3. Box Regression Loss on positive cells
        box_preds = ui_box_regression.permute(0, 2, 3, 1)[pos_mask]  # (N_pos, 4)
        box_targets = target_box.permute(0, 2, 3, 1)[pos_mask]       # (N_pos, 4)

        if box_loss_type == "l1":
            loss_box = F.l1_loss(box_preds, box_targets)
        else:
            loss_box = F.smooth_l1_loss(box_preds, box_targets, beta=0.1)
    else:
        # No positive objects in batch (e.g. blank screen batch)
        loss_cls = torch.tensor(0.0, dtype=torch.float32, device=dev)
        loss_box = torch.tensor(0.0, dtype=torch.float32, device=dev)

    total_h2 = loss_obj + loss_cls + loss_box
    return total_h2, loss_obj, loss_cls, loss_box


def compute_head1_loss(
    change_heatmap: torch.Tensor,
    spinner_activity_logit: torch.Tensor,
    target_change_heatmaps: torch.Tensor,
    target_spinner_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Head 1 temporal change heatmap and spinner activity losses.

    Args:
        change_heatmap: (B, 1, 128, 128) sigmoid probabilities in [0.0, 1.0].
        spinner_activity_logit: (B, 1) raw logit.
        target_change_heatmaps: (B, 1, 128, 128) ground truth change in [0.0, 1.0].
        target_spinner_labels: (B, 1) ground truth spinner labels in {0.0, 1.0}.

    Returns:
        (loss_head1, loss_change, loss_spinner)
    """
    # Numerical clamp to prevent log(0) in BCE
    eps = 1e-7
    clamped_heat = torch.clamp(change_heatmap, min=eps, max=1.0 - eps)
    loss_change = F.binary_cross_entropy(clamped_heat, target_change_heatmaps)

    loss_spinner = F.binary_cross_entropy_with_logits(
        spinner_activity_logit,
        target_spinner_labels,
    )

    total_h1 = loss_change + loss_spinner
    return total_h1, loss_change, loss_spinner


def compute_head3_loss(
    post_action_response_logit: torch.Tensor,
    target_response_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute Head 3 post-action visual response loss.

    Args:
        post_action_response_logit: (B, 1) raw logit.
        target_response_labels: (B, 1) ground truth response in {0.0, 1.0}.

    Returns:
        loss_head3 scalar tensor.
    """
    return F.binary_cross_entropy_with_logits(
        post_action_response_logit,
        target_response_labels,
    )


def compute_future_latent_loss(
    predicted_next_latent: torch.Tensor,
    actual_next_latent: torch.Tensor,
) -> torch.Tensor:
    r"""Compute normalized MSE loss between actual and predicted next latents.

    Args:
        predicted_next_latent: (B, 96) predicted latent \hat{z}_{t+1}.
        actual_next_latent: (B, 96) actual next latent z_{t+1}.

    Returns:
        Normalized MSE loss scalar tensor.
    """
    mse = F.mse_loss(predicted_next_latent, actual_next_latent)
    # Scale normalization factor to avoid scale-dependent drift
    norm_factor = actual_next_latent.detach().pow(2).mean() + 1e-6
    return mse / norm_factor


# ==============================================================================
# Single Multi-Task Training Step Execution
# ==============================================================================


def execute_training_step(
    model: StreamingVisualCortexV2,
    preprocessor: EdgeLuminancePreprocessor,
    optimizer: torch.optim.Optimizer,
    batch_ui: Dict[str, Any],
    batch_temporal: Dict[str, torch.Tensor],
    batch_action: Dict[str, torch.Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
    device: Optional[torch.device] = None,
    clip_grad_norm: float = 1.0,
) -> Tuple[MultiTaskLossResult, float]:
    """Execute a single multi-task training step with genuine backpropagation.

    Ingests:
      1. Arm D2 multi-generator batch (UI Grounding + Cleanroom) -> Head 2 loss.
      2. Procedural temporal sequence batch -> Head 1 loss + Future latent prediction loss.
      3. Procedural action pair batch -> Head 3 post-action visual response loss.

    Returns:
        (MultiTaskLossResult, grad_norm_float)
    """
    dev = device or next(model.parameters()).device
    weights = loss_weights or {"h1": 1.0, "h2": 1.0, "h3": 1.0, "latent": 0.5}

    optimizer.zero_grad()

    # -------------------------------------------------------------------------
    # 1. Head 2: Arm D2 UI Region Detection
    # -------------------------------------------------------------------------
    ui_images_rgb = batch_ui["images"].to(dev)  # (B, 3, 128, 128)
    target_boxes = batch_ui["target_boxes"].to(dev)
    target_classes = batch_ui["target_classes"].to(dev)
    box_mask = batch_ui["box_mask"].to(dev)

    # 4-Channel Preprocessor: (B, 3, 128, 128) -> (B, 4, 128, 128)
    ui_images_4ch = preprocessor(ui_images_rgb)

    # Forward pass on UI images (single observation, initial state)
    out_ui = model.forward_step(ui_images_4ch, prev_state=None)

    # Grid target generation and loss
    t_obj, t_cls, t_box, p_mask = build_head2_targets(
        target_boxes=target_boxes,
        target_classes=target_classes,
        box_mask=box_mask,
        grid_size=16,
        device=dev,
    )
    loss_h2, loss_obj, loss_cls, loss_box = compute_head2_loss(
        ui_objectness_logits=out_ui.ui_objectness_logits,
        ui_class_logits=out_ui.ui_class_logits,
        ui_box_regression=out_ui.ui_box_regression,
        target_obj=t_obj,
        target_cls=t_cls,
        target_box=t_box,
        pos_mask=p_mask,
    )

    # -------------------------------------------------------------------------
    # 2. Head 1 & Future Latent: Procedural Temporal Sequences
    # -------------------------------------------------------------------------
    temp_seqs_rgb = batch_temporal["sequences"].to(dev)          # (B, T, 3, 128, 128)
    temp_change_maps = batch_temporal["change_heatmaps"].to(dev) # (B, 1, 128, 128)
    temp_spinner_labels = batch_temporal["spinner_labels"].to(dev) # (B, 1)

    # We evaluate a two-frame transition t0 -> t1
    frame_0_rgb = temp_seqs_rgb[:, 0]
    frame_1_rgb = temp_seqs_rgb[:, 1]

    frame_0_4ch = preprocessor(frame_0_rgb)
    frame_1_4ch = preprocessor(frame_1_rgb)

    out_0 = model.forward_step(frame_0_4ch, prev_state=None, timestamp=0.0)
    out_1 = model.forward_step(frame_1_4ch, prev_state=out_0.temporal_state, timestamp=0.1)

    loss_h1, loss_change, loss_spinner = compute_head1_loss(
        change_heatmap=out_1.change_heatmap,
        spinner_activity_logit=out_1.spinner_activity_logit,
        target_change_heatmaps=temp_change_maps,
        target_spinner_labels=temp_spinner_labels,
    )

    loss_latent = compute_future_latent_loss(
        predicted_next_latent=out_0.predicted_next_latent,
        actual_next_latent=out_1.latent,
    )

    # -------------------------------------------------------------------------
    # 3. Head 3: Procedural Pre/Post Action Pairs
    # -------------------------------------------------------------------------
    act_pre_rgb = batch_action["pre_frames"].to(dev)    # (B, 3, 128, 128)
    act_post_rgb = batch_action["post_frames"].to(dev)  # (B, 3, 128, 128)
    act_resp_labels = batch_action["response_labels"].to(dev)  # (B, 1)

    pre_4ch = preprocessor(act_pre_rgb)
    post_4ch = preprocessor(act_post_rgb)

    out_pre = model.forward_step(pre_4ch, prev_state=None, timestamp=0.0)
    out_post = model.forward_step(post_4ch, prev_state=out_pre.temporal_state, timestamp=0.2)

    resp_logit = model.post_action_head(out_pre.latent, out_post.latent)
    loss_h3 = compute_head3_loss(resp_logit, act_resp_labels)

    # -------------------------------------------------------------------------
    # 4. Joint Multi-Task Objective & Backpropagation
    # -------------------------------------------------------------------------
    total_loss = (
        weights["h1"] * loss_h1
        + weights["h2"] * loss_h2
        + weights["h3"] * loss_h3
        + weights["latent"] * loss_latent
    )

    total_loss.backward()

    # Gradient clipping for stable training
    if clip_grad_norm > 0.0:
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm).item())
    else:
        grad_norm = 0.0

    optimizer.step()

    result = MultiTaskLossResult(
        total_loss=total_loss.detach(),
        loss_head1=loss_h1.detach(),
        loss_head2=loss_h2.detach(),
        loss_head3=loss_h3.detach(),
        loss_latent=loss_latent.detach(),
        loss_change=loss_change.detach(),
        loss_spinner=loss_spinner.detach(),
        loss_obj=loss_obj.detach(),
        loss_cls=loss_cls.detach(),
        loss_box=loss_box.detach(),
    )
    return result, grad_norm


# ==============================================================================
# Full Multi-Task Training Pipeline & Staging
# ==============================================================================


def train_streaming_v2(
    steps: int = 500,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: str = "cpu",
    lr_scheduler_type: str = "cosine",
    loss_weights: Optional[Dict[str, float]] = None,
    output_dir: str = "runs/weights/staging",
    checkpoint_name: str = "candidate_streaming_v2.pt",
    save_checkpoint: bool = True,
    seed: int = 42,
    log_interval: int = 25,
) -> Dict[str, Any]:
    """Train StreamingVisualCortexV2 and serialize candidate weights to staging.

    Args:
        steps: Total training optimization steps (default: 500).
        batch_size: Batch size for UI and procedural generators (default: 16).
        lr: Initial learning rate (default: 1e-3).
        weight_decay: AdamW weight decay (default: 1e-4).
        device_name: "cpu" or "cuda" (default: "cpu").
        lr_scheduler_type: "cosine" or "step" (default: "cosine").
        loss_weights: Dict of loss coefficients.
        output_dir: Directory to save staged checkpoint and metadata.
        checkpoint_name: Filename of model weights (.pt).
        save_checkpoint: If True, saves candidate checkpoint and sidecar JSON.
        seed: Random seed for reproducibility.
        log_interval: Interval in steps to print progress.

    Returns:
        Dict with comprehensive training telemetry, final metrics, and artifact metadata.
    """
    # 1. Deterministic Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # 2. Select Device
    device = torch.device(device_name if (torch.cuda.is_available() and device_name.startswith("cuda")) else "cpu")

    # 3. Baseline Weights Protection Gate
    print(f"[{datetime.now(timezone.utc).isoformat()}] Verifying baseline weight hashes...")
    verify_baseline_weights_untouched()
    print("  -> Baseline weights verified bit-for-bit identical.")

    # 4. Instantiate Model & Preprocessor
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)

    total_params = model.total_parameters
    param_breakdown = model.get_parameter_breakdown()
    print(f"StreamingVisualCortexV2 instantiated: {total_params:,} parameters on {device}.")
    assert total_params == 400765, f"Expected 400,765 parameters, got {total_params}"

    # Take snapshot of initial weights to measure parameter divergence
    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    # 5. Optimizer & LR Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)
    else:
        scheduler = StepLR(optimizer, step_size=max(1, steps // 3), gamma=0.5)

    # 6. Data Loaders & Procedural Samplers (Composite Batching)
    num_neg = max(1, batch_size // 4) if batch_size >= 4 else 1
    num_pos = max(1, batch_size - num_neg) if batch_size >= 2 else 1
    print(f"Preparing Arm D2 balanced DataLoader with composite ratio ({num_pos} pos : {num_neg} hard neg)...")
    ui_loader = create_arm_d2_dataloader(
        split="train",
        batch_size=num_pos,
        target_image_size=(128, 128),
        apply_augmentation=True,
        hard_negative_prob=0.0,
        rng_seed=seed,
    )
    ui_iter = iter(ui_loader)

    # 7. Training Loop Execution
    weights = loss_weights or {"h1": 1.0, "h2": 1.0, "h3": 1.0, "latent": 0.5}
    loss_history: List[float] = []
    step_times: List[float] = []

    print(f"Starting {steps}-step multi-task training run on {device} (batch_size={batch_size})...")
    start_time = time.perf_counter()

    model.train()
    preprocessor.eval()  # Preprocessor is deterministic and parameter-free

    first_loss_result: Optional[MultiTaskLossResult] = None
    last_loss_result: Optional[MultiTaskLossResult] = None

    for step in range(1, steps + 1):
        step_start = time.perf_counter()

        # Ingest Arm D2 positive UI batch (cycle iterator if exhausted)
        try:
            batch_ui_pos = next(ui_iter)
        except StopIteration:
            ui_iter = iter(ui_loader)
            batch_ui_pos = next(ui_iter)

        # Ingest procedural hard negative screens across 6 texture regimes
        step_rng = random.Random(seed + step * 37)
        neg_images = torch.stack([
            generate_hard_negative_screen(size=(128, 128), rng=step_rng)
            for _ in range(num_neg)
        ], dim=0)

        # Compose batch with zero UI annotations for negatives
        max_objs = batch_ui_pos["target_boxes"].shape[1]
        neg_boxes = torch.zeros((num_neg, max_objs, 4), dtype=torch.float32)
        neg_classes = torch.full((num_neg, max_objs), -1, dtype=torch.int64)
        neg_centroids = torch.zeros((num_neg, max_objs, 2), dtype=torch.float32)
        neg_box_mask = torch.zeros((num_neg, max_objs), dtype=torch.bool)
        neg_has_target = torch.zeros((num_neg,), dtype=torch.float32)
        neg_gen_id = torch.full((num_neg,), -1, dtype=torch.int64)
        neg_is_neg = torch.ones((num_neg,), dtype=torch.bool)

        batch_ui = {
            "images": torch.cat([batch_ui_pos["images"], neg_images], dim=0),
            "target_boxes": torch.cat([batch_ui_pos["target_boxes"], neg_boxes], dim=0),
            "target_classes": torch.cat([batch_ui_pos["target_classes"], neg_classes], dim=0),
            "target_centroids": torch.cat([batch_ui_pos["target_centroids"], neg_centroids], dim=0),
            "box_mask": torch.cat([batch_ui_pos["box_mask"], neg_box_mask], dim=0),
            "has_target": torch.cat([batch_ui_pos["has_target"], neg_has_target], dim=0),
            "generator_id": torch.cat([batch_ui_pos["generator_id"], neg_gen_id], dim=0),
            "is_negative": torch.cat([batch_ui_pos["is_negative"], neg_is_neg], dim=0),
        }

        # Ingest procedural temporal sequences
        batch_temporal = sample_temporal_batch(
            batch_size=batch_size,
            sequence_length=2,
            image_size=(128, 128),
            spinner_ratio=0.35,
            moving_ratio=0.35,
        )

        # Ingest procedural action pairs
        batch_action = sample_action_pair_batch(
            batch_size=batch_size,
            image_size=(128, 128),
            positive_ratio=0.5,
        )

        # Execute training step
        step_result, grad_norm = execute_training_step(
            model=model,
            preprocessor=preprocessor,
            optimizer=optimizer,
            batch_ui=batch_ui,
            batch_temporal=batch_temporal,
            batch_action=batch_action,
            loss_weights=weights,
            device=device,
            clip_grad_norm=1.0,
        )

        scheduler.step()

        step_elapsed = time.perf_counter() - step_start
        step_times.append(step_elapsed)
        loss_val = float(step_result.total_loss.item())
        loss_history.append(loss_val)

        if step == 1:
            first_loss_result = step_result
        last_loss_result = step_result

        # Periodic telemetry logging
        if step % log_interval == 0 or step == steps or step == 1:
            curr_lr = scheduler.get_last_lr()[0]
            avg_step_time = np.mean(step_times[-log_interval:]) * 1000.0
            print(
                f"Step {step:03d}/{steps:03d} | Total Loss: {loss_val:.4f} | "
                f"H1: {step_result.loss_head1.item():.4f} | H2: {step_result.loss_head2.item():.4f} | "
                f"H3: {step_result.loss_head3.item():.4f} | Latent: {step_result.loss_latent.item():.4f} | "
                f"GradNorm: {grad_norm:.3f} | {avg_step_time:.1f} ms/step | LR: {curr_lr:.6f}"
            )

    total_elapsed = time.perf_counter() - start_time
    avg_throughput_fps = (steps * batch_size) / total_elapsed
    print(
        f"Training complete in {total_elapsed:.2f}s ({steps / total_elapsed:.1f} steps/s, "
        f"{avg_throughput_fps:.1f} frames/s)."
    )

    # 8. Measure Parameter Divergence from Initial Weights
    total_l2_divergence = 0.0
    module_divergence: Dict[str, float] = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            diff = (param - initial_params[name]).pow(2).sum().item()
            total_l2_divergence += diff
            module_name = name.split(".")[0]
            module_divergence[module_name] = module_divergence.get(module_name, 0.0) + diff

    total_l2_divergence = math.sqrt(total_l2_divergence)
    print(f"Total parameter L2 divergence from clean init: {total_l2_divergence:.4f}")
    for mod_name, diff_val in module_divergence.items():
        print(f"  {mod_name}: L2 delta = {math.sqrt(diff_val):.4f}")

    # Verify that all major modules actually updated
    for required_mod in ("stem", "conv_gru", "change_spinner_head", "ui_detector_head", "post_action_head"):
        assert module_divergence.get(required_mod, 0.0) > 1e-6, (
            f"Module {required_mod} had zero parameter update! Divergence: {module_divergence.get(required_mod, 0.0)}"
        )

    # 9. Staging Checkpoint Serialization
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_path / checkpoint_name
    sidecar_file = out_path / (checkpoint_name.replace(".pt", ".json"))

    if save_checkpoint:
        print(f"Serializing candidate model weights to {checkpoint_file}...")
        torch.save(
            {
                "model_name": "StreamingVisualCortexV2",
                "version": "2.0.0",
                "parameter_count": total_params,
                "state_dict": model.state_dict(),
                "training_steps": steps,
                "hyperparameters": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "scheduler": lr_scheduler_type,
                    "loss_weights": weights,
                    "seed": seed,
                },
                "final_metrics": last_loss_result.to_metrics_dict() if last_loss_result else {},
            },
            checkpoint_file,
        )

        checkpoint_bytes = checkpoint_file.read_bytes()
        candidate_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        print(f"Candidate Checkpoint SHA-256: {candidate_sha256} ({len(checkpoint_bytes):,} bytes)")

        # Generate sidecar JSON metadata
        sidecar_metadata: Dict[str, Any] = {
            "schema_version": "2.0.0",
            "model_name": "StreamingVisualCortexV2",
            "checkpoint_file": checkpoint_name,
            "sha256": candidate_sha256,
            "parameter_count": total_params,
            "parameter_breakdown": param_breakdown,
            "total_steps": steps,
            "device": str(device),
            "hyperparameters": {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "lr_scheduler": lr_scheduler_type,
                "loss_weights": weights,
                "seed": seed,
            },
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "min_loss": float(min(loss_history)),
            "loss_history_sampled": [round(loss_history[i], 4) for i in range(0, steps, max(1, steps // 20))],
            "final_metrics": last_loss_result.to_metrics_dict() if last_loss_result else {},
            "initial_metrics": first_loss_result.to_metrics_dict() if first_loss_result else {},
            "parameter_divergence_l2": round(total_l2_divergence, 6),
            "cpu_throughput": {
                "total_elapsed_seconds": round(total_elapsed, 2),
                "steps_per_second": round(steps / total_elapsed, 2),
                "frames_per_second": round(avg_throughput_fps, 2),
                "p50_ms_per_step": round(float(np.median(step_times) * 1000.0), 2),
                "p95_ms_per_step": round(float(np.percentile(step_times, 95) * 1000.0), 2),
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        sidecar_file.write_text(json.dumps(sidecar_metadata, indent=2), encoding="utf-8")
        print(f"Sidecar metadata written to {sidecar_file}")

    # 10. Final Baseline Weights Protection Gate
    print(f"[{datetime.now(timezone.utc).isoformat()}] Re-verifying baseline weight hashes post-training...")
    verify_baseline_weights_untouched()
    print("  -> Baseline weights remain strictly bit-for-bit identical.")

    return {
        "model": model,
        "total_parameters": total_params,
        "steps": steps,
        "loss_history": loss_history,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "total_elapsed": total_elapsed,
        "l2_divergence": total_l2_divergence,
        "candidate_sha256": candidate_sha256 if save_checkpoint else None,
        "checkpoint_file": str(checkpoint_file) if save_checkpoint else None,
        "sidecar_file": str(sidecar_file) if save_checkpoint else None,
    }


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Task Training Pipeline for StreamingVisualCortexV2"
    )
    parser.add_argument("--steps", type=int, default=500, help="Training optimization steps (default: 500)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cpu", help="Target device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"], help="LR scheduler")
    parser.add_argument("--out-dir", type=str, default="runs/weights/staging", help="Output staging directory")
    parser.add_argument("--checkpoint-name", type=str, default="candidate_streaming_v2.pt", help="Checkpoint filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=25, help="Step logging interval")
    parser.add_argument("--no-save", action="store_true", help="Disable checkpoint saving (dry-run mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_streaming_v2(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device_name=args.device,
        lr_scheduler_type=args.scheduler,
        output_dir=args.out_dir,
        checkpoint_name=args.checkpoint_name,
        save_checkpoint=not args.no_save,
        seed=args.seed,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
