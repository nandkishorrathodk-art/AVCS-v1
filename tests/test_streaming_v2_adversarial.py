"""Adversarial Stress Test Suite for StreamingVisualCortexV2 Multi-Task Training.

Author: challenger_m4_1 (Empirical Challenger)
Milestone: M4 — Multi-Task Training Pipeline & Staging for StreamingVisualCortexV2

Adversarial Stress Test Dimensions:
  1. Empty-Target Batches:
     - Direct loss & grid target generation on all-negative screens (0 objects).
     - Full execute_training_step with empty-target batches: finite loss, non-zero gradients,
       proper objectness suppression without NaNs or Infs.
  2. Static Action Pairs:
     - Verification when pre_frame == post_frame (pixel-for-pixel identical) with response_label == 0.
     - Loss computation, optical delta zeroing, gradient flow into PostActionVisualResponseHead.
     - Multi-step convergence showing predicted response probability decays toward 0.0.
  3. Noisy / Corrupted Inputs:
     - NaNs in input image tensors propagate cleanly without silent masking.
     - Extreme pixel values (e.g. +/-1e4) handled with numerical stability.
     - Corrupted / out-of-bounds class IDs filtered cleanly without index-out-of-bounds panics.
     - Inverted (x2 < x1) and out-of-bounds boxes clamped safely to valid grid indices and positive dimensions.
     - Invalid target values outside [0, 1] rejected cleanly by loss functions.
     - NaNs in bounding box coordinates raise clean ValueError.
  4. Gradient Clipping Effectiveness:
     - Artificial 100,000x loss spike producing multi-billion gradient norms strictly capped at 1.0.
     - Small gradients below threshold preserved exactly without distortion.
  5. Multi-Step Loss Monotonicity:
     - 10-step optimization run on a fixed batch strictly decreases total loss without divergence.
     - Substantial overall loss reduction (> 75%).
     - All sub-task head losses (Head 1, Head 2, Head 3, Future Latent) decrease from step 1 to step 10.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.dataset.unified.temporal_sampler import (
    sample_action_pair_batch,
    sample_temporal_batch,
)
from avcs.streaming_v2.model import StreamingVisualCortexV2
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.training.train_streaming_v2 import (
    build_head2_targets,
    compute_future_latent_loss,
    compute_head1_loss,
    compute_head2_loss,
    compute_head3_loss,
    execute_training_step,
)


# ==============================================================================
# 1. Adversarial Dimension 1: Empty-Target Batches
# ==============================================================================


def test_adversarial_empty_target_batch_direct_loss() -> None:
    """Verify Head 2 loss computation and backward pass when all frames have 0 targets.

    Asserts:
      - pos_mask is completely False across all batch items and grid cells.
      - target_obj is all 0.0.
      - loss_cls == 0.0 and loss_box == 0.0.
      - loss_obj > 0.0, is finite, and produces non-zero gradients on objectness logits.
      - No gradients backpropagate into class or box regression heads (since no objects exist).
    """
    b_size = 8
    grid_size = 16

    # Empty target annotations
    target_boxes = torch.zeros((b_size, 0, 4), dtype=torch.float32)
    target_classes = torch.zeros((b_size, 0), dtype=torch.int64)
    box_mask = torch.zeros((b_size, 0), dtype=torch.bool)

    t_obj, t_cls, t_box, p_mask = build_head2_targets(
        target_boxes=target_boxes,
        target_classes=target_classes,
        box_mask=box_mask,
        grid_size=grid_size,
    )

    assert p_mask.sum().item() == 0, "pos_mask should be empty for all-negative batch"
    assert t_obj.sum().item() == 0.0, "target_obj should be all zeros"
    assert (t_cls == -1).all(), "target_cls should be all background (-1)"

    pred_obj = torch.randn(b_size, 1, grid_size, grid_size, requires_grad=True)
    pred_cls = torch.randn(b_size, 5, grid_size, grid_size, requires_grad=True)
    pred_box = torch.randn(b_size, 4, grid_size, grid_size, requires_grad=True)

    loss_h2, l_obj, l_cls, l_box = compute_head2_loss(
        ui_objectness_logits=pred_obj,
        ui_class_logits=pred_cls,
        ui_box_regression=pred_box,
        target_obj=t_obj,
        target_cls=t_cls,
        target_box=t_box,
        pos_mask=p_mask,
    )

    assert torch.isfinite(loss_h2) and loss_h2.item() > 0.0
    assert l_cls.item() == 0.0, "Classification loss must be 0.0 on empty batch"
    assert l_box.item() == 0.0, "Box regression loss must be 0.0 on empty batch"
    assert l_obj.item() == loss_h2.item()

    loss_h2.backward()
    assert pred_obj.grad is not None and torch.isfinite(pred_obj.grad).all()
    assert pred_obj.grad.abs().sum().item() > 0.0, "Objectness must receive non-zero gradients"
    assert pred_cls.grad is None or pred_cls.grad.abs().sum().item() == 0.0
    assert pred_box.grad is None or pred_box.grad.abs().sum().item() == 0.0


def test_adversarial_empty_target_batch_full_step_and_convergence() -> None:
    """Verify execute_training_step end-to-end on empty-target batch and multi-step convergence."""
    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    b_size = 4
    batch_ui_empty = {
        "images": torch.rand(b_size, 3, 128, 128),
        "target_boxes": torch.zeros(b_size, 0, 4),
        "target_classes": torch.zeros(b_size, 0, dtype=torch.long),
        "box_mask": torch.zeros(b_size, 0, dtype=torch.bool),
    }
    batch_temporal = sample_temporal_batch(b_size, 2, (128, 128))
    batch_action = sample_action_pair_batch(b_size, (128, 128))

    # Single step verification
    res, grad_norm = execute_training_step(
        model=model,
        preprocessor=preprocessor,
        optimizer=optimizer,
        batch_ui=batch_ui_empty,
        batch_temporal=batch_temporal,
        batch_action=batch_action,
        device=device,
        clip_grad_norm=1.0,
    )

    assert torch.isfinite(res.total_loss)
    assert res.loss_cls.item() == 0.0
    assert res.loss_box.item() == 0.0
    assert grad_norm > 0.0
    assert model.ui_detector_head.conv_objectness.weight.grad is not None
    assert torch.isfinite(model.ui_detector_head.conv_objectness.weight.grad).all()

    # Multi-step convergence on empty screens: objectness loss should strictly decrease
    obj_losses: List[float] = [float(res.loss_obj.item())]
    for _ in range(4):
        step_res, _ = execute_training_step(
            model=model,
            preprocessor=preprocessor,
            optimizer=optimizer,
            batch_ui=batch_ui_empty,
            batch_temporal=batch_temporal,
            batch_action=batch_action,
            device=device,
            clip_grad_norm=1.0,
        )
        obj_losses.append(float(step_res.loss_obj.item()))

    assert obj_losses[-1] < obj_losses[0], (
        f"Objectness loss did not decrease on empty batch: {obj_losses}"
    )
    assert all(torch.isfinite(torch.tensor(obj_losses)))


# ==============================================================================
# 2. Adversarial Dimension 2: Static Action Pairs
# ==============================================================================


def test_adversarial_static_action_pair_loss_and_gradients() -> None:
    """Verify loss computation when pre_frame == post_frame (pixel-identical, response_label == 0)."""
    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)

    b_size = 4
    static_frame = torch.rand(b_size, 3, 128, 128)
    pre_4ch = preprocessor(static_frame)
    post_4ch = preprocessor(static_frame.clone())

    out_pre = model.forward_step(pre_4ch, prev_state=None, timestamp=0.0)
    out_post = model.forward_step(post_4ch, prev_state=out_pre.temporal_state, timestamp=0.2)

    # In static action pairs, optical feature delta is 0
    f_delta = torch.abs(out_post.temporal_state.previous_features - out_pre.temporal_state.previous_features)
    assert torch.allclose(f_delta, torch.zeros_like(f_delta), atol=1e-6), (
        "Feature delta between identical frames must be 0"
    )

    resp_logit = model.post_action_head(out_pre.latent, out_post.latent)
    target_resp = torch.zeros(b_size, 1, dtype=torch.float32)

    loss_h3 = compute_head3_loss(resp_logit, target_resp)
    assert torch.isfinite(loss_h3) and loss_h3.item() >= 0.0

    loss_h3.backward()
    for name, p in model.post_action_head.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(p.grad).all(), f"Gradient non-finite for {name}"


def test_adversarial_static_action_pair_convergence() -> None:
    """Verify that training on static action pairs suppresses response probability toward 0.0."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    b_size = 4
    static_frame = torch.rand(b_size, 3, 128, 128)
    batch_action_static = {
        "pre_frames": static_frame,
        "post_frames": static_frame.clone(),
        "response_labels": torch.zeros(b_size, 1),
    }
    batch_ui = {
        "images": torch.rand(b_size, 3, 128, 128),
        "target_boxes": torch.zeros(b_size, 0, 4),
        "target_classes": torch.zeros(b_size, 0, dtype=torch.long),
        "box_mask": torch.zeros(b_size, 0, dtype=torch.bool),
    }
    batch_temporal = sample_temporal_batch(b_size, 2, (128, 128))

    losses_h3: List[float] = []
    probs: List[float] = []

    for _ in range(5):
        res, _ = execute_training_step(
            model=model,
            preprocessor=preprocessor,
            optimizer=optimizer,
            batch_ui=batch_ui,
            batch_temporal=batch_temporal,
            batch_action=batch_action_static,
            device=device,
            clip_grad_norm=1.0,
        )
        losses_h3.append(float(res.loss_head3.item()))

        with torch.no_grad():
            f4 = preprocessor(static_frame)
            o_pre = model(f4)
            o_post = model(f4, prev_state=o_pre.temporal_state, timestamp=0.2)
            p = torch.sigmoid(model.post_action_head(o_pre.latent, o_post.latent)).mean().item()
            probs.append(p)

    assert losses_h3[-1] < losses_h3[0], f"Head 3 loss did not decrease: {losses_h3}"
    assert probs[-1] < probs[0], f"Response probability did not decay: {probs}"
    assert probs[-1] < 0.05, f"Expected suppressed probability < 0.05, got {probs[-1]}"


# ==============================================================================
# 3. Adversarial Dimension 3: Noisy / Corrupted Inputs
# ==============================================================================


def test_adversarial_nan_input_detection_and_no_silent_corruption() -> None:
    """Verify that input NaNs propagate cleanly to output/loss and are NOT silently masked."""
    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    b_size = 2
    batch_ui = {
        "images": torch.rand(b_size, 3, 128, 128),
        "target_boxes": torch.zeros(b_size, 0, 4),
        "target_classes": torch.zeros(b_size, 0, dtype=torch.long),
        "box_mask": torch.zeros(b_size, 0, dtype=torch.bool),
    }
    # Corrupt a pixel in UI images with NaN
    batch_ui["images"][0, 0, 32, 32] = float("nan")

    batch_temporal = sample_temporal_batch(b_size, 2, (128, 128))
    batch_action = sample_action_pair_batch(b_size, (128, 128))

    res, _ = execute_training_step(
        model=model,
        preprocessor=preprocessor,
        optimizer=optimizer,
        batch_ui=batch_ui,
        batch_temporal=batch_temporal,
        batch_action=batch_action,
        device=device,
        clip_grad_norm=1.0,
    )

    # Loss MUST reflect NaN and NOT return a fake finite loss
    assert torch.isnan(res.total_loss).item() is True, (
        "Silent corruption violation: NaN in input was swallowed into finite loss!"
    )


def test_adversarial_extreme_pixel_values_numerical_stability() -> None:
    """Verify that extreme pixel values (+/-1e4) are handled without crashes or uncaught exceptions."""
    preprocessor = EdgeLuminancePreprocessor()
    model = StreamingVisualCortexV2()

    # Extreme high and negative values
    extreme_images = torch.full((2, 3, 128, 128), 1e4, dtype=torch.float32)
    extreme_images[1] = -1e4

    four_ch = preprocessor(extreme_images)
    assert four_ch.shape == (2, 4, 128, 128)
    assert torch.isfinite(four_ch).all(), "Preprocessor produced non-finite values on extreme inputs"

    out = model(four_ch)
    assert torch.isfinite(out.change_heatmap).all()
    assert torch.isfinite(out.ui_objectness_logits).all()
    assert torch.isfinite(out.post_action_response_logit).all()


def test_adversarial_corrupted_class_ids_filtered_cleanly() -> None:
    """Verify that invalid/corrupted class IDs (-100, 5, 999) are filtered cleanly without panics."""
    b_size = 2
    target_boxes = torch.tensor([
        [[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]],
        [[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]],
    ])
    # All corrupted class IDs (valid are 0..4)
    target_classes = torch.tensor([[-100, 5], [999, -1]])
    box_mask = torch.tensor([[True, True], [True, True]])

    t_obj, t_cls, t_box, p_mask = build_head2_targets(
        target_boxes=target_boxes,
        target_classes=target_classes,
        box_mask=box_mask,
        grid_size=16,
    )

    assert p_mask.sum().item() == 0, "Corrupted class IDs should have been filtered out"
    assert t_obj.sum().item() == 0.0


def test_adversarial_inverted_and_out_of_bounds_boxes_clamped() -> None:
    """Verify that inverted boxes (x2 < x1) and out-of-bounds coordinates are safely clamped."""
    target_boxes = torch.tensor([[
        [-10.0, -20.0, 50.0, 80.0],  # Out of bounds [0, 1]
        [0.8, 0.8, 0.2, 0.2],         # Inverted box (x2 < x1, y2 < y1)
    ]])
    target_classes = torch.tensor([[0, 1]])
    box_mask = torch.tensor([[True, True]])

    t_obj, t_cls, t_box, p_mask = build_head2_targets(
        target_boxes=target_boxes,
        target_classes=target_classes,
        box_mask=box_mask,
        grid_size=16,
    )

    assert p_mask.sum().item() == 2
    # Inverted box width/height must be clamped to minimum 1e-4
    widths = t_box[0, 2, p_mask[0]]
    heights = t_box[0, 3, p_mask[0]]
    assert (widths >= 1e-4).all()
    assert (heights >= 1e-4).all()

    # Compute loss and check backward pass
    pred_obj = torch.randn(1, 1, 16, 16, requires_grad=True)
    pred_cls = torch.randn(1, 5, 16, 16, requires_grad=True)
    pred_box = torch.randn(1, 4, 16, 16, requires_grad=True)

    loss_h2, _, _, _ = compute_head2_loss(
        pred_obj, pred_cls, pred_box, t_obj, t_cls, t_box, p_mask
    )
    assert torch.isfinite(loss_h2)
    loss_h2.backward()
    assert torch.isfinite(pred_box.grad).all()


def test_adversarial_corrupted_targets_rejected_by_bce() -> None:
    """Verify that invalid change heatmap targets (<0 or >1) are rejected with RuntimeError."""
    heat = torch.sigmoid(torch.randn(2, 1, 128, 128))
    spin = torch.randn(2, 1)
    bad_heat_target = torch.full((2, 1, 128, 128), 2.5)
    bad_spin_target = torch.tensor([[1.0], [0.0]])

    with pytest.raises(RuntimeError):
        compute_head1_loss(heat, spin, bad_heat_target, bad_spin_target)


def test_adversarial_nan_in_target_boxes_raises_error() -> None:
    """Verify that NaNs in bounding box coordinates raise clean ValueError."""
    boxes = torch.tensor([[[float("nan"), 0.2, 0.4, 0.4]]])
    classes = torch.tensor([[0]])
    mask = torch.tensor([[True]])

    with pytest.raises(ValueError):
        build_head2_targets(boxes, classes, mask)


# ==============================================================================
# 4. Adversarial Dimension 4: Gradient Clipping Effectiveness
# ==============================================================================


def test_adversarial_gradient_clipping_under_massive_loss_spike() -> None:
    """Verify that gradient norms are strictly capped at 1.0 even under a multi-billion loss spike."""
    model = StreamingVisualCortexV2()
    x = torch.rand(2, 4, 128, 128)
    out = model(x)

    # Induce massive loss spike by scaling by 100,000x
    artificial_loss_spike = (out.ui_objectness_logits.sum() + out.post_action_response_logit.sum()) * 100000.0
    artificial_loss_spike.backward()

    # Pre-clipping norm check
    unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    assert unclipped_norm > 1e6, f"Expected massive unclipped norm, got {unclipped_norm}"

    # Compute actual post-clipping total gradient norm across all model parameters
    post_clip_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]),
        2,
    ).item()

    assert post_clip_norm <= 1.0001, (
        f"Gradient clipping failed: post-clip norm {post_clip_norm} exceeds 1.0!"
    )
    assert math.isclose(post_clip_norm, 1.0, rel_tol=1e-4)


def test_adversarial_gradient_clipping_preserves_small_gradients() -> None:
    """Verify that gradient clipping is a no-op when the gradient norm is already below max_norm."""
    model = StreamingVisualCortexV2()
    x = torch.rand(2, 4, 128, 128)
    out = model(x)

    # Tiny loss producing small gradient norm
    tiny_loss = out.post_action_response_logit.mean() * 1e-4
    tiny_loss.backward()

    pre_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]),
        2,
    ).item()

    assert pre_norm < 1.0, f"Expected small pre-clip norm < 1.0, got {pre_norm}"

    reported_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    post_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]),
        2,
    ).item()

    assert math.isclose(pre_norm, post_norm, rel_tol=1e-4), (
        f"Clipping altered gradient norm below threshold: pre={pre_norm}, post={post_norm}"
    )


# ==============================================================================
# 5. Adversarial Dimension 5: Multi-Step Loss Monotonicity
# ==============================================================================


def test_adversarial_10_step_loss_monotonicity_fixed_batch() -> None:
    """Verify that a 10-step optimization run on a fixed batch strictly decreases loss without divergence.

    Empirical Invariants:
      1. Every single optimization step strictly decreases loss: losses[i] > losses[i+1].
      2. Overall loss reduction is substantial (> 75% reduction from initial loss).
      3. No NaNs or Infs occur at any step.
      4. Sub-component head losses (Head 1, Head 2, Head 3, Latent) are lower at step 10 than step 1.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    b_size = 4
    batch_ui = {
        "images": torch.rand(b_size, 3, 128, 128),
        "target_boxes": torch.tensor([
            [[0.2, 0.2, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
            [[0.6, 0.6, 0.8, 0.8], [0.1, 0.1, 0.3, 0.3]],
            [[0.3, 0.3, 0.7, 0.7], [0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]],
        ]),
        "target_classes": torch.tensor([[0, -1], [1, 2], [3, -1], [4, 0]]),
        "box_mask": torch.tensor([[True, False], [True, True], [True, False], [True, True]]),
    }
    batch_temporal = sample_temporal_batch(batch_size=b_size, sequence_length=2, image_size=(128, 128), rng=rng)
    batch_action = sample_action_pair_batch(batch_size=b_size, image_size=(128, 128), rng=rng)

    losses: List[float] = []
    first_res = None
    last_res = None

    for step in range(10):
        res, grad_norm = execute_training_step(
            model=model,
            preprocessor=preprocessor,
            optimizer=optimizer,
            batch_ui=batch_ui,
            batch_temporal=batch_temporal,
            batch_action=batch_action,
            device=device,
            clip_grad_norm=1.0,
        )
        val = float(res.total_loss.item())
        losses.append(val)
        if step == 0:
            first_res = res
        if step == 9:
            last_res = res

    # 1. No divergence / finite checks
    for i, loss_val in enumerate(losses):
        assert not math.isnan(loss_val) and not math.isinf(loss_val), f"Step {i} produced non-finite loss"
        assert loss_val > 0.0, f"Step {i} produced non-positive loss {loss_val}"

    # 2. Strict step-by-step monotonicity
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1], (
            f"Loss failed strict monotonicity at step {i} -> {i+1}: {losses[i]:.4f} <= {losses[i+1]:.4f}. "
            f"Full loss trajectory: {[round(x, 4) for x in losses]}"
        )

    # 3. Overall loss reduction > 75%
    initial_loss = losses[0]
    final_loss = losses[-1]
    pct_reduction = (initial_loss - final_loss) / initial_loss
    assert pct_reduction > 0.75, (
        f"Expected > 75% loss reduction, got {pct_reduction:.2%} ({initial_loss:.4f} -> {final_loss:.4f})"
    )

    # 4. All specialized heads improved
    assert last_res.loss_head1.item() < first_res.loss_head1.item()
    assert last_res.loss_head2.item() < first_res.loss_head2.item()
    assert last_res.loss_head3.item() < first_res.loss_head3.item()
    assert last_res.loss_latent.item() < first_res.loss_latent.item()
