"""Comprehensive Unit Test Suite for StreamingVisualCortexV2 Multi-Task Training.

Milestone M4 Verification:
  1. Multi-task loss functions with synthetic batches (finite, non-negative, non-zero gradients).
  2. Single training step weight updates across stem, ConvGRU, and all 3 specialized heads.
  3. Short training run convergence on CPU without NaNs or Infs.
  4. Candidate checkpoint serialization, SHA-256 digest, and clean state-dict loading.
  5. Pure-Pixel AST audit across avcs/training/train_streaming_v2.py (0 prohibited imports).
  6. Frozen baseline weight guardian (verifies legacy checkpoints remain bit-for-bit identical).
  7. Empty-batch hard negative safety.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Dict, List, Set

import numpy as np
import pytest
import torch

from avcs.streaming_v2.model import StreamingVisualCortexV2
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.training.train_streaming_v2 import (
    BASELINE_WEIGHT_HASHES,
    build_head2_targets,
    compute_future_latent_loss,
    compute_head1_loss,
    compute_head2_loss,
    compute_head3_loss,
    execute_training_step,
    train_streaming_v2,
    verify_baseline_weights_untouched,
)


# ==============================================================================
# 1. Baseline Weights Integrity Guardian
# ==============================================================================


def test_baseline_weights_integrity_guardian() -> None:
    """Verify that all baseline weights exist and match their expected SHA-256 digests."""
    verified = verify_baseline_weights_untouched()
    assert len(verified) == 3
    for rel_path, expected_hash in BASELINE_WEIGHT_HASHES.items():
        assert rel_path in verified
        assert verified[rel_path] == expected_hash, f"Hash mismatch for baseline {rel_path}"


# ==============================================================================
# 2. Head 2 Anchor-Free Grid Target Generation
# ==============================================================================


def test_head2_target_generation_and_shapes() -> None:
    """Test 16x16 grid target generation with positive objects and empty negatives."""
    b_size = 2
    max_objs = 3

    # Sample 0 has 2 objects; Sample 1 has 0 objects (negative screen)
    target_boxes = torch.zeros((b_size, max_objs, 4), dtype=torch.float32)
    target_classes = torch.full((b_size, max_objs), -1, dtype=torch.int64)
    box_mask = torch.zeros((b_size, max_objs), dtype=torch.bool)

    # Sample 0, Object 0: Button in center [0.4, 0.4, 0.6, 0.6] -> centroid (0.5, 0.5) -> grid (8, 8)
    target_boxes[0, 0] = torch.tensor([0.4, 0.4, 0.6, 0.6])
    target_classes[0, 0] = 0  # Button
    box_mask[0, 0] = True

    # Sample 0, Object 1: Input box in top-left [0.05, 0.05, 0.25, 0.15] -> centroid (0.15, 0.10) -> grid (2, 1)
    target_boxes[0, 1] = torch.tensor([0.05, 0.05, 0.25, 0.15])
    target_classes[0, 1] = 1  # Input box
    box_mask[0, 1] = True

    t_obj, t_cls, t_box, p_mask = build_head2_targets(
        target_boxes=target_boxes,
        target_classes=target_classes,
        box_mask=box_mask,
        grid_size=16,
    )

    assert t_obj.shape == (2, 1, 16, 16)
    assert t_cls.shape == (2, 16, 16)
    assert t_box.shape == (2, 4, 16, 16)
    assert p_mask.shape == (2, 16, 16)

    # Sample 0 must have exactly 2 positive grid cells
    assert p_mask[0].sum().item() == 2
    assert t_obj[0, 0, 8, 8].item() == 1.0
    assert t_cls[0, 8, 8].item() == 0  # Button class
    # Offsets at center (0.5, 0.5) in 16x16 grid: stride = 0.0625, cell_cx = 8.5 * 0.0625 = 0.53125
    assert abs(t_box[0, 2, 8, 8].item() - 0.2) < 1e-4  # width = 0.6 - 0.4 = 0.2
    assert abs(t_box[0, 3, 8, 8].item() - 0.2) < 1e-4  # height = 0.6 - 0.4 = 0.2

    # Sample 1 (negative) must have 0 positive cells
    assert p_mask[1].sum().item() == 0
    assert t_obj[1].sum().item() == 0.0


# ==============================================================================
# 3. Multi-Task Losses: Finite, Non-Negative, and Gradients
# ==============================================================================


def test_multi_task_losses_finite_non_negative_and_gradients() -> None:
    """Verify that all multi-task loss functions are finite, >= 0, and backprop non-zero gradients."""
    b_size = 2

    # 1. Head 1 Loss
    raw_heatmap = torch.randn(b_size, 1, 128, 128, requires_grad=True)
    pred_heatmap = torch.sigmoid(raw_heatmap)
    pred_spinner = torch.randn(b_size, 1, requires_grad=True)
    target_heat = torch.rand(b_size, 1, 128, 128)
    target_spin = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss_h1, l_change, l_spin = compute_head1_loss(pred_heatmap, pred_spinner, target_heat, target_spin)
    assert torch.isfinite(loss_h1) and loss_h1.item() >= 0.0
    assert torch.isfinite(l_change) and l_change.item() >= 0.0
    assert torch.isfinite(l_spin) and l_spin.item() >= 0.0

    loss_h1.backward()
    assert raw_heatmap.grad is not None and raw_heatmap.grad.abs().sum() > 0.0
    assert pred_spinner.grad is not None and pred_spinner.grad.abs().sum() > 0.0

    # 2. Head 2 Loss
    pred_obj = torch.randn(b_size, 1, 16, 16, requires_grad=True)
    pred_cls = torch.randn(b_size, 5, 16, 16, requires_grad=True)
    pred_box = torch.randn(b_size, 4, 16, 16, requires_grad=True)

    t_obj = torch.zeros(b_size, 1, 16, 16)
    t_obj[0, 0, 4, 4] = 1.0
    t_cls = torch.full((b_size, 16, 16), -1, dtype=torch.long)
    t_cls[0, 4, 4] = 2
    t_box = torch.zeros(b_size, 4, 16, 16)
    t_box[0, :, 4, 4] = torch.tensor([0.1, -0.1, 0.25, 0.3])
    pos_mask = (t_obj[:, 0] > 0.5)

    loss_h2, l_obj, l_cls, l_box = compute_head2_loss(
        pred_obj, pred_cls, pred_box, t_obj, t_cls, t_box, pos_mask
    )
    assert torch.isfinite(loss_h2) and loss_h2.item() >= 0.0
    assert torch.isfinite(l_obj) and l_obj.item() >= 0.0
    assert torch.isfinite(l_cls) and l_cls.item() >= 0.0
    assert torch.isfinite(l_box) and l_box.item() >= 0.0

    loss_h2.backward()
    assert pred_obj.grad is not None and pred_obj.grad.abs().sum() > 0.0
    assert pred_cls.grad is not None and pred_cls.grad.abs().sum() > 0.0
    assert pred_box.grad is not None and pred_box.grad.abs().sum() > 0.0

    # 3. Head 3 Loss
    pred_resp = torch.randn(b_size, 1, requires_grad=True)
    target_resp = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    loss_h3 = compute_head3_loss(pred_resp, target_resp)
    assert torch.isfinite(loss_h3) and loss_h3.item() >= 0.0

    loss_h3.backward()
    assert pred_resp.grad is not None and pred_resp.grad.abs().sum() > 0.0

    # 4. Future Latent Loss
    z_hat = torch.randn(b_size, 96, requires_grad=True)
    z_act = torch.randn(b_size, 96, requires_grad=True)
    loss_latent = compute_future_latent_loss(z_hat, z_act)
    assert torch.isfinite(loss_latent) and loss_latent.item() >= 0.0

    loss_latent.backward()
    assert z_hat.grad is not None and z_hat.grad.abs().sum() > 0.0


# ==============================================================================
# 4. Single Training Step Weight Updates
# ==============================================================================


def test_single_training_step_weight_updates() -> None:
    """Verify that a single training step updates weights across stem, ConvGRU, and all 3 heads."""
    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    preprocessor = EdgeLuminancePreprocessor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Initial parameter snapshot
    initial_weights = {name: p.clone().detach() for name, p in model.named_parameters()}

    # Synthetic batches
    b_size = 2
    batch_ui = {
        "images": torch.rand(b_size, 3, 128, 128),
        "target_boxes": torch.tensor([
            [[0.2, 0.2, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
            [[0.6, 0.6, 0.8, 0.8], [0.1, 0.1, 0.3, 0.3]],
        ]),
        "target_classes": torch.tensor([[0, -1], [1, 2]]),
        "box_mask": torch.tensor([[True, False], [True, True]]),
    }

    batch_temporal = {
        "sequences": torch.rand(b_size, 2, 3, 128, 128),
        "change_heatmaps": torch.rand(b_size, 1, 128, 128),
        "spinner_labels": torch.tensor([[1.0], [0.0]]),
    }

    batch_action = {
        "pre_frames": torch.rand(b_size, 3, 128, 128),
        "post_frames": torch.rand(b_size, 3, 128, 128),
        "response_labels": torch.tensor([[1.0], [0.0]]),
    }

    # Execute training step
    loss_res, grad_norm = execute_training_step(
        model=model,
        preprocessor=preprocessor,
        optimizer=optimizer,
        batch_ui=batch_ui,
        batch_temporal=batch_temporal,
        batch_action=batch_action,
        device=device,
        clip_grad_norm=1.0,
    )

    assert torch.isfinite(loss_res.total_loss)
    assert grad_norm > 0.0

    # Verify that every component has updated weights
    component_updated: Dict[str, bool] = {
        "stem": False,
        "conv_gru": False,
        "latent_projection": False,
        "future_predictor": False,
        "change_spinner_head": False,
        "ui_detector_head": False,
        "post_action_head": False,
    }

    for name, param in model.named_parameters():
        init_param = initial_weights[name]
        diff = (param - init_param).abs().max().item()
        if diff > 1e-6:
            for comp in component_updated:
                if name.startswith(comp):
                    component_updated[comp] = True

    for comp, updated in component_updated.items():
        assert updated, f"Component {comp} did NOT update weights during execute_training_step!"


# ==============================================================================
# 5. Short Training Run Convergence on CPU
# ==============================================================================


def test_short_train_run_convergence_on_cpu() -> None:
    """Verify that a short 5-step CPU training run completes with finite losses and parameter divergence."""
    res = train_streaming_v2(
        steps=5,
        batch_size=4,
        lr=1e-3,
        device_name="cpu",
        save_checkpoint=False,
        seed=42,
        log_interval=2,
    )

    assert res["steps"] == 5
    assert len(res["loss_history"]) == 5
    for loss_val in res["loss_history"]:
        assert not np.isnan(loss_val) and not np.isinf(loss_val)
        assert loss_val > 0.0

    assert res["l2_divergence"] > 0.1, "Model parameters failed to diverge from initial weights"


# ==============================================================================
# 6. Candidate Checkpoint Save & Load Verification
# ==============================================================================


def test_candidate_checkpoint_save_and_load() -> None:
    """Verify that training serializes candidate weights and sidecar JSON cleanly and loadable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = train_streaming_v2(
            steps=2,
            batch_size=4,
            lr=1e-3,
            device_name="cpu",
            output_dir=tmp_dir,
            checkpoint_name="test_candidate.pt",
            save_checkpoint=True,
            seed=42,
        )

        ckpt_path = Path(tmp_dir) / "test_candidate.pt"
        json_path = Path(tmp_dir) / "test_candidate.json"

        assert ckpt_path.exists(), "Checkpoint .pt file was not written"
        assert json_path.exists(), "Sidecar .json file was not written"

        # Verify sidecar JSON
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        assert meta["parameter_count"] == 400765
        assert meta["total_steps"] == 2
        assert "cpu_throughput" in meta

        # Verify SHA-256 match
        computed_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        assert meta["sha256"] == computed_hash

        # Load weights into fresh model instance
        fresh_model = StreamingVisualCortexV2()
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        fresh_model.load_state_dict(ckpt_data["state_dict"])

        # Test inference with loaded weights
        dummy_input = torch.rand(1, 4, 128, 128)
        out = fresh_model(dummy_input)
        assert out.change_heatmap.shape == (1, 1, 128, 128)
        assert out.ui_objectness_logits.shape == (1, 1, 16, 16)
        assert out.post_action_response_logit.shape == (1, 1)
        assert torch.isfinite(out.change_heatmap).all()


# ==============================================================================
# 7. Pure-Pixel AST Audit Across Training Module
# ==============================================================================


def test_pure_pixel_ast_audit_training_module() -> None:
    """Verify that avcs/training/train_streaming_v2.py contains 0 prohibited imports."""
    prohibited_substrings = {
        "tesseract",
        "easyocr",
        "paddleocr",
        "pytesseract",
        "ocr",
        "playwright",
        "selenium",
        "uiautomation",
        "pywinauto",
        "accessibility",
        "win32gui",
        "win32con",
        "msaa",
    }

    train_file = Path("avcs/training/train_streaming_v2.py")
    assert train_file.exists(), f"File {train_file} does not exist"

    tree = ast.parse(train_file.read_text(encoding="utf-8"), filename=str(train_file))

    imported_names: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.lower())

    violations: List[str] = []
    for mod in imported_names:
        for prohibited in prohibited_substrings:
            if prohibited in mod:
                violations.append(f"Prohibited import: {mod!r} matches forbidden pattern {prohibited!r}")

    assert len(violations) == 0, f"Pure-Pixel AST Audit failed with violations:\n" + "\n".join(violations)


# ==============================================================================
# 8. Hard Negative & Empty Batch Safety
# ==============================================================================


def test_empty_batch_hard_negative_safety() -> None:
    """Verify that Head 2 loss handles empty batches (all negative screens) safely without NaNs."""
    b_size = 2
    grid_size = 16

    target_boxes = torch.zeros((b_size, 0, 4), dtype=torch.float32)
    target_classes = torch.zeros((b_size, 0), dtype=torch.int64)
    box_mask = torch.zeros((b_size, 0), dtype=torch.bool)

    t_obj, t_cls, t_box, p_mask = build_head2_targets(
        target_boxes=target_boxes,
        target_classes=target_classes,
        box_mask=box_mask,
        grid_size=grid_size,
    )

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

    assert torch.isfinite(loss_h2) and loss_h2.item() >= 0.0
    assert l_cls.item() == 0.0
    assert l_box.item() == 0.0

    loss_h2.backward()
    assert pred_obj.grad is not None and torch.isfinite(pred_obj.grad).all()


# ==============================================================================
# 9. Adaptive Hard-Negative Weighting & Loss Dynamics
# ==============================================================================


def test_adaptive_hard_negative_weighting_loss() -> None:
    """Verify that compute_head2_loss boosts hard negative cells and negative frames."""
    b_size = 4
    grid_size = 16

    pred_obj = torch.full((b_size, 1, grid_size, grid_size), -4.0, requires_grad=True)
    pred_cls = torch.zeros(b_size, 5, grid_size, grid_size)
    pred_box = torch.zeros(b_size, 4, grid_size, grid_size)

    target_obj = torch.zeros(b_size, 1, grid_size, grid_size)
    target_cls = torch.full((b_size, grid_size, grid_size), -1, dtype=torch.long)
    target_box = torch.zeros(b_size, 4, grid_size, grid_size)
    pos_mask = torch.zeros(b_size, grid_size, grid_size, dtype=torch.bool)

    # Frame 0: Positive frame with 1 object at (4, 4)
    target_obj[0, 0, 4, 4] = 1.0
    target_cls[0, 4, 4] = 0
    pos_mask[0, 4, 4] = True

    # Frame 1: Negative frame with 0 objects, but has a hard negative false alarm spike at (8, 8)
    with torch.no_grad():
        pred_obj[1, 0, 8, 8] = 2.0  # p = sigmoid(2.0) = 0.88 >= 0.25

    loss_h2, l_obj, l_cls, l_box = compute_head2_loss(
        ui_objectness_logits=pred_obj,
        ui_class_logits=pred_cls,
        ui_box_regression=pred_box,
        target_obj=target_obj,
        target_cls=target_cls,
        target_box=target_box,
        pos_mask=pos_mask,
        hard_neg_boost=10.0,
        margin_threshold=0.25,
        neg_frame_multiplier=2.0,
    )

    loss_h2.backward()

    # Gradient on easy negative cell on pos frame: base weight 1.0
    grad_easy_pos_frame = pred_obj.grad[0, 0, 0, 0].item()

    # Gradient on easy negative cell on neg frame: base weight 1.0 * 2.0 = 2.0
    grad_easy_neg_frame = pred_obj.grad[1, 0, 0, 0].item()

    # Gradient on hard negative cell on neg frame: (1.0 + 10.0) * 2.0 = 22.0
    grad_hard_neg_frame = pred_obj.grad[1, 0, 8, 8].item()

    # Easy neg frame gradient must be exactly 2.0x of easy pos frame gradient (same logit -4.0)
    assert abs(grad_easy_neg_frame / (grad_easy_pos_frame + 1e-12) - 2.0) < 1e-3

    # Hard neg cell must receive a massive gradient compared to easy cells
    assert grad_hard_neg_frame > 50.0 * grad_easy_pos_frame


# ==============================================================================
# 10. Telemetry p95 Scaling Unit Test
# ==============================================================================


def test_telemetry_scaling_p95_format() -> None:
    """Verify that candidate sidecar telemetry formats p95_ms_per_step in milliseconds."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = train_streaming_v2(
            steps=3,
            batch_size=4,
            lr=1e-3,
            device_name="cpu",
            output_dir=tmp_dir,
            checkpoint_name="telemetry_test.pt",
            save_checkpoint=True,
            seed=42,
        )
        json_path = Path(tmp_dir) / "telemetry_test.json"
        meta = json.loads(json_path.read_text(encoding="utf-8"))

        p50 = meta["cpu_throughput"]["p50_ms_per_step"]
        p95 = meta["cpu_throughput"]["p95_ms_per_step"]

        assert p95 >= p50, f"p95 ({p95}) must be >= p50 ({p50})"
        assert p95 > 50.0, f"p95 ({p95} ms) was not scaled to milliseconds properly"
