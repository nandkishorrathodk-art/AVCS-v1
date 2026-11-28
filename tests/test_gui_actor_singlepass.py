"""Unit tests for MSGUIActorSinglePassVLA (0-generate GUI-Actor-3B + ParallelActionHeads)."""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import torch

from avcs.models.gui_actor_singlepass import (
    MSGUIActorSinglePassVLA,
    ms_region_point_from_attn,
    preserve_aspect_resize,
)
from avcs.models.vla_cortex import ACTION_VOCAB_10, UnifiedMultiTaskLoss


def test_ms_region_point_from_attn_picks_brightest_connected_region():
    attn = torch.zeros((1, 4 * 5), dtype=torch.float32)
    # Two adjacent high patches in row 2, cols 3 and 4 (grid 5x4 -> idx = y*5 + x)
    attn[0, 2 * 5 + 3] = 0.45
    attn[0, 2 * 5 + 4] = 0.45
    # Isolated lower noise patch at (0, 0)
    attn[0, 0] = 0.10

    x, y = ms_region_point_from_attn(attn, n_width=5, n_height=4, activation_threshold=0.3)
    assert abs(x - 0.80) < 1e-4  # center between col 3 (0.7) and col 4 (0.9)
    assert abs(y - 0.625) < 1e-4  # row 2 -> (2.5 / 4)


def test_preserve_aspect_resize_does_not_squish_16_9():
    img_16_9 = Image.new("RGB", (1920, 1080), color=(30, 60, 90))
    resized = preserve_aspect_resize(img_16_9, max_w=1280, max_h=800)
    w, h = resized.size
    assert w <= 1280 and h <= 800
    assert abs((w / h) - (1920 / 1080)) < 0.02

    small = Image.new("RGB", (640, 360), color=(10, 20, 30))
    assert preserve_aspect_resize(small, max_w=1280, max_h=800).size == (640, 360)


def test_ms_gui_actor_single_pass_dry_forward_and_loss_backward():
    model = MSGUIActorSinglePassVLA(dry_run=True, query_dim=256, backbone_hidden_dim=512)
    imgs = torch.rand(2, 3, 280, 448)
    out = model(imgs, queries=["click Save button", "scroll down"])

    assert out["generate_calls"] == 0
    assert out["action_logits"].shape == (2, len(ACTION_VOCAB_10))
    assert out["subpixel_coords"].shape == (2, 2)
    assert out["target_coords"].shape == (2, 2)
    assert out["attn_probs"].shape[0] == 2
    assert out["attn_logits"].shape == out["attn_probs"].shape

    criterion = UnifiedMultiTaskLoss(weight_action=5.0, weight_coord_l1=10.0, weight_coord_l2=10.0, weight_patch_ce=5.0)
    loss_dict = criterion(
        model_out=out,
        target_actions=torch.tensor([0, 5], dtype=torch.long),
        target_coords=torch.tensor([[0.7, 0.5], [0.5, 0.5]], dtype=torch.float32),
        target_bboxes=torch.tensor([[0.65, 0.45, 0.75, 0.55], [0.45, 0.45, 0.55, 0.55]], dtype=torch.float32),
    )
    loss_dict["loss"].backward()
    assert any(p.grad is not None for p in model.action_heads.parameters())
    assert model.backbone.multi_patch_pointer_head.proj_enc.weight.grad is not None


def test_train_ddp_dry_run_with_gui_actor_3b_backend(tmp_path: Path):
    import importlib.util

    train_path = Path(__file__).resolve().parent.parent / "kaggle" / "train_vla_gui_actor.py"
    spec = importlib.util.spec_from_file_location("avcs_kaggle_train_vla_gui_actor", train_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    save_dir = tmp_path / "ckpts"
    mod.run_ddp_training(
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        save_dir=str(save_dir),
        dry_run=True,
        max_samples=16,
        dataset_source="synthetic",
        model_backend="gui_actor_3b",
        log_interval=4,
    )
    ckpt = save_dir / "gui_actor_step6_2_epoch_1.pt"
    assert ckpt.is_file()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert blob["model_backend"] == "gui_actor_3b"
    assert any("multi_patch_pointer_head." in k for k in blob["model_state_dict"])
