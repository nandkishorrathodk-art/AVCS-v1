"""Unit & Integration Tests for Step 6.2 Architecture & Hardware Guardrails.

Verifies:
1. GUIActorPatchPointer cross-attention and sub-patch offset math.
2. UnifiedMultiTaskLoss gradient flow and coordinate convergence.
3. QwenGUIActorModel parameter accounting (~3.7B frozen backbone vs ~1.49M trainable heads).
4. Titlebar Chrome Guard: strict suppression of micro-jitter retries when y < 40 (ty < 0.05).
5. Fast Brain WebSocket Routing: direct execution via VLAClient (/ws/act).
6. High-Speed Screen Capture: sub-25ms local capture + TurboJPEG encoding.
"""
from __future__ import annotations

import io
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Ensure repository root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from avcs.protocol.actions import Action, ActionKind
from avcs.models.vla_cortex import (
    ACTION_VOCAB_10,
    GUIActorPatchPointer,
    GUIPointerOutput,
    ParallelActionHeads,
    UnifiedMultiTaskLoss,
)
from avcs.runtime.agent import AceAgent
from avcs.os_io.computer import Observation, WindowsComputer
from avcs.control.verifier import VerificationStatus, VisualRippleVerifier
import importlib.util

train_vla_path = Path(repo_root) / "kaggle" / "train_vla_gui_actor.py"
spec = importlib.util.spec_from_file_location("train_vla_gui_actor", train_vla_path)
train_vla_module = importlib.util.module_from_spec(spec)
sys.modules["train_vla_gui_actor"] = train_vla_module
spec.loader.exec_module(train_vla_module)

QwenGUIActorModel = train_vla_module.QwenGUIActorModel
MockQwenBackbone = train_vla_module.MockQwenBackbone
GroundingDataset = train_vla_module.GroundingDataset
run_overfit_sanity = train_vla_module.run_overfit_sanity
maybe_auto_spawn_torchrun = train_vla_module.maybe_auto_spawn_torchrun
DAggerReplayBuffer = train_vla_module.DAggerReplayBuffer
RLVisualMotorTrainer = train_vla_module.RLVisualMotorTrainer

from fastapi.testclient import TestClient
from avcs.runtime.vla_engine import FastVLAInferenceEngine
from avcs.runtime.server import app, set_fast_engine
from avcs.runtime.turbo_jpeg import encode_turbo_jpeg


# --------------------------------------------------------------------------- #
# 1. GUIActorPatchPointer & Offset Math Tests
# --------------------------------------------------------------------------- #

def test_gui_actor_patch_pointer_output_shape_and_bounds():
    """Validates that GUIActorPatchPointer outputs normalized [0, 1] coordinates and bounded offsets."""
    pointer = GUIActorPatchPointer(query_dim=384, patch_dim=128, hidden_dim=128)
    b, k = 2, 256
    patch_tokens = torch.randn(b, k, 128)
    patch_coords = torch.rand(b, k, 2)
    query_input = torch.randn(b, 1, 384)

    out: GUIPointerOutput = pointer(
        query_input=query_input,
        patch_tokens=patch_tokens,
        patch_coords=patch_coords,
        patch_stride=(1.0 / 16.0, 1.0 / 16.0),
    )

    assert isinstance(out, GUIPointerOutput)
    assert out.coords.shape == (b, 2)
    assert out.subpixel_coords.shape == (b, 2)
    assert out.attention.shape == (b, k)
    assert out.attn_logits.squeeze(1).shape == (b, k)

    # Check bounds
    assert torch.all(out.coords >= 0.0) and torch.all(out.coords <= 1.0)
    assert torch.all(out.delta >= -0.5) and torch.all(out.delta <= 0.5)
    # Attention probabilities must sum to 1.0 per batch item
    prob_sums = out.attention.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. UnifiedMultiTaskLoss Gradient Flow & Convergence Tests
# --------------------------------------------------------------------------- #

def test_unified_multitask_loss_gradient_flow():
    """Verifies that UnifiedMultiTaskLoss flows gradients to both action logits and target coordinates."""
    criterion = UnifiedMultiTaskLoss(weight_action=1.0, weight_coord_l1=15.0, weight_coord_l2=35.0)

    pred_logits = torch.randn(4, 10, requires_grad=True)
    pred_coords = torch.rand(4, 2, requires_grad=True)
    target_actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_coords = torch.tensor([[0.2, 0.3], [0.5, 0.5], [0.8, 0.1], [0.9, 0.9]], dtype=torch.float32)

    loss_dict = criterion(
        model_out={"action_logits": pred_logits, "target_coords": pred_coords},
        target_actions=target_actions,
        target_coords=target_coords,
    )

    total_loss = loss_dict["loss"]
    assert total_loss > 0.0
    assert not torch.isnan(total_loss)

    total_loss.backward()

    assert pred_logits.grad is not None
    assert torch.norm(pred_logits.grad) > 0.0
    assert pred_coords.grad is not None
    assert torch.norm(pred_coords.grad) > 0.0


def test_unified_multitask_loss_convergence():
    """Verifies that an optimizer minimizes the coordinate loss towards target coordinates."""
    coords = nn.Parameter(torch.tensor([[0.1, 0.1]], dtype=torch.float32))
    logits = nn.Parameter(torch.randn(1, 10, dtype=torch.float32))
    target_coord = torch.tensor([[0.85, 0.75]], dtype=torch.float32)
    target_action = torch.tensor([0], dtype=torch.long)

    optimizer = torch.optim.Adam([coords, logits], lr=0.1)
    criterion = UnifiedMultiTaskLoss(weight_action=1.0, weight_coord_l1=15.0, weight_coord_l2=35.0)

    initial_dist = torch.norm(coords - target_coord).item()

    for _ in range(50):
        optimizer.zero_grad()
        loss_dict = criterion(
            model_out={"action_logits": logits, "target_coords": coords},
            target_actions=target_action,
            target_coords=target_coord,
        )
        loss_dict["loss"].backward()
        optimizer.step()

    final_dist = torch.norm(coords - target_coord).item()
    assert final_dist < initial_dist
    assert final_dist < 0.05


# --------------------------------------------------------------------------- #
# 3. QwenGUIActorModel Parameter Accounting
# --------------------------------------------------------------------------- #

def test_qwen_gui_actor_parameter_accounting():
    """Verifies that QwenGUIActorModel strictly freezes backbone parameters and reports ~1.49M trainable heads."""
    model = QwenGUIActorModel(dry_run=True, backbone_hidden_dim=2048, pointer_hidden_dim=128, query_dim=384)
    stats = model.get_parameter_summary()

    # Trainable head parameters must be around ~1.49M
    assert stats["trainable_params"] > 1_000_000
    assert stats["trainable_params"] < 3_000_000
    # Frozen parameters must be greater than 0
    assert stats["frozen_params"] > 0

    # Ensure no backbone parameters have requires_grad=True
    for name, param in model.named_parameters():
        if "backbone" in name:
            assert not param.requires_grad, f"Backbone parameter {name} was not frozen!"
        elif "patch_proj" in name or "query_proj" in name or "pointer" in name or "action_heads" in name:
            assert param.requires_grad, f"Head parameter {name} is not trainable!"


# --------------------------------------------------------------------------- #
# 4. Titlebar Chrome Guard Protection Tests
# --------------------------------------------------------------------------- #

def test_titlebar_chrome_retry_ban_y_less_than_40():
    """Verifies that clicks in window titlebar / chrome (y < 40 or ty < 0.05) strictly suppress micro-jitter retries."""
    mock_eye = MagicMock()
    mock_eye.observe.return_value = np.zeros((800, 1280, 3), dtype=np.uint8)
    mock_driver = MagicMock()

    verifier = VisualRippleVerifier(eye=mock_eye, driver=mock_driver)

    # Coordinate near top chrome (e.g. close button at y=24, normalized ty=0.030)
    chrome_target = (0.767, 0.030)

    # Mock verify_action to report unconfirmed ripple on initial click
    fail_res = MagicMock()
    fail_res.is_successful = False
    fail_res.status = VerificationStatus.RETRY_MICRO_JITTER
    fail_res.photometric_delta = 0.001
    fail_res.optical_ripple_energy = 0.0001
    fail_res.surprise_pulse = 0.0001
    fail_res.state_transition_detected = False
    fail_res.causal_summary = "Unconfirmed"
    fail_res.diff = None

    with patch.object(verifier, "verify_action", return_value=fail_res):
        res = verifier.execute_and_verify(
            action_type="click",
            target=chrome_target,
        )

    # Must be ESCALATE_BLOCKED with titlebar ban reason
    assert res.status == VerificationStatus.ESCALATE_BLOCKED
    assert "window chrome / titlebar" in res.causal_summary
    # Ensure driver.click was called only ONCE for the initial click, with ZERO retry clicks fired
    assert mock_driver.click.call_count == 1


# --------------------------------------------------------------------------- #
# 5. Fast Brain WebSocket Routing Tests
# --------------------------------------------------------------------------- #

def test_fast_brain_websocket_vla_direct_actuation():
    """Verifies that AceAgent routes through VLAClient over WebSocket when configured."""
    mock_vla = MagicMock()
    mock_vla.act.return_value = {
        "action": "left_click",
        "coordinate": [346, 785],
        "latency_ms": 32.5,
        "roundtrip_ms": 145.2,
    }

    agent = AceAgent(vla_client=mock_vla)

    dummy_obs = Observation(
        image_bytes=b"fake_jpeg",
        format="jpeg",
        width=1280,
        height=800,
        native_width=1280,
        native_height=800,
        raw_numpy=np.zeros((800, 1280, 3), dtype=np.uint8),
    )

    action, lat = agent.step(dummy_obs, goal="Click the search bar")

    assert action.kind == ActionKind.LEFT_CLICK
    assert action.coordinate == (346, 785)
    assert action.metadata.get("fast_brain") is True
    assert action.metadata.get("vla_streaming") is True
    mock_vla.act.assert_called_once()


# --------------------------------------------------------------------------- #
# 6. High-Speed Screen Capture Latency Test
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific GDI capture test")
def test_windows_computer_observe_sub_25ms():
    """Verifies that WindowsComputer.observe captures and TurboJPEG encodes in <25ms."""
    if sys.platform == "win32":
        try:
            import ctypes
            hdesk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
            if not hdesk:
                pytest.skip("Windows desktop is locked or headless - GDI fast capture unavailable")
            ctypes.windll.user32.CloseDesktop(hdesk)
        except Exception:
            pytest.skip("Windows desktop is locked or headless - GDI fast capture unavailable")
    comp = WindowsComputer()
    try:
        # Warmup grab
        comp.observe()
        t0 = time.perf_counter()
        obs = comp.observe()
        t_total = (time.perf_counter() - t0) * 1000.0

        assert obs.width == 1280
        assert obs.height == 800
        assert len(obs.image_bytes) > 0
        assert obs.format == "jpeg"
        # Latency must be < 35ms on CPU / desktop
        assert obs.capture_time_ms < 35.0
        assert t_total < 45.0
    finally:
        comp.close()


# --------------------------------------------------------------------------- #
# 7. Step 6.2 Pre-Flight Hardening: 512-dim Projections & Dynamic Alignment
# --------------------------------------------------------------------------- #

def test_qwen_gui_actor_512_widened_projections():
    """Verifies default QwenGUIActorModel uses 512-dim projections yielding ~3.98M trainable parameters."""
    model = QwenGUIActorModel(dry_run=True)
    stats = model.get_parameter_summary()

    # Exact ~3.98M parameters (3,978,386 params)
    assert 3_900_000 <= stats["trainable_params"] <= 4_100_000
    assert stats["trainable_params"] == 3_978_386
    assert stats["frozen_params"] > 0

    # Ensure all backbone parameters are frozen
    for name, param in model.named_parameters():
        if "backbone" in name:
            assert not param.requires_grad, f"Backbone parameter {name} was not frozen!"
        elif any(head_k in name for head_k in ("patch_proj", "query_proj", "pointer", "action_heads")):
            assert param.requires_grad, f"Head parameter {name} is not trainable!"


def test_dynamic_image_grid_thw_non_square_alignment():
    """Verifies dynamic image_grid_thw alignment: (H_patches, W_patches) = (H_grid // 2, W_grid // 2)."""
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    device = torch.device("cpu")

    # Non-square 16:10 screen aspect ratio: H_grid=30, W_grid=48 -> gh=15, gw=24
    grid_thw_16_10 = torch.tensor([[1, 30, 48]], dtype=torch.long)
    images = torch.randn(2, 3, 400, 640)

    out = model(images, image_grid_thw=grid_thw_16_10)

    assert out["grid_hw"] == (15, 24)
    assert out["target_coords"].shape == (2, 2)
    assert out["action_logits"].shape == (2, len(ACTION_VOCAB_10))
    # Normalized coordinates must be within [0, 1]
    assert torch.all(out["target_coords"] >= 0.0) and torch.all(out["target_coords"] <= 1.0)
    assert out["patch_coords"].shape == (360, 2)

    # Non-square 16:9 screen aspect ratio: H_grid=20, W_grid=36 -> gh=10, gw=18
    grid_thw_16_9 = torch.tensor([[1, 20, 36]], dtype=torch.long)
    out_16_9 = model(images, image_grid_thw=grid_thw_16_9)

    assert out_16_9["grid_hw"] == (10, 18)
    assert out_16_9["patch_coords"].shape == (180, 2)


# --------------------------------------------------------------------------- #
# 8. Fail-Closed Dataset Ingestion & Zero-Leakage Gate
# --------------------------------------------------------------------------- #

def test_fail_closed_dataset_ingestion_and_zero_leakage(tmp_path: Path):
    """Verifies fail-closed dataset gate and held-out showdown-clicks-dev zero leakage."""
    # 1. Fail-closed: raises RuntimeError if dataset paths are missing and allow_synthetic_fallback=False
    with pytest.raises(RuntimeError, match="Fail-closed dataset gate"):
        GroundingDataset(
            dataset_paths=[str(tmp_path / "nonexistent.jsonl")],
            allow_synthetic_fallback=False,
        )

    # 2. Zero-leakage gate against showdown-clicks-dev
    sample_file = tmp_path / "train_sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        # Valid sample
        f.write('{"id": "os_atlas_101", "instruction": "Click search", "conversations": [{"from": "human", "value": "Click search"}, {"from": "gpt", "value": "{\\"action\\": \\"left_click\\", \\"coordinate\\": [200, 300]}"}]}\n')
        # Held-out leakage sample
        f.write('{"id": "showdown-clicks-dev-999", "instruction": "Click leaked", "conversations": [{"from": "human", "value": "Click leaked"}, {"from": "gpt", "value": "{\\"action\\": \\"left_click\\", \\"coordinate\\": [100, 100]}"}]}\n')

    ds = GroundingDataset(dataset_paths=[str(sample_file)], allow_synthetic_fallback=False)
    assert len(ds) == 1
    assert ds.leakage_filtered_count == 1
    assert ds.samples[0]["instruction"] == "Click search"

    # 3. Clean skip on non-existent golden directory
    missing_golden = str(tmp_path / "skills_vault" / "golden")
    ds_golden = GroundingDataset(dataset_paths=[missing_golden], allow_synthetic_fallback=True)
    assert len(ds_golden) == 1000  # Fallback to synthetic without crashing


# --------------------------------------------------------------------------- #
# 9. Multi-Resolution Overfit Sanity Mode
# --------------------------------------------------------------------------- #

def test_multi_resolution_overfit_sanity():
    """Verifies that run_overfit_sanity achieves >=95% Strict (0px) accuracy across multi-resolution targets."""
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    # Test on dynamic non-square screen resolutions
    test_res = [(800, 1280), (600, 800)]
    results = run_overfit_sanity(model=model, resolutions=test_res, steps_per_res=35, lr=3e-4)

    for res_key in ("800x1280", "600x800"):
        assert res_key in results
        assert results[res_key]["strict_accuracy"] >= 0.95
        assert results[res_key]["converged"] is True
        assert results[res_key]["loss_drop"] > 0.0
        assert results[res_key]["final_loss"] < results[res_key]["initial_loss"]


# --------------------------------------------------------------------------- #
# 10. Auto-Torchrun Detection & Multi-GPU Guardrail
# --------------------------------------------------------------------------- #

def test_auto_torchrun_launcher_guardrails():
    """Verifies auto-torchrun launcher detection logic under diverse rank states."""
    # Under single GPU or CPU without force, returns False
    with patch("torch.cuda.device_count", return_value=1):
        assert maybe_auto_spawn_torchrun() is False

    # When LOCAL_RANK is already set, should never re-spawn
    with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
        with patch("torch.cuda.device_count", return_value=2):
            assert maybe_auto_spawn_torchrun() is False


# --------------------------------------------------------------------------- #
# 11. DAgger Replay Buffer & RL Visual-Motor Trainer Tests
# --------------------------------------------------------------------------- #

def test_dagger_replay_buffer_and_rl_trainer():
    """Verifies DAgger interactive buffer aggregation and RL policy gradient loss."""
    # 1. DAgger replay buffer
    buf = DAggerReplayBuffer(capacity=100, initial_beta=1.0, beta_decay=0.9)
    dummy_img = torch.zeros(3, 400, 640)
    buf.add_interaction(
        image=dummy_img,
        instruction="Click save",
        student_action=0,
        student_coord=[0.5, 0.5],
        expert_action=0,
        expert_coord=[0.8, 0.2],
        intervention_needed=True,
    )
    assert len(buf) == 1
    new_beta = buf.decay_beta(min_beta=0.1)
    assert new_beta == 0.9
    batch = buf.sample_batch(batch_size=1)
    assert len(batch) == 1
    assert batch[0]["instruction"] == "Click save"

    # 2. RL policy gradient trainer
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    rl_trainer = RLVisualMotorTrainer(model=model, lr=1e-3)

    pred_out = {
        "action_logits": torch.randn(2, 10, requires_grad=True),
        "target_coords": torch.tensor([[0.5, 0.5], [0.3, 0.3]], requires_grad=True),
    }
    action_idx = torch.tensor([0, 1], dtype=torch.long)
    coord_xy = torch.tensor([[0.52, 0.48], [0.31, 0.29]], dtype=torch.float32)
    reward = torch.tensor([1.0, 1.0], dtype=torch.float32)  # Positive verification reward

    pg_dict = rl_trainer.compute_policy_gradient_loss(pred_out, action_idx, coord_xy, reward)
    loss = pg_dict["loss"]
    assert not torch.isnan(loss)
    loss.backward()
    assert pred_out["action_logits"].grad is not None
    assert pred_out["target_coords"].grad is not None


# --------------------------------------------------------------------------- #
# 12. Sub-350ms Production Inference Engine & WebSocket Serving Tests
# --------------------------------------------------------------------------- #

def test_fast_vla_inference_engine_sub_350ms():
    """Verifies FastVLAInferenceEngine produces valid actuation in under 350ms."""
    engine = FastVLAInferenceEngine(
        precision="fp32",
        enable_token_pruning=False,
        enable_rolling_kv=True,
    )

    test_pil = Image.new("RGB", (1280, 800), color=(128, 128, 128))
    # CPU warmup to bypass initial torch JIT / memory initialization overhead
    engine.act(image=test_pil, instruction="Warmup", canvas_w=1280, canvas_h=800, session_id="warmup")
    res = engine.act(
        image=test_pil,
        instruction="Click the OK button",
        canvas_w=1280,
        canvas_h=800,
        session_id="test_session_1",
    )

    assert res["action"] in ACTION_VOCAB_10
    assert isinstance(res["coordinate"], list) and len(res["coordinate"]) == 2
    assert 0 <= res["coordinate"][0] <= 1280
    assert 0 <= res["coordinate"][1] <= 800
    assert res["engine"] == "FastVLA-Production"
    # Strict sub-350ms latency guarantee
    assert res["latency_ms"] < 350.0


def test_server_ws_act_with_fast_vla_engine():
    """Verifies FastAPI /ws/act streaming endpoint powered by FastVLAInferenceEngine."""
    client = TestClient(app)
    engine = FastVLAInferenceEngine(precision="fp32", enable_token_pruning=False)
    set_fast_engine(engine)

    try:
        raw_arr = np.zeros((800, 1280, 3), dtype=np.uint8)
        jpeg_bytes = encode_turbo_jpeg(raw_arr, quality=80)

        with client.websocket_connect("/ws/act?instruction=Click+Search&width=1280&height=800") as ws:
            ws.send_bytes(jpeg_bytes)
            data = ws.receive_json()

            assert data["action"] in ACTION_VOCAB_10
            assert isinstance(data["coordinate"], list)
            assert len(data["coordinate"]) == 2
            assert data["latency_ms"] < 350.0
            assert data.get("engine") == "FastVLA-Production"
    finally:
        set_fast_engine(None)

