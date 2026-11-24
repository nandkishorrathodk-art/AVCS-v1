"""Pre-Flight Verification Tests for Step 6.2 GUI-Actor Training Script.

Validates the 5 essential pre-flight requirements for kaggle/train_vla_gui_actor.py:
1. Auto-torchrun launcher self-spawning logic and environment checks.
2. Dynamic image_grid_thw non-square aspect ratio alignment across diverse screen resolutions.
3. 512-dim widened projections and parameter accounting (~3.98M trainable parameters).
4. Fail-closed dataset ingestion and zero-leakage gate against showdown-clicks-dev.
5. Multi-resolution overfit sanity verification mode.
6. Verification that token pruning and rolling KV memory are strictly paused by default.
"""
from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from avcs.models.vla_cortex import ACTION_VOCAB_10

train_vla_path = Path(repo_root) / "kaggle" / "train_vla_gui_actor.py"
spec = importlib.util.spec_from_file_location("train_vla_gui_actor", train_vla_path)
train_vla_module = importlib.util.module_from_spec(spec)
sys.modules["train_vla_gui_actor"] = train_vla_module
spec.loader.exec_module(train_vla_module)

GroundingDataset = train_vla_module.GroundingDataset
MockQwenBackbone = train_vla_module.MockQwenBackbone
QwenGUIActorModel = train_vla_module.QwenGUIActorModel
maybe_auto_spawn_torchrun = train_vla_module.maybe_auto_spawn_torchrun
setup_distributed = train_vla_module.setup_distributed
run_overfit_sanity = train_vla_module.run_overfit_sanity


# --------------------------------------------------------------------------- #
# 1. Auto-Torchrun Launcher Tests
# --------------------------------------------------------------------------- #

def test_auto_torchrun_spawns_when_multigpu_unranked():
    """Verifies that maybe_auto_spawn_torchrun invokes subprocess.run with torch.distributed.run when 2+ GPUs."""
    with patch.dict(os.environ, {}, clear=True):
        # Ensure RANK and LOCAL_RANK are absent
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)

        with patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("subprocess.run") as mock_run, \
             patch("sys.exit") as mock_exit:
            mock_run.return_value = MagicMock(returncode=0)

            maybe_auto_spawn_torchrun()

            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            assert "-m" in called_cmd
            assert "torch.distributed.run" in called_cmd
            assert "--nproc_per_node=2" in called_cmd
            assert "--no-auto-spawn" in called_cmd
            mock_exit.assert_called_once_with(0)


def test_auto_torchrun_skips_when_ranked_or_single_gpu():
    """Verifies that maybe_auto_spawn_torchrun does nothing when LOCAL_RANK is set, only 1 GPU, or --no-auto-spawn."""
    # Case 1: LOCAL_RANK is already set
    with patch.dict(os.environ, {"LOCAL_RANK": "0", "RANK": "0"}):
        with patch("torch.cuda.device_count", return_value=2):
            assert maybe_auto_spawn_torchrun() is False

    # Case 2: Only 1 GPU without force
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)
        with patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.is_available", return_value=True):
            assert maybe_auto_spawn_torchrun() is False

    # Case 3: --no-auto-spawn explicitly in sys.argv
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)
        with patch.object(sys, "argv", ["train_vla_gui_actor.py", "--no-auto-spawn"]), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.is_available", return_value=True):
            assert maybe_auto_spawn_torchrun() is False


# --------------------------------------------------------------------------- #
# 2. Dynamic image_grid_thw Alignment Across Diverse Resolutions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "height,width,expected_gh,expected_gw",
    [
        (400, 640, 14, 22),     # 16:10 canonical
        (800, 1280, 28, 45),    # 16:10 desktop
        (600, 800, 21, 28),     # 4:3 classic
        (540, 960, 19, 34),     # 16:9 widescreen
        (720, 1280, 25, 45),    # 720p widescreen
        (256, 256, 9, 9),       # square small
    ],
)
def test_dynamic_image_grid_thw_alignment_resolutions(height, width, expected_gh, expected_gw):
    """Verifies that dynamic image_grid_thw derives exact patch grid dimensions matching screen aspect ratio."""
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    images = torch.randn(2, 3, height, width)

    # Calling without explicit grid_thw should derive dynamically
    out = model(images)

    assert out["grid_hw"] == (expected_gh, expected_gw)
    expected_k = expected_gh * expected_gw
    assert out["patch_coords"].shape == (expected_k, 2)
    assert out["target_coords"].shape == (2, 2)
    assert out["action_logits"].shape == (2, len(ACTION_VOCAB_10))
    assert torch.all(out["target_coords"] >= 0.0) and torch.all(out["target_coords"] <= 1.0)


# --------------------------------------------------------------------------- #
# 3. 512-dim Widened Projections & Parameter Invariants
# --------------------------------------------------------------------------- #

def test_512_dim_projections_trainable_param_exact():
    """Verifies that 512-dim projections result in exactly 3,978,386 trainable parameters."""
    model = QwenGUIActorModel(
        dry_run=True,
        backbone_hidden_dim=2048,
        pointer_hidden_dim=512,
        query_dim=512,
        use_token_pruning=False,
        use_rolling_kv=False,
    )
    stats = model.get_parameter_summary()

    # Exact trainable parameter invariant
    assert stats["trainable_params"] == 3_978_386
    assert stats["frozen_params"] > 0
    # Against 3.74B parameter Qwen2.5-VL-3B backbone
    trainable_pct_real_backbone = (stats["trainable_params"] / 3.74e9) * 100.0
    assert trainable_pct_real_backbone < 0.15  # <0.15% of entire 3.74B model

    # Check projection layer dimensions
    assert model.patch_proj[0].in_features == 2048
    assert model.patch_proj[0].out_features == 512
    assert model.query_proj[0].in_features == 2048
    assert model.query_proj[0].out_features == 512


# --------------------------------------------------------------------------- #
# 4. Fail-Closed Dataset Ingestion & Zero-Leakage Gate
# --------------------------------------------------------------------------- #

def test_fail_closed_dataset_ingestion_empty_raises(tmp_path: Path):
    """Verifies that empty dataset paths raise RuntimeError when allow_synthetic_fallback=False."""
    with pytest.raises(RuntimeError, match="Fail-closed dataset gate"):
        GroundingDataset(
            dataset_paths=[str(tmp_path / "nonexistent.jsonl")],
            allow_synthetic_fallback=False,
        )


def test_zero_leakage_gate_comprehensive(tmp_path: Path):
    """Verifies zero leakage against all variants of showdown-clicks-dev and held-out benchmarks."""
    sample_jsonl = tmp_path / "leakage_test.jsonl"
    records = [
        # Valid training samples
        {"id": "os_atlas_valid_1", "instruction": "Click File", "conversations": [{"from": "gpt", "value": '{"action": "left_click", "coordinate": [100, 200]}'}]},
        {"id": "showui_desktop_valid_2", "source": "showui_train", "instruction": "Click OK", "conversations": [{"from": "gpt", "value": '{"action": "left_click", "coordinate": [0.5, 0.5]}'}]},
        # Leaked held-out samples that MUST be filtered
        {"id": "showdown-clicks-dev-001", "instruction": "Leaked sample 1"},
        {"id": "valid_id", "image": "showdown_dev/shot1.png", "instruction": "Leaked sample 2"},
        {"id": "valid_id_2", "source": "eval_dev_benchmark", "instruction": "Leaked sample 3"},
        {"id": "valid_id_3", "dataset": "showdown_clicks_eval", "instruction": "Leaked sample 4"},
        {"id": "valid_id_4", "custom_meta": {"benchmark": "showdown-clicks-dev"}, "instruction": "Leaked sample 5"},
    ]
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds = GroundingDataset(dataset_paths=[str(sample_jsonl)], allow_synthetic_fallback=False)
    assert len(ds) == 2
    assert ds.leakage_filtered_count == 5


def test_golden_directory_clean_skip_and_normalization(tmp_path: Path):
    """Verifies missing golden trajectory directories cleanly skip, and ingested coordinates are normalized."""
    # 1. Missing directory cleanly skips
    missing_dir = tmp_path / "missing_golden"
    ds = GroundingDataset(dataset_paths=[str(missing_dir)], allow_synthetic_fallback=True)
    assert len(ds) == 1000  # Fallback triggered cleanly without crash

    # 2. Existing golden directory with mixed pixel and normalized coords + leakage
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    sample_traj = {
        "goal": "Test Golden Trajectory",
        "steps": [
            {"action": {"kind": "left_click", "coordinate": [640, 400]}},  # pixel coords
            {"action": {"kind": "right_click", "coordinate": [0.25, 0.75]}},  # normalized coords
        ],
    }
    with open(golden_dir / "sample_traj.json", "w", encoding="utf-8") as f:
        json.dump(sample_traj, f)

    # Leaked golden trajectory file must be filtered
    leaked_traj = {
        "goal": "Held-out showdown-clicks-dev trajectory",
        "steps": [{"action": {"kind": "left_click", "coordinate": [100, 100]}}],
    }
    with open(golden_dir / "showdown-clicks-dev_test.json", "w", encoding="utf-8") as f:
        json.dump(leaked_traj, f)

    ds_golden = GroundingDataset(dataset_paths=[str(golden_dir)], allow_synthetic_fallback=False)
    assert len(ds_golden) == 2
    assert ds_golden.leakage_filtered_count == 1
    # First step: 640/1280 = 0.5, 400/800 = 0.5
    assert ds_golden.samples[0]["target_xy"] == [0.5, 0.5]
    # Second step: already normalized
    assert ds_golden.samples[1]["target_xy"] == [0.25, 0.75]


def test_openai_and_sharegpt_format_ingestion_and_normalization(tmp_path: Path):
    """Verifies that OpenAI/HF (role/content) and ShareGPT (from/value) formats are ingested with dynamic resolution scaling."""
    sample_jsonl = tmp_path / "mixed_format.jsonl"
    dummy_img = tmp_path / "test_frame.png"
    from PIL import Image
    Image.new("RGB", (128, 128), color=(255, 0, 0)).save(dummy_img)

    records = [
        # OpenAI/HF format with 128x128 resolution specified in user prompt
        {
            "id": "hf_sample_1",
            "image": str(dummy_img),
            "conversations": [
                {
                    "role": "user",
                    "content": "<image>\nInstruction: Click the tab at (16, 64)\nScreen Resolution: 128x128\nCoordinate Space: absolute screen pixels [0 to 128, 0 to 128]\nOutput the physical action:"
                },
                {
                    "role": "assistant",
                    "content": '{"action": "left_click", "coordinate": [16, 64]}'
                }
            ]
        },
        # ShareGPT format with 1280x800 resolution
        {
            "id": "sharegpt_sample_2",
            "image": str(dummy_img),
            "width": 1280,
            "height": 800,
            "conversations": [
                {"from": "human", "value": "Click Submit button"},
                {"from": "gpt", "value": '{"action": "double_click", "coordinate": [640, 400]}'}
            ]
        },
        # Root-level bbox format
        {
            "id": "root_bbox_sample_3",
            "instruction": "Click Search Icon",
            "width": 1000,
            "height": 500,
            "bbox": [100, 200, 200, 300],
        }
    ]
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds = GroundingDataset(dataset_paths=[str(sample_jsonl)], allow_synthetic_fallback=False)
    assert len(ds) == 3

    # Sample 1: 16/128 = 0.125, 64/128 = 0.5
    assert ds.samples[0]["instruction"] == "Click the tab at (16, 64)"
    assert ds.samples[0]["action_idx"] == ACTION_VOCAB_10.index("left_click")
    assert ds.samples[0]["target_xy"] == [0.125, 0.5]

    # Sample 2: 640/1280 = 0.5, 400/800 = 0.5
    assert ds.samples[1]["instruction"] == "Click Submit button"
    assert ds.samples[1]["action_idx"] == ACTION_VOCAB_10.index("double_click")
    assert ds.samples[1]["target_xy"] == [0.5, 0.5]

    # Sample 3: center of [100, 200, 200, 300] is [150, 250] -> 150/1000 = 0.15, 250/500 = 0.5
    assert ds.samples[2]["instruction"] == "Click Search Icon"
    assert ds.samples[2]["target_xy"][0] == pytest.approx(0.15)
    assert ds.samples[2]["target_xy"][1] == pytest.approx(0.5)

    # Verify __getitem__ loads real image pixels rather than zeros
    item = ds[0]
    assert item["image"].shape == (3, 800, 1280)
    assert (item["image"] > 0).sum() > 0


# --------------------------------------------------------------------------- #
# 5. Token Pruning and KV Memory Pause Verification
# --------------------------------------------------------------------------- #

def test_token_pruning_and_kv_memory_strictly_paused():
    """Verifies that token pruning and rolling KV memory are strictly disabled by default."""
    model = QwenGUIActorModel(dry_run=True)
    assert model.use_token_pruning is False
    assert model.use_rolling_kv is False
    assert model.pruner is None
    assert model.memory is None

    # Forward pass should not invoke pruner or memory
    images = torch.randn(1, 3, 400, 640)
    out = model(images)
    assert out["memory_state"] is None


def test_grid_hw_and_image_grid_thw_mutual_derivation():
    """Verifies mutual derivation: grid_hw from image_grid_thw, and image_grid_thw from grid_hw."""
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    images = torch.randn(2, 3, 400, 640)

    # 1. Provide only grid_hw
    out1 = model(images, grid_hw=(14, 22))
    assert out1["grid_hw"] == (14, 22)

    # 2. Provide 2D image_grid_thw (B, 3)
    thw_2d = torch.tensor([[1, 28, 44]], dtype=torch.long)
    out2 = model(images, image_grid_thw=thw_2d)
    assert out2["grid_hw"] == (14, 22)

    # 3. Provide 1D image_grid_thw (3,)
    thw_1d = torch.tensor([1, 28, 44], dtype=torch.long)
    out3 = model(images, image_grid_thw=thw_1d)
    assert out3["grid_hw"] == (14, 22)


def test_ddp_multi_step_backward_with_unused_parameters(tmp_path: Path):
    """Verifies that DDP operates across multiple backward steps with find_unused_parameters=True."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    store_file = tmp_path / "ddp_store"
    store = dist.FileStore(str(store_file), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        ddp_model = DDP(model, find_unused_parameters=True)
        optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=1e-3)
        images = torch.randn(2, 3, 400, 640)
        target_actions = torch.tensor([0, 1], dtype=torch.long)

        # Must run multiple iterations without throwing "reduction in prior iteration" error
        for step in range(3):
            optimizer.zero_grad()
            out = ddp_model(images)
            loss = torch.nn.functional.cross_entropy(out["action_logits"], target_actions)
            loss.backward()
            optimizer.step()
            assert loss.item() > 0.0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_setup_distributed_reentry_when_initialized(tmp_path: Path):
    """Verifies that setup_distributed cleanly reuses an existing initialized process group."""
    import torch.distributed as dist

    store_file = tmp_path / "reentry_store"
    store = dist.FileStore(str(store_file), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        rank, world_size, local_rank, is_dist = setup_distributed()
        assert rank == 0
        assert world_size == 1
        assert is_dist is True
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# 6. Multi-Resolution Overfit Sanity Clean Convergence
# --------------------------------------------------------------------------- #

def test_overfit_sanity_clean_convergence():
    """Verifies that run_overfit_sanity converges cleanly across all target resolutions."""
    test_resolutions = [(400, 640), (600, 800)]
    res = run_overfit_sanity(resolutions=test_resolutions, steps_per_res=25, lr=1e-3)

    assert len(res) == 2
    for r_key in ("400x640", "600x800"):
        assert r_key in res
        assert res[r_key]["converged"] is True
        assert res[r_key]["loss_drop"] > 0.0
        assert res[r_key]["final_loss"] < res[r_key]["initial_loss"]


def test_extract_features_and_precomputed_forwarding():
    """Verifies extract_features works with PIL images and precomputed feats forward correctly."""
    from PIL import Image
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    pil_img = Image.new("RGB", (640, 400), color=(100, 150, 200))

    # Extract features
    feats = model.extract_features(images=[pil_img], queries=["Click the file menu"])
    assert "patch_feats" in feats
    assert "pooled_feats" in feats
    assert "grid_hw" in feats
    gh, gw = feats["grid_hw"]
    assert gh * gw == feats["patch_feats"].shape[1]

    # Forward with precomputed features
    out = model(
        patch_feats=feats["patch_feats"],
        pooled_feats=feats["pooled_feats"],
        grid_hw=(gh, gw),
    )
    assert "action_logits" in out
    assert "target_coords" in out
    assert out["target_coords"].shape == (1, 2)
    assert 0.0 <= out["target_coords"][0, 0].item() <= 1.0
    assert 0.0 <= out["target_coords"][0, 1].item() <= 1.0


def test_overfit_sanity_with_hf_sample_structure():
    """Verifies run_overfit_sanity with mock HF desktop samples achieving >=95% strict accuracy."""
    from PIL import Image
    model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)
    samples = [
        {
            "id": f"test_sample_{i}",
            "image": Image.new("RGB", (640, 400), color=(i * 20, 100, 150)),
            "width": 640,
            "height": 400,
            "source": "showlab/ShowUI-desktop",
            "instruction": f"Click icon {i}",
            "target_action": 0,
            "target_coord": [0.35 + i * 0.05, 0.45],
            "target_bbox": [0.30 + i * 0.05, 0.40, 0.40 + i * 0.05, 0.50],
        }
        for i in range(4)
    ]

    res = run_overfit_sanity(model=model, samples=samples, steps_per_res=35, lr=1e-3)
    assert "hf_preflight" in res
    assert res["hf_preflight"]["converged"] is True
    assert res["hf_preflight"]["strict_accuracy"] >= 0.95
    assert res["hf_preflight"]["hits"] == 4
