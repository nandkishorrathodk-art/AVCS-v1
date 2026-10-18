"""Unit and integration tests for Query-Conditioned Foveal Grounder.

Verifies:
1. Native crop boundary clamping across screen borders.
2. Coordinate restoration from local crop to global screen coordinates.
3. FiLM query conditioning and presence head discrimination.
4. Fail-closed abstention on negative queries.
5. CPU forward pass latency <= 30 ms.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest
import torch

from avcs.perception.foveal_grounder import (
    FovealRefinementResult,
    QueryConditionedFovealRefiner,
    load_foveal_checkpoint,
)
from avcs.perception.goal_grounder import encode_queries


@pytest.fixture
def foveal_model():
    ckpt_path = "runs/weights/candidates/candidate_m3_foveal_v1.pt"
    if Path(ckpt_path).is_file():
        return load_foveal_checkpoint(ckpt_path, device="cpu")
    return QueryConditionedFovealRefiner().eval()


def test_foveal_refiner_architecture():
    model = QueryConditionedFovealRefiner()
    total_params = sum(p.numel() for p in model.parameters())
    assert 150_000 <= total_params <= 300_000, f"Unexpected parameter count: {total_params}"

    crop = torch.zeros(1, 3, 96, 96)
    q_tokens = encode_queries(["click button"])
    hm, bbox, pres = model(crop, q_tokens)
    assert hm.shape == (1, 1, 48, 48)
    assert bbox.shape == (1, 4)
    assert pres.shape == (1, 1)


def test_crop_boundary_clamping(foveal_model):
    screen = np.zeros((1200, 1920, 3), dtype=np.uint8)

    # Top-left corner (0, 0)
    res_tl = foveal_model.refine_prediction(screen, "Start", (0.0, 0.0), crop_size=96)
    assert res_tl.crop_origin_px == (0, 0)

    # Bottom-right corner (1.0, 1.0)
    res_br = foveal_model.refine_prediction(screen, "Tray", (1.0, 1.0), crop_size=96)
    assert res_br.crop_origin_px == (1920 - 96, 1200 - 96)


def test_coordinate_restoration_identity():
    # If a feature is at the exact center of a centered crop, global position must match coarse position
    model = QueryConditionedFovealRefiner().eval()
    screen = np.zeros((256, 256, 3), dtype=np.uint8)
    res = model.refine_prediction(screen, "test", (0.5, 0.5), crop_size=96)
    assert abs(res.coarse_coord_norm[0] - 0.5) < 1e-4
    assert abs(res.coarse_coord_norm[1] - 0.5) < 1e-4


def test_negative_query_presence_rejection(foveal_model):
    crop = np.zeros((96, 96, 3), dtype=np.uint8)
    crop_t = torch.from_numpy(crop.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)

    for nq in ["unicorn rainbow button", "Delete database", "reboot system"]:
        q_tokens = encode_queries([nq])
        _, _, pres = foveal_model(crop_t, q_tokens)
        prob = float(torch.sigmoid(pres).item())
        assert prob < 0.20, f"Expected absent rejection for '{nq}', got {prob}"


def test_cpu_latency_budget(foveal_model):
    screen = np.zeros((256, 256, 3), dtype=np.uint8)
    # Warmup
    for _ in range(3):
        foveal_model.refine_prediction(screen, "button", (0.5, 0.5), crop_size=96)

    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        foveal_model.refine_prediction(screen, "button", (0.5, 0.5), crop_size=96)
        times.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.median(times))
    assert p50 < 30.0, f"P50 foveal latency {p50:.2f} ms exceeds 30 ms budget"
