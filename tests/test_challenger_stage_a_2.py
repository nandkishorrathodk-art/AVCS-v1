"""Adversarial Empirical Challenge Suite for Stage A: AVCS System 1 Live-Stream Visual Cortex.

Author: challenger_stage_a_2 (Stage A Mathematical & Temporal Dynamics Challenger)
Execution target: Local CPU, PyTorch 2.14.1+cpu
Scope: Mathematical & temporal stress tests on StreamingVisualCortexV1 & ConvGRUCell in avcs/streaming_v1/model.py

Evaluates:
1. Orthogonality & Spectral Radius Stress Test (Frobenius norm, spectral radius, update gate bias = 1.0).
2. Temporal Stability Stress Test (500 sequential ticks under random, synthetic, and extreme static inputs).
3. Extreme Timestamp Delta Stress Test (0.001s, 10.0s, 3600.0s, jittered delta_t).
4. Clean Random Initialization Invariant (zero identical weights vs legacy checkpoints).
5. Boundary and Negative Input Validation (invalid ranges, NaN/Inf rejection, minimum dimensions).
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from avcs.streaming_v1.model import (
    ConvGRUCell,
    CortexState,
    StreamingVisualCortexV1,
    create_clean_random_cortex,
)
from avcs.streaming_v1.sources import SyntheticCaptureSource


# ==============================================================================
# 1. Orthogonality & Spectral Radius Stress Test
# ==============================================================================

@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 777, 999, 2026, 4096, 99999])
def test_convgru_recurrent_orthogonality_and_spectral_radius_across_seeds(seed: int):
    """Stress-test orthogonality and spectral radius across 10 distinct random seeds."""
    torch.manual_seed(seed)
    hidden_dim = 48
    input_dim = 48
    cell = ConvGRUCell(input_dim, hidden_dim)

    eye = torch.eye(hidden_dim, dtype=torch.float32)

    # 1. Candidate recurrent kernel W_hh: shape (hidden_dim, hidden_dim * 3 * 3) -> (48, 432)
    w_cand = cell.candidate.weight[:, input_dim:].flatten(1)
    prod_cand = torch.matmul(w_cand, w_cand.T)
    diff_cand = prod_cand - eye

    # Empirical measurements
    max_err_cand = diff_cand.abs().max().item()
    frob_cand = torch.linalg.norm(diff_cand, ord="fro").item()
    norm_frob_cand = frob_cand / hidden_dim
    sv_cand = torch.linalg.svdvals(w_cand)
    spectral_radius_error_cand = abs(sv_cand.max().item() - 1.0)

    # Maximum elementwise error must be at float32 machine precision limit (< 2e-6)
    assert max_err_cand < 2e-6, f"Seed {seed}: Cand max error {max_err_cand} exceeds 2e-6"
    # Spectral radius error |sigma_max - 1.0| must be strictly < 1e-6
    assert spectral_radius_error_cand < 1e-6, f"Seed {seed}: Cand spectral radius error {spectral_radius_error_cand} exceeds 1e-6"
    # Normalized Frobenius norm error must be well below 1e-6 (approx 4e-8)
    assert norm_frob_cand < 1e-6, f"Seed {seed}: Cand normalized frob {norm_frob_cand} exceeds 1e-6"
    # Note: Unnormalized Frobenius norm in float32 for 48x48 matrix has sqrt(48)*eps ~ 1.9e-6.
    assert frob_cand < 3.0e-6, f"Seed {seed}: Cand Frobenius norm {frob_cand} exceeds float32 bounds"

    # 2. Gates recurrent kernels (reset and update slices)
    for i in range(2):
        w_gate = cell.gates.weight[i * hidden_dim : (i + 1) * hidden_dim, input_dim:].flatten(1)
        prod_gate = torch.matmul(w_gate, w_gate.T)
        diff_gate = prod_gate - eye

        max_err_gate = diff_gate.abs().max().item()
        frob_gate = torch.linalg.norm(diff_gate, ord="fro").item()
        norm_frob_gate = frob_gate / hidden_dim
        sv_gate = torch.linalg.svdvals(w_gate)
        spectral_radius_error_gate = abs(sv_gate.max().item() - 1.0)

        assert max_err_gate < 2e-6, f"Seed {seed}: Gate {i} max error {max_err_gate} exceeds 2e-6"
        assert spectral_radius_error_gate < 1e-6, f"Seed {seed}: Gate {i} spectral error {spectral_radius_error_gate} exceeds 1e-6"
        assert norm_frob_gate < 1e-6, f"Seed {seed}: Gate {i} normalized frob {norm_frob_gate} exceeds 1e-6"
        assert frob_gate < 3.0e-6, f"Seed {seed}: Gate {i} Frobenius norm {frob_gate} exceeds float32 bounds"

    # 3. Update gate bias invariant: must be exactly 1.0
    update_bias = cell.gates.bias[hidden_dim:]
    assert (update_bias == 1.0).all().item(), f"Seed {seed}: Update gate bias not exactly 1.0"

    # 4. Reset gate and candidate bias invariants: must be exactly 0.0
    reset_bias = cell.gates.bias[:hidden_dim]
    assert (reset_bias == 0.0).all().item(), f"Seed {seed}: Reset gate bias not exactly 0.0"
    assert (cell.candidate.bias == 0.0).all().item(), f"Seed {seed}: Candidate bias not exactly 0.0"


# ==============================================================================
# 2. Temporal Stability Stress Test (500 Sequential Ticks)
# ==============================================================================

def test_temporal_stability_500_ticks_random_noise():
    """Verify strictly finite representations across 500 consecutive random frames."""
    model = create_clean_random_cortex(seed=42, feature_dim=48, hidden_dim=48, latent_dim=64)
    model.eval()

    state = None
    latent_history = []
    hidden_norms = []

    with torch.inference_mode():
        for tick in range(500):
            frame = torch.rand(1, 3, 128, 128)
            z, state, out = model.forward(frame, prev_state=state, delta_t=0.033)

            # Strict finiteness assertions
            assert torch.isfinite(z).all().item(), f"Tick {tick}: NaN or Inf encountered in latent representation z!"
            assert torch.isfinite(state.hidden).all().item(), f"Tick {tick}: NaN or Inf encountered in recurrent state hidden!"
            assert torch.isfinite(state.previous_features).all().item(), f"Tick {tick}: NaN or Inf encountered in previous_features!"

            # Range bounds: hidden state must be strictly bounded within (-1, 1) by GRU tanh/convex update
            assert state.hidden.min().item() > -1.0, f"Tick {tick}: Hidden state underflowed tanh lower bound -1.0"
            assert state.hidden.max().item() < 1.0, f"Tick {tick}: Hidden state overflowed tanh upper bound 1.0"

            latent_history.append(z.squeeze(0))
            hidden_norms.append(state.hidden.norm().item())

    all_z = torch.stack(latent_history)
    # Bounded variance assertions: variance across time must be finite and positive
    var_per_dim = all_z.var(dim=0)
    assert (var_per_dim > 0.0).all().item(), "Latent representation collapsed to zero variance across time"
    assert var_per_dim.mean().item() < 10.0, "Latent variance exploded"
    # Recurrent hidden norm must remain bounded and stable
    assert 10.0 < hidden_norms[-1] < 100.0, f"Hidden norm {hidden_norms[-1]} outside stable operating envelope"


def test_temporal_stability_500_ticks_synthetic_stream():
    """Verify strictly finite representations across 500 frames of realistic synthetic stream."""
    model = create_clean_random_cortex(seed=101, feature_dim=48, hidden_dim=48, latent_dim=64)
    model.eval()

    src = SyntheticCaptureSource(width=128, height=128, total_frames=500)
    src.open()

    state = None
    latent_history = []
    change_scores = []

    try:
        with torch.inference_mode():
            for tick in range(500):
                rgb = src.grab()
                frame = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                z, state, out = model.forward(frame, prev_state=state, delta_t=0.033)

                assert torch.isfinite(z).all().item(), f"Tick {tick}: Non-finite z"
                assert torch.isfinite(state.hidden).all().item(), f"Tick {tick}: Non-finite hidden state"
                assert 0.0 <= out["change_score"].item() <= 1.0, f"Tick {tick}: change_score out of [0, 1]"

                latent_history.append(z.squeeze(0))
                change_scores.append(out["change_score"].item())
    finally:
        src.close()

    all_z = torch.stack(latent_history)
    var_per_dim = all_z.var(dim=0)
    assert (var_per_dim > 0.0).all().item(), "Latent representation collapsed on synthetic stream"
    # Change score on dynamic moving synthetic box must show variation
    assert min(change_scores) < max(change_scores), "Change scores are static across dynamic stream"


def test_temporal_stability_500_ticks_extreme_static_screens():
    """Verify stability on extreme degenerate inputs: 500 ticks of all-black and all-white screens."""
    model = create_clean_random_cortex(seed=777, feature_dim=48, hidden_dim=48, latent_dim=64)
    model.eval()

    with torch.inference_mode():
        # Case A: 500 consecutive black frames (all 0.0)
        state_black = None
        for tick in range(500):
            frame_black = torch.zeros(1, 3, 128, 128)
            z_black, state_black, out_black = model.forward(frame_black, prev_state=state_black, delta_t=0.033)
            assert torch.isfinite(z_black).all().item()
            assert torch.isfinite(state_black.hidden).all().item()

        # Case B: 500 consecutive white frames (all 1.0)
        state_white = None
        for tick in range(500):
            frame_white = torch.ones(1, 3, 128, 128)
            z_white, state_white, out_white = model.forward(frame_white, prev_state=state_white, delta_t=0.033)
            assert torch.isfinite(z_white).all().item()
            assert torch.isfinite(state_white.hidden).all().item()
            assert state_white.hidden.min().item() > -1.0
            assert state_white.hidden.max().item() < 1.0


# ==============================================================================
# 3. Extreme Timestamp Delta Stress Test
# ==============================================================================

@pytest.mark.parametrize("delta_t", [0.001, 10.0, 3600.0])
def test_extreme_timestamp_delta_values(delta_t: float):
    """Stress-test forward() with extreme delta_t values (1ms, 10s, 3600s)."""
    model = create_clean_random_cortex(seed=42, feature_dim=48, hidden_dim=48, latent_dim=64)
    model.eval()

    state = None
    with torch.inference_mode():
        for tick in range(50):
            frame = torch.rand(1, 3, 64, 64)
            z, state, out = model.forward(frame, prev_state=state, delta_t=delta_t)
            assert torch.isfinite(z).all().item(), f"delta_t={delta_t}: Non-finite z at tick {tick}"
            assert torch.isfinite(state.hidden).all().item(), f"delta_t={delta_t}: Non-finite hidden at tick {tick}"
            assert state.hidden.min().item() > -1.0
            assert state.hidden.max().item() < 1.0


def test_jittered_and_interrupted_timestamp_deltas():
    """Stress-test forward() with stochastic, violently fluctuating delta_t values."""
    model = create_clean_random_cortex(seed=2026, feature_dim=48, hidden_dim=48, latent_dim=64)
    model.eval()

    # Sequence of varying delta_t: alternating ultra-fast, typical, interrupted, massive gap
    deltas = [0.001, 0.016, 0.033, 0.1, 2.5, 10.0, 0.001, 3600.0, 0.033, 86400.0] * 10
    state = None
    with torch.inference_mode():
        for tick, dt in enumerate(deltas):
            frame = torch.rand(1, 3, 64, 64)
            z, state, out = model.forward(frame, prev_state=state, delta_t=dt)
            assert torch.isfinite(z).all().item(), f"Jitter tick {tick} (dt={dt}): Non-finite z"
            assert torch.isfinite(state.hidden).all().item(), f"Jitter tick {tick} (dt={dt}): Non-finite hidden"


# ==============================================================================
# 4. Clean Random Initialization Invariant Tests
# ==============================================================================

def test_clean_random_init_invariant_vs_legacy_checkpoints():
    """Verify model parameters are not identical to any legacy weights in runs/weights."""
    repo_root = Path(__file__).resolve().parent.parent
    legacy_candidate = repo_root / "runs" / "weights" / "candidates" / "avcs_system1_v1.pt"
    legacy_human_eye = repo_root / "runs" / "weights" / "human_eye_weights.pt"

    assert legacy_candidate.exists(), f"Legacy checkpoint missing: {legacy_candidate}"
    assert legacy_human_eye.exists(), f"Legacy checkpoint missing: {legacy_human_eye}"

    legacy_1 = torch.load(legacy_candidate, map_location="cpu", weights_only=False)
    legacy_2 = torch.load(legacy_human_eye, map_location="cpu", weights_only=False)

    sd1 = legacy_1.get("model_state_dict", legacy_1.get("state_dict", legacy_1))
    sd2 = legacy_2.get("model_state_dict", legacy_2.get("state_dict", legacy_2))

    # Test clean random model
    model = create_clean_random_cortex(seed=42)
    model_sd = model.state_dict()

    for m_name, m_tensor in model_sd.items():
        if not isinstance(m_tensor, torch.Tensor):
            continue
        # Compare against legacy candidate
        if isinstance(sd1, dict):
            for l_name, l_tensor in sd1.items():
                if isinstance(l_tensor, torch.Tensor) and m_tensor.shape == l_tensor.shape:
                    assert not torch.equal(m_tensor, l_tensor), (
                        f"Taint violation! {m_name} is identical to legacy parameter {l_name} in avcs_system1_v1.pt"
                    )
        # Compare against legacy human_eye_weights
        if isinstance(sd2, dict):
            for l_name, l_tensor in sd2.items():
                if isinstance(l_tensor, torch.Tensor) and m_tensor.shape == l_tensor.shape:
                    assert not torch.equal(m_tensor, l_tensor), (
                        f"Taint violation! {m_name} is identical to legacy parameter {l_name} in human_eye_weights.pt"
                    )


# ==============================================================================
# 5. Boundary & Adversarial Input Rejection Tests
# ==============================================================================

def test_forward_rejects_negative_or_unbounded_rgb():
    """Verify forward() strictly rejects out-of-range pixels ([0, 1] requirement)."""
    model = create_clean_random_cortex(seed=42)
    model.eval()

    # Negative pixel
    frame_neg = torch.full((1, 3, 32, 32), -0.01)
    with pytest.raises(ValueError, match="frame RGB values must lie in"):
        model.forward(frame_neg)

    # Over 1.0 pixel
    frame_over = torch.full((1, 3, 32, 32), 1.01)
    with pytest.raises(ValueError, match="frame RGB values must lie in"):
        model.forward(frame_over)

    # Frame containing NaN
    frame_nan = torch.rand(1, 3, 32, 32)
    frame_nan[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="frame must contain finite floating-point"):
        model.forward(frame_nan)

    # Frame containing Inf
    frame_inf = torch.rand(1, 3, 32, 32)
    frame_inf[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="frame must contain finite floating-point"):
        model.forward(frame_inf)


def test_forward_rejects_negative_delta_t():
    """Verify forward() rejects retrograde timestamp deltas (delta_t < 0)."""
    model = create_clean_random_cortex(seed=42)
    model.eval()
    frame = torch.rand(1, 3, 32, 32)

    with pytest.raises(ValueError, match="timestamp_delta values must be finite and non-negative"):
        model.forward(frame, delta_t=-0.001)

    with pytest.raises(ValueError, match="timestamp_delta values must be finite and non-negative"):
        model.forward(frame, delta_t=float("nan"))
