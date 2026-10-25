"""Unit and Integration Tests for the Consumer 'avc' Package."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import avc
from avc import (
    AFFORDANCE_CLASSES,
    INTERACTIVE_CLASSES,
    STABILITY_STATES,
    UI_CATEGORIES,
    ArtificialVisualCortex,
    CortexConfig,
    CortexState,
    ModelConfig,
    RetinaConfig,
    ScreenEye,
    load_cortex,
    realtime_config,
)


def test_package_exports():
    """Verify top-level public package exports."""
    assert avc.__version__ == "1.0.0"
    assert len(AFFORDANCE_CLASSES) == 5
    assert len(UI_CATEGORIES) == 8
    assert len(STABILITY_STATES) == 4
    assert "clickable" in INTERACTIVE_CLASSES


def test_model_initialization_and_weights():
    """Verify ArtificialVisualCortex instantiates and loads pretrained weights."""
    cortex = load_cortex(device="cpu")
    assert isinstance(cortex, ArtificialVisualCortex)

    total_params = sum(p.numel() for p in cortex.parameters())
    # 1.14M core foundation + learned heads = 1.68M total params
    assert total_params == 1678780


def test_exact_neural_shapes():
    """Verify all tensor shapes strictly match the architectural specification."""
    cortex = load_cortex(device="cpu")

    # Peripheral 256x256, Foveal 256x256
    p = torch.randn(1, 3, 256, 256)
    f = torch.randn(1, 3, 256, 256)
    text_emb = torch.randn(1, 256)

    out = cortex(p, f, text_embedding=text_emb)

    # 1. Cortical Backbone Fusion (1, 128, 16, 16)
    assert out["fused_feat"].shape == (1, 128, 16, 16)

    # 2. Dorsal Stream Affordances (1, 5, 32, 32)
    assert out["affordance_logits"].shape == (1, 5, 32, 32)

    # 3. Ventral Stream Semantics: latent (1, 256), cat (1, 8), text_density (1, 1)
    assert out["scene_latent"].shape == (1, 256)
    assert out["category_logits"].shape == (1, 8)
    assert out["text_density"].shape == (1, 1)

    # 4. Language Modulation: rel_map (1, 1, 16, 16), coords (1, 2)
    assert out["relevance_map"].shape == (1, 1, 16, 16)
    assert out["query_coords"].shape == (1, 2)

    # 5. Predictive Coding: mem (1, 256), surprise (1, 1), stability (1, 4)
    assert out["temporal_memory"].shape == (1, 256)
    assert out["surprise_score"].shape == (1, 1)
    assert out["stability_logits"].shape == (1, 4)


def test_cortex_perceive_interface():
    """Verify perception from full 1080p screen frame array."""
    cortex = load_cortex(device="cpu")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    state = cortex.perceive(
        frame=frame,
        cursor_position=(0.5, 0.5),
        device="cpu",
    )

    assert isinstance(state, CortexState)
    assert state.primary_category in UI_CATEGORIES
    assert 0.0 <= state.category_confidence <= 1.0
    assert state.stability_state in STABILITY_STATES
    assert 0.0 <= state.surprise_score <= 1.0
    assert state.affordance_map.shape == (32, 32)
    assert state.visual_latent.shape == (256,)
    assert isinstance(state.motor_flow, tuple)
    assert len(state.motor_flow) == 2
    assert state.inference_ms > 0.0

    state_dict = state.to_dict()
    assert "primary_category" in state_dict
    assert "stability_state" in state_dict
    assert "affordance_map" not in state_dict  # excluded from lightweight summary dict


def test_realtime_config_preset():
    """Verify realtime_config preset switches to 128x128 retina."""
    cfg = realtime_config()
    assert cfg.model.retina.peripheral_size == (128, 128)
    assert cfg.model.retina.foveal_size == (128, 128)

    cortex = load_cortex(config=cfg, device="cpu")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = cortex.perceive(frame=frame)
    assert state.primary_category in UI_CATEGORIES


def test_screen_eye_interface():
    """Verify ScreenEye developer SDK interface."""
    with ScreenEye(load_pretrained=True) as eye:
        assert eye.weights_loaded is True
        # Synthetic frame perceive
        synth_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        state = eye.perceive(synth_frame)
        assert state.primary_category in UI_CATEGORIES
