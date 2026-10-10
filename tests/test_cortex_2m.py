"""Tests for Scaled 2.0M Parameter Cortical Visual Foundation Model (Cortex-2M)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from avcs.cortex_2m import (
    AffordanceHead2M,
    CorticalBackbone2M,
    HumanEye2M,
    LanguageGrounder2M,
    UnifiedCortex2M,
    UNIVERSAL_AFFORDANCE_CLASSES,
    UNIVERSAL_APP_CATEGORIES,
    VentralStream2M,
)
from forge.vision.capture import SyntheticScreen


def test_universal_ontology_counts():
    """Validates that universal desktop ontology covers 32 apps and 24 affordance geometries."""
    assert len(UNIVERSAL_APP_CATEGORIES) == 32
    assert len(UNIVERSAL_AFFORDANCE_CLASSES) == 24
    assert "editor_code" in UNIVERSAL_APP_CATEGORIES
    assert "cad_3d_modeling" in UNIVERSAL_APP_CATEGORIES
    assert "data_spreadsheet" in UNIVERSAL_APP_CATEGORIES
    assert "terminal_shell" in UNIVERSAL_APP_CATEGORIES
    assert "button_primary" in UNIVERSAL_AFFORDANCE_CLASSES
    assert "code_editor_line" in UNIVERSAL_AFFORDANCE_CLASSES
    assert "spreadsheet_cell" in UNIVERSAL_AFFORDANCE_CLASSES


def test_cortex_2m_parameter_scale():
    """Asserts that UnifiedCortex2M scales to ~2.40M parameters."""
    cortex = UnifiedCortex2M(feature_dim=256)
    total_params = cortex.total_parameters

    # Verify scaling between 2.2M and 2.6M
    assert 2_200_000 <= total_params <= 2_600_000, f"Expected ~2.4M params, got {total_params:,}"

    backbone_p = sum(p.numel() for p in cortex.backbone.parameters())
    affordance_p = sum(p.numel() for p in cortex.affordance.parameters())
    ventral_p = sum(p.numel() for p in cortex.ventral.parameters())
    grounder_p = sum(p.numel() for p in cortex.grounder.parameters())

    assert backbone_p > 700_000
    assert affordance_p > 1_000_000
    assert ventral_p > 200_000
    assert grounder_p > 100_000


def test_cortex_2m_forward_pass():
    """Tests forward pass tensor dimensions."""
    cortex = UnifiedCortex2M(feature_dim=256)
    cortex.eval()

    x = torch.randn(2, 3, 128, 128)
    text_emb = torch.randn(2, 128)

    with torch.no_grad():
        out = cortex(x, text_emb=text_emb)

    assert "feat" in out
    assert out["feat"].shape == (2, 256, 8, 8)

    assert "affordance_logits" in out
    assert out["affordance_logits"].shape == (2, 24, 8, 8)

    assert "visual_latent" in out
    assert out["visual_latent"].shape == (2, 256)

    assert "category_logits" in out
    assert out["category_logits"].shape == (2, 32)

    assert "grounding_map" in out
    assert out["grounding_map"].shape == (2, 1, 8, 8)


def test_cortex_2m_affordance_hotspots():
    """Tests hotspot extraction on synthetic feature map."""
    cortex = UnifiedCortex2M(feature_dim=256)
    cortex.eval()

    feat = torch.randn(1, 256, 8, 8)
    hotspots = cortex.affordance.find_hotspots(feat, top_k=3, threshold=0.0)

    assert len(hotspots) <= 3
    for x, y, aff_name, conf in hotspots:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert aff_name in UNIVERSAL_AFFORDANCE_CLASSES
        assert 0.0 <= conf <= 1.0


def test_human_eye_2m_perception():
    """Tests continuous real-time perception loop of HumanEye2M."""
    class StaticSource:
        def __init__(self):
            self.frame = np.full((480, 640, 3), 60, dtype=np.uint8)
            # Add static shapes
            self.frame[100:200, 150:350] = 200
        def grab(self):
            return self.frame

    source = StaticSource()
    eye = HumanEye2M(source=source)

    # Frame 1: Dynamic initial pass (runs full 2.4M forward)
    res1 = eye.perceive()
    assert res1["frame_index"] == 1
    assert not res1["is_static"]
    assert res1["primary_category"] in UNIVERSAL_APP_CATEGORIES
    assert res1["parameters"] > 2_200_000
    assert len(res1["affordances"]) > 0

    # Frame 2: Trigger static screen early exit (<0.2ms)
    res2 = eye.perceive()
    assert res2["frame_index"] == 2
    assert res2["is_static"]
    assert res2["latency_ms"] < 10.0  # Ultra-fast early bypass

    eye.close()
