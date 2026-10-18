"""Unit and regression tests for AVCS Visual Understanding Pilot & Visual State Observation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch

from avcs.perception.visual_pilot import ELEMENT_KINDS, UI_CATEGORIES, VisualUnderstandingPilot
from avcs.perception.visual_state import (
    SpatialCandidateInfo,
    TemporalChangeInfo,
    VisualRegion,
    VisualStateObservation,
)


def test_visual_pilot_architecture_and_forward_shapes():
    model = VisualUnderstandingPilot(feature_dim=128)
    params = sum(p.numel() for p in model.parameters())
    assert 200_000 < params < 500_000, f"Expected params between 200k and 500k, got {params}"

    x = torch.zeros(2, 3, 256, 256)
    out = model(x)

    assert out["features"].shape == (2, 128, 16, 16)
    assert out["category_logits"].shape == (2, len(UI_CATEGORIES))
    assert out["affordance_logits"].shape == (2, 3, 16, 16)
    assert out["box_pred"].shape == (2, 4, 16, 16)
    assert out["kind_logits"].shape == (2, len(ELEMENT_KINDS), 16, 16)


def test_visual_pilot_predict_structure_returns_valid_regions():
    model = VisualUnderstandingPilot(feature_dim=128)
    model.eval()

    dummy_frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    # Add a mock bright button
    dummy_frame[50:90, 50:150] = 220

    cat, conf, regions = model.predict_visual_structure(dummy_frame)

    assert cat in UI_CATEGORIES
    assert 0.0 <= conf <= 1.0
    assert isinstance(regions, list)
    for r in regions:
        assert isinstance(r, VisualRegion)
        assert r.element_kind in ELEMENT_KINDS
        assert 0.0 <= r.confidence <= 1.0
        assert len(r.bbox_norm) == 4
        assert len(r.bbox_px) == 4
        assert r.bbox_norm[0] <= r.bbox_norm[2]
        assert r.bbox_norm[1] <= r.bbox_norm[3]


def test_visual_state_observation_to_dict_and_privacy_boundary():
    temp_info = TemporalChangeInfo(
        stability_state="stable",
        mean_motion=0.0012,
        optical_flow_energy=0.000001,
        surprise_score=0.02,
        overlay_suspected=False,
        appeared_regions_count=0,
        disappeared_regions_count=0,
    )

    region = VisualRegion(
        region_id="reg_0",
        element_kind="button",
        bbox_norm=(0.1, 0.1, 0.2, 0.3),
        bbox_px=(36.0, 64.0, 72.0, 192.0),
        confidence=0.91,
        is_interactive=True,
        is_focused=False,
    )

    candidate = SpatialCandidateInfo(
        candidate_id=0,
        coord_norm=(0.2, 0.15),
        coord_px=(128.0, 54.0),
        coarse_logit=4.12,
        rerank_score=0.95,
        rerank_prob=0.97,
    )

    obs = VisualStateObservation(
        frame_sequence=1,
        timestamp=1700000000.0,
        screen_dimensions=(640, 360),
        coordinate_convention="normalized_0_1_origin_top_left",
        primary_category="terminal_shell",
        category_confidence=0.985,
        detected_regions=[region],
        target_presence="PRESENT",
        presence_confidence=0.95,
        active_query="Terminal icon",
        target_candidates=[candidate],
        selected_target_norm=(0.2, 0.15),
        selected_target_px=(128.0, 54.0),
        target_margin=0.35,
        is_ambiguous=False,
        action_authorized=False, # Fail-closed invariant
        temporal_change=temp_info,
        uncertainty_flags=[],
    )

    data = obs.to_dict()

    # Serialization validation
    json_str = json.dumps(data)
    loaded = json.loads(json_str)

    assert loaded["frame_sequence"] == 1
    assert loaded["primary_category"] == "terminal_shell"
    assert loaded["action_authorized"] is False
    assert len(loaded["detected_regions"]) == 1
    assert loaded["temporal_change"]["stability_state"] == "stable"

    # Anti-cheating & privacy check: zero raw pixel arrays in dictionary
    for k, v in data.items():
        assert not isinstance(v, (np.ndarray, torch.Tensor))

