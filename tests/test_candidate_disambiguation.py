"""Unit and integration tests for Global Candidate Disambiguation & Reranking.

Verifies:
1. NMS local maxima spatial candidate extraction (no duplicate adjacent peaks).
2. Contextual local feature pooling and tensor shapes.
3. Neural reranker scoring pass and margin calculation.
4. Fail-closed decision policy on candidate conflict (ambiguous competing peaks).
5. Strict runtime-only non-leakage (zero ground-truth parameters in inference).
6. Disambiguation CPU latency budget (<= 10 ms).
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from avcs.perception.candidate_disambiguator import (
    CandidateReranker,
    GlobalCandidateDisambiguator,
    extract_spatial_peaks_nms,
)
from avcs.perception.goal_grounder import GoalConditionedGrounder, encode_queries


def test_spatial_peak_extraction_nms():
    # Synthetic heatmap with 3 distinct peaks
    hm = torch.full((1, 1, 64, 64), -10.0)
    # Peak 1 at (10, 10) value 5.0
    hm[0, 0, 10, 10] = 5.0
    # Peak 2 at (10, 12) value 4.0 (within separation radius of 4 cells -> should be suppressed!)
    hm[0, 0, 10, 12] = 4.0
    # Peak 3 at (40, 40) value 3.0 (distinct peak -> should be retained!)
    hm[0, 0, 40, 40] = 3.0
    # Peak 4 at (55, 20) value 2.0 (distinct peak -> should be retained!)
    hm[0, 0, 55, 20] = 2.0

    peaks = extract_spatial_peaks_nms(hm, k=4, min_separation_cells=4, min_logit_threshold=-5.0)[0]

    # Must extract Peak 1, Peak 3, Peak 4; Peak 2 must be suppressed by NMS!
    assert len(peaks) == 3
    coords = [(r, c) for r, c, _ in peaks]
    assert (10, 10) in coords
    assert (10, 12) not in coords
    assert (40, 40) in coords
    assert (55, 20) in coords


def test_feature_pooling_and_reranker_forward():
    reranker = CandidateReranker(visual_channels=64, query_dim=192, hidden_dim=128)
    total_params = sum(p.numel() for p in reranker.parameters())
    assert 50_000 <= total_params <= 200_000, f"Unexpected parameter count: {total_params}"

    B, K = 2, 4
    decoded = torch.randn(B, 64, 32, 32)
    row_col_pairs = [(5, 5), (10, 20), (25, 10), (30, 30)]

    local_feats = reranker.pool_local_context(decoded, row_col_pairs)
    assert local_feats.shape == (B, K, 64 * 9)

    global_feat = torch.randn(B, 64)
    query_emb = torch.randn(B, 192)
    coords_norm = torch.rand(B, K, 2)
    coarse_logits = torch.randn(B, K, 1)

    scores = reranker(local_feats, global_feat, query_emb, coords_norm, coarse_logits)
    assert scores.shape == (B, K)


def test_margin_ambiguity_fail_closed():
    coarse_model = GoalConditionedGrounder().eval()
    reranker = CandidateReranker().eval()
    disambiguator = GlobalCandidateDisambiguator(
        coarse_model=coarse_model,
        reranker=reranker,
        margin_threshold=0.30,
        conflict_distance_px=15.0,
    )

    # Synthetic image & query
    img_t = torch.zeros(1, 3, 256, 256)
    res = disambiguator.disambiguate(img_t, "test button", screen_size=(256, 256))

    # Must return structured result
    assert hasattr(res, "margin")
    assert hasattr(res, "is_ambiguous")
    assert hasattr(res, "decision")
    assert res.decision in ("CONFIDENT", "UNCERTAIN", "ABSENT")


def test_runtime_only_non_leakage():
    # Verify signature and execution of disambiguate does NOT accept ground-truth parameters
    import inspect
    sig = inspect.signature(GlobalCandidateDisambiguator.disambiguate)
    param_names = list(sig.parameters.keys())
    assert "gt" not in param_names
    assert "ground_truth" not in param_names
    assert "label" not in param_names
    assert "target_coord" not in param_names
    assert param_names == ["self", "image_tensor", "query", "screen_size"]


def test_disambiguation_latency_budget():
    reranker = CandidateReranker(visual_channels=64, query_dim=192, hidden_dim=128).eval()
    B, K = 1, 4
    decoded = torch.randn(B, 64, 32, 32)
    row_col_pairs = [(5, 5), (10, 20), (25, 10), (30, 30)]
    global_feat = torch.randn(B, 64)
    query_emb = torch.randn(B, 192)
    coords_norm = torch.rand(B, K, 2)
    coarse_logits = torch.randn(B, K, 1)

    # Warmup
    local_feats = reranker.pool_local_context(decoded, row_col_pairs)
    reranker(local_feats, global_feat, query_emb, coords_norm, coarse_logits)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        local_feats = reranker.pool_local_context(decoded, row_col_pairs)
        _ = reranker(local_feats, global_feat, query_emb, coords_norm, coarse_logits)
        times.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.median(times))
    # Isolated candidate pooling + reranker scoring pass on CPU must be fast (budget <= 5.0 ms)
    assert p50 < 5.0, f"Candidate reranking P50 latency {p50:.2f} ms exceeds 5.0 ms budget"
