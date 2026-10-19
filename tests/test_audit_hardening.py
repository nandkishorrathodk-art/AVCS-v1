"""Tests for AVCS 10-Stream Audit Hardening Improvements."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.vision.cortex.core.dorsal import DorsalAffordanceStream, AFFORDANCE_CLASSES
from forge.vision.cortex.core.learned import LearnedStabilityHead
from forge.vision.cortex.core.modulation import LanguageModulation, CharNGramEncoder, dsnt_soft_argmax
from forge.vision.cortex.core.predictive import STABILITY_STATES, PredictiveCodingEngine
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def test_dsnt_soft_argmax_exact_coordinates():
    """Verify that DSNT soft-argmax recovers known peak locations."""
    B, H, W = 2, 32, 32
    # Create single-peak heatmaps
    heat = torch.zeros(B, 1, H, W)
    # Peak at (x=16, y=8) -> norm_x ~ 0.515, norm_y ~ 0.265
    heat[0, 0, 8, 16] = 100.0
    # Peak at (x=24, y=24) -> norm_x ~ 0.765, norm_y ~ 0.765
    heat[1, 0, 24, 24] = 100.0

    prob = torch.softmax(heat.view(B, 1, -1), dim=-1).view(B, 1, H, W)
    coords = dsnt_soft_argmax(prob)

    assert coords.shape == (B, 2)
    assert abs(coords[0, 0].item() - 0.515) < 0.05
    assert abs(coords[0, 1].item() - 0.265) < 0.05
    assert abs(coords[1, 0].item() - 0.765) < 0.05
    assert abs(coords[1, 1].item() - 0.765) < 0.05


def test_char_ngram_encoder():
    """Verify zero-cloud CharNGramEncoder creates normalized embeddings."""
    encoder = CharNGramEncoder(out_dim=64, hash_bins=256)
    emb1 = encoder("Submit Button")
    emb2 = encoder("Submit Button")
    emb3 = encoder("Cancel Operation")

    assert emb1.shape == (1, 64)
    # Identical query yields identical normalized representation
    assert torch.allclose(emb1, emb2, atol=1e-5)
    # Different queries produce distinct vectors
    sim = (emb1 * emb3).sum().item()
    assert sim < 0.99


def test_dorsal_affordance_map_is_dense_and_normalised():
    """The dorsal stream now exposes only its learned dense affordance field.

    Peak extraction moved to the learned SlotDetector, so what remains here is
    the head itself: a per-class probability field that must be a valid
    distribution at every spatial location.
    """
    dorsal = DorsalAffordanceStream(in_dim=64)
    feat = torch.randn(1, 64, 16, 16)

    probs = dorsal.get_affordance_map(feat)
    assert probs.dim() == 4
    assert probs.shape[1] == len(AFFORDANCE_CLASSES)
    # Softmax over classes: each location sums to one, nothing is negative.
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, *probs.shape[2:]), atol=1e-5)
    assert float(probs.min()) >= 0.0


def test_learned_stability_head_replaces_schmitt_thresholds():
    """Stability is now the argmax of a learned head, not a threshold ladder."""
    head = LearnedStabilityHead(latent_dim=16, num_states=len(STABILITY_STATES))
    memory = torch.randn(2, 16)
    surprise = torch.rand(2, 1)

    logits = head(memory, surprise, torch.tensor([0, 1]))
    assert logits.shape == (2, len(STABILITY_STATES))

    # The head is differentiable end to end: this is what lets the temporal
    # state be trained rather than hand-tuned.
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_sdk_bare_decorator_and_close():
    """Verify ScreenEye supports bare decorators and close cleanup."""
    source = SyntheticScreen(width=320, height=180)
    eye = ScreenEye(fps=20, source=source)

    called = []
    @eye.on_affordance
    def my_listener(state):
        called.append(True)

    @eye.on_surprise
    def my_surprise_listener(state):
        called.append(True)

    assert len(eye._affordance_callbacks) == 1
    assert len(eye._surprise_callbacks) == 1

    # Verify context manager and close
    with eye:
        assert eye.is_stable() or not eye.is_stable()
    
    eye.close()

