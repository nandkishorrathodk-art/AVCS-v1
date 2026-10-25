"""Tests for the opt-in ONNX cortex accelerator.

The accelerator is a performance path, so the properties that matter are:
  1. It is off by default (the eager path is unchanged unless asked).
  2. When enabled, it reproduces the eager outputs (parity), so enabling it
     cannot change what the model reports.
  3. It can be disabled again.
  4. It fails closed if the optional dependency is missing.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from avcs.perception.human_eye import HumanEye
from avcs.training.gui_dataset import DesktopGUIGenerator

_HAS_ONNX = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)

requires_onnx = pytest.mark.skipif(
    not _HAS_ONNX, reason="optional 'onnx'/'onnxruntime' not installed"
)


@pytest.fixture(scope="module")
def frames():
    gen = DesktopGUIGenerator()
    return [gen.generate_gui_screen(seed=31000 + i).frame for i in range(4)]


def test_accelerator_is_off_by_default():
    eye = HumanEye(source="synthetic", load_weights=True)
    assert eye._use_onnx is False
    assert eye._onnx_accel is None


@requires_onnx
def test_accelerator_parity_with_eager(frames):
    """Enabling the accelerator must not change reported perception."""
    eager = HumanEye(source="synthetic", load_weights=True)
    eager.eval()
    accel = HumanEye(source="synthetic", load_weights=True, onnx_accelerator=True)
    accel.eval()
    assert accel._use_onnx is True

    for f in frames:
        a = eager.observe(f)
        b = accel.observe(f)
        assert a.primary_category == b.primary_category
        assert a.category_confidence == pytest.approx(b.category_confidence, abs=1e-4)
        assert a.affordance_hotspots == b.affordance_hotspots


@requires_onnx
def test_accelerator_can_be_disabled(frames):
    eye = HumanEye(source="synthetic", load_weights=True, onnx_accelerator=True)
    eye.eval()
    eye.disable_onnx_accelerator()
    assert eye._use_onnx is False
    state = eye.observe(frames[0])
    assert state.primary_category


@requires_onnx
def test_accelerator_rejects_non_parity_graph(monkeypatch):
    """A graph that does not reproduce eager output must be refused, not used.

    The gate compares the exported graph against the eager modules on a probe
    input. Corrupting the graph's outputs must therefore raise, not silently
    produce a model that disagrees with the eager path.
    """
    from avcs.perception import onnx_accelerator as mod

    eye = HumanEye(source="synthetic", load_weights=True)
    eye.eval()

    original = mod.OnnxCortexAccelerator.run_torch

    def corrupted(self, pixels):
        cat, aff, feat = original(self, pixels)
        return cat + 1.0, aff, feat

    monkeypatch.setattr(mod.OnnxCortexAccelerator, "run_torch", corrupted)
    with pytest.raises(RuntimeError, match="parity"):
        mod.OnnxCortexAccelerator.from_eye(eye, input_size=(128, 128))


@requires_onnx
def test_accelerator_run_shapes_match_eager():
    from avcs.perception.onnx_accelerator import OnnxCortexAccelerator

    eye = HumanEye(source="synthetic", load_weights=True)
    eye.eval()
    accel = OnnxCortexAccelerator.from_eye(eye, input_size=(128, 128))

    pixels = torch.rand(1, 3, 128, 128)
    cat, aff, feat = accel.run_torch(pixels)
    with torch.inference_mode():
        e_cat = eye.ventral(eye.backbone(pixels))["category_logits"]
        e_feat = eye.backbone(pixels)
    assert torch.allclose(cat, e_cat, atol=1e-4)
    assert torch.allclose(feat, e_feat, atol=1e-4)
    assert aff.shape[0] == 1 and aff.shape[1] == 3
