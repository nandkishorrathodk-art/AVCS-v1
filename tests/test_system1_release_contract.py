"""Regression tests for the standalone System 1 release API contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from avcs.perception.system1_v1 import AVCSSystem1V1


def test_inference_json_and_screen_pixel_coordinate_contract():
    model = AVCSSystem1V1().eval()
    frame = np.zeros((48, 80, 3), dtype=np.uint8)
    observation = model.observe_frame(frame, active_query="Save button")
    payload = observation.to_dict()

    assert payload["screen_dimensions"] == [80, 48]
    assert payload["coordinate_convention"] == "normalized_0_1_origin_top_left"
    assert payload["active_query"] == "Save button"
    assert payload["target_presence"] in {"PRESENT", "ABSENT", "UNCERTAIN"}
    assert 0.0 <= payload["presence_confidence"] <= 1.0
    if payload["selected_target_norm"] is not None:
        nx, ny = payload["selected_target_norm"]
        px, py = payload["selected_target_px"]
        assert 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        assert px == round(nx * 80, 1)
        assert py == round(ny * 48, 1)
    json.dumps(payload, allow_nan=False)


def test_inference_accepts_grayscale_and_rgba_inputs():
    model = AVCSSystem1V1().eval()
    gray = np.zeros((32, 40), dtype=np.uint8)
    rgba = np.zeros((32, 40, 4), dtype=np.uint8)

    assert model.observe_frame(gray).screen_dimensions == (40, 32)
    assert model.observe_frame(rgba).screen_dimensions == (40, 32)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10, 2), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.full((10, 10, 3), 256, dtype=np.uint16),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_inference_rejects_unsupported_or_invalid_numpy_frames(frame):
    model = AVCSSystem1V1().eval()
    with pytest.raises((TypeError, ValueError)):
        model.observe_frame(frame)


def test_loader_rejects_checkpoint_digest_mismatch():
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "bad.pt"
        checkpoint.write_bytes(b"not a checkpoint")
        checkpoint.with_suffix(".json").write_text(
            json.dumps(
                {
                    "architecture": "AVCSSystem1V1",
                    "parameter_count": 1_663_564,
                    "sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            AVCSSystem1V1.load_from_checkpoint(str(checkpoint))


def test_loader_preserves_compatible_non_default_architecture_checkpoint():
    model = AVCSSystem1V1(pyramid_dim=24, grounder_dim=96)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "custom.pt"
        model.save_checkpoint(str(checkpoint))
        loaded = AVCSSystem1V1.load_from_checkpoint(str(checkpoint))

    assert loaded.pyramid_dim == 24
    assert loaded.grounder_dim == 96
    assert loaded.total_parameters == model.total_parameters
