"""Unit tests for AVCS System 1 standalone demo CLI and packaging entry points."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from avcs import AVCSSystem1V1, VisualStateObservation, load_system1
from avcs.demo_system1 import format_observation_text


def test_load_system1_factory(tmp_path: Path):
    checkpoint = tmp_path / "system1.pt"
    AVCSSystem1V1().save_checkpoint(str(checkpoint))
    model = load_system1(checkpoint_path=str(checkpoint), device="cpu")
    assert isinstance(model, AVCSSystem1V1)
    assert model.total_parameters == 1_663_564


def test_format_observation_text():
    model = AVCSSystem1V1().eval()
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    obs = model.observe_frame(img, active_query="Save button")
    txt = format_observation_text(obs, latency_ms=45.2)

    assert "AVCS SYSTEM 1 V1" in txt
    assert "Active Query Grounding" in txt
    assert "Inference Latency" in txt
    assert "Interactive Elements Detected" in txt


def test_demo_system1_cli_execution(tmp_path: Path):
    checkpoint = tmp_path / "system1.pt"
    image_path = tmp_path / "input.png"
    AVCSSystem1V1().save_checkpoint(str(checkpoint))
    Image.fromarray(np.zeros((48, 80, 3), dtype=np.uint8)).save(image_path)
    cmd = [
        sys.executable,
        "-m",
        "avcs.demo_system1",
        "--image",
        str(image_path),
        "--query",
        "Save button",
        "--checkpoint",
        str(checkpoint),
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["screen_dimensions"] == [80, 48]
    assert data["coordinate_convention"] == "normalized_0_1_origin_top_left"
    assert data["active_query"] == "Save button"
    assert data["target_presence"] in {"PRESENT", "ABSENT", "UNCERTAIN"}
    assert isinstance(data["action_authorized"], bool)
