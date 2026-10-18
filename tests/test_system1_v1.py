"""Unit tests for AVCSSystem1V1 neural perception architecture.

Verifies:
1. Architecture instantiation and total parameter count (~1.66M).
2. Multi-scale proposal head output shapes (P2, P3, P4, category logits).
3. Query-conditioned grounding forward pass (128x128 heatmap, presence logits).
4. Pixel word recognizer forward pass (132-class vocabulary).
5. Magnocellular temporal motion engine and state transitions.
6. Unified observe_frame API and fail-closed action authorization.
7. Pure-pixel AST audit (zero OCR / UIA / DOM libraries).
8. Checkpoint save/load integrity and SHA-256 provenance.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

from avcs.perception.system1_v1 import (
    AVCSSystem1V1,
    SYSTEM1_ELEMENT_KINDS,
    KIND_TO_IDX,
    SYSTEM1_V1_SCHEMA_VERSION,
)
from avcs.perception.visual_state import (
    VisualStateObservation,
    VisualRegion,
    SpatialCandidateInfo,
    TemporalChangeInfo,
)
from avcs.perception.goal_grounder import encode_queries
from avcs.perception.visual_pilot import UI_CATEGORIES
from avcs.perception.word_recognizer import UI_WORD_VOCABULARY


class TestAVCSSystem1V1Architecture:
    """Verifies neural model initialization and forward pass contracts."""

    @pytest.fixture
    def model(self) -> AVCSSystem1V1:
        return AVCSSystem1V1()

    def test_parameter_count(self, model: AVCSSystem1V1):
        total_params = model.total_parameters
        assert 1_500_000 <= total_params <= 1_800_000, (
            f"Expected parameter count ~1.66M, got {total_params:,}"
        )

    def test_proposals_forward_shapes(self, model: AVCSSystem1V1):
        x = torch.randn(2, 3, 256, 256)
        out = model.forward_proposals(x)

        assert "cat_logits" in out
        assert out["cat_logits"].shape == (2, len(UI_CATEGORIES))

        assert "p2" in out
        assert out["p2"].shape == (2, 10, 64, 64)

        assert "p3" in out
        assert out["p3"].shape == (2, 10, 32, 32)

        assert "p4" in out
        assert out["p4"].shape == (2, 10, 16, 16)

    def test_grounding_forward_shapes(self, model: AVCSSystem1V1):
        x = torch.randn(2, 3, 256, 256)
        tokens = encode_queries(["Save button", "File menu"])
        heatmaps, presence = model.forward_grounding(x, tokens)

        assert heatmaps.shape == (2, 1, 128, 128)
        assert presence.shape == (2,)

    def test_word_recognition_forward_shapes(self, model: AVCSSystem1V1):
        crops = torch.randn(4, 3, 32, 96)
        logits = model.forward_word_recognition(crops)
        assert logits.shape == (4, len(UI_WORD_VOCABULARY))


class TestMagnocellularMotionEngine:
    """Verifies temporal perception and optical flow state tracking."""

    def test_optical_flow_and_stability(self):
        model = AVCSSystem1V1()
        engine = model.temporal_engine

        # Frame 1: uniform grey
        f1 = np.full((100, 100, 3), 128, dtype=np.uint8)
        engine.process(f1)
        assert engine.stability_state == "stable"
        assert engine.optical_flow_energy == 0.0

        # Frame 2: identical frame -> stable
        f2 = np.full((100, 100, 3), 128, dtype=np.uint8)
        engine.process(f2)
        assert engine.stability_state == "stable"
        assert engine.optical_flow_energy < 0.01

        # Frame 3: massive change (modal pop-up / inversion)
        f3 = np.full((100, 100, 3), 0, dtype=np.uint8)
        engine.process(f3)
        assert engine.mean_motion > 0.4
        assert engine.surprise_score > 0.3
        assert engine.stability_state in ("changed", "overlay", "surprise")


class TestObserveFrameAPI:
    """Verifies the unified observation contract and fail-closed safety."""

    @pytest.fixture
    def model(self) -> AVCSSystem1V1:
        return AVCSSystem1V1()

    def test_observe_without_query(self, model: AVCSSystem1V1):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        obs = model.observe_frame(frame, active_query=None)

        assert isinstance(obs, VisualStateObservation)
        assert obs.screen_dimensions == (1280, 720)
        assert obs.primary_category in UI_CATEGORIES
        assert obs.selected_target_px is None
        assert obs.target_presence == "ABSENT"
        assert obs.action_authorized is False  # Fail-closed invariant

    def test_observe_with_query_fail_closed(self, model: AVCSSystem1V1):
        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        obs = model.observe_frame(frame, active_query="Non-existent unicorn element")

        assert isinstance(obs, VisualStateObservation)
        assert obs.screen_dimensions == (800, 600)
        assert obs.active_query == "Non-existent unicorn element"

        # Action authorization must fail-closed if presence < 0.50 or uncertainty flags present
        if obs.presence_confidence < 0.50:
            assert obs.action_authorized is False
            assert obs.target_presence in ("ABSENT", "UNCERTAIN")
        else:
            if obs.is_ambiguous or len(obs.uncertainty_flags) > 0:
                assert obs.action_authorized is False

    def test_coordinate_scaling(self, model: AVCSSystem1V1):
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        obs = model.observe_frame(frame, active_query="Settings")

        if obs.selected_target_px is not None:
            tx, ty = obs.selected_target_px
            assert 0 <= tx <= 1920, f"Grounded X coordinate out of bounds: {tx}"
            assert 0 <= ty <= 1080, f"Grounded Y coordinate out of bounds: {ty}"

        for reg in obs.detected_regions:
            assert 0.0 <= reg.bbox_norm[0] <= reg.bbox_norm[2] <= 1.0, f"Region norm bbox invalid: {reg.bbox_norm}"
            assert 0.0 <= reg.bbox_norm[1] <= reg.bbox_norm[3] <= 1.0, f"Region norm bbox invalid: {reg.bbox_norm}"


class TestPurePixelIntegrity:
    """Verifies that no forbidden OCR or accessibility libraries are imported."""

    FORBIDDEN_MODULES = {
        "pytesseract",
        "easyocr",
        "paddleocr",
        "winocr",
        "uiautomation",
        "pywinauto",
        "playwright",
        "selenium",
    }

    def _check_file_ast(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in self.FORBIDDEN_MODULES, (
                        f"Forbidden library {root_pkg} imported in {path.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    assert root_pkg not in self.FORBIDDEN_MODULES, (
                        f"Forbidden library {root_pkg} imported in {path.name}"
                    )

    def test_perception_pure_pixel(self):
        repo_root = Path(__file__).resolve().parents[1]
        self._check_file_ast(repo_root / "avcs" / "perception" / "system1_v1.py")
        trainer = repo_root / "avcs" / "training" / "train_system1_v1.py"
        if trainer.is_file():
            self._check_file_ast(trainer)
        benchmark = repo_root / "benchmarks" / "benchmark_system1_v1.py"
        if benchmark.is_file():
            self._check_file_ast(benchmark)


class TestCheckpointSerialization:
    """Verifies checkpoint saving, hash calculation, and reloading."""

    def test_save_relative_checkpoint_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = AVCSSystem1V1()
        sha256 = model.save_checkpoint("relative.pt")

        assert len(sha256) == 64
        assert Path("relative.pt").is_file()
        assert Path("relative.json").is_file()
        assert AVCSSystem1V1.load_from_checkpoint("relative.pt").total_parameters == (
            model.total_parameters
        )

    def test_save_and_load_roundtrip(self):
        model = AVCSSystem1V1()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_sys1.pt"
            sha256 = model.save_checkpoint(
                str(ckpt_path), training_metadata={"torch": torch.__version__}
            )

            assert ckpt_path.exists()
            assert (Path(tmpdir) / "test_sys1.json").exists()
            assert len(sha256) == 64

            # Reload into fresh instance
            fresh_model = AVCSSystem1V1.load_from_checkpoint(str(ckpt_path))

            assert fresh_model._checkpoint_metadata["schema_version"] == SYSTEM1_V1_SCHEMA_VERSION
            assert fresh_model._checkpoint_metadata["architecture"] == "AVCSSystem1V1"
            assert fresh_model.total_parameters == model.total_parameters

            # Test forward pass with loaded model
            x = torch.randn(1, 3, 256, 256)
            tokens = encode_queries(["Test query"])
            heat, pres = fresh_model.forward_grounding(x, tokens)
            assert heat.shape == (1, 1, 128, 128)
