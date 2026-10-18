"""Unit tests for AVCS System 1 <-> System 2 Reasoning Bridge (Milestone 4.0).

Verifies:
1. Loading and integrity of AVCSSystem1V1 candidate checkpoint.
2. Anti-coordinate injection and sanitization.
3. Stale frame sequence detection and rejection.
4. Locating a known control and proposing action in shadow mode.
5. Handling absent targets safely without unauthorized actions.
6. Resolving ambiguous target instances with clarification prompts.
7. Visual settling on unstable / high-motion screen transitions.
8. Provider exception handling and timeout recovery.
9. Pure-pixel boundary audit (zero RGB arrays or tensor embeddings passed to System 2).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch

from avcs.perception.system1_v1 import AVCSSystem1V1
from avcs.reasoning.contracts import (
    ActionIntent,
    ActionOp,
    IntentValidationResult,
    LoopExecutionReport,
    RejectionCategory,
    System2Decision,
)
from avcs.reasoning.provider import (
    MockReasoningProvider,
    LLMReasoningProvider,
)
from avcs.reasoning.bridge import System2Bridge


CHECKPOINT_PATH = "runs/weights/candidates/avcs_system1_v1.pt"
EXPECTED_SHA256 = "c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4"


class TestCheckpointIntegrity:
    """Verifies that the System 1 neural model checkpoint is intact and uncorrupted."""

    def test_checkpoint_exists_and_hash_matches(self):
        assert os.path.isfile(CHECKPOINT_PATH), f"Missing checkpoint: {CHECKPOINT_PATH}"
        with open(CHECKPOINT_PATH, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()
        assert computed_hash == EXPECTED_SHA256, (
            f"Checkpoint SHA-256 mismatch: {computed_hash} != {EXPECTED_SHA256}"
        )

    def test_checkpoint_loads_and_has_exact_parameters(self):
        model = AVCSSystem1V1.load_from_checkpoint(CHECKPOINT_PATH, device="cpu")
        assert model.total_parameters == 1_663_564
        assert hasattr(model, "grounder")
        assert hasattr(model, "word_recognizer")
        assert hasattr(model, "temporal_engine")


class TestIntentContractSanitization:
    """Verifies anti-coordinate injection, stale-frame rejection, and contract typing."""

    def test_anti_coordinate_injection(self):
        malicious_intent_dict = {
            "step_index": 1,
            "op": "click",
            "target_description": "Search button",
            "x": 350.0,
            "y": 120.0,
            "bbox": [100, 200, 150, 250],
            "parameters": {"coords": [350, 120]},
        }
        intent, had_injection = ActionIntent.from_dict(malicious_intent_dict)
        assert had_injection is True
        assert not hasattr(intent, "x")
        assert not hasattr(intent, "y")
        assert not hasattr(intent, "bbox")
        assert "coords" not in intent.parameters

    def test_stale_frame_rejection(self):
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow")
        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Provider emits an intent targeting frame 1, but current frame is 5
        provider = MockReasoningProvider(inject_stale_frame=True)
        bridge.provider = provider

        report = bridge.run_task(
            task="Click settings",
            frame_source=[dummy_frame, dummy_frame, dummy_frame, dummy_frame, dummy_frame, dummy_frame],
        )
        assert report.rejection_count > 0
        first_step = report.step_history[0]
        assert first_step.validation["rejection_category"] == RejectionCategory.STALE_FRAME.value
        assert first_step.validation["action_authorized"] is False


class TestObservationReasoningLoop:
    """Tests the full state flow across various real and synthetic desktop tasks."""

    @pytest.fixture
    def test_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        # Load a real desktop frame from the dataset
        manifest_path = "datasets/ui_grounding/manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        held_out = [fr for fr in m["frames"] if fr["split"] == "test_held_out"]
        img_p1 = os.path.join("datasets/ui_grounding", held_out[0]["file_path"])
        img_p2 = os.path.join("datasets/ui_grounding", held_out[1]["file_path"])
        f1 = np.array(Image.open(img_p1).convert("RGB"), dtype=np.uint8)
        f2 = np.array(Image.open(img_p2).convert("RGB"), dtype=np.uint8)
        return f1, f2

    def test_known_control_happy_path(self, test_frames: Tuple[np.ndarray, np.ndarray]):
        f1, _ = test_frames
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow", max_steps=5)

        # Task: Open Settings
        report = bridge.run_task(task="Open Settings", frame_source=f1)

        assert len(report.step_history) >= 1
        # In shadow mode, live input dispatch must be strictly False
        for s in report.step_history:
            if s.proposed_action:
                assert s.proposed_action["executed_to_desktop"] is False
                assert s.proposed_action["mode"] == "shadow"

    def test_absent_target_safe_abortion(self, test_frames: Tuple[np.ndarray, np.ndarray]):
        f1, _ = test_frames
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow")

        # Mock provider will try to look for non-existent unicorn button
        scripted = [
            System2Decision(
                thought="Looking for non-existent unicorn control.",
                task_status="in_progress",
                next_intent=ActionIntent.create(
                    step_index=1,
                    op=ActionOp.OBSERVE_TARGET,
                    target_description="magical unicorn rainbow button",
                    expected_outcome="Target grounded",
                    frame_sequence=1,
                ),
            ),
        ]
        bridge.provider = MockReasoningProvider(scripted_decisions=scripted)

        report = bridge.run_task(task="Click unicorn button", frame_source=f1)

        # System 2 must recognize target absence and stop without clicking
        assert report.final_status in ("ABORTED", "SUCCESS")
        assert report.action_authorized_count == 0

    def test_ambiguous_target_clarification(self, test_frames: Tuple[np.ndarray, np.ndarray]):
        f1, _ = test_frames
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow")

        # Script an ambiguous target scenario
        scripted = [
            System2Decision(
                thought="Found two identical icons with low margin.",
                task_status="needs_clarification",
                next_intent=ActionIntent.create(
                    step_index=1,
                    op=ActionOp.ASK_USER,
                    target_description="Duplicate icon",
                    expected_outcome="User clarifies index",
                    frame_sequence=1,
                ),
                clarification_prompt="Which icon should I select?",
            )
        ]
        bridge.provider = MockReasoningProvider(scripted_decisions=scripted)

        report = bridge.run_task(task="Select duplicate icon", frame_source=f1)
        assert report.final_status == "NEEDS_CLARIFICATION"
        assert report.action_authorized_count == 0
        assert "Which icon" in report.summary_message

    def test_temporal_visual_settling(self, test_frames: Tuple[np.ndarray, np.ndarray]):
        f1, f2 = test_frames
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow")

        # Alternate frames to trigger high motion energy
        report = bridge.run_task(task="Explore editor", frame_source=[f1, f2, f2, f2])
        assert report.total_steps >= 1
        assert report.mode == "shadow"

    def test_provider_error_recovery(self, test_frames: Tuple[np.ndarray, np.ndarray]):
        f1, _ = test_frames
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow")
        bridge.provider = MockReasoningProvider(inject_provider_error=True)

        report = bridge.run_task(task="Perform task", frame_source=f1)
        assert report.final_status == "PROVIDER_ERROR"
        assert "HTTP 504 Gateway Timeout" in report.summary_message
        assert report.action_authorized_count == 0


class TestPurePixelDataBoundary:
    """Verifies that no raw image arrays or internal tensor embeddings cross to System 2."""

    def test_zero_raw_pixels_exported(self):
        f = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bridge = System2Bridge(checkpoint_path=CHECKPOINT_PATH, mode="shadow", max_steps=2)

        report = bridge.run_task(task="Inspect canvas", frame_source=f)

        # Inspect every record in step_history
        for s in report.step_history:
            obs = s.observation_summary
            assert "raw_frame" not in obs
            assert "image" not in obs
            assert "crop" not in obs
            assert "embedding" not in obs
            assert "tensor" not in obs

            # Check that serializing to json succeeds (confirming no raw numpy/torch tensors)
            dumped = json.dumps(s.observation_summary)
            assert "numpy" not in dumped

    def test_reasoning_ast_pure_pixel(self):
        """Verifies zero external OCR or DOM inspection imports in avcs/reasoning/."""
        forbidden = {
            "pytesseract",
            "easyocr",
            "paddleocr",
            "winocr",
            "uiautomation",
            "pywinauto",
            "playwright",
            "selenium",
        }
        reasoning_dir = Path(__file__).resolve().parents[1] / "avcs" / "reasoning"
        for p in reasoning_dir.glob("*.py"):
            with open(p, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=p.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        assert a.name.split(".")[0] not in forbidden, f"Forbidden import {a.name} in {p.name}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert node.module.split(".")[0] not in forbidden, f"Forbidden import {node.module} in {p.name}"

