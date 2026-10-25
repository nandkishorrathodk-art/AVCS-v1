"""Live End-to-End Runtime Integration Tests for StreamingV2Adapter and StreamingVisualCortexV2.

Exercises:
  1. Live integration of StreamingV2Adapter with StreamingVisualCortexV2.
  2. Safe checkpoint loading from runs/weights/avcs_v2_weights.pt (weights_loaded == True).
  3. Processing synthetic 128x128 screen frame ticks (topological detection, change maps, post-action response).
  4. System 2 Encapsulation Contract via adapter.to_system2_summary() (strictly zero coordinate leakage,
     only high-level booleans and categorical strings).
  5. Recurrent temporal state persistence and fail-closed 2.0s gap / dropped frame reset.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from avcs.streaming_v2.integration import (
    DEFAULT_WEIGHTS_PATH,
    StreamingV2Adapter,
    StreamingV2Observation,
)
from avcs.streaming_v2.model import StreamingVisualCortexV2, System2SymbolicSummary
from avcs.training.multi_generator_dataset import ArmD2MultiGeneratorDataset, SPLIT_TEST_HELD_OUT


@pytest.fixture
def weights_path() -> str:
    path = DEFAULT_WEIGHTS_PATH
    if not os.path.exists(path):
        pytest.skip(f"Weights file not found at {path}")
    return path


@pytest.fixture
def live_adapter(weights_path: str) -> StreamingV2Adapter:
    adapter = StreamingV2Adapter(weights_path=weights_path, presence_threshold=0.35)
    return adapter


class TestStreamingV2LiveIntegration:
    """End-to-End integration test suite for StreamingV2Adapter."""

    def test_weights_load_cleanly(self, live_adapter: StreamingV2Adapter, weights_path: str) -> None:
        """Verify weights load cleanly into model with zero mismatch and weights_loaded is True."""
        assert live_adapter.weights_loaded is True
        assert isinstance(live_adapter.model, StreamingVisualCortexV2)
        assert live_adapter.model.total_parameters >= 400_000
        assert live_adapter.model.total_parameters <= 500_000

    def test_process_synthetic_128x128_frame_tick(self, live_adapter: StreamingV2Adapter) -> None:
        """Verify processing synthetic 128x128 screen frame tick produces valid StreamingV2Observation."""
        # Generate synthetic desktop with title bar, buttons, and content region
        img = Image.new("RGB", (128, 128), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 128, 16], fill=(50, 70, 120))  # title bar
        draw.rectangle([15, 25, 113, 45], fill=(255, 255, 255), outline=(100, 100, 100))  # text field
        draw.rectangle([15, 55, 60, 75], fill=(70, 130, 220))  # button
        frame1 = np.array(img, dtype=np.uint8)

        obs1 = live_adapter.process_frame(frame1, timestamp=1.0)
        assert isinstance(obs1, StreamingV2Observation)
        assert isinstance(obs1.spinner_active, bool)
        assert 0.0 <= obs1.spinner_probability <= 1.0
        assert isinstance(obs1.change_detected, bool)
        assert obs1.mean_change_energy >= 0.0
        assert isinstance(obs1.visual_response_verified, bool)
        assert 0.0 <= obs1.visual_response_confidence <= 1.0
        assert obs1.forward_latency_ms > 0.0
        assert isinstance(obs1.detected_ui_count, int)
        assert isinstance(obs1.detected_classes, list)
        assert isinstance(obs1.candidate_boxes, list)

        # Frame 2: simulated button interaction (trigger visual change)
        draw.rectangle([15, 55, 60, 75], fill=(30, 80, 180))
        frame2 = np.array(img, dtype=np.uint8)

        obs2 = live_adapter.process_frame(frame2, timestamp=1.05)
        assert isinstance(obs2, StreamingV2Observation)
        assert obs2.forward_latency_ms < 100.0  # reasonable CPU latency threshold
        text_summary = obs2.to_system2_summary()
        assert isinstance(text_summary, str)
        assert len(text_summary) > 0

    def test_topological_detection_on_heldout_dataset(self, live_adapter: StreamingV2Adapter) -> None:
        """Verify topological detection and classification on actual held-out UI stream samples."""
        ds = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_HELD_OUT, apply_augmentation=False, preload_images=True)
        sample = ds[0]
        image_tensor = sample["image"]  # (3, 128, 128)
        
        obs = live_adapter.process_frame(image_tensor, timestamp=10.0)
        assert isinstance(obs, StreamingV2Observation)
        assert obs.detected_ui_count > 0
        assert len(obs.detected_classes) == obs.detected_ui_count
        assert len(obs.candidate_boxes) == obs.detected_ui_count
        for cls_id, conf, (dx, dy, w, h) in obs.candidate_boxes:
            assert 0 <= cls_id <= 4
            assert conf >= live_adapter.presence_threshold
            assert -1.0 <= dx <= 1.0
            assert -1.0 <= dy <= 1.0
            assert w >= 0.0
            assert h >= 0.0

    def test_adapter_to_system2_summary_encapsulation(self, live_adapter: StreamingV2Adapter) -> None:
        """Verify adapter.to_system2_summary() strictly encapsulates internal coordinates.

        Must expose only high-level status booleans, categorical strings, and scalar percentages.
        """
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        live_adapter.process_frame(img, timestamp=20.0)

        sym_summary = live_adapter.to_system2_summary()
        assert isinstance(sym_summary, System2SymbolicSummary)

        # High-level status booleans
        assert isinstance(sym_summary.has_button, bool)
        assert isinstance(sym_summary.has_input_box, bool)
        assert isinstance(sym_summary.has_icon_toggle, bool)
        assert isinstance(sym_summary.has_dialog_popup, bool)
        assert isinstance(sym_summary.has_text_area, bool)
        assert isinstance(sym_summary.has_interactive_control, bool)
        assert isinstance(sym_summary.has_text_input_field, bool)
        assert isinstance(sym_summary.has_navigation_toggle, bool)
        assert isinstance(sym_summary.has_modal_dialog, bool)
        assert isinstance(sym_summary.has_content_region, bool)
        assert isinstance(sym_summary.visual_change_detected, bool)
        assert isinstance(sym_summary.spinner_active, bool)
        assert isinstance(sym_summary.post_action_response_detected, bool)

        # Categorical strings
        assert sym_summary.uncertainty_level in ("LOW", "MEDIUM", "HIGH")
        if sym_summary.primary_ui_class is not None:
            assert isinstance(sym_summary.primary_ui_class, str)

        # Strict coordinate leak check on dictionary export
        d = sym_summary.to_dict()
        prohibited_keys = {"x", "y", "w", "h", "bbox", "box", "coordinates", "coords"}
        for key in d:
            assert key not in prohibited_keys, f"Coordinate key {key} leaked to System 2!"
        for key, val in d.items():
            assert not isinstance(val, (list, tuple, np.ndarray, torch.Tensor)), (
                f"Raw coordinate or array data leaked in field {key}: {val}"
            )

    def test_temporal_persistence_and_fail_closed_reset(self, live_adapter: StreamingV2Adapter) -> None:
        """Verify recurrent temporal state persistence and fail-closed reset upon 2.0s gap or dropped frame."""
        live_adapter.reset_temporal_state()
        assert live_adapter.temporal_state is None

        frame = np.zeros((128, 128, 3), dtype=np.uint8)

        # Step 1: Initial cold-start
        live_adapter.process_frame(frame, timestamp=0.0)
        assert live_adapter.temporal_state is not None
        assert live_adapter.temporal_state.step_count == 1

        # Step 2: Normal sequential frame (delta_t = 0.033s)
        live_adapter.process_frame(frame, timestamp=0.033)
        assert live_adapter.temporal_state.step_count == 2

        # Step 3: Normal sequential frame (delta_t = 0.066s)
        live_adapter.process_frame(frame, timestamp=0.066)
        assert live_adapter.temporal_state.step_count == 3

        # Step 4: Temporal gap >= 2.0s (timestamp jump from 0.066s to 2.500s, delta_t = 2.434s)
        live_adapter.process_frame(frame, timestamp=2.500)
        assert live_adapter.temporal_state.step_count == 1, (
            f"Expected fail-closed state reset (step_count == 1) after 2.434s gap, got {live_adapter.temporal_state.step_count}"
        )

        # Step 5: Sequential step after gap reset
        live_adapter.process_frame(frame, timestamp=2.533)
        assert live_adapter.temporal_state.step_count == 2

        # Step 6: Explicit dropped_frame flag
        live_adapter.process_frame(frame, timestamp=2.566, dropped_frame=True)
        assert live_adapter.temporal_state.step_count == 1, (
            f"Expected fail-closed state reset (step_count == 1) on dropped_frame=True, got {live_adapter.temporal_state.step_count}"
        )

        # Step 7: Manual reset_temporal_state()
        live_adapter.reset_temporal_state()
        assert live_adapter.temporal_state is None
        assert live_adapter.last_output is None
        assert live_adapter.last_observation is None
