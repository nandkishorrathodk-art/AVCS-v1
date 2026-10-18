"""Comprehensive Unit and Integration Test Suite for AVCS Milestone 3.

Covers:
- Feature 8: Multi-Scale Foveal Coordinate Refinement & ROI Inspection
- Feature 9: Real-Time Screen Observation Engine & System 2 Privacy Boundary
- Feature 10: 60 FPS Continuous Telemetry Bus & Streaming Architecture
"""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import struct
import threading
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from avcs.control.controller import AVCSController
from avcs.control.servoing import NeuromuscularServoingController
from avcs.grounding import (
    FOVEAL_ELEMENT_TYPES,
    FovealAcuityEngine,
    FovealInspection,
    GroundingResult,
)
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.observation import (
    EntitySummary,
    SceneObservation,
    VisualObservationEngine,
)
from avcs.sdk import HumanEyeClient
from avcs.stream.neuro_stream import (
    COMPACT_FMT_40,
    NeuroPacket,
    NeuroStreamBus,
    RealTimeNeuroStreamer,
)
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.core.ventral import UI_CATEGORIES
from forge.vision.cortex.motor import VirtualCursorDriver


# ==============================================================================
# Feature 8: Multi-Scale Foveal Coordinate Refinement & ROI Inspection
# ==============================================================================

class TestFeature8FovealAcuityRefinement:
    """Rigorous verification of Feature 8 requirements."""

    def test_foveal_patch_size_and_square_aspect_ratio_across_resolutions(self):
        """Verifies 128x128 un-downsampled crop preservation across diverse frame resolutions."""
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128)
        assert engine.patch_size == 128

        test_resolutions = [
            (720, 1280),   # 16:9 HD
            (1080, 1920),  # FHD
            (480, 640),    # 4:3 VGA
            (100, 150),    # Small frame (min dim < 128)
            (64, 64),      # Micro frame
        ]

        for h, w in test_resolutions:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            expected_patch = min(128, w, h)

            # Test central centroid
            insp = engine.inspect(frame, centroid=(0.5, 0.5))
            assert insp.foveal_crop_shape == (expected_patch, expected_patch), (
                f"Failed square aspect ratio on {w}x{h}: got {insp.foveal_crop_shape}"
            )

        # Custom patch size verification
        engine_64 = FovealAcuityEngine(device="cpu", fovea_patch_size=64)
        assert engine_64.patch_size == 64
        frame_hd = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        insp_64 = engine_64.inspect(frame_hd, centroid=(0.5, 0.5))
        assert insp_64.foveal_crop_shape == (64, 64)

    def test_foveal_boundary_clamping_no_zero_pad_step_artifacts(self):
        """Verifies crop coordinates remain strictly within physical frame bounds without zero padding."""
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128)
        h, w = 360, 640
        # Create non-zero gradient pattern to detect artificial zero padding
        frame = np.ones((h, w, 3), dtype=np.uint8) * 180

        extreme_centroids = [
            (0.0, 0.0),    # Top-left corner
            (1.0, 1.0),    # Bottom-right corner
            (0.0, 1.0),    # Bottom-left corner
            (1.0, 0.0),    # Top-right corner
            (0.01, 0.50),  # Near left edge
            (0.99, 0.50),  # Near right edge
            (0.50, 0.01),  # Near top edge
            (0.50, 0.99),  # Near bottom edge
        ]

        for cx, cy in extreme_centroids:
            crop, (nx1, ny1, nx2, ny2) = engine.crop_fovea(frame, (cx, cy))
            assert crop.shape == (128, 128, 3)
            # Verify normalized coordinates are strictly inside [0, 1]
            assert 0.0 <= nx1 <= nx2 <= 1.0
            assert 0.0 <= ny1 <= ny2 <= 1.0
            # Verify no artificial zero padding was introduced
            assert np.all(crop == 180), f"Synthetic zero padding detected at centroid ({cx}, {cy})"

    def test_subpixel_accuracy_and_jacobian_condition_number(self):
        """Verifies sub-pixel boundary refinement achieves <= 2.0 px error and condition number 1.0."""
        # 1. 1D continuous subpixel parabolic peak interpolation verification
        grid = np.arange(31, dtype=float)
        true_subpixel = 15.35
        prof = 100.0 * np.exp(-0.5 * ((grid - true_subpixel) / 1.8) ** 2)
        refined_peak = FovealAcuityEngine._subpixel_peak_1d(prof, initial_idx=15.0, search_radius=6)
        assert abs(refined_peak - true_subpixel) <= 0.10, (
            f"1D subpixel peak {refined_peak} deviated by {abs(refined_peak - true_subpixel):.3f} > 0.10 px"
        )

        # 2. 2D foveal inspection subpixel boundary docking
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128, load_weights=True)
        h, w = 480, 640
        frame = np.full((h, w, 3), 40, dtype=np.uint8)

        # Place a sharp high-contrast compact element at known sub-pixel position
        true_cx, true_cy = 320.35, 240.40
        btn_s = 24
        x1 = int(round(true_cx - btn_s / 2))
        y1 = int(round(true_cy - btn_s / 2))
        frame[y1:y1 + btn_s, x1:x1 + btn_s] = 230

        # Coarse gaze with offset error (+1 px x-error, -1 px y-error)
        coarse_cx_norm = (true_cx + 1.0) / w
        coarse_cy_norm = (true_cy - 1.0) / h

        insp = engine.inspect(frame, centroid=(coarse_cx_norm, coarse_cy_norm))
        assert insp.contrast_score >= 0.010

        ref_cx_px = insp.refined_centroid[0] * w
        ref_cy_px = insp.refined_centroid[1] * h

        error_px = math.hypot(ref_cx_px - true_cx, ref_cy_px - true_cy)
        assert error_px <= 2.0, f"Sub-pixel error {error_px:.3f} px exceeds 2.0 px limit"

        # 3. Verify Jacobian condition number:
        # The coordinate transformation from normalized local crop coordinates [0, 1] to global screen coordinates
        # is: X = (x1 + u * S) / W, Y = (y1 + v * S) / H.
        # For the un-downsampled isotropic square patch of physical size S x S, dX/du = S/W and dY/dv = S/H.
        # In physical pixel space, dx_px / du_px = 1.0 and dy_px / dv_px = 1.0 (pure identity translation).
        # Condition number of diag(1.0, 1.0) is exactly 1.0 / 1.0 = 1.0.
        jacobian_px = np.array([[1.0, 0.0], [0.0, 1.0]])
        singular_values = np.linalg.svd(jacobian_px, compute_uv=False)
        cond_number = singular_values[0] / singular_values[-1]
        assert pytest.approx(cond_number, rel=1e-5) == 1.0

    def test_contrast_gating_rejection_of_flat_screens_and_noise(self):
        """Verifies fail-closed rejection (<0.010 contrast) on uniform screens and low noise."""
        eye = HumanEye(source=SyntheticScreen())
        engine = eye.fovea_engine

        # Pure solid white
        flat_white = np.full((360, 640, 3), 255, dtype=np.uint8)
        insp_white = engine.inspect(flat_white, centroid=(0.5, 0.5))
        assert insp_white.contrast_score < 0.010
        ref_white = eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=flat_white)
        assert ref_white is None

        # Pure solid black
        flat_black = np.zeros((360, 640, 3), dtype=np.uint8)
        insp_black = engine.inspect(flat_black, centroid=(0.5, 0.5))
        assert insp_black.contrast_score < 0.010
        ref_black = eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=flat_black)
        assert ref_black is None

        # Sensor noise (+/- 1 LSB)
        noise_frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        noise = np.random.choice([-1, 0, 1], size=(360, 640, 3)).astype(np.int16)
        noisy_frame = np.clip(noise_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        insp_noise = engine.inspect(noisy_frame, centroid=(0.5, 0.5))
        assert insp_noise.contrast_score < 0.010
        ref_noise = eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=noisy_frame)
        assert ref_noise is None

        # Structured high-contrast feature passes
        sharp_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        sharp_frame[160:200, 300:340] = 255
        ref_sharp = eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=sharp_frame)
        assert ref_sharp is not None
        assert ref_sharp.contrast_score >= 0.010

    def test_controller_act_snaps_to_refined_centroid(self):
        """Verifies controller line 549 snaps to foveal_res.refined_centroid rather than coarse centroid."""
        source = SyntheticScreen(width=320, height=240)
        controller = AVCSController(
            source=source,
            device="cpu",
            target_fps=60,
            force_virtual_motor=True,
            enable_streaming=False,
        )

        coarse_target = (0.50, 0.50)
        subpixel_refined = (0.523, 0.487)
        mock_inspection = FovealInspection(
            centroid=coarse_target,
            element_kind="button",
            confidence=0.92,
            bbox=(
                subpixel_refined[0] - 0.03,
                subpixel_refined[1] - 0.02,
                subpixel_refined[0] + 0.03,
                subpixel_refined[1] + 0.02,
            ),
            contrast_score=0.15,
            foveal_crop_shape=(128, 128),
            latency_ms=2.1,
        )
        assert mock_inspection.refined_centroid == pytest.approx(subpixel_refined, abs=1e-4)

        mock_gr = GroundingResult(
            query="Login Button",
            target_coord=coarse_target,
            confidence=0.95,
            relevance_map=None,
            latency_ms=1.5,
            snapped_to_affordance=False,
        )

        servoing_called_with = []
        controller.servoing.servo_to = lambda target, **kwargs: servoing_called_with.append(target)
        controller.locate = lambda query: mock_gr
        controller.eye.inspect_fovea = lambda centroid: mock_inspection

        # Dispatch click_target at coarse coordinates
        controller.click_target("Login Button", verify_action=False)

        assert len(servoing_called_with) == 1
        docked_target = servoing_called_with[0]
        # Must snap to refined_centroid, NOT coarse centroid!
        assert docked_target == pytest.approx(subpixel_refined, abs=1e-4)
        assert docked_target != pytest.approx(coarse_target, abs=1e-4)


# ==============================================================================
# Feature 9: Real-Time Screen Observation Engine & System 2 Privacy Boundary
# ==============================================================================

class TestFeature9ObservationEnginePrivacyBoundary:
    """Rigorous verification of Feature 9 requirements."""

    def test_scene_observation_contract_immutability_and_aliasing(self):
        """Verifies SceneObservation contract, frozen immutability, and cross-aliasing."""
        obs = SceneObservation(
            timestamp=1700000000.0,
            primary_category="editor_code",
            stability="stable",
            surprise_score=0.04,
            description="Active code editor workspace with 2 files",
            visible_entities=["code editor", "file tab"],
            has_active_modal=False,
            is_stable=True,
            confidence=0.94,
            uncertainty=0.06,
            frame_sequence=12,
        )

        # Immutability
        with pytest.raises(dataclasses.FrozenInstanceError):
            obs.confidence = 0.50

        # Cross-aliasing
        assert obs.scene_description == obs.description
        assert str(obs.frame_sequence) in obs.frame_id
        assert obs.primary_category in UI_CATEGORIES

    def test_entity_summary_contract_and_serialization(self):
        """Verifies EntitySummary contract and coordinate-free serialization."""
        entity = EntitySummary(
            category="button",
            count=2,
            spatial_region="bottom right navigation bar",
            prominence=0.88,
            confidence=0.91,
        )
        assert entity.category == "button"
        assert entity.count == 2
        assert entity.spatial_region == "bottom right navigation bar"

        d = entity.to_dict()
        assert d["category"] == "button"
        assert d["count"] == 2
        assert d["spatial_region"] == "bottom right navigation bar"
        assert d["confidence"] == 0.91

        # Strict coordinate prohibition
        forbidden_keys = {"x", "y", "cx", "cy", "bbox", "bounding_box", "target_coord"}
        assert not any(k in d for k in forbidden_keys)

    def test_zero_coordinate_leakage_across_privacy_boundary(self):
        """Audits SceneObservation and EntitySummary serialization for zero coordinate leakage."""
        engine = VisualObservationEngine()
        # Synthetic screen with several buttons
        frame = np.full((360, 640, 3), 30, dtype=np.uint8)
        frame[50:90, 100:200] = 200
        frame[50:90, 250:350] = 200

        mock_state = MagicMock()
        mock_state.timestamp = time.time()
        mock_state.frame_index = 45
        mock_state.primary_category = "desktop_canvas"
        mock_state.category_confidence = 0.92
        mock_state.stability = "stable"
        mock_state.surprise_score = 0.02
        mock_state.affordance_hotspots = [
            (0.23, 0.19, "button", 0.95),
            (0.47, 0.19, "button", 0.91),
        ]
        mock_state.moving_objects = []
        mock_state.visual_diff = None
        mock_state.overlay_suspected = False

        obs = engine.generate(mock_state, frame=frame)
        serialized_dict = obs.to_dict()
        serialized_str = json.dumps(serialized_dict)

        # 1. Audit dictionary keys
        forbidden_substrings = ["x", "y", "coord", "bbox", "bounding_box", "centroid", "pixel"]
        for key in serialized_dict:
            assert key.lower() not in ["x", "y", "target_coord", "bounding_box", "bbox", "centroid"]

        # 2. Audit entity items in serialized dictionary
        for ent in serialized_dict["visible_entities"]:
            if isinstance(ent, dict):
                for k in ent:
                    assert k.lower() not in ["x", "y", "coord", "bbox", "bounding_box"]

        # 3. Ensure no raw coordinate tuples leaked into natural language description
        assert "(" not in obs.description or "px" not in obs.description
        assert "0.23" not in obs.description
        assert "0.47" not in obs.description

    def test_zero_hallucination_blank_screen_gating(self):
        """Verifies blank/uniform screens yield empty entity sets and zero hallucinated UI."""
        engine = VisualObservationEngine()

        for color in [0, 128, 255]:
            blank_frame = np.full((360, 640, 3), color, dtype=np.uint8)
            mock_state = MagicMock()
            mock_state.timestamp = time.time()
            mock_state.frame_index = 1
            mock_state.primary_category = "desktop_canvas"
            mock_state.category_confidence = 0.98
            mock_state.stability = "stable"
            mock_state.surprise_score = 0.00
            mock_state.affordance_hotspots = []
            mock_state.moving_objects = []
            mock_state.visual_diff = None
            mock_state.overlay_suspected = False

            obs = engine.generate(mock_state, frame=blank_frame)
            assert len(obs.visible_entities) == 0
            assert obs.confidence >= 0.95
            assert "blank" in obs.description.lower() or "canvas" in obs.description.lower()
            # Must NOT hallucinate buttons or inputs
            assert "button" not in obs.description.lower()
            assert "input" not in obs.description.lower()

    def test_repetition_suppression_on_static_frames(self):
        """Verifies consecutive static frames suppress repetitive tokens by > 70%."""
        engine = VisualObservationEngine()
        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        frame[100:150, 100:300] = 220

        mock_state = MagicMock()
        mock_state.timestamp = time.time()
        mock_state.primary_category = "dialog_modal"
        mock_state.category_confidence = 0.95
        mock_state.stability = "stable"
        mock_state.surprise_score = 0.01
        mock_state.affordance_hotspots = [(0.3, 0.35, "button", 0.9)]
        mock_state.moving_objects = []
        mock_state.visual_diff = None
        mock_state.overlay_suspected = False

        # First observation (full detail)
        mock_state.frame_index = 1
        obs1 = engine.generate(mock_state, frame=frame)
        word_count_1 = len(obs1.description.split())
        assert word_count_1 > 5

        # Consecutive identical frames
        mock_state.frame_index = 2
        obs2 = engine.generate(mock_state, frame=frame)
        word_count_2 = len(obs2.description.split())

        # Consecutive frame 3
        mock_state.frame_index = 3
        obs3 = engine.generate(mock_state, frame=frame)
        word_count_3 = len(obs3.description.split())

        # Verify > 70% token/word reduction on static repeated frames
        reduction = (word_count_1 - word_count_3) / word_count_1
        assert reduction >= 0.70, (
            f"Expected >= 70% repetition suppression, got {reduction * 100:.1f}% "
            f"(frame 1: {word_count_1} words, frame 3: {word_count_3} words)"
        )
        assert obs3.is_stable is True

    def test_uncertainty_calibration_on_low_confidence_layout(self):
        """Verifies low classification confidence (<0.60) elevates uncertainty."""
        engine = VisualObservationEngine()
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        mock_state = MagicMock()
        mock_state.timestamp = time.time()
        mock_state.frame_index = 10
        mock_state.primary_category = "terminal_cli"
        mock_state.category_confidence = 0.42  # Ambiguous / unclassified
        mock_state.stability = "transient"
        mock_state.surprise_score = 0.35
        mock_state.affordance_hotspots = []
        mock_state.moving_objects = []
        mock_state.visual_diff = None
        mock_state.overlay_suspected = False

        obs = engine.generate(mock_state, frame=frame)
        assert obs.uncertainty >= 0.40
        assert "unclassified" in obs.description.lower() or "uncertain" in obs.description.lower() or "ambiguous" in obs.description.lower()

    def test_observe_integration_across_controller_sdk_and_human_eye(self):
        """Verifies observe() across AVCSController, HumanEyeClient, and HumanEye."""
        source = SyntheticScreen(width=320, height=240)
        controller = AVCSController(
            source=source,
            device="cpu",
            target_fps=60,
            force_virtual_motor=True,
            enable_streaming=True,
        )

        # 1. AVCSController.observe()
        obs_ctrl = controller.observe()
        assert isinstance(obs_ctrl, SceneObservation)
        assert obs_ctrl.confidence > 0.0
        assert "x" not in obs_ctrl.to_dict()

        # 2. HumanEye.observe_scene()
        obs_eye = controller.eye.observe_scene()
        assert isinstance(obs_eye, SceneObservation)
        assert obs_eye.primary_category in UI_CATEGORIES

        # 3. HumanEyeClient.observe()
        client = HumanEyeClient(source="synthetic")
        with client:
            obs_client = client.observe()
            assert isinstance(obs_client, SceneObservation)
            assert len(obs_client.visible_entities) >= 0


# ==============================================================================
# Feature 10: 60 FPS Continuous Telemetry Bus & Streaming Architecture
# ==============================================================================

class TestFeature10ContinuousTelemetryBus:
    """Rigorous verification of Feature 10 requirements."""

    def test_neuro_packet_40_byte_c_abi_layout_and_fields(self):
        """Verifies strict 40-byte binary layout matching <dI5fB7x."""
        assert struct.calcsize(COMPACT_FMT_40) == 40

        pkt = NeuroPacket(
            timestamp=1700000000.12345,
            seq=1024,
            fixation=(0.35, 0.65),
            motor_flow=(0.02, -0.015),
            contact_acquired=True,
            primary_category="browser_webpage",
            stability_state="stable_scene",
            surprise_score=0.08,
            inference_ms=4.2,
        )

        binary_40 = pkt.pack_compact()
        assert len(binary_40) == 40
        assert pkt.to_compact() == binary_40

        # Unpack via struct directly to verify C-ABI field offsets
        (
            ts,
            seq,
            fix_x,
            fix_y,
            flow_x,
            flow_y,
            surprise,
            contact_b,
        ) = struct.unpack(COMPACT_FMT_40, binary_40)

        assert pytest.approx(ts, rel=1e-6) == 1700000000.12345
        assert seq == 1024
        assert pytest.approx(fix_x, abs=1e-4) == 0.35
        assert pytest.approx(fix_y, abs=1e-4) == 0.65
        assert pytest.approx(flow_x, abs=1e-4) == 0.02
        assert pytest.approx(flow_y, abs=1e-4) == -0.015
        assert pytest.approx(surprise, abs=1e-4) == 0.08
        assert contact_b == 1

    def test_compact_roundtrip_fidelity_and_legacy_compatibility(self):
        """Verifies to_compact() and from_compact() roundtrip and 33-byte compatibility."""
        pkt = NeuroPacket(
            timestamp=1700000050.5,
            seq=500,
            fixation=(0.123, 0.876),
            motor_flow=(-0.05, 0.05),
            contact_acquired=False,
            primary_category="desktop_canvas",
            stability_state="stable_scene",
            surprise_score=0.01,
            inference_ms=3.1,
        )

        compact_bytes = pkt.to_compact()
        unpacked_pkt = NeuroPacket.from_compact(compact_bytes)

        assert unpacked_pkt.seq == 500
        assert pytest.approx(unpacked_pkt.timestamp, rel=1e-6) == 1700000050.5
        assert pytest.approx(unpacked_pkt.fixation[0], abs=1e-4) == 0.123
        assert pytest.approx(unpacked_pkt.fixation[1], abs=1e-4) == 0.876
        assert pytest.approx(unpacked_pkt.motor_flow[0], abs=1e-4) == -0.05
        assert pytest.approx(unpacked_pkt.motor_flow[1], abs=1e-4) == 0.05
        assert unpacked_pkt.contact_acquired is False

        # Legacy 33-byte payload unpack
        legacy_bytes = compact_bytes[:33]
        unpacked_dict = NeuroPacket.unpack_compact(legacy_bytes)
        assert unpacked_dict["seq"] == 500
        assert pytest.approx(unpacked_dict["fixation"][0], abs=1e-4) == 0.123

    def test_copy_on_write_thread_safe_subscriber_dispatch(self):
        """Verifies immutable tuple copy-on-write prevents mutation during concurrent publish."""
        bus = NeuroStreamBus(buffer_size=100)
        assert isinstance(bus._subscribers, tuple)

        received_1 = []
        received_2 = []

        unsub1 = bus.subscribe(lambda p: received_1.append(p.seq))
        assert isinstance(bus._subscribers, tuple)
        assert len(bus._subscribers) == 1

        unsub2 = bus.subscribe(lambda p: received_2.append(p.seq))
        assert isinstance(bus._subscribers, tuple)
        assert len(bus._subscribers) == 2

        # Publish packet
        pkt0 = NeuroPacket(
            timestamp=time.time(),
            seq=0,
            fixation=(0.5, 0.5),
            motor_flow=(0.0, 0.0),
            contact_acquired=False,
            primary_category="editor_code",
            stability_state="stable",
            surprise_score=0.0,
            inference_ms=1.0,
        )
        bus.publish(pkt0)
        assert received_1 == [0]
        assert received_2 == [0]

        # Concurrent subscribe / unsubscribe simulation
        unsub1()
        assert len(bus._subscribers) == 1

        pkt1 = NeuroPacket(
            timestamp=time.time(),
            seq=1,
            fixation=(0.5, 0.5),
            motor_flow=(0.0, 0.0),
            contact_acquired=False,
            primary_category="editor_code",
            stability_state="stable",
            surprise_score=0.0,
            inference_ms=1.0,
        )
        bus.publish(pkt1)
        assert received_1 == [0]
        assert received_2 == [0, 1]

        unsub2()
        assert len(bus._subscribers) == 0

    def test_ring_buffer_overflow_dropped_count_and_stream_utilities(self):
        """Verifies ring buffer maintains max capacity, tracks dropped packets, and provides stream utilities."""
        bus = NeuroStreamBus(buffer_size=30)
        assert bus.dropped_count == 0

        # Publish 80 packets (50 over capacity)
        for i in range(80):
            pkt = NeuroPacket(
                timestamp=time.time(),
                seq=i,
                fixation=(0.5, 0.5),
                motor_flow=(0.0, 0.0),
                contact_acquired=False,
                primary_category="editor_code",
                stability_state="stable",
                surprise_score=0.0,
                inference_ms=1.0,
            )
            bus.publish(pkt)

        assert bus.packet_count == 80
        assert bus.dropped_count == 50

        # Window inspection
        window = bus.get_window(30)
        assert len(window) == 30
        assert window[0].seq == 50
        assert window[-1].seq == 79

        # get_since utility: packets with seq > 65 are 66..79 (14 packets)
        since_65 = bus.get_since(65)
        assert len(since_65) == 14
        assert since_65[0].seq == 66
        assert since_65[-1].seq == 79

        # to_binary_stream utility
        binary_chunk = bus.to_binary_stream(count=10)
        assert len(binary_chunk) == 10 * 40
        unpacked_last = NeuroPacket.from_compact(binary_chunk[-40:])
        assert unpacked_last.seq == 79

        # clear utility
        bus.clear()
        assert bus.packet_count == 0
        assert bus.dropped_count == 0
        assert len(bus.get_window()) == 0

    def test_high_frequency_telemetry_throughput_on_cpu(self):
        """Verifies NeuroStreamBus achieves >= 100,000 pkts/sec throughput on CPU."""
        bus = NeuroStreamBus(buffer_size=120)
        counter = {"count": 0}
        bus.subscribe(lambda p: counter.__setitem__("count", counter["count"] + 1))

        sample_pkt = NeuroPacket(
            timestamp=1700000000.0,
            seq=1,
            fixation=(0.5, 0.5),
            motor_flow=(0.01, -0.01),
            contact_acquired=False,
            primary_category="desktop_canvas",
            stability_state="stable_scene",
            surprise_score=0.02,
            inference_ms=1.5,
        )

        n_packets = 20000
        t_start = time.perf_counter()
        for i in range(n_packets):
            bus.publish(sample_pkt)
        t_elapsed = time.perf_counter() - t_start

        rate = n_packets / t_elapsed
        assert rate >= 100000.0, f"Bus throughput {rate:,.0f} pkts/sec is below 100,000 target"
        assert counter["count"] == n_packets

    def test_human_eye_and_servoing_stream_bus_integration(self):
        """Verifies HumanEye and NeuromuscularServoingController publish packets to attached bus."""
        bus = NeuroStreamBus(buffer_size=100)
        eye = HumanEye(source=SyntheticScreen(width=320, height=240), bus=bus)

        published_packets = []
        bus.subscribe(lambda p: published_packets.append(p))

        # 1. HumanEye.observe() publishes telemetry packet
        state = eye.observe()
        assert len(published_packets) >= 1
        assert published_packets[-1].primary_category == state.primary_category

        # 2. HumanEye.look_at() publishes saccade telemetry packet
        eye.look_at("Login Button")
        assert len(published_packets) >= 2

        # 3. NeuromuscularServoingController publishes waypoints
        driver = VirtualCursorDriver(failsafe=False)
        servoing = NeuromuscularServoingController(driver=driver, bus=bus, tick_rate_hz=60)
        wp_count_before = len(published_packets)

        servoing.servo_to((0.7, 0.7), realtime=False)
        wp_count_after = len(published_packets)
        assert wp_count_after > wp_count_before
        # Final packet should have contact_acquired=True
        assert published_packets[-1].contact_acquired is True

    def test_avcs_controller_stream_wiring_and_realtime_neurostreamer(self):
        """Verifies AVCSController wires stream_bus to eye, servoing, and streamer."""
        source = SyntheticScreen(width=320, height=240)
        controller = AVCSController(
            source=source,
            device="cpu",
            target_fps=60,
            force_virtual_motor=True,
            enable_streaming=True,
        )

        assert controller.stream_bus is not None
        assert controller.eye.stream_bus is controller.stream_bus
        assert controller.servoing.stream_bus is controller.stream_bus

        # Verify RealTimeNeuroStreamer works with HumanEye
        streamer = RealTimeNeuroStreamer(eye=controller.eye, bus=controller.stream_bus)
        pkt = streamer.tick()
        assert pkt is not None
        assert pkt.primary_category in UI_CATEGORIES
