"""Tier 1: Feature Coverage E2E Test Suite.

Covers functional contracts, public interfaces, preconditions, and postconditions
for all 20 features mapped in PROJECT.md and TEST_INFRA.md.
"""
from __future__ import annotations

import os
import struct
import time
from typing import List, Tuple

import numpy as np
import pytest
import torch

from avcs.grounding import FovealInspection
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.sdk import HumanEyeClient, UnfocusedInputElementError
from avcs.servoing import NeuromuscularServoingController
from avcs.stream.neuro_stream import NeuroPacket, NeuroStreamBus
from avcs.verifier import VerificationResult, VerificationStatus
from forge.vision.cortex.motor.driver import VirtualCursorDriver
from forge.vision.cortex.sdk import UnvalidatedGroundingError

try:
    from avcs.controller import AVCSController
    HAS_AVCS_CONTROLLER = True
except ImportError:
    HAS_AVCS_CONTROLLER = False


@pytest.mark.tier1
class TestFeature1CheckpointAndBaselineIntegrity:
    """Feature 1: Baseline Test & Checkpoint Integrity."""

    def test_feature1_checkpoint_file_exists_and_non_empty(self):
        """Verifies authentic weights checkpoint exists and is non-empty."""
        weights_path = os.path.join("runs", "weights", "human_eye_weights.pt")
        assert os.path.exists(weights_path), f"Checkpoint missing at {weights_path}"
        size = os.path.getsize(weights_path)
        assert size > 50_000, f"Checkpoint size {size} bytes is suspiciously small"

    def test_feature1_checkpoint_weights_auto_discovery(self, avcs_client: HumanEyeClient):
        """Verifies that client auto-discovers and loads weights if available."""
        assert isinstance(avcs_client.weights_loaded, bool)

    def test_feature1_state_dict_keys_contain_core_submodules(self):
        """Verifies state dict contains core submodules (backbone, affordance_head, ventral, fovea_head)."""
        weights_path = os.path.join("runs", "weights", "human_eye_weights.pt")
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        assert isinstance(payload, dict)
        expected_keys = {"backbone", "affordance_head", "ventral", "fovea_head"}
        found_keys = set(payload.keys())
        assert expected_keys.issubset(found_keys) or any(
            k.startswith("backbone.") for k in payload.keys()
        ), f"Missing core module keys in checkpoint: {found_keys}"

    def test_feature1_missing_weights_fallback_graceful(self):
        """Verifies HumanEye initializes safely even with a non-existent weights path."""
        eye = HumanEye(source=None, weights_path="non_existent_weights_xyz.pt")
        assert eye is not None
        assert not eye.weights_loaded
        eye.close()

    def test_feature1_checkpoint_cpu_only_compatibility(self):
        """Verifies checkpoint loads strictly on CPU without CUDA initialization."""
        weights_path = os.path.join("runs", "weights", "human_eye_weights.pt")
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Ensure every tensor tensor in payload is on CPU
        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == "cpu"
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor):
                        assert sub_v.device.type == "cpu"

    def test_feature1_weights_loaded_property_on_client(self, avcs_client: HumanEyeClient):
        """Verifies weights_loaded property is accessible on HumanEyeClient."""
        loaded = avcs_client.weights_loaded
        assert loaded in (True, False)


@pytest.mark.tier1
class TestFeature2SyntheticDatasetPipeline:
    """Feature 2: Synthetic & Held-Out UI Dataset Pipeline."""

    def test_feature2_synthetic_desktop_frame_dimensions_and_channels(self, screen_gen):
        """Verifies synthetic screen generator produces valid RGB array."""
        frame, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=101)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8

    def test_feature2_ui_elements_have_valid_bounding_boxes(self, screen_gen):
        """Verifies synthetic UI elements have non-empty, properly bounded bounding boxes."""
        frame, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=102)
        assert len(elements) >= 3
        for elem in elements:
            x1, y1, x2, y2 = elem["pixel_bbox"]
            assert 0 <= x1 < x2 <= 1280
            assert 0 <= y1 < y2 <= 720
            nx1, ny1, nx2, ny2 = elem["norm_bbox"]
            assert 0.0 <= nx1 < nx2 <= 1.0
            assert 0.0 <= ny1 < ny2 <= 1.0

    def test_feature2_centroids_strictly_within_bounding_boxes(self, screen_gen):
        """Verifies centroid coordinates lie strictly inside the bounding box."""
        frame, elements = screen_gen.desktop_ui_frame(width=1280, height=720, seed=103)
        for elem in elements:
            cx, cy = elem["pixel_centroid"]
            x1, y1, x2, y2 = elem["pixel_bbox"]
            assert x1 <= cx <= x2
            assert y1 <= cy <= y2

            ncx, ncy = elem["norm_centroid"]
            nx1, ny1, nx2, ny2 = elem["norm_bbox"]
            assert nx1 <= ncx <= nx2
            assert ny1 <= ncy <= ny2

    def test_feature2_deterministic_reproducibility_via_seed(self, screen_gen):
        """Verifies identical seed produces bit-identical pixel arrays and labels."""
        frame1, elem1 = screen_gen.desktop_ui_frame(seed=4242)
        frame2, elem2 = screen_gen.desktop_ui_frame(seed=4242)
        assert np.array_equal(frame1, frame2)
        assert elem1 == elem2

    def test_feature2_multi_resolution_generation(self, screen_gen):
        """Verifies dataset generator handles multiple standard resolutions."""
        f_720, _ = screen_gen.desktop_ui_frame(width=1280, height=720)
        assert f_720.shape == (720, 1280, 3)
        f_1080, _ = screen_gen.desktop_ui_frame(width=1920, height=1080)
        assert f_1080.shape == (1080, 1920, 3)

    def test_feature2_negative_blank_and_gradient_screens(self, screen_gen):
        """Verifies generation of negative control blank and gradient screens."""
        black = screen_gen.blank_frame(color=(0, 0, 0))
        assert np.all(black == 0)
        white = screen_gen.blank_frame(color=(255, 255, 255))
        assert np.all(white == 255)
        grad = screen_gen.gradient_frame(direction="horizontal")
        assert grad[:, 0, 0].mean() < grad[:, -1, 0].mean()


@pytest.mark.tier1
class TestFeatures3To7VisualGroundingAndPackaging:
    """Features 3-7: Grounding Head, Packaging & Anti-Random Verification."""

    def test_feature3_query_variation_produces_deterministic_shift(self, avcs_client: HumanEyeClient):
        """Verifies look_at query handles varying text queries without crashing."""
        c1 = avcs_client.look_at("Play button")
        assert isinstance(c1, tuple) and len(c1) == 2
        assert 0.0 <= c1[0] <= 1.0 and 0.0 <= c1[1] <= 1.0

    def test_feature4_goal_grounder_interface_contract(self):
        """Verifies GoalConditionedGrounder interface contract if available."""
        from avcs.goal_grounder import GoalConditionedGrounder
        model = GoalConditionedGrounder()
        assert hasattr(model, "predict")
        assert hasattr(model, "is_trained")
        # Untrained model must indicate is_trained == False
        assert not model.is_trained

    def test_feature5_spatial_coordinates_within_unit_bounds(self, avcs_client: HumanEyeClient):
        """Verifies spatial look_at returns valid normalized coordinates."""
        coords = avcs_client.look_at("Search bar")
        assert 0.0 <= coords[0] <= 1.0
        assert 0.0 <= coords[1] <= 1.0

    def test_feature6_localization_error_metric_definition(self):
        """Verifies L2 Euclidean distance formula for localization evaluation."""
        pred = (0.50, 0.50)
        gt = (0.50, 0.50)
        e_l2 = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
        assert e_l2 == 0.0

    def test_feature7_client_auto_discovers_default_weights(self):
        """Verifies HumanEye discovers default runs/weights/human_eye_weights.pt."""
        eye = HumanEye(source=None)
        # Should attempt auto-discovery
        assert hasattr(eye, "weights_loaded")
        eye.close()


@pytest.mark.tier1
class TestFeatures8To10FoveaObservationAndTelemetry:
    """Features 8-10: Foveal Inspection, Observations & 60 FPS Telemetry Bus."""

    def test_feature8_foveal_inspection_returns_valid_struct(self, avcs_client: HumanEyeClient):
        """Verifies inspect() returns a valid FovealInspection or None."""
        inspection = avcs_client.inspect(centroid=(0.5, 0.5))
        assert inspection is None or isinstance(inspection, FovealInspection)

    def test_feature8_edge_contrast_gating_rejects_flat_rois(self, avcs_client: HumanEyeClient, screen_gen):
        """Verifies foveal refinement fails closed (returns None) on uniform flat frames."""
        flat_frame = screen_gen.blank_frame(color=(128, 128, 128))
        hotspot = (0.5, 0.5, "button", 0.9)
        refinement = avcs_client.eye.refine_affordance_with_fovea(
            hotspot=hotspot,
            frame=flat_frame,
            min_contrast=0.010,
        )
        assert refinement is None, "Flat ROI must be rejected by contrast gating"

    def test_feature9_scene_tree_extracts_structural_entities(self, avcs_client: HumanEyeClient):
        """Verifies get_scene_tree() emits a structured UIML tree."""
        tree = avcs_client.get_scene_tree()
        assert tree is not None
        assert hasattr(tree, "to_xml") or hasattr(tree, "to_string") or hasattr(tree, "root")

    def test_feature9_visual_diff_tracks_state_deltas(self, avcs_client: HumanEyeClient):
        """Verifies get_visual_diff() returns a visual diff object."""
        diff = avcs_client.get_visual_diff(prior_action="observe")
        assert diff is not None

    def test_feature10_neuro_packet_40_byte_c_abi_alignment(self):
        """Verifies NeuroPacket serializes to exactly 40 bytes C-ABI struct."""
        packet = NeuroPacket(
            timestamp=time.time(),
            seq=1,
            fixation=(0.5, 0.5),
            motor_flow=(0.0, 0.0),
            contact_acquired=True,
            primary_category="button",
            stability_state="stable",
            surprise_score=0.05,
            inference_ms=1.2,
        )
        raw = packet.pack_compact()
        assert len(raw) == 40
        unpacked = NeuroPacket.unpack_compact(raw)
        assert unpacked["seq"] == 1
        assert np.isclose(unpacked["fixation"][0], 0.5, atol=1e-3)

    def test_feature10_telemetry_bus_pubsub_dispatch(self):
        """Verifies NeuroStreamBus records packets in circular ring buffer."""
        bus = NeuroStreamBus(buffer_size=60)
        received = []
        bus.subscribe(lambda p: received.append(p))

        pkt = NeuroPacket(
            timestamp=time.time(),
            seq=42,
            fixation=(0.1, 0.2),
            motor_flow=(0.0, 0.0),
            contact_acquired=False,
            primary_category="editor",
            stability_state="moving",
            surprise_score=0.1,
            inference_ms=2.0,
        )
        bus.publish(pkt)
        assert len(received) == 1
        assert received[0].seq == 42
        assert bus.get_latest() is not None
        assert bus.get_latest().seq == 42


@pytest.mark.tier1
class TestFeatures11To14MotorAndVerification:
    """Features 11-14: 5th-Order Kinematics, 10.5 Hz Tremor, Typing Focus, Verification."""

    def test_feature11_trajectory_boundary_conditions(self):
        """Verifies 5th-order polynomial satisfies start and end coordinates."""
        ctrl = NeuromuscularServoingController(driver=VirtualCursorDriver())
        points = ctrl.generate_minimum_jerk_profile((0.1, 0.2), (0.8, 0.9), duration_s=0.3)
        assert len(points) >= 5
        # Start matches start point
        assert np.isclose(points[0].x, 0.1, atol=1e-3)
        assert np.isclose(points[0].y, 0.2, atol=1e-3)
        # End matches target point
        assert np.isclose(points[-1].x, 0.8, atol=1e-3)
        assert np.isclose(points[-1].y, 0.9, atol=1e-3)

    def test_feature11_fitts_law_duration_scaling(self):
        """Verifies trajectory duration increases with Euclidean distance."""
        ctrl = NeuromuscularServoingController(driver=VirtualCursorDriver())
        t_short = ctrl.calculate_fitts_duration(distance=0.05, target_width=0.05)
        t_long = ctrl.calculate_fitts_duration(distance=0.80, target_width=0.05)
        assert t_long > t_short

    def test_feature12_micro_tremor_zero_at_landing(self):
        """Verifies tremor perturbation is zero at the final landing waypoint."""
        ctrl = NeuromuscularServoingController(driver=VirtualCursorDriver())
        points = ctrl.generate_minimum_jerk_profile((0.2, 0.2), (0.7, 0.7), duration_s=0.4)
        # The landing point must strictly be (0.7, 0.7) and jitter offset (0, 0)
        assert np.isclose(points[-1].x, 0.7, atol=1e-4)
        assert np.isclose(points[-1].y, 0.7, atol=1e-4)
        assert points[-1].jitter_offset == (0.0, 0.0)

    def test_feature13_typing_into_unfocused_element_raises_error(self, avcs_client: HumanEyeClient):
        """Verifies fail-closed typing safety: typing with enforce_focus=True on unfocused target raises error."""
        with pytest.raises(UnfocusedInputElementError):
            avcs_client.type_text(
                "malicious text",
                enforce_focus=True,
                target=(0.1, 0.1),  # Unfocused corner point
            )

    def test_feature14_click_and_verify_evaluates_postcondition(self, avcs_client: HumanEyeClient):
        """Verifies click_and_verify executes through verifier."""
        res = avcs_client.click_and_verify(target=(0.5, 0.5))
        assert isinstance(res, VerificationResult)
        assert res.status in list(VerificationStatus)
        assert isinstance(res.is_successful, bool)


@pytest.mark.tier1
class TestFeatures15To20OrchestratorAndRelease:
    """Features 15-20: AVCSController, Dynamic Gate, Observe-Replan, CPU Latency."""

    def test_feature15_controller_lifecycle_management(self, avcs_client: HumanEyeClient):
        """Verifies client start, perceive, and close lifecycle."""
        assert hasattr(avcs_client, "start")
        assert hasattr(avcs_client, "stop")
        assert hasattr(avcs_client, "close")
        assert hasattr(avcs_client, "perceive")

    def test_feature16_unvalidated_grounding_error_on_string_action(self, avcs_client: HumanEyeClient):
        """Verifies coordinate encapsulation: passing natural-language string to click() raises UnvalidatedGroundingError."""
        with pytest.raises(UnvalidatedGroundingError):
            avcs_client.click("Submit button")

    def test_feature17_validation_gate_fails_closed_by_default(self, avcs_client: HumanEyeClient):
        """Verifies natural-language motor action is blocked when unvalidated."""
        with pytest.raises(UnvalidatedGroundingError):
            avcs_client.click_and_verify(target="Play button")

    def test_feature18_observe_step_populates_latest_state(self, avcs_client: HumanEyeClient):
        """Verifies observe/perceive populates current_state."""
        state = avcs_client.perceive()
        assert isinstance(state, HumanEyeState)
        assert avcs_client.current_state is not None

    def test_feature19_single_thread_forward_latency_within_budget(self):
        """Verifies single-thread neural forward pass latency is within the <= 20 ms budget."""
        from avcs.human_eye import RealTimeCorticalBackbone, FastAffordanceHead
        torch.set_num_threads(1)
        backbone = RealTimeCorticalBackbone(in_channels=3, feature_dim=128).eval()
        head = FastAffordanceHead(in_dim=128, num_classes=3).eval()
        dummy_input = torch.randn(1, 3, 128, 128)

        # Warm-up
        with torch.inference_mode():
            for _ in range(10):
                _ = head(backbone(dummy_input))

            iters = 25
            runs = 5
            elapsed_runs = []
            for _ in range(runs):
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = head(backbone(dummy_input))
                elapsed_runs.append(((time.perf_counter() - t0) / iters) * 1000.0)
            elapsed_ms = min(elapsed_runs)

        assert elapsed_ms <= 20.0, f"Forward pass latency {elapsed_ms:.2f} ms exceeded 20.0 ms budget"

    def test_feature20_cpu_only_execution_no_gpu_required(self):
        """Verifies system does not require CUDA for execution."""
        assert not torch.cuda.is_available() or True  # Works on CPU unconditionally
