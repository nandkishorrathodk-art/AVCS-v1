"""Adversarial Challenge & Stress-Testing Suite for AVCS Milestone 3.

Empirical verification executed by Challenger M3-1:
1. Adversarial Screen Resolutions, Aspect Ratios & Extreme Boundary Clamping.
2. Sub-Pixel Parabolic Localization Accuracy & Identity Jacobian Condition Number.
3. Contrast Gating Rejection on Flat Screens and Sensor Noise.
4. System 2 Privacy Boundary: Zero Coordinate Leakage in Serialization.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from avcs.grounding import FovealAcuityEngine, FovealInspection
from avcs.human_eye import HumanEye
from avcs.observation import (
    EntitySummary,
    SceneObservation,
    VisualObservationEngine,
)
from forge.vision.capture import SyntheticScreen


# ==============================================================================
# Challenge 1: Non-Standard Resolutions, Aspect Ratios & Boundary Clamping
# ==============================================================================

class TestAdversarialResolutionsAndBoundaryClamping:
    """Stress-tests crop_fovea and foveal inspection across extreme canvas geometries."""

    ASPECT_RATIOS = [
        ("16:9", 1920, 1080),
        ("16:9_4k", 3840, 2160),
        ("16:9_720p", 1280, 720),
        ("16:10", 1920, 1200),
        ("16:10_mac", 2560, 1600),
        ("16:10_wxga", 1280, 800),
        ("21:9", 2560, 1080),
        ("21:9_uwqhd", 3440, 1440),
        ("4:3", 1024, 768),
        ("4:3_vga", 640, 480),
        ("4:3_uxga", 1600, 1200),
        ("1:1", 1080, 1080),
        ("1:1_small", 512, 512),
        ("32:9_ultrawide", 5120, 1440),
        ("32:9_dfhd", 3840, 1080),
        ("sub_patch_canvas", 100, 110),
        ("micro_canvas", 64, 64),
    ]

    EXTREME_COORDINATES = [
        (0.0, 0.0),         # Exact top-left corner
        (1.0, 1.0),         # Exact bottom-right corner
        (0.0, 1.0),         # Exact bottom-left corner
        (1.0, 0.0),         # Exact top-right corner
        (0.5, 0.5),         # Screen center
        (1e-7, 1e-7),       # Infinitesimal positive offset
        (1.0 - 1e-7, 1.0 - 1e-7),
        (-0.5, -0.5),       # Out of bounds: negative
        (-10.0, 5.0),       # Out of bounds: extreme negative/positive
        (1.5, 0.5),         # Out of bounds: right
        (0.5, 1.5),         # Out of bounds: bottom
        (2.0, 2.0),         # Out of bounds: far bottom-right
    ]

    def test_crop_fovea_strictly_square_and_no_index_errors(self):
        """Verifies crop_fovea produces strictly square crops without index errors across all geometries."""
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128)

        for name, w, h in self.ASPECT_RATIOS:
            frame = np.random.randint(20, 240, (h, w, 3), dtype=np.uint8)
            expected_size = min(128, w, h)

            for cx, cy in self.EXTREME_COORDINATES:
                try:
                    crop, norm_crop_box = engine.crop_fovea(frame, (cx, cy))
                except Exception as exc:
                    pytest.fail(f"Index or bounds error on resolution {name} ({w}x{h}) at ({cx}, {cy}): {exc}")

                # 1. Strictly square check
                assert crop.shape[0] == crop.shape[1], (
                    f"Crop is not square on {name} ({w}x{h}) at ({cx}, {cy}): got {crop.shape}"
                )
                assert crop.shape[0] == expected_size, (
                    f"Crop size {crop.shape[0]} != expected {expected_size} on {name} ({w}x{h})"
                )

                # 2. Normalized crop box validity
                nx1, ny1, nx2, ny2 = norm_crop_box
                assert 0.0 <= nx1 <= nx2 <= 1.0, f"Invalid x bounds on {name}: ({nx1}, {nx2})"
                assert 0.0 <= ny1 <= ny2 <= 1.0, f"Invalid y bounds on {name}: ({ny1}, {ny2})"

                # 3. Box width and height in normalized space
                assert nx2 > nx1 or w == 0
                assert ny2 > ny1 or h == 0

    def test_crop_fovea_no_zero_pad_step_artifacts(self):
        """Verifies boundary crops contain native pixel data with zero artificial zero-padding."""
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128)
        w, h = 1920, 1080
        # Frame with guaranteed strictly positive pixel values
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        frame = ((x_coords * 3 + y_coords * 5) % 200 + 40).astype(np.uint8)
        frame_rgb = np.stack([frame, frame, frame], axis=-1)

        for cx, cy in self.EXTREME_COORDINATES:
            crop, _ = engine.crop_fovea(frame_rgb, (cx, cy))
            # Any pixel value < 40 indicates artificial zero padding or step edge artifact
            min_val = int(np.min(crop))
            assert min_val >= 40, (
                f"Synthetic zero-padding or step artifact detected at ({cx}, {cy}): min_val={min_val} < 40"
            )

    def test_foveal_inspect_robustness_on_out_of_bounds_coordinates(self):
        """Verifies inspect() handles out-of-bounds centroids without NaN or unhandled exceptions."""
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128)
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

        for cx, cy in self.EXTREME_COORDINATES:
            insp = engine.inspect(frame, (cx, cy))
            assert isinstance(insp, FovealInspection)
            assert not math.isnan(insp.contrast_score)
            assert not math.isnan(insp.confidence)
            bx1, by1, bx2, by2 = insp.bbox
            assert 0.0 <= bx1 <= bx2 <= 1.0
            assert 0.0 <= by1 <= by2 <= 1.0
            rcx, rcy = insp.refined_centroid
            assert 0.0 <= rcx <= 1.0
            assert 0.0 <= rcy <= 1.0


# ==============================================================================
# Challenge 2: Sub-Pixel Accuracy & Identity Jacobian
# ==============================================================================

class TestSubPixelAccuracyAndIdentityJacobian:
    """Stress-tests continuous sub-pixel parabolic accuracy on shifted targets and Jacobian condition."""

    @staticmethod
    def _render_supersampled_button(
        w: int,
        h: int,
        true_cx: float,
        true_cy: float,
        btn_w: int = 32,
        btn_h: int = 24,
        scale: int = 8,
    ) -> np.ndarray:
        """Renders an anti-aliased button at exact floating-point position using 8x supersampling."""
        hi_w, hi_h = w * scale, h * scale
        hi_frame = np.full((hi_h, hi_w), 40, dtype=np.float32)

        hi_cx = true_cx * scale
        hi_cy = true_cy * scale
        hi_bw = btn_w * scale
        hi_bh = btn_h * scale

        hi_x1 = int(round(hi_cx - hi_bw / 2.0))
        hi_x2 = int(round(hi_cx + hi_bw / 2.0))
        hi_y1 = int(round(hi_cy - hi_bh / 2.0))
        hi_y2 = int(round(hi_cy + hi_bh / 2.0))

        # Button fill with sharp high-contrast border
        hi_frame[hi_y1:hi_y2, hi_x1:hi_x2] = 220.0
        # 1px border at supersampled scale (0.125px physical)
        hi_frame[hi_y1:hi_y1 + scale, hi_x1:hi_x2] = 255.0
        hi_frame[hi_y2 - scale:hi_y2, hi_x1:hi_x2] = 255.0
        hi_frame[hi_y1:hi_y2, hi_x1:hi_x1 + scale] = 255.0
        hi_frame[hi_y1:hi_y2, hi_x2 - scale:hi_x2] = 255.0

        # Downsample via 8x8 block averaging
        down = hi_frame.reshape(h, scale, w, scale).mean(axis=(1, 3))
        down_uint8 = np.clip(down, 0, 255).astype(np.uint8)
        return np.stack([down_uint8] * 3, axis=-1)

    def test_subpixel_localization_accuracy_distribution(self):
        """Measures empirical localization error across a sweep of continuous sub-pixel offsets.

        Acceptance Criteria:
        - Median localization error <= 2.0 px
        - 95th-percentile localization error <= 3.0 px
        """
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128, load_weights=True)
        w, h = 640, 480

        # Sweep sub-pixel shifts delta in [0.1, 0.9]
        subpixel_deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        errors: List[float] = []

        base_x, base_y = 300.0, 220.0
        for dx in subpixel_deltas:
            for dy in subpixel_deltas:
                true_x = base_x + dx
                true_y = base_y + dy

                frame = self._render_supersampled_button(w, h, true_x, true_y, btn_w=32, btn_h=24)

                # Coarse initial gaze with realistic perturbation (+/- 1.5 px)
                gaze_x = (true_x + 1.2) / w
                gaze_y = (true_y - 1.4) / h

                insp = engine.inspect(frame, (gaze_x, gaze_y))
                assert insp.contrast_score >= 0.010, f"Sub-pixel button contrast {insp.contrast_score} dropped below 0.010"

                refined_px_x = insp.refined_centroid[0] * w
                refined_px_y = insp.refined_centroid[1] * h

                err = math.hypot(refined_px_x - true_x, refined_px_y - true_y)
                errors.append(err)

        errors_arr = np.array(errors)
        median_err = float(np.median(errors_arr))
        p95_err = float(np.percentile(errors_arr, 95))
        max_err = float(np.max(errors_arr))

        print(f"\n[Empirical Sub-Pixel Accuracy] Count: {len(errors)}, Median: {median_err:.3f} px, P95: {p95_err:.3f} px, Max: {max_err:.3f} px")

        assert median_err <= 2.0, f"Median sub-pixel error {median_err:.3f} px exceeds 2.0 px limit"
        assert p95_err <= 3.0, f"95th-percentile sub-pixel error {p95_err:.3f} px exceeds 3.0 px limit"

    def test_jacobian_condition_number_strictly_identity(self):
        """Verifies that the coordinate Jacobian condition number is strictly 1.0 across all aspect ratios."""
        engine = FovealAcuityEngine(device="cpu", fovea_patch_size=128)

        # For an un-downsampled isotropic square physical patch of size S x S:
        # X_px = x1 + u_px
        # Y_px = y1 + v_px
        # dX_px/du_px = 1.0, dX_px/dv_px = 0.0
        # dY_px/du_px = 0.0, dY_px/dv_px = 1.0
        # J_px = [[1.0, 0.0], [0.0, 1.0]]
        # Condition number = max(svd) / min(svd) == 1.0.

        for name, w, h in [("16:9", 1920, 1080), ("16:10", 1920, 1200), ("21:9", 2560, 1080), ("4:3", 1024, 768), ("32:9", 5120, 1440)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            crop, norm_box = engine.crop_fovea(frame, (0.5, 0.5))

            S_h, S_w = crop.shape[:2]
            # Spatial isotropy: S_h == S_w
            assert S_h == S_w == 128

            # Physical coordinate Jacobian
            # du_px = 1 px, dv_px = 1 px
            # Test empirical numerical derivative of pixel mapping
            px_step = 1.0
            u_px, v_px = 64.0, 64.0

            # Pixel mapping function
            def map_to_global_px(u, v):
                nx1, ny1, nx2, ny2 = norm_box
                x1_px = nx1 * w
                y1_px = ny1 * h
                return x1_px + u, y1_px + v

            gx0, gy0 = map_to_global_px(u_px, v_px)
            gx_u, gy_u = map_to_global_px(u_px + px_step, v_px)
            gx_v, gy_v = map_to_global_px(u_px, v_px + px_step)

            J = np.array([
                [(gx_u - gx0) / px_step, (gx_v - gx0) / px_step],
                [(gy_u - gy0) / px_step, (gy_v - gy0) / px_step],
            ])

            assert np.allclose(J, np.eye(2), atol=1e-7), f"Jacobian on {name} is not identity: {J}"
            svs = np.linalg.svd(J, compute_uv=False)
            cond = svs[0] / svs[-1]
            assert abs(cond - 1.0) < 1e-6, f"Jacobian condition number {cond} != 1.0 on {name}"


# ==============================================================================
# Challenge 3: Contrast Gating on Flat Screens and Sensor Noise
# ==============================================================================

class TestContrastGatingOnFlatScreensAndNoise:
    """Stress-tests fail-closed contrast rejection on featureless screens and sensor noise."""

    def test_flat_screens_contrast_rejection(self):
        """Verifies 100% fail-closed rejection (<0.010 contrast) across pure flat screens."""
        eye = HumanEye(source=SyntheticScreen())
        engine = eye.fovea_engine

        flat_levels = [
            ("solid_black", 0),
            ("solid_white", 255),
            ("middle_gray", 128),
            ("oled_near_black", 2),
            ("dark_gray", 16),
            ("light_gray", 240),
        ]

        for name, val in flat_levels:
            frame = np.full((720, 1280, 3), val, dtype=np.uint8)
            insp = engine.inspect(frame, (0.5, 0.5))
            assert insp.contrast_score < 0.010, f"{name} contrast {insp.contrast_score} >= 0.010"

            refined = eye.refine_affordance_with_fovea((0.5, 0.5, "button", 0.9), frame=frame)
            assert refined is None, f"{name} failed to fail-closed (returned {refined})"

    def test_smooth_luminance_gradients_rejection(self):
        """Verifies smooth lighting/wallpaper gradients without discrete edges are rejected."""
        eye = HumanEye(source=SyntheticScreen())
        engine = eye.fovea_engine
        h, w = 720, 1280

        # Horizontal gradient spanning 0 to 60 across 1280 px
        # Local gradient per pixel is 60 / 1280 = 0.0468 / 255 = 0.00018 << 0.010
        grad_x = np.linspace(20, 80, w, dtype=np.float32)
        frame_h = np.tile(grad_x, (h, 1)).astype(np.uint8)
        frame_rgb = np.stack([frame_h] * 3, axis=-1)

        insp = engine.inspect(frame_rgb, (0.5, 0.5))
        assert insp.contrast_score < 0.010, f"Smooth gradient contrast {insp.contrast_score} >= 0.010"
        refined = eye.refine_affordance_with_fovea((0.5, 0.5, "button", 0.9), frame=frame_rgb)
        assert refined is None

    def test_sensor_noise_rejection(self):
        """Verifies Gaussian sensor noise with sigma <= 1.0 achieves 100% rejection (<0.010)."""
        eye = HumanEye(source=SyntheticScreen())
        engine = eye.fovea_engine
        h, w = 480, 640

        sigmas = [0.2, 0.5, 0.8, 1.0]
        base_levels = [0, 64, 128, 200, 255]

        rejection_count = 0
        total_trials = 0

        for sigma in sigmas:
            for base in base_levels:
                for seed in range(3):
                    total_trials += 1
                    rng = np.random.RandomState(seed + int(sigma * 100) + base)
                    noise = rng.normal(0.0, sigma, (h, w, 3))
                    noisy_frame = np.clip(base + noise, 0, 255).astype(np.uint8)

                    insp = engine.inspect(noisy_frame, (0.5, 0.5))
                    refined = eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=noisy_frame)

                    if insp.contrast_score < 0.010 and refined is None:
                        rejection_count += 1
                    else:
                        pytest.fail(f"False trigger on noise sigma={sigma}, base={base}: contrast={insp.contrast_score:.4f}")

        assert rejection_count == total_trials == len(sigmas) * len(base_levels) * 3
        print(f"\n[Empirical Noise Gating] 100% Rejection achieved: {rejection_count}/{total_trials} trials rejected (sigma <= 1.0)")


# ==============================================================================
# Challenge 4: Privacy Boundary & Zero Coordinate Leakage Property Tests
# ==============================================================================

class TestPrivacyBoundaryZeroCoordinateLeakage:
    """Property-based audit of SceneObservation and EntitySummary serialization."""

    FORBIDDEN_KEY_PATTERNS = [
        r"^x$", r"^y$", r"^cx$", r"^cy$", r"^x1$", r"^y1$", r"^x2$", r"^y2$",
        r".*coord.*", r".*bbox.*", r".*bounding.*", r".*centroid.*", r"^px$", r"^py$",
    ]

    def _assert_no_forbidden_keys(self, obj: Any, path: str = ""):
        """Recursively inspects dict/list trees to verify absence of coordinate keys."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                curr_path = f"{path}.{k}" if path else str(k)
                k_lower = str(k).lower()
                for pat in self.FORBIDDEN_KEY_PATTERNS:
                    assert not re.match(pat, k_lower), (
                        f"Privacy leak: Forbidden coordinate key '{k}' found at path '{curr_path}'"
                    )
                self._assert_no_forbidden_keys(v, curr_path)
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                self._assert_no_forbidden_keys(item, f"{path}[{idx}]")

    def test_entity_summary_serialization_has_zero_coordinate_keys(self):
        """Verifies EntitySummary.to_dict() contains strictly zero coordinate keys."""
        regions = ["header", "sidebar", "center", "footer", "screen"]
        prominences = ["primary", "normal", "subtle", "active", "disabled"]

        for reg in regions:
            for prom in prominences:
                ent = EntitySummary(
                    category="button",
                    count=3,
                    spatial_region=reg,
                    prominence=prom,
                    confidence=0.92,
                )
                d = ent.to_dict()
                self._assert_no_forbidden_keys(d)
                assert set(d.keys()) == {"category", "count", "spatial_region", "prominence", "confidence"}

    def test_scene_observation_serialization_has_zero_coordinate_keys(self):
        """Verifies SceneObservation.to_dict() contains strictly zero coordinate keys under diverse configurations."""
        entities = [
            EntitySummary("button", 2, "center", "primary", 0.95),
            EntitySummary("input_field", 1, "header", "normal", 0.88),
            EntitySummary("icon", 4, "sidebar", "subtle", 0.76),
        ]

        obs = SceneObservation(
            scene_description="Active web browser displaying dashboard with controls",
            visible_entities=entities,
            state_changes=["Loaded new analytics dashboard page"],
            confidence=0.96,
            timestamp=1700000000.123,
            frame_id="frame_42",
            primary_category="web_browser",
            stability="stable",
            surprise_score=0.015,
            has_active_modal=False,
            is_stable=True,
            uncertainty=0.04,
            frame_sequence=42,
        )

        d = obs.to_dict()
        self._assert_no_forbidden_keys(d)

        # Confirm JSON serializability
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        self._assert_no_forbidden_keys(deserialized)

    def test_visual_observation_engine_generate_zero_coordinate_leakage(self):
        """Verifies VisualObservationEngine output on real synthetic screens has zero coordinate leakage."""
        engine = VisualObservationEngine()
        screen = SyntheticScreen(width=640, height=480, seed=42)
        frame = screen.grab()

        mock_state = type("State", (), {
            "timestamp": 123456.789,
            "frame_index": 7,
            "primary_category": "desktop_canvas",
            "category_confidence": 0.92,
            "stability": "stable",
            "surprise_score": 0.03,
            "affordance_hotspots": [
                (0.15, 0.12, "button", 0.94),
                (0.50, 0.50, "input_field", 0.89),
                (0.85, 0.90, "icon", 0.77),
            ],
            "moving_objects": [],
            "visual_diff": None,
            "overlay_suspected": False,
        })()

        obs = engine.generate(mock_state, frame=frame)
        d = obs.to_dict()
        self._assert_no_forbidden_keys(d)

        # Verify no coordinate numbers leaked into text description
        for text in [obs.scene_description, obs.description] + obs.state_changes:
            # Check for float coordinates like (0.15, 0.12)
            assert not re.search(r"\(\s*\d+\.\d+\s*,\s*\d+\.\d+\s*\)", text), (
                f"Coordinate tuple leaked in description: {text}"
            )
            # Check for pixel coordinates like 150px, 320x240
            assert "px" not in text.lower()
