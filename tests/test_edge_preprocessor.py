"""Comprehensive unit tests and benchmarks for 4-Channel Edge & Theme Preprocessor.

Verifies:
  1. Mathematical edge invariance: max |M(X) - M(1 - X)| < 1e-5.
  2. Thresholded edge map IoU: IoU(E(X), E(1 - X)) >= 0.90 across UI samples.
  3. Output tensor dimensions [B, 4, 128, 128] and value bounds.
  4. Single-thread and 4-thread CPU forward latency benchmarks (p50 <= 1.0 ms).
  5. Directional gradient orientation and reflection padding boundary integrity.
  6. Zero learned parameters and stateless deterministic execution.
  7. Vectorized theme augmentations (photometric inversion, brightness, contrast, gamma, composite).
  8. Fail-closed input validation on malformed inputs.
"""

import time
import pytest
import torch
import torch.nn.functional as F

from avcs.perception.edge_preprocessor import (
    EdgeLuminancePreprocessor,
    ThemeAugmentation,
    brightness_jitter,
    compute_edge_iou,
    compute_edge_map,
    contrast_jitter,
    gamma_jitter,
    photometric_inversion,
)


def _generate_synthetic_ui_frame(
    bg_color: float = 0.95,
    btn_color: float = 0.20,
    text_color: float = 0.05,
    size: int = 128,
) -> torch.Tensor:
    """Generate a realistic synthetic UI frame [1, 3, size, size] with buttons and text lines."""
    frame = torch.full((1, 3, size, size), bg_color, dtype=torch.float32)

    # Title bar / header
    frame[:, :, 0:16, :] = 0.85

    # Button 1: (top 30:50, left 20:60)
    frame[:, :, 30:50, 20:60] = btn_color
    # Button text line inside button 1
    frame[:, :, 38:42, 28:52] = 0.95

    # Button 2: (top 30:50, left 70:110)
    frame[:, :, 30:50, 70:110] = 0.40

    # Text input field: (top 65:85, left 20:108) with darker border
    frame[:, :, 65:85, 20:108] = 1.0
    frame[:, :, 65:67, 20:108] = 0.50  # top border
    frame[:, :, 83:85, 20:108] = 0.50  # bottom border
    frame[:, :, 65:85, 20:22] = 0.50  # left border
    frame[:, :, 65:85, 106:108] = 0.50  # right border

    # Text line inside input field
    frame[:, :, 73:77, 26:60] = text_color

    # Horizontal divider rule
    frame[:, :, 95:97, 10:118] = 0.70

    # Small icon / checkbox at bottom
    frame[:, :, 105:117, 20:32] = 0.30
    frame[:, :, 107:115, 22:30] = 0.95

    return frame


# ==============================================================================
# 1. Mathematical Invariance Tests
# ==============================================================================


class TestEdgeMathematicalInvariance:
    """Tests verifying max |M(X) - M(1 - X)| < 1e-5."""

    @pytest.mark.parametrize("seed", [42, 101, 2026, 777])
    def test_random_uniform_invariance(self, seed: int):
        torch.manual_seed(seed)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        x = torch.rand(4, 3, 128, 128, dtype=torch.float32)
        x_inv = 1.0 - x

        out_orig = preprocessor(x)
        out_inv = preprocessor(x_inv)

        mag_orig = out_orig[:, 3:4]
        mag_inv = out_inv[:, 3:4]

        max_diff = torch.max(torch.abs(mag_orig - mag_inv)).item()
        assert max_diff < 1e-5, f"Gradient magnitude diff {max_diff:.8e} exceeded 1e-5 limit"

    def test_synthetic_ui_invariance(self):
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        x = _generate_synthetic_ui_frame()
        x_inv = 1.0 - x

        mag_orig = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        max_diff = torch.max(torch.abs(mag_orig - mag_inv)).item()
        assert max_diff < 1e-5, f"UI frame gradient magnitude diff {max_diff:.8e} exceeded 1e-5 limit"

    def test_flat_and_extreme_screens_invariance(self):
        """Invariance on pure black, pure white, and high-contrast checkerboard."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # Pure black and pure white
        black = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        white = torch.ones(1, 3, 128, 128, dtype=torch.float32)
        mag_black = preprocessor.compute_gradient_magnitude(black)
        mag_white = preprocessor.compute_gradient_magnitude(white)
        diff_bw = torch.max(torch.abs(mag_black - mag_white)).item()
        assert diff_bw < 1e-5

        # Checkerboard pattern
        checker = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        checker[:, :, 0::2, 0::2] = 1.0
        checker[:, :, 1::2, 1::2] = 1.0
        checker_inv = 1.0 - checker

        mag_chk = preprocessor.compute_gradient_magnitude(checker)
        mag_chk_inv = preprocessor.compute_gradient_magnitude(checker_inv)
        diff_chk = torch.max(torch.abs(mag_chk - mag_chk_inv)).item()
        assert diff_chk < 1e-5


# ==============================================================================
# 2. Thresholded Edge Map IoU Tests
# ==============================================================================


class TestEdgeMapIoU:
    """Tests verifying thresholded edge map IoU >= 0.90."""

    def test_edge_iou_on_ui_inversion(self):
        x = _generate_synthetic_ui_frame()
        x_inv = 1.0 - x

        edge_orig = compute_edge_map(x, threshold=0.05)
        edge_inv = compute_edge_map(x_inv, threshold=0.05)

        iou = compute_edge_iou(edge_orig, edge_inv)
        assert iou >= 0.90, f"Edge map IoU {iou:.4f} is below 0.90 target"
        # In fact, with mathematical invariance within 1e-6, IoU should be virtually 1.0
        assert iou >= 0.999, f"Exact inversion edge IoU should be ~1.0, got {iou:.6f}"

    def test_edge_iou_under_photometric_jitter(self):
        """Tests that edge IoU remains robust (>= 0.90) under moderate contrast/brightness shifts."""
        torch.manual_seed(42)
        x = _generate_synthetic_ui_frame()
        # Light theme vs darkened theme (brightness -10%, contrast +10%)
        x_darkened = torch.clamp((x - 0.1) * 1.05, 0.0, 1.0)

        edge_a = compute_edge_map(x, threshold=0.04)
        edge_b = compute_edge_map(x_darkened, threshold=0.04)

        iou = compute_edge_iou(edge_a, edge_b)
        assert iou >= 0.90, f"Edge IoU under illumination shift is {iou:.4f}, expected >= 0.90"

    def test_edge_iou_empty_blank_screens(self):
        blank_a = torch.zeros(1, 3, 128, 128)
        blank_b = torch.ones(1, 3, 128, 128)
        edge_a = compute_edge_map(blank_a, threshold=0.05)
        edge_b = compute_edge_map(blank_b, threshold=0.05)
        iou = compute_edge_iou(edge_a, edge_b)
        assert iou == 1.0, "Empty edge maps must yield IoU 1.0"


# ==============================================================================
# 3. Output Dimensions & Value Bounds Tests
# ==============================================================================


class TestOutputDimensionsAndBounds:
    """Tests verifying output tensor shapes [B, 4, 128, 128] and value bounds."""

    def test_batch_dimensions(self):
        preprocessor = EdgeLuminancePreprocessor()
        for b in [1, 2, 4, 8]:
            inp = torch.rand(b, 3, 128, 128, dtype=torch.float32)
            out = preprocessor(inp)
            assert out.shape == (b, 4, 128, 128)

    def test_single_image_3d_support(self):
        preprocessor = EdgeLuminancePreprocessor()
        inp = torch.rand(3, 128, 128, dtype=torch.float32)
        out = preprocessor(inp)
        assert out.shape == (4, 128, 128)

    def test_channel_value_bounds_default(self):
        """In default mode:
        Channel 0 (Luminance) in [0.0, 1.0]
        Channel 1 (Sobel X) in [-0.5, 0.5]
        Channel 2 (Sobel Y) in [-0.5, 0.5]
        Channel 3 (Gradient Magnitude) in [0.0, 1.0]
        """
        torch.manual_seed(42)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6, normalize_to_unit=False)
        inp = torch.rand(8, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(inp)

        lum = out[:, 0]
        gx = out[:, 1]
        gy = out[:, 2]
        mag = out[:, 3]

        # Luminance bounds
        assert lum.min() >= -1e-6
        assert lum.max() <= 1.0 + 1e-6

        # Sobel directional bounds
        assert gx.min() >= -0.5 - 1e-6
        assert gx.max() <= 0.5 + 1e-6
        assert gy.min() >= -0.5 - 1e-6
        assert gy.max() <= 0.5 + 1e-6

        # Gradient magnitude bounds
        assert mag.min() >= 0.0
        assert mag.max() <= 1.0 + 1e-6

        # Overall tensor is strictly bounded in [-0.5, 1.0]
        assert out.min() >= -0.5 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_channel_value_bounds_normalized_to_unit(self):
        """When normalize_to_unit=True: all 4 channels are strictly in [0.0, 1.0]."""
        torch.manual_seed(42)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6, normalize_to_unit=True)
        inp = torch.rand(8, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(inp)

        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6


# ==============================================================================
# 4. Latency Benchmark Tests (1 CPU thread & 4 CPU threads)
# ==============================================================================


class TestLatencyBenchmarks:
    """Measures CPU forward latency (20 warmup, 50 iterations), asserting p50 <= 1.0 ms."""

    def test_forward_latency_single_thread(self):
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
            inp = torch.rand(1, 3, 128, 128, dtype=torch.float32)

            # Warmup: 20 cycles
            for _ in range(20):
                _ = preprocessor(inp)

            # Benchmark: 50 iterations
            timings = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = preprocessor(inp)
                timings.append((time.perf_counter() - t0) * 1000.0)

            timings.sort()
            p50 = timings[len(timings) // 2]
            min_t = timings[0]
            p95 = timings[int(len(timings) * 0.95)]

            print(f"\n[CPU 1-Thread Benchmark] min: {min_t:.3f} ms, p50: {p50:.3f} ms, p95: {p95:.3f} ms")
            assert p50 <= 1.0, f"1-Thread CPU latency p50 ({p50:.3f} ms) exceeded 1.0 ms threshold"
        finally:
            torch.set_num_threads(orig_threads)

    def test_forward_latency_four_threads(self):
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
            inp = torch.rand(1, 3, 128, 128, dtype=torch.float32)

            # Warmup: 20 cycles
            for _ in range(20):
                _ = preprocessor(inp)

            # Benchmark: 50 iterations
            timings = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = preprocessor(inp)
                timings.append((time.perf_counter() - t0) * 1000.0)

            timings.sort()
            p50 = timings[len(timings) // 2]
            min_t = timings[0]
            p95 = timings[int(len(timings) * 0.95)]

            print(f"\n[CPU 4-Thread Benchmark] min: {min_t:.3f} ms, p50: {p50:.3f} ms, p95: {p95:.3f} ms")
            assert p50 <= 1.0, f"4-Thread CPU latency p50 ({p50:.3f} ms) exceeded 1.0 ms threshold"
        finally:
            torch.set_num_threads(orig_threads)


# ==============================================================================
# 5. Reflection Padding and Sobel Filter Properties
# ==============================================================================


class TestSobelPropertiesAndPadding:
    """Tests reflection padding boundary behavior and directional filter responses."""

    def test_reflection_padding_prevents_border_artifacts(self):
        """On a uniform white screen, reflection padding must yield near-zero edge response at boundaries."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        white = torch.ones(1, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(white)

        gx = out[:, 1]
        gy = out[:, 2]
        mag = out[:, 3]

        # Gradients on a flat screen must be 0 everywhere, including borders
        assert torch.max(torch.abs(gx)).item() < 1e-6
        assert torch.max(torch.abs(gy)).item() < 1e-6
        # Magnitude is sqrt(0 + 0 + 1e-6) = 1e-3
        assert torch.allclose(mag, torch.tensor(1e-3), atol=1e-5)

    def test_directional_horizontal_and_vertical_edges(self):
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # Pure vertical split (horizontal gradient): Left half 0.0, Right half 1.0
        v_split = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        v_split[:, :, :, 64:] = 1.0
        out_v = preprocessor(v_split)
        gx_v = out_v[:, 1, :, :]
        gy_v = out_v[:, 2, :, :]

        # Gx must have positive response at boundary (col 63-64), Gy must be zero everywhere
        assert gx_v[:, :, 63:65].abs().max() > 0.20
        assert gy_v.abs().max() < 1e-6

        # Pure horizontal split (vertical gradient): Top half 0.0, Bottom half 1.0
        h_split = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        h_split[:, :, 64:, :] = 1.0
        out_h = preprocessor(h_split)
        gx_h = out_h[:, 1, :, :]
        gy_h = out_h[:, 2, :, :]

        assert gy_h[:, 63:65, :].abs().max() > 0.20
        assert gx_h.abs().max() < 1e-6


# ==============================================================================
# 6. Zero Parameters and Statelessness Tests
# ==============================================================================


class TestStatelessnessAndNoLearnedWeights:
    """Verifies preprocessor has zero trainable parameters and maintains no mutable state."""

    def test_zero_parameters(self):
        preprocessor = EdgeLuminancePreprocessor()
        params = list(preprocessor.parameters())
        assert len(params) == 0, f"Expected 0 parameters, found {len(params)}"

    def test_registered_buffer(self):
        preprocessor = EdgeLuminancePreprocessor()
        buffers = dict(preprocessor.named_buffers())
        assert "sobel_weights" in buffers
        assert buffers["sobel_weights"].shape == (2, 1, 3, 3)

    def test_stateless_determinism(self):
        preprocessor = EdgeLuminancePreprocessor()
        x = torch.rand(2, 3, 128, 128)
        out1 = preprocessor(x)
        out2 = preprocessor(x)
        assert torch.equal(out1, out2), "Preprocessor must be strictly deterministic"


# ==============================================================================
# 7. Theme Augmentations Suite Tests
# ==============================================================================


class TestThemeAugmentations:
    """Verifies vectorized photometric and theme augmentation suite."""

    def test_photometric_inversion_certain(self):
        x = torch.rand(4, 3, 64, 64)
        inv = photometric_inversion(x, p=1.0)
        assert torch.allclose(inv, 1.0 - x, atol=1e-6)

        no_inv = photometric_inversion(x, p=0.0)
        assert torch.equal(no_inv, x)

    def test_photometric_inversion_probability(self):
        torch.manual_seed(42)
        x = torch.rand(100, 3, 16, 16)
        inv = photometric_inversion(x, p=0.5)
        # Verify that roughly 50% are inverted
        diffs = (inv - x).abs().mean(dim=(1, 2, 3))
        inverted_count = (diffs > 0.01).sum().item()
        assert 30 <= inverted_count <= 70, f"Expected ~50 inverted, got {inverted_count}"

    def test_brightness_jitter_bounds(self):
        torch.manual_seed(42)
        x = torch.full((4, 3, 32, 32), 0.5)
        jittered = brightness_jitter(x, factor_range=(-0.2, 0.2))
        assert jittered.min() >= 0.3 - 1e-5
        assert jittered.max() <= 0.7 + 1e-5
        # Verify edge clamping on extreme input
        x_extreme = torch.full((4, 3, 32, 32), 0.95)
        jittered_extreme = brightness_jitter(x_extreme, factor_range=(0.1, 0.2))
        assert jittered_extreme.max() <= 1.0

    def test_contrast_jitter_bounds(self):
        torch.manual_seed(42)
        x = torch.rand(4, 3, 32, 32)
        jittered = contrast_jitter(x, factor_range=(0.8, 1.2))
        assert jittered.min() >= 0.0
        assert jittered.max() <= 1.0

    def test_gamma_jitter_bounds(self):
        torch.manual_seed(42)
        x = torch.rand(4, 3, 32, 32)
        jittered = gamma_jitter(x, gamma_range=(0.8, 1.2))
        assert jittered.min() >= 0.0
        assert jittered.max() <= 1.0

    def test_theme_augmentation_composite(self):
        torch.manual_seed(42)
        aug = ThemeAugmentation(p_invert=0.5, p_jitter=0.5)
        x = torch.rand(8, 3, 128, 128)
        out = aug(x)
        assert out.shape == x.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ==============================================================================
# 8. Fail-Closed Input Validation Tests
# ==============================================================================


class TestInputValidationFailClosed:
    """Verifies that invalid inputs raise descriptive ValueErrors fail-closed."""

    def test_invalid_dimension_raises(self):
        preprocessor = EdgeLuminancePreprocessor()
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            preprocessor(torch.rand(2, 3))
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            preprocessor(torch.rand(1, 1, 3, 128, 128))

    def test_invalid_channels_raises(self):
        preprocessor = EdgeLuminancePreprocessor()
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor(torch.rand(1, 1, 128, 128))
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor(torch.rand(1, 4, 128, 128))
