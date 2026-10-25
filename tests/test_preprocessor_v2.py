"""Comprehensive unit tests and benchmarks for StreamingVisualCortexV2 Preprocessor.

Verifies:
  1. Input shape (B, 3, 128, 128) -> output shape (B, 4, 128, 128), and 3D (3, 128, 128) support.
  2. Output ranges: I_lum in [0, 1], G_x, G_y in [-1, 1], M in [0, sqrt(2)].
  3. Reflection padding border behavior: no artificial border spike on solid or border-touching patterns.
  4. Numerical stability: finite non-zero gradients on pure black (0,0,0), pure white (1,1,1),
     and mid-gray (0.5,0.5,0.5) without NaN or Inf during forward and backward passes.
  5. Photometric color inversion invariance: max |M(x) - M(1 - x)| < 1e-5 on random and synthetic UI tensors.
  6. CPU forward latency benchmark: 50+ iterations after warmup; assert latency <= 1.5 ms on CPU at 128x128.
  7. Negative test: assert ValueError on non-3-channel or invalid-dimension input.
  8. Architectural integrity: zero trainable parameters, buffer registration, and zero dependencies
     on PixelWordRecognizer.
  9. Vectorized theme augmentations: photometric inversion, brightness, contrast, and gamma jitter.
"""

import sys
import time
from typing import Tuple
import pytest
import torch
import torch.nn.functional as F

from avcs.streaming_v2.preprocessor import (
    EdgeLuminancePreprocessor,
    ThemeAugmentation,
    brightness_jitter,
    compute_edge_iou,
    compute_edge_map,
    contrast_jitter,
    gamma_jitter,
    photometric_inversion,
)


def _build_synthetic_ui_frame(
    bg_color: float = 0.95,
    btn_color: float = 0.20,
    text_color: float = 0.05,
    size: int = 128,
) -> torch.Tensor:
    """Construct a deterministic synthetic UI frame [1, 3, size, size] with standard UI elements."""
    frame = torch.full((1, 3, size, size), bg_color, dtype=torch.float32)

    # Title bar / header
    frame[:, :, 0:16, :] = 0.85

    # Button 1 (top 30:50, left 20:60)
    frame[:, :, 30:50, 20:60] = btn_color
    frame[:, :, 38:42, 28:52] = 0.95

    # Button 2 (top 30:50, left 70:110)
    frame[:, :, 30:50, 70:110] = 0.40

    # Input field (top 65:85, left 20:108) with darker border
    frame[:, :, 65:85, 20:108] = 1.0
    frame[:, :, 65:67, 20:108] = 0.50
    frame[:, :, 83:85, 20:108] = 0.50
    frame[:, :, 65:85, 20:22] = 0.50
    frame[:, :, 65:85, 106:108] = 0.50

    # Text line inside input
    frame[:, :, 73:77, 26:60] = text_color

    # Horizontal divider rule
    frame[:, :, 95:97, 10:118] = 0.70

    # Small toggle / icon at bottom
    frame[:, :, 105:117, 20:32] = 0.30
    frame[:, :, 107:115, 22:30] = 0.95

    return frame


# ==============================================================================
# 1. Input Shape & Dimensionality Tests
# ==============================================================================


class TestInputOutputShapes:
    """Verifies input shape [B, 3, 128, 128] -> output shape [B, 4, 128, 128]."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batched_input_shapes_128x128(self, batch_size: int):
        preprocessor = EdgeLuminancePreprocessor()
        x = torch.rand(batch_size, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(x)
        assert out.shape == (batch_size, 4, 128, 128), (
            f"Expected shape ({batch_size}, 4, 128, 128), got {tuple(out.shape)}"
        )
        assert out.dtype == torch.float32

    def test_unbatched_3d_input_support(self):
        preprocessor = EdgeLuminancePreprocessor()
        x = torch.rand(3, 128, 128, dtype=torch.float32)
        out = preprocessor(x)
        assert out.shape == (4, 128, 128), f"Expected shape (4, 128, 128), got {tuple(out.shape)}"

    @pytest.mark.parametrize("resolution", [64, 128, 256])
    def test_variable_spatial_resolutions(self, resolution: int):
        preprocessor = EdgeLuminancePreprocessor()
        x = torch.rand(2, 3, resolution, resolution, dtype=torch.float32)
        out = preprocessor(x)
        assert out.shape == (2, 4, resolution, resolution)


# ==============================================================================
# 2. Output Ranges & Normalization Tests
# ==============================================================================


class TestOutputRanges:
    """Verifies output ranges: I_lum in [0, 1], G_x, G_y in [-1, 1], M in [0, sqrt(2)]."""

    def test_output_channel_ranges_default(self):
        torch.manual_seed(2026)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6, normalize_to_unit=False)
        x = torch.rand(8, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(x)

        lum = out[:, 0]
        gx = out[:, 1]
        gy = out[:, 2]
        mag = out[:, 3]

        # Channel 0: Grayscale luminance in [0, 1]
        assert lum.min() >= -1e-6, f"Luminance min {lum.min().item()} < 0.0"
        assert lum.max() <= 1.0 + 1e-6, f"Luminance max {lum.max().item()} > 1.0"

        # Channels 1 & 2: Sobel directional gradients in [-1, 1] (and strictly within [-0.5, 0.5])
        assert gx.min() >= -1.0, f"Gx min {gx.min().item()} < -1.0"
        assert gx.max() <= 1.0, f"Gx max {gx.max().item()} > 1.0"
        assert gy.min() >= -1.0, f"Gy min {gy.min().item()} < -1.0"
        assert gy.max() <= 1.0, f"Gy max {gy.max().item()} > 1.0"

        # Specifically with normalized Sobel kernel / 8.0:
        assert gx.min() >= -0.5 - 1e-6
        assert gx.max() <= 0.5 + 1e-6
        assert gy.min() >= -0.5 - 1e-6
        assert gy.max() <= 0.5 + 1e-6

        # Channel 3: Gradient magnitude M in [0, sqrt(2)]
        sqrt_2 = 2.0 ** 0.5
        assert mag.min() >= 0.0, f"Magnitude min {mag.min().item()} < 0.0"
        assert mag.max() <= sqrt_2 + 1e-6, f"Magnitude max {mag.max().item()} > sqrt(2)"
        assert mag.max() <= 1.0 + 1e-6

    def test_output_channel_ranges_normalized_to_unit(self):
        torch.manual_seed(2026)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6, normalize_to_unit=True)
        x = torch.rand(8, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(x)

        assert out.min() >= 0.0, f"Normalized output min {out.min().item()} < 0.0"
        assert out.max() <= 1.0 + 1e-6, f"Normalized output max {out.max().item()} > 1.0"


# ==============================================================================
# 3. Reflection Padding Border Behavior Tests
# ==============================================================================


class TestReflectionPaddingBorderBehavior:
    """Verifies reflection padding eliminates artificial border spikes on solid or boundary patterns."""

    def test_solid_white_no_border_spike(self):
        """Uniform white frame must produce zero gradient response across entire image including borders."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        white = torch.ones(1, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(white)

        gx = out[:, 1]
        gy = out[:, 2]
        mag = out[:, 3]

        assert gx.abs().max().item() < 1e-6, f"Expected 0 Gx on flat white, got {gx.abs().max().item()}"
        assert gy.abs().max().item() < 1e-6, f"Expected 0 Gy on flat white, got {gy.abs().max().item()}"
        # With eps=1e-6, mag = sqrt(0 + 0 + 1e-6) = 1e-3 everywhere
        expected_mag = torch.tensor(1e-3)
        assert torch.allclose(mag, expected_mag, atol=1e-5), "Expected uniform magnitude = 1e-3"

    def test_solid_black_no_border_spike(self):
        """Uniform black frame must produce zero gradient response across entire image including borders."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        black = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        out = preprocessor(black)

        gx = out[:, 1]
        gy = out[:, 2]
        mag = out[:, 3]

        assert gx.abs().max().item() < 1e-6
        assert gy.abs().max().item() < 1e-6
        assert torch.allclose(mag, torch.tensor(1e-3), atol=1e-5)

    def test_solid_mid_gray_no_border_spike(self):
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        gray = torch.full((1, 3, 128, 128), 0.5, dtype=torch.float32)
        out = preprocessor(gray)

        assert out[:, 1].abs().max().item() < 1e-6
        assert out[:, 2].abs().max().item() < 1e-6
        assert torch.allclose(out[:, 3], torch.tensor(1e-3), atol=1e-5)

    def test_border_touching_rectangle_behavior(self):
        """A solid patch touching the screen border has zero gradient along its boundary edge,

        and gradient response only at the interior transition step.
        """
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        # Background 0.0, top-left rectangle [0:32, 0:32] = 1.0 (touches top and left borders)
        x = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        x[:, :, 0:32, 0:32] = 1.0
        out = preprocessor(x)

        gx = out[:, 1]
        gy = out[:, 2]

        # Top-left corner pixel (0, 0) is surrounded by 1.0s via reflection padding:
        # Gradients at (0, 0) should be near 0
        assert gx[:, 0, 0].abs().item() < 1e-6
        assert gy[:, 0, 0].abs().item() < 1e-6

        # Interior step transition at row 31-32 and col 31-32 must have significant gradient
        assert gy[:, 31:33, 10:20].abs().max().item() > 0.20
        assert gx[:, 10:20, 31:33].abs().max().item() > 0.20

    def test_directional_sobel_orthogonality(self):
        """Vertical edge produces non-zero Gx and zero Gy; Horizontal edge produces non-zero Gy and zero Gx."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # Vertical transition: left half 0.0, right half 1.0
        v_edge = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        v_edge[:, :, :, 64:] = 1.0
        out_v = preprocessor(v_edge)
        gx_v = out_v[:, 1]
        gy_v = out_v[:, 2]

        assert gx_v[:, :, 63:65].abs().max().item() > 0.20
        assert gy_v.abs().max().item() < 1e-6

        # Horizontal transition: top half 0.0, bottom half 1.0
        h_edge = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        h_edge[:, :, 64:, :] = 1.0
        out_h = preprocessor(h_edge)
        gx_h = out_h[:, 1]
        gy_h = out_h[:, 2]

        assert gy_h[:, 63:65, :].abs().max().item() > 0.20
        assert gx_h.abs().max().item() < 1e-6


# ==============================================================================
# 4. Numerical Stability & Autograd Tests
# ==============================================================================


class TestNumericalStability:
    """Verifies finite non-zero gradients on pure black (0,0,0), pure white (1,1,1),

    and mid-gray (0.5,0.5,0.5) without NaN or Inf during forward and backward passes.
    """

    @pytest.mark.parametrize("fill_value", [0.0, 1.0, 0.5])
    def test_solid_color_gradients_stability(self, fill_value: float):
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        x = torch.full((2, 3, 128, 128), fill_value, dtype=torch.float32, requires_grad=True)

        out = preprocessor(x)

        # Forward outputs must be finite without NaN or Inf
        assert not torch.isnan(out).any().item(), f"NaN detected in forward output for fill={fill_value}"
        assert not torch.isinf(out).any().item(), f"Inf detected in forward output for fill={fill_value}"

        # Magnitude channel must be strictly positive
        assert (out[:, 3] > 0.0).all().item()

        # Backward autograd pass
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "x.grad is None after backward()"
        assert not torch.isnan(x.grad).any().item(), f"NaN detected in x.grad for fill={fill_value}"
        assert not torch.isinf(x.grad).any().item(), f"Inf detected in x.grad for fill={fill_value}"

        # Gradients must be finite and non-zero across the tensor
        grad_abs_sum = x.grad.abs().sum().item()
        assert grad_abs_sum > 0.0, f"Expected non-zero gradient sum, got {grad_abs_sum}"

    def test_gradient_magnitude_only_backward_pass(self):
        """Specifically verifies backward pass through magnitude channel sqrt(Gx^2 + Gy^2 + eps)."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        # Uniform black with requires_grad
        x_black = torch.zeros(1, 3, 128, 128, dtype=torch.float32, requires_grad=True)
        mag_black = preprocessor.compute_gradient_magnitude(x_black)
        loss_black = mag_black.sum()
        loss_black.backward()

        assert x_black.grad is not None
        assert not torch.isnan(x_black.grad).any().item(), "NaN in x.grad for magnitude-only backward"
        assert not torch.isinf(x_black.grad).any().item(), "Inf in x.grad for magnitude-only backward"

        # Non-uniform image backward pass
        torch.manual_seed(42)
        x_rand = torch.rand(2, 3, 128, 128, dtype=torch.float32, requires_grad=True)
        mag_rand = preprocessor.compute_gradient_magnitude(x_rand)
        loss_rand = mag_rand.sum()
        loss_rand.backward()

        assert x_rand.grad is not None
        assert not torch.isnan(x_rand.grad).any().item()
        assert not torch.isinf(x_rand.grad).any().item()
        assert x_rand.grad.abs().sum().item() > 0.0


# ==============================================================================
# 5. Photometric Color Inversion Invariance Tests
# ==============================================================================


class TestPhotometricInvariance:
    """Verifies max |M(x) - M(1 - x)| < 1e-5 on random and synthetic UI tensors."""

    @pytest.mark.parametrize("seed", [42, 101, 2026, 777, 9999])
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
        x = _build_synthetic_ui_frame()
        x_inv = 1.0 - x

        mag_orig = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        max_diff = torch.max(torch.abs(mag_orig - mag_inv)).item()
        assert max_diff < 1e-5, f"Synthetic UI magnitude diff {max_diff:.8e} exceeded 1e-5 limit"

    def test_high_contrast_checkerboard_invariance(self):
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        checker = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        checker[:, :, 0::4, 0::4] = 1.0
        checker[:, :, 1::4, 1::4] = 1.0
        checker_inv = 1.0 - checker

        mag = preprocessor.compute_gradient_magnitude(checker)
        mag_inv = preprocessor.compute_gradient_magnitude(checker_inv)

        max_diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert max_diff < 1e-5

    def test_thresholded_edge_iou_invariance(self):
        """Thresholded binary edge map IoU on UI frame under inversion must be >= 0.99."""
        x = _build_synthetic_ui_frame()
        x_inv = 1.0 - x

        edge_x = compute_edge_map(x, threshold=0.05)
        edge_inv = compute_edge_map(x_inv, threshold=0.05)

        iou = compute_edge_iou(edge_x, edge_inv)
        assert iou >= 0.99, f"Edge map IoU under inversion was {iou:.6f}, expected >= 0.99"


# ==============================================================================
# 6. CPU Forward Latency Benchmark Tests
# ==============================================================================


class TestCPULatencyBenchmark:
    """Measures 50+ iterations after warmup; asserts latency <= 1.5 ms on CPU at 128x128."""

    def test_cpu_latency_single_thread(self):
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
            inp = torch.rand(1, 3, 128, 128, dtype=torch.float32)

            # Warmup: 20 iterations
            for _ in range(20):
                _ = preprocessor(inp)

            # Benchmark: 60 iterations (exceeding 50+ requirement)
            timings = []
            for _ in range(60):
                t0 = time.perf_counter()
                _ = preprocessor(inp)
                timings.append((time.perf_counter() - t0) * 1000.0)

            timings.sort()
            min_t = timings[0]
            p50 = timings[len(timings) // 2]
            p95 = timings[int(len(timings) * 0.95)]

            print(f"\n[CPU 1-Thread 128x128] min: {min_t:.3f} ms, p50: {p50:.3f} ms, p95: {p95:.3f} ms")
            assert p50 <= 1.5, f"Median CPU latency {p50:.3f} ms exceeded 1.5 ms threshold"
        finally:
            torch.set_num_threads(orig_threads)

    def test_cpu_latency_four_threads(self):
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
            inp = torch.rand(1, 3, 128, 128, dtype=torch.float32)

            # Warmup: 20 iterations
            for _ in range(20):
                _ = preprocessor(inp)

            # Benchmark: 60 iterations
            timings = []
            for _ in range(60):
                t0 = time.perf_counter()
                _ = preprocessor(inp)
                timings.append((time.perf_counter() - t0) * 1000.0)

            timings.sort()
            min_t = timings[0]
            p50 = timings[len(timings) // 2]
            p95 = timings[int(len(timings) * 0.95)]

            print(f"\n[CPU 4-Thread 128x128] min: {min_t:.3f} ms, p50: {p50:.3f} ms, p95: {p95:.3f} ms")
            assert p50 <= 1.5, f"4-Thread median CPU latency {p50:.3f} ms exceeded 1.5 ms threshold"
        finally:
            torch.set_num_threads(orig_threads)


# ==============================================================================
# 7. Fail-Closed Negative Tests
# ==============================================================================


class TestFailClosedValidation:
    """Asserts descriptive ValueError on non-3-channel or non-4D/3D input."""

    def test_negative_invalid_dimensions(self):
        preprocessor = EdgeLuminancePreprocessor()
        # 1D tensor
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            preprocessor(torch.rand(3))
        # 2D tensor
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            preprocessor(torch.rand(128, 128))
        # 5D tensor
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            preprocessor(torch.rand(1, 1, 3, 128, 128))

    def test_negative_invalid_channels(self):
        preprocessor = EdgeLuminancePreprocessor()
        # 1 channel (grayscale)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor(torch.rand(1, 1, 128, 128))
        # 4 channels (RGBA)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor(torch.rand(1, 4, 128, 128))
        # 2 channels
        with pytest.raises(ValueError, match="Expected 3 channels"):
            preprocessor(torch.rand(1, 2, 128, 128))


# ==============================================================================
# 8. Decoupling & Architectural Integrity Tests
# ==============================================================================


class TestDecouplingAndIntegrity:
    """Verifies complete decoupling from PixelWordRecognizer, zero trainable parameters,

    and strict deterministic behavior.
    """

    def test_zero_trainable_parameters(self):
        preprocessor = EdgeLuminancePreprocessor()
        params = list(preprocessor.parameters())
        assert len(params) == 0, f"Expected 0 trainable parameters, found {len(params)}"

    def test_registered_buffer_sobel_weights(self):
        preprocessor = EdgeLuminancePreprocessor()
        buffers = dict(preprocessor.named_buffers())
        assert "sobel_weights" in buffers
        assert buffers["sobel_weights"].shape == (2, 1, 3, 3)
        assert not buffers["sobel_weights"].requires_grad

    def test_strictly_deterministic_execution(self):
        preprocessor = EdgeLuminancePreprocessor()
        x = torch.rand(2, 3, 128, 128)
        out1 = preprocessor(x)
        out2 = preprocessor(x)
        assert torch.equal(out1, out2), "Preprocessor must be bit-for-bit deterministic"

    def test_pixel_word_recognizer_omitted(self):
        """Verify avcs.streaming_v2.preprocessor has zero references or imports to PixelWordRecognizer."""
        import avcs.streaming_v2.preprocessor as mod

        assert not hasattr(mod, "PixelWordRecognizer"), "PixelWordRecognizer found in preprocessor module!"
        # Check source text of the file directly
        import inspect
        source = inspect.getsource(mod)
        assert "import PixelWordRecognizer" not in source
        assert "from" not in source or "word_recognizer" not in source


# ==============================================================================
# 9. Vectorized Theme Augmentation Suite Tests
# ==============================================================================


class TestThemeAugmentations:
    """Verifies photometric inversion, brightness, contrast, gamma jitter, and composite transforms."""

    def test_photometric_inversion_deterministic_cases(self):
        x = torch.rand(4, 3, 64, 64)
        inv = photometric_inversion(x, p=1.0)
        assert torch.allclose(inv, 1.0 - x, atol=1e-6)

        no_inv = photometric_inversion(x, p=0.0)
        assert torch.equal(no_inv, x)

    def test_photometric_inversion_random_ratio(self):
        torch.manual_seed(42)
        x = torch.rand(100, 3, 16, 16)
        inv = photometric_inversion(x, p=0.5)
        diffs = (inv - x).abs().mean(dim=(1, 2, 3))
        inverted_count = (diffs > 0.01).sum().item()
        assert 35 <= inverted_count <= 65, f"Expected ~50 inverted items, got {inverted_count}"

    def test_brightness_jitter_ranges(self):
        torch.manual_seed(42)
        x = torch.full((4, 3, 32, 32), 0.5)
        jittered = brightness_jitter(x, factor_range=(-0.2, 0.2))
        assert jittered.min() >= 0.3 - 1e-5
        assert jittered.max() <= 0.7 + 1e-5

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

    def test_composite_theme_augmentation(self):
        torch.manual_seed(42)
        aug = ThemeAugmentation(p_invert=0.5, p_jitter=0.5)
        x = torch.rand(8, 3, 128, 128)
        out = aug(x)
        assert out.shape == x.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0

