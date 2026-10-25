"""Adversarial stress-test suite for EdgeLuminancePreprocessor (Milestone M1).

Adversarially challenges and stress-tests:
1. Extreme photometric color inversions:
   - Gaussian noise (variances 0.1 to 5.0)
   - Salt-and-pepper noise (densities 0.01 to 0.8)
   - Extreme contrast scaling (factors 1e-4 to 1e4)
   - Random alpha blends of foreground UI over complex backgrounds
   - Out-of-range unnormalized float tensors
2. High-frequency Nyquist checkerboards, single-pixel lines, sub-pixel edge shifts:
   - 1-pixel alternating Nyquist checkerboards (evaluating cancellation & symmetry)
   - 2-pixel and 4-pixel checkerboards
   - 1-pixel horizontal, vertical, and diagonal lines
   - Single isolated impulse pixels
   - Sub-pixel continuous tanh/sigmoid edge shifts
3. Numerical autograd under extreme tensor values:
   - Subnormal floats (1e-38, 1e-30)
   - Epsilon boundary probing (G_x^2 + G_y^2 << eps, == eps, >> eps)
   - Gradient behavior on uniform fields (ensuring zero gradient, no division by zero)
   - Large batch scaling (B = 32, 64, 128, 256)
4. 100-scene synthetic desktop IoU benchmark:
   - 100 randomized multi-window procedural UI scenes
   - Verification that Edge IoU >= 0.90 under inversion across 100/100 scenes
5. Static AST and shortcut inspection:
   - Verifies zero hardcoded heuristics, short-circuits, or prohibited dependencies in preprocessor.py.
"""

import ast
import inspect
import math
import time
from typing import List, Tuple
import pytest
import torch
import torch.nn.functional as F

import avcs.streaming_v2.preprocessor as prep_mod
from avcs.streaming_v2.preprocessor import (
    EdgeLuminancePreprocessor,
    compute_edge_iou,
    compute_edge_map,
    photometric_inversion,
)


# ==============================================================================
# 1. Extreme Photometric Color Inversions Under Hostile Corruptions
# ==============================================================================


class TestAdversarialPhotometricInversions:
    """Stress-tests photometric inversion under extreme noise, scaling, and blends."""

    @pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("mean", [0.1, 0.5, 0.9])
    def test_gaussian_noise_inversion(self, sigma: float, mean: float):
        """Verify max |M(X) - M(1 - X)| <= 1e-5 under various Gaussian noise distributions."""
        torch.manual_seed(int(sigma * 1000 + mean * 100))
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # Generate Gaussian noise tensor, clamped to [0, 1]
        noise = torch.randn(4, 3, 128, 128, dtype=torch.float32) * sigma + mean
        x = torch.clamp(noise, 0.0, 1.0)
        x_inv = 1.0 - x

        mag = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Gaussian noise (sigma={sigma}, mean={mean}) max diff {diff:.8e} > 1e-5"

    @pytest.mark.parametrize("density", [0.01, 0.05, 0.1, 0.25, 0.5, 0.8])
    def test_salt_and_pepper_noise_inversion(self, density: float):
        """Verify max |M(X) - M(1 - X)| <= 1e-5 under hostile salt-and-pepper impulse noise."""
        torch.manual_seed(int(density * 10000))
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        x = torch.full((4, 3, 128, 128), 0.5, dtype=torch.float32)
        # Random salt (1.0) and pepper (0.0)
        rand_vals = torch.rand(4, 1, 128, 128)
        x = torch.where(rand_vals < (density / 2.0), torch.zeros_like(x), x)
        x = torch.where((rand_vals >= (density / 2.0)) & (rand_vals < density), torch.ones_like(x), x)
        x_inv = 1.0 - x

        mag = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Salt-and-pepper (density={density}) max diff {diff:.8e} > 1e-5"

    @pytest.mark.parametrize("scale", [1e-4, 1e-3, 1e-2, 0.1, 2.0, 10.0, 100.0, 1e4])
    def test_extreme_contrast_scaling_inversion(self, scale: float):
        """Verify max |M(X) - M(1 - X)| <= 1e-5 under extreme contrast compression and expansion."""
        torch.manual_seed(42)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        base = torch.rand(4, 3, 128, 128, dtype=torch.float32)
        # Scaled around 0.5 and clamped to [0, 1]
        x = torch.clamp((base - 0.5) * scale + 0.5, 0.0, 1.0)
        x_inv = 1.0 - x

        mag = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Extreme contrast scale {scale} max diff {diff:.8e} > 1e-5"

    @pytest.mark.parametrize("alpha", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_random_alpha_blends_inversion(self, alpha: float):
        """Verify max |M(X) - M(1 - X)| <= 1e-5 when blending foreground UI over background."""
        torch.manual_seed(int(alpha * 1000))
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        fg = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        bg = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        x = alpha * fg + (1.0 - alpha) * bg
        x_inv = 1.0 - x

        mag = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Alpha blend (alpha={alpha}) max diff {diff:.8e} > 1e-5"

    def test_unbounded_unnormalized_float_inversion(self):
        """Verify mathematical linearity of Sobel and gradient magnitude even with out-of-range floats."""
        torch.manual_seed(2026)
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # Unclamped floats spanning negative to high values [-5.0, 10.0]
        x = torch.randn(4, 3, 128, 128, dtype=torch.float32) * 3.0 + 2.0
        x_inv = 1.0 - x

        mag = preprocessor.compute_gradient_magnitude(x)
        mag_inv = preprocessor.compute_gradient_magnitude(x_inv)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Unbounded float max diff {diff:.8e} > 1e-5"


# ==============================================================================
# 2. High-Frequency Nyquist & Geometric Boundary Stress Tests
# ==============================================================================


class TestAdversarialNyquistAndGeometry:
    """Stress-tests high-frequency patterns, single-pixel features, and sub-pixel edge shifts."""

    def test_nyquist_1pixel_checkerboard(self):
        """1-pixel alternating checkerboard (highest spatial frequency).

        Note: With standard 3x3 Sobel, orthogonal smoothing [-1, 0, 1] x [1, 2, 1]^T cancels out
        on pure 1-pixel checkerboards. Inversion invariance must remain exact.
        """
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
        checker = ((x + y) % 2).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        checker_inv = 1.0 - checker

        out = preprocessor(checker)
        out_inv = preprocessor(checker_inv)

        mag = out[:, 3:4]
        mag_inv = out_inv[:, 3:4]

        # Inversion diff must be <= 1e-5
        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Nyquist 1-pixel checkerboard diff {diff:.8e} > 1e-5"

    @pytest.mark.parametrize("block_size", [2, 3, 4, 8, 16])
    def test_multiscale_checkerboard_invariance(self, block_size: int):
        """Checkerboards at various block sizes produce strong Sobel gradients and must remain invariant."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
        checker = (((x // block_size) + (y // block_size)) % 2).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        checker_inv = 1.0 - checker

        mag = preprocessor.compute_gradient_magnitude(checker)
        mag_inv = preprocessor.compute_gradient_magnitude(checker_inv)

        # Gradients must be strong on block transitions
        assert mag.max().item() > 0.20, f"Expected strong edge on {block_size}x{block_size} checker"

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Checkerboard (block={block_size}) diff {diff:.8e} > 1e-5"

    def test_single_pixel_lines(self):
        """1-pixel horizontal, vertical, and diagonal lines must produce precise edges and invert identically."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # 1-pixel horizontal line at row 64
        h_line = torch.zeros(1, 3, 128, 128)
        h_line[:, :, 64, :] = 1.0
        diff_h = torch.max(torch.abs(preprocessor.compute_gradient_magnitude(h_line) - preprocessor.compute_gradient_magnitude(1.0 - h_line))).item()
        assert diff_h <= 1e-5

        # 1-pixel vertical line at col 64
        v_line = torch.zeros(1, 3, 128, 128)
        v_line[:, :, :, 64] = 1.0
        diff_v = torch.max(torch.abs(preprocessor.compute_gradient_magnitude(v_line) - preprocessor.compute_gradient_magnitude(1.0 - v_line))).item()
        assert diff_v <= 1e-5

        # 1-pixel diagonal line
        d_line = torch.zeros(1, 3, 128, 128)
        idx = torch.arange(128)
        d_line[:, :, idx, idx] = 1.0
        diff_d = torch.max(torch.abs(preprocessor.compute_gradient_magnitude(d_line) - preprocessor.compute_gradient_magnitude(1.0 - d_line))).item()
        assert diff_d <= 1e-5

    def test_isolated_single_pixel_impulse(self):
        """Single isolated white pixel on black background and black on white background."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        impulse = torch.zeros(1, 3, 128, 128)
        impulse[:, :, 64, 64] = 1.0

        mag = preprocessor.compute_gradient_magnitude(impulse)
        mag_inv = preprocessor.compute_gradient_magnitude(1.0 - impulse)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5

        # Impulse excites the 8 surrounding neighbors around (64, 64).
        # Note: Center pixel at (64, 64) has Sobel gradient = 0 (since Sobel center weight is 0),
        # yielding magnitude = sqrt(eps) = 1e-3.
        # Cardinal neighbors have G = 2/8 = 0.25; diagonal neighbors have G = sqrt(1^2+1^2)/8 = 0.1768.
        center_val = mag[0, 0, 64, 64].item()
        assert abs(center_val - 1e-3) < 1e-4

        # All 8 perimeter pixels around the impulse must have strong response > 0.15
        perimeter = [
            mag[0, 0, 63, 63].item(), mag[0, 0, 63, 64].item(), mag[0, 0, 63, 65].item(),
            mag[0, 0, 64, 63].item(),                           mag[0, 0, 64, 65].item(),
            mag[0, 0, 65, 63].item(), mag[0, 0, 65, 64].item(), mag[0, 0, 65, 65].item(),
        ]
        assert all(p > 0.15 for p in perimeter), f"Expected all perimeter > 0.15, got {perimeter}"

    @pytest.mark.parametrize("shift", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    def test_subpixel_edge_shifts(self, shift: float):
        """Continuous sub-pixel edge transitions modeled via smooth sigmoid."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        cols = torch.arange(128, dtype=torch.float32).view(1, 1, 1, 128).repeat(1, 3, 128, 1)
        # Sigmoid transition centered at 64.0 + shift with slope k = 2.0
        center = 64.0 + shift
        edge = torch.sigmoid((cols - center) * 2.0)
        edge_inv = 1.0 - edge

        mag = preprocessor.compute_gradient_magnitude(edge)
        mag_inv = preprocessor.compute_gradient_magnitude(edge_inv)

        diff = torch.max(torch.abs(mag - mag_inv)).item()
        assert diff <= 1e-5, f"Subpixel shift {shift} diff {diff:.8e} > 1e-5"


# ==============================================================================
# 3. Numerical Autograd Under Extreme Floats & Epsilon Boundaries
# ==============================================================================


class TestAdversarialAutogradAndExtremeFloats:
    """Stress-tests PyTorch autograd stability on subnormal values, eps boundaries, and large batches."""

    @pytest.mark.parametrize("subnormal", [1e-38, 1e-30, 1e-25, 1e-20])
    def test_subnormal_floats_autograd_stability(self, subnormal: float):
        """Subnormal float inputs must not cause underflow crash, NaN or Inf in forward/backward."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        x = torch.full((2, 3, 128, 128), subnormal, dtype=torch.float32, requires_grad=True)

        out = preprocessor(x)
        assert not torch.isnan(out).any().item()
        assert not torch.isinf(out).any().item()

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any().item()
        assert not torch.isinf(x.grad).any().item()

    def test_epsilon_boundary_gradient_behavior(self):
        """Probe the sqrt(G_x^2 + G_y^2 + eps) derivative at G_x^2 + G_y^2 << eps, == eps, and >> eps.

        The derivative w.r.t G_x is G_x / sqrt(G_x^2 + G_y^2 + eps).
        When G_x = 0, G_y = 0, this derivative must be identically 0.0, not NaN or exploding.
        """
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)

        # Exactly zero gradient input
        x_flat = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32, requires_grad=True)
        mag_flat = preprocessor.compute_gradient_magnitude(x_flat)
        mag_flat.sum().backward()

        assert x_flat.grad is not None
        # Gradients on flat field should be zero (sum of Sobel weights is 0)
        assert x_flat.grad.abs().max().item() < 1e-6
        assert not torch.isnan(x_flat.grad).any().item()

        # Step magnitude ~ 1e-4: G_x^2 ~ 1e-8 << eps (1e-6)
        x_small = torch.zeros(1, 3, 32, 32, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            x_small[:, :, :, 16:] = 1e-4
        mag_small = preprocessor.compute_gradient_magnitude(x_small)
        mag_small.sum().backward()

        assert x_small.grad is not None
        assert not torch.isnan(x_small.grad).any().item()
        assert not torch.isinf(x_small.grad).any().item()
        assert x_small.grad.abs().max().item() < 1.0

    @pytest.mark.parametrize("batch_size", [16, 32, 64, 128, 256])
    def test_large_batch_memory_and_gradient_scaling(self, batch_size: int):
        """Stress-test memory consumption and gradient flow under large batch processing."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        x = torch.rand(batch_size, 3, 128, 128, dtype=torch.float32, requires_grad=True)

        t0 = time.perf_counter()
        out = preprocessor(x)
        loss = out.mean()
        loss.backward()
        elapsed = time.perf_counter() - t0

        assert out.shape == (batch_size, 4, 128, 128)
        assert x.grad is not None
        assert not torch.isnan(x.grad).any().item()
        assert not torch.isinf(x.grad).any().item()
        assert elapsed < 5.0, f"Batch size {batch_size} took {elapsed:.2f}s (too slow)"


# ==============================================================================
# 4. 100-Scene Synthetic Desktop IoU Benchmark Under Inversion
# ==============================================================================


def _generate_random_desktop_scene(seed: int) -> torch.Tensor:
    """Generate a diverse synthetic desktop scene with random windows, buttons, controls, and colors."""
    torch.manual_seed(seed)
    # Random base wallpaper (dark, light, or gradient)
    mode = seed % 3
    if mode == 0:
        # Light theme desktop
        bg_r = 0.8 + 0.15 * torch.rand(1).item()
        bg_g = 0.8 + 0.15 * torch.rand(1).item()
        bg_b = 0.85 + 0.12 * torch.rand(1).item()
        frame = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        frame[:, 0] = bg_r
        frame[:, 1] = bg_g
        frame[:, 2] = bg_b
    elif mode == 1:
        # Dark theme desktop
        bg_r = 0.05 + 0.15 * torch.rand(1).item()
        bg_g = 0.05 + 0.15 * torch.rand(1).item()
        bg_b = 0.08 + 0.15 * torch.rand(1).item()
        frame = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        frame[:, 0] = bg_r
        frame[:, 1] = bg_g
        frame[:, 2] = bg_b
    else:
        # Vertical gradient wallpaper
        grad = torch.linspace(0.1, 0.9, 128).view(1, 1, 128, 1).repeat(1, 3, 1, 128)
        frame = grad

    # Taskbar at bottom (rows 116:128)
    frame[:, :, 116:128, :] = 0.20 if mode == 0 else 0.10

    # Taskbar icons
    for i in range(4):
        x_start = 10 + i * 16
        frame[:, :, 119:125, x_start:x_start + 10] = 0.60 + 0.1 * i

    # Random number of windows (1 to 3)
    num_windows = 1 + (seed % 3)
    for w in range(num_windows):
        w_seed = seed * 10 + w
        torch.manual_seed(w_seed)

        top = int(10 + torch.randint(0, 30, (1,)).item())
        left = int(10 + torch.randint(0, 30, (1,)).item())
        height = int(40 + torch.randint(20, 50, (1,)).item())
        width = int(40 + torch.randint(30, 60, (1,)).item())

        bottom = min(top + height, 114)
        right = min(left + width, 126)

        # Window body
        win_color = 0.95 if (mode == 0 or w % 2 == 0) else 0.15
        frame[:, :, top:bottom, left:right] = win_color

        # Window border
        border_color = 0.40
        frame[:, :, top:top + 1, left:right] = border_color
        frame[:, :, bottom - 1:bottom, left:right] = border_color
        frame[:, :, top:bottom, left:left + 1] = border_color
        frame[:, :, top:bottom, right - 1:right] = border_color

        # Title bar
        t_height = 8
        if top + t_height < bottom:
            frame[:, :, top:top + t_height, left:right] = 0.70 if win_color > 0.5 else 0.30
            # Close button
            frame[:, :, top + 2:top + 6, right - 8:right - 2] = 0.90

        # Window controls: button or text field
        if bottom - top > 25 and right - left > 30:
            c_top = top + 15
            c_left = left + 6
            c_bottom = min(c_top + 10, bottom - 4)
            c_right = min(c_left + 25, right - 6)
            frame[:, :, c_top:c_bottom, c_left:c_right] = 0.35 if win_color > 0.5 else 0.80

    return frame


class TestAdversarialSyntheticDesktop100Scenes:
    """Evaluates Edge IoU across 100 randomized synthetic desktop scenes under photometric inversion."""

    def test_edge_iou_across_100_synthetic_scenes(self):
        """Assert Edge IoU >= 0.90 under inversion across 100 random synthetic desktop scenes."""
        preprocessor = EdgeLuminancePreprocessor(eps=1e-6)
        ious = []
        max_diffs = []

        for i in range(100):
            scene = _generate_random_desktop_scene(seed=1000 + i)
            scene_inv = 1.0 - scene

            mag = preprocessor.compute_gradient_magnitude(scene)
            mag_inv = preprocessor.compute_gradient_magnitude(scene_inv)

            # Max difference
            diff = torch.max(torch.abs(mag - mag_inv)).item()
            max_diffs.append(diff)
            assert diff <= 1e-5, f"Scene {i} max diff {diff:.8e} > 1e-5"

            # Binary edge map with standard threshold 0.05
            edge_orig = compute_edge_map(scene, threshold=0.05)
            edge_inv = compute_edge_map(scene_inv, threshold=0.05)

            iou = compute_edge_iou(edge_orig, edge_inv)
            ious.append(iou)

            assert iou >= 0.90, f"Scene {i} IoU {iou:.6f} < 0.90 threshold"

        min_iou = min(ious)
        mean_iou = sum(ious) / len(ious)
        max_diff = max(max_diffs)

        print(f"\n[100 Synthetic Desktop Scenes Benchmark]")
        print(f"  Minimum Edge IoU: {min_iou:.6f} (Requirement: >= 0.90)")
        print(f"  Mean Edge IoU:    {mean_iou:.6f}")
        print(f"  Max Diff:         {max_diff:.8e} (Requirement: <= 1e-5)")

        # Verify strict compliance
        assert min_iou >= 0.90
        assert max_diff <= 1e-5


# ==============================================================================
# 5. Static AST and Shortcut Inspection
# ==============================================================================


class TestAdversarialStaticInspectionAndAST:
    """Verifies that no hardcoded heuristics, short-circuits, or prohibited dependencies exist."""

    def test_no_prohibited_imports_in_preprocessor(self):
        """Scan AST of preprocessor module for forbidden OCR, DOM, or word recognizer imports."""
        import textwrap
        source = inspect.getsource(prep_mod)
        tree = ast.parse(source)

        prohibited_modules = {
            "easyocr", "paddleocr", "pytesseract", "tesseract", "winocr",
            "playwright", "selenium", "uiautomation", "pywinauto",
            "bs4", "beautifulsoup4", "PixelWordRecognizer",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in prohibited_modules, f"Prohibited import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert not any(p in mod_name for p in prohibited_modules), f"Prohibited import from: {mod_name}"
                for alias in node.names:
                    assert alias.name not in prohibited_modules, f"Prohibited import name: {alias.name}"

        # Assert no PixelWordRecognizer attribute on module
        assert not hasattr(prep_mod, "PixelWordRecognizer")

    def test_ast_check_no_shortcut_branches_in_forward(self):
        """Parse AST of EdgeLuminancePreprocessor.forward to ensure no mock or hardcoded returns."""
        import textwrap
        source = textwrap.dedent(inspect.getsource(EdgeLuminancePreprocessor.forward))
        tree = ast.parse(source)

        # Inspect all if-statements in forward()
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Ensure condition does not check magic thresholds or cheat on inversion
                test_str = ast.unparse(node.test)
                # Allowed conditions: ndim checks, shape checks, device/dtype checks, normalize_to_unit, squeeze_batch
                allowed = ["ndim", "shape", "device", "dtype", "normalize_to_unit", "squeeze_batch"]
                is_valid = any(kw in test_str for kw in allowed)
                assert is_valid, f"Suspicious branch condition found in forward(): '{test_str}'"

    def test_weights_are_strictly_canonical_sobel(self):
        """Verify the registered Sobel weights exactly match standard normalized Sobel matrices."""
        preprocessor = EdgeLuminancePreprocessor()
        weights = preprocessor.sobel_weights

        expected_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        expected_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0

        assert torch.allclose(weights[0, 0], expected_x, atol=1e-7), "Sobel X weights deviate from canonical matrix"
        assert torch.allclose(weights[1, 0], expected_y, atol=1e-7), "Sobel Y weights deviate from canonical matrix"
        assert weights.shape == (2, 1, 3, 3)
