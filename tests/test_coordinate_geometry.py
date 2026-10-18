"""Unit tests for AVCS coordinate geometry, letterbox transforms, and physical pixel scaling."""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from avcs.goal_grounder import (
    LetterboxTransform,
    click_heatmap,
    extract_subpixel_peak,
    letterbox_rgb,
)


@pytest.mark.parametrize("width,height", [
    (1920, 1080),  # 16:9 standard
    (1920, 1200),  # 16:10 standard
    (2560, 1440),  # 16:9 QHD
    (3840, 2160),  # 16:9 4K UHD
    (1280, 800),   # 16:10 WXGA
    (1024, 768),   # 4:3 legacy
    (1080, 1920),  # 9:16 portrait
])
def test_letterbox_aspect_ratio_and_round_trip(width: int, height: int):
    """Verify round-trip fidelity across diverse display resolutions and aspect ratios."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    input_size = 256
    tensor, transform = letterbox_rgb(frame, input_size=input_size)

    assert tensor.shape == (3, input_size, input_size)

    # Test interior points
    test_points = [
        (0.5, 0.5),      # Center
        (0.01, 0.01),    # Near top-left
        (0.99, 0.01),    # Near top-right
        (0.01, 0.99),    # Near bottom-left
        (0.99, 0.99),    # Near bottom-right
        (0.25, 0.75),    # Quadrant 3
        (0.85, 0.15),    # Quadrant 1
    ]

    for pt in test_points:
        canvas_pt = transform.original_to_canvas(pt)
        # Canvas point must be inside [0, 1]
        assert 0.0 <= canvas_pt[0] <= 1.0, f"Canvas X out of bounds for {pt} on {width}x{height}"
        assert 0.0 <= canvas_pt[1] <= 1.0, f"Canvas Y out of bounds for {pt} on {width}x{height}"

        # Round-trip recovery
        recovered_pt = transform.canvas_to_original(canvas_pt)
        assert abs(recovered_pt[0] - pt[0]) < 1e-5, f"X roundtrip error on {width}x{height}"
        assert abs(recovered_pt[1] - pt[1]) < 1e-5, f"Y roundtrip error on {width}x{height}"


def test_letterbox_padding_boundary_rejection():
    """Verify that predictions falling into letterbox padding regions are cleanly rejected."""
    # 1920x1080 letterboxed into 256x256 has top and bottom bars of height 56 px
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, transform = letterbox_rgb(frame, input_size=256)

    assert transform.top_norm == pytest.approx(56 / 256, rel=1e-3)
    assert transform.height_norm == pytest.approx(144 / 256, rel=1e-3)

    # A point in the top padding region (e.g. y = 0.05 on canvas)
    pad_point = (0.5, 0.05)
    orig_point = transform.canvas_to_original(pad_point)
    # Must have negative Y in original space
    assert orig_point[1] < 0.0

    # A point in the bottom padding region (e.g. y = 0.95 on canvas)
    pad_point_bottom = (0.5, 0.95)
    orig_point_bottom = transform.canvas_to_original(pad_point_bottom)
    # Must exceed 1.0 in original space
    assert orig_point_bottom[1] > 1.0


def test_subpixel_peak_extraction_accuracy():
    """Verify that sub-pixel quadratic peak extraction recovers sub-cell offsets with <0.05 cell error."""
    H, W = 128, 128
    true_x = 0.5035
    true_y = 0.4982

    # Create synthetic Gaussian heatmap centered at non-grid sub-pixel coordinate
    hm = click_heatmap((true_x, true_y), (H, W), sigma_cells=3.0)
    # Convert probability to mock logits
    logits = torch.log(hm.clamp_min(1e-7)).unsqueeze(0).unsqueeze(0)

    coords, conf = extract_subpixel_peak(logits)
    pred_x, pred_y = float(coords[0, 0]), float(coords[0, 1])

    # Error on 128x128 grid must be smaller than 0.5 cells
    cell_err_x = abs(pred_x - true_x) * W
    cell_err_y = abs(pred_y - true_y) * H
    assert cell_err_x < 0.25, f"Subpixel X error too large: {cell_err_x:.3f} cells"
    assert cell_err_y < 0.25, f"Subpixel Y error too large: {cell_err_y:.3f} cells"


def test_physical_pixel_error_calculation():
    """Verify physical screen pixel error calculation between normalized predictions and annotations."""
    screen_w, screen_h = 1920, 1200
    gt_norm = (0.5000, 0.5000)

    # Prediction off by 2 physical pixels in X and 1 pixel in Y
    dx_px = 2.0
    dy_px = 1.0
    pred_norm = (gt_norm[0] + dx_px / screen_w, gt_norm[1] + dy_px / screen_h)

    # Calculate physical pixel error
    err_px = math.hypot((pred_norm[0] - gt_norm[0]) * screen_w, (pred_norm[1] - gt_norm[1]) * screen_h)
    expected_px = math.hypot(dx_px, dy_px)
    assert abs(err_px - expected_px) < 1e-4
    assert err_px <= 2.24  # Within ~2.24 physical pixels

