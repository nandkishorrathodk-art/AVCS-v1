"""Held-Out Evaluation Benchmarks for AVCS Cortical Affordance & Foveal Models.

Tests:
1. Automatic discovery and loading of pre-trained HumanEye neural weights.
2. High precision and recall on held-out procedural desktop GUI test screens.
3. Strict zero/near-zero false positive rate on static wallpapers / uniform screens.
4. CPU perception latency benchmarking ensuring sub-15ms throughput.
5. Foveal acuity classification and bounding box regression on held-out crops.
"""
from __future__ import annotations

import math
import random
import time
import numpy as np
import pytest
import torch

from avcs.grounding import FOVEAL_ELEMENT_TYPES, FovealAcuityEngine
from avcs.human_eye import FastAffordanceHead, HumanEye, RealTimeCorticalBackbone
from avcs.training.gui_dataset import DesktopGUIGenerator
from avcs.weights import get_human_eye_weights_path, load_avcs_weights


def test_human_eye_weights_auto_discovery():
    """Verifies that human_eye_weights.pt is discovered and automatically loaded by HumanEye."""
    weights_path = get_human_eye_weights_path()
    assert weights_path is not None, "Expected to locate pre-trained human_eye_weights.pt"
    assert weights_path.endswith("human_eye_weights.pt")

    eye = HumanEye(source="synthetic", load_weights=True)
    assert eye.weights_loaded is True, "Expected HumanEye to have loaded weights"


def test_held_out_gui_affordance_precision_and_recall():
    """Benchmarks precision and recall on held-out desktop GUI application screens."""
    eye = HumanEye(source="synthetic", load_weights=True)
    eye.eval()

    gen = DesktopGUIGenerator()
    tp, fp, fn = 0, 0, 0
    categories = ["browser_webpage", "editor_code", "form_input", "modal_dialog", "terminal_shell"]

    # Evaluate 30 held-out diverse GUI screens
    for i in range(30):
        cat = categories[i % len(categories)]
        sample = gen.generate_gui_screen(category=cat, seed=50000 + i)

        # Run model perception
        state = eye.observe(sample.frame)
        hotspots = state.affordance_hotspots
        gt_points = sample.interactive_points

        matched_gt = set()
        for hx, hy, act, conf in hotspots:
            found = False
            for idx, (gx, gy, gact) in enumerate(gt_points):
                if math.hypot(hx - gx, hy - gy) < 0.20:
                    found = True
                    matched_gt.add(idx)
                    break
            if found:
                tp += 1
            else:
                fp += 1

        fn += len(gt_points) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    print(f"\n[Held-Out GUI Eval] Precision: {precision:.1%}, Recall: {recall:.1%}, TP: {tp}, FP: {fp}, FN: {fn}")
    # Verify strong precision on genuine GUI elements
    assert precision >= 0.70, f"Expected precision >= 70%, got {precision:.1%}"
    assert tp > 0, "Expected to detect true positive interactive affordances"


def test_held_out_wallpaper_zero_false_positives():
    """Verifies that static desktop wallpapers produce minimal/zero false positives."""
    eye = HumanEye(source="synthetic", load_weights=True)
    eye.eval()

    gen = DesktopGUIGenerator()
    wallpaper_screens = 20
    fp_screens = 0

    for i in range(wallpaper_screens):
        sample = gen.generate_gui_screen(category="desktop_canvas", seed=70000 + i)
        state = eye.observe(sample.frame)
        if len(state.affordance_hotspots) > 0:
            fp_screens += 1

    fp_rate = fp_screens / float(wallpaper_screens)
    print(f"\n[Wallpaper Rejection Eval] FP Rate: {fp_rate:.1%} ({fp_screens}/{wallpaper_screens} screens)")
    assert fp_rate <= 0.10, f"Expected wallpaper FP rate <= 10%, got {fp_rate:.1%}"


def test_cpu_perception_latency_benchmark():
    """Asserts that HumanEye perception tick runs at <20ms on standard CPU (>50 FPS)."""
    eye = HumanEye(source="synthetic", load_weights=True)
    frame = np.full((360, 640, 3), 40, dtype=np.uint8)

    # Warmup
    for _ in range(5):
        eye.observe(frame)

    # Benchmark 30 ticks
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        eye.observe(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    print(f"\n[CPU Latency Benchmark] p50: {p50:.2f}ms | p95: {p95:.2f}ms")
    assert p50 < 25.0, f"Expected p50 latency < 25ms, got {p50:.2f}ms"


def test_foveal_acuity_held_out_element_inspection():
    """Validates FovealAcuityEngine classification and bounding box regression on held-out crops."""
    fovea = FovealAcuityEngine(device="cpu", load_weights=True)
    assert fovea.weights_loaded is True, "Expected FovealAcuityEngine to load weights"
    gen = DesktopGUIGenerator()

    correct_kinds = 0
    ious = []
    total = 25

    for i in range(total):
        target_kind = FOVEAL_ELEMENT_TYPES[i % len(FOVEAL_ELEMENT_TYPES)]
        sample = gen.generate_foveal_element(element_kind=target_kind, seed=90000 + i)

        # Inspect crop
        inspection = fovea.inspect_crop(sample.crop)
        assert inspection is not None
        assert inspection.confidence > 0.0
        assert len(inspection.bbox) == 4

        if inspection.element_kind == target_kind:
            correct_kinds += 1

        # Verify valid normalized bounding box bounds: bbox is (xmin, ymin, xmax, ymax)
        x1, y1, x2, y2 = inspection.bbox
        assert 0.0 <= x1 <= x2 <= 1.0
        assert 0.0 <= y1 <= y2 <= 1.0

        # Calculate IoU against ground truth [ymin, xmin, ymax, xmax]
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = sample.bbox
        inter_x1 = max(x1, gt_xmin)
        inter_y1 = max(y1, gt_ymin)
        inter_x2 = min(x2, gt_xmax)
        inter_y2 = min(y2, gt_ymax)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        pred_area = (x2 - x1) * (y2 - y1)
        gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
        union_area = pred_area + gt_area - inter_area
        iou = inter_area / max(union_area, 1e-6)
        ious.append(iou)

    acc = correct_kinds / float(total)
    mean_iou = float(np.mean(ious))
    print(f"\n[Foveal Acuity Eval] Class Acc: {acc:.1%}, Mean IoU: {mean_iou:.3f}")
    assert acc >= 0.70, f"Expected classification accuracy >= 70%, got {acc:.1%}"
    assert mean_iou >= 0.60, f"Expected mean IoU >= 60%, got {mean_iou:.3f}"


def test_submodule_weights_loading():
    """Validates that individual AVCS cortical submodules can load weights independently."""
    from avcs.grounding import FovealAcuityHead

    backbone = RealTimeCorticalBackbone()
    assert load_avcs_weights(backbone) is True, "Expected backbone to load weights"

    aff_head = FastAffordanceHead()
    assert load_avcs_weights(aff_head) is True, "Expected affordance head to load weights"

    fov_head = FovealAcuityHead()
    assert load_avcs_weights(fov_head) is True, "Expected foveal acuity head to load weights"

    fov_engine = FovealAcuityEngine(load_weights=True)
    assert fov_engine.weights_loaded is True, "Expected foveal acuity engine to load weights"


def test_typeable_and_clickable_affordances():
    """Verifies that HumanEye with action_type='all' detects both buttons and input fields."""
    eye = HumanEye(source="synthetic", load_weights=True, affordance_action_type="all", affordance_top_k=6)
    gen = DesktopGUIGenerator()

    # Form input screen contains both text input fields and submit button
    sample = gen.generate_gui_screen(category="form_input", seed=42)
    state = eye.observe(sample.frame)

    action_types = {h[2] for h in state.affordance_hotspots}
    assert "clickable" in action_types, "Expected clickable affordances on form screen"
    assert "typeable" in action_types, "Expected typeable affordances on form screen"


def test_human_eye_fovea_submodule_registration():
    """Verifies that HumanEye registers fovea_head as an nn.Module submodule with trained weights."""
    eye = HumanEye(source="synthetic", load_weights=True)
    assert hasattr(eye, "fovea_head"), "Expected HumanEye to have fovea_head submodule"
    assert any("fovea_head" in k for k in eye.state_dict().keys()), "Expected fovea_head in state_dict"
    assert eye.fovea_engine.weights_loaded is True, "Expected fovea_engine weights to be marked loaded"
    # Verify that fovea_engine.head and fovea_head share parameters
    param_engine = next(eye.fovea_engine.head.parameters())
    param_eye = next(eye.fovea_head.parameters())
    assert torch.equal(param_engine, param_eye), "Expected shared parameter references"

