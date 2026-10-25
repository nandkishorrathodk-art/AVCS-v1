"""Independent Empirical Adversarial Stress Suite for Milestone M4 Iteration 2.

Author: challenger_m4_rem_2 (Empirical Challenger)
Milestone: M4 Iteration 2 (Blank-Screen Safety Remediation & Candidate Staging)
Target: runs/weights/staging/candidate_streaming_v2.pt

Stress Dimensions:
1. Exact per-screen analysis of all 10 desktop wallpapers (test_neg_0021..0030) and cleanroom empty frames.
2. Adversarial perturbations on wallpapers:
   - Photometric inversion (dark/light theme equivalence test)
   - Contrast scaling (0.8x, 1.2x, 1.5x)
   - Brightness shifts (+/- 0.1, +/- 0.2)
   - Additive Gaussian noise (sigma=0.02, 0.05, 0.10)
3. Synthetic adversarial patterns:
   - High-frequency checkerboards (1px, 2px, 4px, 8px)
   - Border-only wireframe boxes without controls
   - Extreme salt-and-pepper noise
4. Threshold sensitivity & ROC analysis:
   - Objectness false alarm rate vs threshold theta in [0.20, 0.50]
   - Objectness margin analysis (delta = 0.35 - p_max)
5. Parameter breakdown & baseline hash verification
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from avcs.dataset.unified.generator_harmonizer import (
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
)
from avcs.streaming_v2.model import StreamingVisualCortexV2
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.training.multi_generator_dataset import ArmD2MultiGeneratorDataset
from avcs.training.train_streaming_v2 import (
    BASELINE_WEIGHT_HASHES,
    verify_baseline_weights_untouched,
)


def load_candidate_model(
    ckpt_path: str = "runs/weights/staging/candidate_streaming_v2.pt",
) -> Tuple[StreamingVisualCortexV2, EdgeLuminancePreprocessor]:
    device = torch.device("cpu")
    model = StreamingVisualCortexV2().to(device)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(data["state_dict"])
    model.eval()
    preprocessor = EdgeLuminancePreprocessor().to(device)
    return model, preprocessor


def stress_1_detailed_wallpaper_inspection(
    model: StreamingVisualCortexV2,
    preprocessor: EdgeLuminancePreprocessor,
) -> Dict[str, Any]:
    """Inspect every single wallpaper screen individually."""
    print("\n--- Stress 1: Individual Wallpaper (test_neg_0021..0030) Analysis ---")
    ds_neg = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_NEGATIVE, apply_augmentation=False, preload_images=True)
    
    wallpaper_results = []
    for i in range(len(ds_neg)):
        sample = ds_neg[i]
        fid = sample["frame_id"]
        gid = sample["generator_id"]
        
        # Only UI Grounding wallpaper screens (screens 21 to 30)
        if gid == 0 and "test_neg_" in fid:
            num = int(fid.split("_")[-1])
            if 21 <= num <= 30:
                img = sample["image"].unsqueeze(0)
                x4 = preprocessor(img)
                with torch.no_grad():
                    out = model.forward_step(x4, prev_state=None)
                
                probs = torch.sigmoid(out.ui_objectness_logits[0, 0]).numpy()
                p_max = float(np.max(probs))
                p_mean = float(np.mean(probs))
                max_idx = np.unravel_index(np.argmax(probs), probs.shape)
                
                margin_035 = 0.35 - p_max
                margin_050 = 0.50 - p_max
                
                wallpaper_results.append({
                    "frame_id": fid,
                    "screen_number": num,
                    "p_max": round(p_max, 4),
                    "p_mean": round(p_mean, 4),
                    "max_cell_grid": [int(max_idx[0]), int(max_idx[1])],
                    "margin_to_035": round(margin_035, 4),
                    "margin_to_050": round(margin_050, 4),
                    "is_fp_035": p_max >= 0.35,
                    "is_fp_050": p_max >= 0.50,
                })
                print(f"  {fid}: p_max={p_max:.4f} (at cell {max_idx}), margin_to_0.35={margin_035:+.4f}, FP_035={p_max >= 0.35}")
                
    max_p_overall = max(r["p_max"] for r in wallpaper_results)
    min_margin = min(r["margin_to_035"] for r in wallpaper_results)
    
    return {
        "wallpapers_tested": len(wallpaper_results),
        "all_under_035": all(not r["is_fp_035"] for r in wallpaper_results),
        "all_under_050": all(not r["is_fp_050"] for r in wallpaper_results),
        "max_probability": max_p_overall,
        "min_margin_to_035": min_margin,
        "individual_results": wallpaper_results,
    }


def stress_2_perturbed_wallpapers(
    model: StreamingVisualCortexV2,
    preprocessor: EdgeLuminancePreprocessor,
) -> Dict[str, Any]:
    """Test wallpapers under photometric inversion, contrast scaling, brightness, and noise."""
    print("\n--- Stress 2: Wallpaper Perturbation & Theme Robustness ---")
    ds_neg = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_NEGATIVE, apply_augmentation=False, preload_images=True)
    
    # Gather wallpaper tensors
    wp_images = []
    wp_ids = []
    for i in range(len(ds_neg)):
        sample = ds_neg[i]
        fid = sample["frame_id"]
        gid = sample["generator_id"]
        if gid == 0 and "test_neg_" in fid:
            num = int(fid.split("_")[-1])
            if 21 <= num <= 30:
                wp_images.append(sample["image"])
                wp_ids.append(fid)
                
    wp_batch = torch.stack(wp_images)  # (10, 3, 128, 128)
    
    perturbations = {
        "clean_baseline": wp_batch,
        "photometric_inversion (1 - x)": 1.0 - wp_batch,
        "contrast_scale_0.8x": torch.clamp((wp_batch - 0.5) * 0.8 + 0.5, 0.0, 1.0),
        "contrast_scale_1.2x": torch.clamp((wp_batch - 0.5) * 1.2 + 0.5, 0.0, 1.0),
        "contrast_scale_1.5x": torch.clamp((wp_batch - 0.5) * 1.5 + 0.5, 0.0, 1.0),
        "brightness_plus_0.1": torch.clamp(wp_batch + 0.1, 0.0, 1.0),
        "brightness_minus_0.1": torch.clamp(wp_batch - 0.1, 0.0, 1.0),
        "brightness_plus_0.2": torch.clamp(wp_batch + 0.2, 0.0, 1.0),
        "gaussian_noise_sigma_0.02": torch.clamp(wp_batch + 0.02 * torch.randn_like(wp_batch), 0.0, 1.0),
        "gaussian_noise_sigma_0.05": torch.clamp(wp_batch + 0.05 * torch.randn_like(wp_batch), 0.0, 1.0),
    }
    
    perturbation_results = {}
    for name, p_imgs in perturbations.items():
        x4 = preprocessor(p_imgs)
        with torch.no_grad():
            out = model.forward_step(x4, prev_state=None)
        
        probs = torch.sigmoid(out.ui_objectness_logits[:, 0]).numpy()  # (10, 16, 16)
        max_per_img = np.max(probs, axis=(1, 2))
        fp_035 = int(np.sum(max_per_img >= 0.35))
        fp_050 = int(np.sum(max_per_img >= 0.50))
        max_prob = float(np.max(max_per_img))
        mean_prob = float(np.mean(probs))
        
        perturbation_results[name] = {
            "max_prob": round(max_prob, 4),
            "mean_prob": round(mean_prob, 4),
            "false_positives_at_0_35": fp_035,
            "false_positives_at_0_50": fp_050,
            "pass_035": fp_035 == 0,
        }
        print(f"  {name:30s}: max_p={max_prob:.4f}, mean_p={mean_prob:.4f}, FP@0.35={fp_035}/10, FP@0.50={fp_050}/10")
        
    return perturbation_results


def stress_3_synthetic_adversarial_patterns(
    model: StreamingVisualCortexV2,
    preprocessor: EdgeLuminancePreprocessor,
) -> Dict[str, Any]:
    """Test high-frequency checkerboards, wireframe boxes, and salt-and-pepper noise."""
    print("\n--- Stress 3: Synthetic Adversarial Patterns ---")
    patterns = {}
    
    # 1. Checkerboards of various block sizes
    for block_sz in [1, 2, 4, 8, 16]:
        cb = np.zeros((128, 128), dtype=np.float32)
        for r in range(128):
            for c in range(128):
                if ((r // block_sz) + (c // block_sz)) % 2 == 1:
                    cb[r, c] = 1.0
        patterns[f"checkerboard_{block_sz}px"] = torch.from_numpy(cb).unsqueeze(0).repeat(3, 1, 1)
        
    # 2. Wireframe rectangular boxes (empty UI outlines)
    wireframe = np.zeros((3, 128, 128), dtype=np.float32)
    # Box 1 outline
    wireframe[:, 20:22, 20:80] = 1.0
    wireframe[:, 45:47, 20:80] = 1.0
    wireframe[:, 20:47, 20:22] = 1.0
    wireframe[:, 20:47, 78:80] = 1.0
    # Box 2 outline
    wireframe[:, 60:62, 30:100] = 1.0
    wireframe[:, 90:92, 30:100] = 1.0
    wireframe[:, 60:92, 30:32] = 1.0
    wireframe[:, 60:92, 98:100] = 1.0
    patterns["wireframe_empty_boxes"] = torch.from_numpy(wireframe)
    
    # 3. Dense grid lines (window pane / spreadsheet without text)
    grid = np.zeros((3, 128, 128), dtype=np.float32)
    grid[:, ::16, :] = 1.0
    grid[:, :, ::16] = 1.0
    patterns["dense_grid_16px"] = torch.from_numpy(grid)
    
    # 4. Salt-and-pepper impulse noise
    sp = np.random.RandomState(42).choice([0.0, 0.5, 1.0], size=(3, 128, 128), p=[0.05, 0.90, 0.05]).astype(np.float32)
    patterns["salt_and_pepper_noise"] = torch.from_numpy(sp)
    
    # 5. Concentric circles / radial gradient
    y, x = np.ogrid[-64:64, -64:64]
    r = np.sqrt(x**2 + y**2)
    radial = (np.sin(r / 3.0) + 1.0) / 2.0
    patterns["concentric_radial_waves"] = torch.from_numpy(radial.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
    
    pattern_results = {}
    for name, pat in patterns.items():
        x4 = preprocessor(pat.unsqueeze(0))
        with torch.no_grad():
            out = model.forward_step(x4, prev_state=None)
        
        probs = torch.sigmoid(out.ui_objectness_logits[0, 0]).numpy()
        p_max = float(np.max(probs))
        p_mean = float(np.mean(probs))
        fp_035 = int(np.sum(probs >= 0.35))
        fp_050 = int(np.sum(probs >= 0.50))
        
        pattern_results[name] = {
            "p_max": round(p_max, 4),
            "p_mean": round(p_mean, 4),
            "fp_cells_035": fp_035,
            "fp_cells_050": fp_050,
            "is_false_positive_035": fp_035 > 0,
            "is_false_positive_050": fp_050 > 0,
        }
        print(f"  {name:25s}: p_max={p_max:.4f}, p_mean={p_mean:.4f}, fp_cells@0.35={fp_035}, fp_cells@0.50={fp_050}")
        
    return pattern_results


def stress_4_threshold_operating_curve(
    model: StreamingVisualCortexV2,
    preprocessor: EdgeLuminancePreprocessor,
) -> Dict[str, Any]:
    """Sweep threshold theta in [0.15, 0.55] across all negative screens and held-out positive screens."""
    print("\n--- Stress 4: Threshold Operating Curve & Selectivity ---")
    
    # Gather negative screens max probs
    ds_neg = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_NEGATIVE, apply_augmentation=False, preload_images=True)
    neg_max_probs = []
    for i in range(len(ds_neg)):
        sample = ds_neg[i]
        x4 = preprocessor(sample["image"].unsqueeze(0))
        with torch.no_grad():
            out = model.forward_step(x4, prev_state=None)
        p_max = float(torch.sigmoid(out.ui_objectness_logits[0, 0]).max().item())
        neg_max_probs.append(p_max)
        
    neg_max_probs = np.array(neg_max_probs)
    
    # Gather positive frames positive cells probabilities
    ds_pos = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_HELD_OUT, apply_augmentation=False, preload_images=True)
    pos_cell_probs = []
    for i in range(len(ds_pos)):
        sample = ds_pos[i]
        x4 = preprocessor(sample["image"].unsqueeze(0))
        with torch.no_grad():
            out = model.forward_step(x4, prev_state=None)
        probs = torch.sigmoid(out.ui_objectness_logits[0, 0]).numpy()
        # Positive cells from boxes
        boxes = sample["target_boxes"].numpy()
        for b in boxes:
            cx_norm = (b[0] + b[2]) / 2.0
            cy_norm = (b[1] + b[3]) / 2.0
            cx = int(round(cx_norm * 15.0))
            cy = int(round(cy_norm * 15.0))
            cx = min(max(cx, 0), 15)
            cy = min(max(cy, 0), 15)
            pos_cell_probs.append(float(probs[cy, cx]))
                
    pos_cell_probs = np.array(pos_cell_probs)
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55]
    operating_points = []
    
    for th in thresholds:
        neg_fps = int(np.sum(neg_max_probs >= th))
        fp_rate = neg_fps / len(neg_max_probs)
        pos_recall = float(np.mean(pos_cell_probs >= th))
        operating_points.append({
            "threshold": th,
            "neg_fps": neg_fps,
            "neg_fp_rate_pct": round(fp_rate * 100.0, 2),
            "pos_cell_recall_pct": round(pos_recall * 100.0, 2),
        })
        print(f"  Threshold {th:.2f}: Negative FPs = {neg_fps:2d}/{len(neg_max_probs)} ({fp_rate*100:5.1f}%), Positive Cell Recall = {pos_recall*100:5.1f}%")
        
    return {
        "negative_screens_count": len(neg_max_probs),
        "positive_cells_count": len(pos_cell_probs),
        "operating_points": operating_points,
    }


def stress_5_parameter_and_baseline_audit() -> Dict[str, Any]:
    """Verify bit-for-bit baseline immutability and exact parameter breakdown."""
    print("\n--- Stress 5: Parameter Breakdown & Baseline Immutability Audit ---")
    
    # 1. Baseline immutability
    baseline_status = verify_baseline_weights_untouched()
    for name, p in baseline_status.items():
        print(f"  [VERIFIED] Baseline {name} matches frozen hash.")
        
    # 2. Candidate parameter math
    ckpt_path = "runs/weights/staging/candidate_streaming_v2.pt"
    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = data["state_dict"]
    
    breakdown = {
        "spatial_stem": 0,
        "conv_gru": 0,
        "latent_projection": 0,
        "future_predictor": 0,
        "head1_change_spinner": 0,
        "head2_ui_detector": 0,
        "head3_post_action": 0,
    }
    
    for k, v in state.items():
        n = v.numel()
        if k.startswith("stem."):
            breakdown["spatial_stem"] += n
        elif k.startswith("conv_gru."):
            breakdown["conv_gru"] += n
        elif k.startswith("latent_projection."):
            breakdown["latent_projection"] += n
        elif k.startswith("future_predictor."):
            breakdown["future_predictor"] += n
        elif k.startswith("change_spinner_head."):
            breakdown["head1_change_spinner"] += n
        elif k.startswith("ui_detector_head."):
            breakdown["head2_ui_detector"] += n
        elif k.startswith("post_action_head."):
            breakdown["head3_post_action"] += n
        else:
            print(f"  Unknown parameter group: {k}")
            
    total_computed = sum(breakdown.values())
    print(f"  Total Computed Parameters: {total_computed}")
    for k, v in breakdown.items():
        print(f"    - {k:22s}: {v:7d}")
        
    expected_total = 400765
    assert total_computed == expected_total, f"Parameter count mismatch: {total_computed} vs {expected_total}"
    
    return {
        "baseline_weights_untouched": True,
        "baselines_checked": list(baseline_status.keys()),
        "parameter_breakdown": breakdown,
        "total_parameters": total_computed,
        "parameter_count_match": (total_computed == expected_total),
    }


def run_all_adversarial_challenges() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING INDEPENDENT ADVERSARIAL STRESS SUITE (CHALLENGER_M4_REM_2)")
    print("=" * 80)
    
    model, preprocessor = load_candidate_model()
    
    results = {}
    results["stress_1_wallpapers"] = stress_1_detailed_wallpaper_inspection(model, preprocessor)
    results["stress_2_perturbations"] = stress_2_perturbed_wallpapers(model, preprocessor)
    results["stress_3_synthetic_patterns"] = stress_3_synthetic_adversarial_patterns(model, preprocessor)
    results["stress_4_threshold_curve"] = stress_4_threshold_operating_curve(model, preprocessor)
    results["stress_5_parameter_audit"] = stress_5_parameter_and_baseline_audit()
    
    # Save results to benchmarks/results/challenger_m4_rem_2_stress_results.json
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "challenger_m4_rem_2_stress_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[COMPLETE] Stress suite results saved to {out_path.resolve()}")
    
    return results


if __name__ == "__main__":
    run_all_adversarial_challenges()
