"""Benchmark harness for Area MT/V5 Optic Flow, Motion Flux, and Head 1 Metrics.

Evaluates:
  1. Head 1 (TemporalChangeSpinnerHead) Spatial Change Heatmap:
     - Heatmap BCE loss verification
     - Pixel-level Sensitivity (Recall), Specificity, Precision, F1, and mean IoU
  2. Cyclical Frequency Detection (Spinner Head):
     - Spinner BCE loss verification
     - Cyclical frequency detection across rotation/pulsation period sweeps (periods 2..16)
     - Spinner classification Sensitivity (Recall), Specificity, Precision, Accuracy, and ROC-AUC
  3. Area MT/V5 Optic Flow & Motion Flux Visual Settling:
     - Kinetic optical flow energy E_flow = mean(vx^2 + vy^2)
     - Temporal predictive coding surprise S_t
     - Visual settling thresholds verification: (surprise <= 0.08, energy <= 0.05)
     - Sensitivity and Recall for visual settling / stability detection
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from avcs.dataset.unified.temporal_sampler import (
    _draw_pulsating_dots_spinner,
    _draw_rotating_arc_spinner,
    _render_base_desktop_canvas,
    generate_temporal_sequence,
    sample_temporal_batch,
)
from avcs.perception.human_eye import HumanEye, SyntheticScreen
from avcs.streaming_v2.model import StreamingVisualCortexV2
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.training.train_streaming_v2 import compute_head1_loss


def compute_roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1.0
    unique_scores, inverse_indices, counts = np.unique(
        y_score, return_inverse=True, return_counts=True
    )
    if len(unique_scores) < len(y_score):
        for idx, count in enumerate(counts):
            if count > 1:
                tie_mask = (inverse_indices == idx)
                ranks[tie_mask] = np.mean(ranks[tie_mask])
    rank_sum_pos = np.sum(ranks[pos_mask])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1.0)) / 2.0
    return float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))


def benchmark_heatmap_and_spinner(
    checkpoint_path: str = "runs/weights/staging/candidate_streaming_v2.pt",
    batch_size: int = 100,
    seed: int = 20261024,
) -> Dict[str, Any]:
    """Benchmark TemporalChangeSpinnerHead heatmap BCE loss and spinner metrics."""
    device = torch.device("cpu")
    torch.set_num_threads(4)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = StreamingVisualCortexV2().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prep = EdgeLuminancePreprocessor().to(device)
    prep.eval()

    rng = random.Random(seed)
    tb = sample_temporal_batch(batch_size=batch_size, sequence_length=2, rng=rng)
    seqs = tb["sequences"].to(device)
    gt_change = tb["change_heatmaps"].to(device)
    gt_spin = tb["spinner_labels"].to(device)

    f0 = prep(seqs[:, 0])
    f1 = prep(seqs[:, 1])

    with torch.no_grad():
        o0 = model.forward_step(f0, prev_state=None, timestamp=0.0)
        o1 = model.forward_step(f1, prev_state=o0.temporal_state, timestamp=0.1)

    total_h1, loss_change, loss_spinner = compute_head1_loss(
        o1.change_heatmap,
        o1.spinner_activity_logit,
        gt_change,
        gt_spin,
    )

    # Spatial Change Heatmap Pixel Metrics @ threshold 0.10
    bin_gt = (gt_change > 0.05).float()
    bin_pred = (o1.change_heatmap > 0.10).float()

    tp_pix = float((bin_pred * bin_gt).sum().item())
    fp_pix = float(((bin_pred == 1) & (bin_gt == 0)).sum().item())
    fn_pix = float(((bin_pred == 0) & (bin_gt == 1)).sum().item())
    tn_pix = float(((bin_pred == 0) & (bin_gt == 0)).sum().item())

    pix_sensitivity = tp_pix / (tp_pix + fn_pix + 1e-8)  # Recall
    pix_specificity = tn_pix / (tn_pix + fp_pix + 1e-8)
    pix_precision = tp_pix / (tp_pix + fp_pix + 1e-8)
    pix_f1 = 2.0 * pix_precision * pix_sensitivity / (pix_precision + pix_sensitivity + 1e-8)
    pix_accuracy = (tp_pix + tn_pix) / (tp_pix + tn_pix + fp_pix + fn_pix + 1e-8)

    # Intersection over Union (IoU)
    intersection = (bin_pred * bin_gt).sum(dim=(-2, -1))
    union = ((bin_pred + bin_gt) > 0).float().sum(dim=(-2, -1))
    ious = torch.where(union > 0, intersection / union, torch.ones_like(union)).squeeze()
    mean_iou = float(ious.mean().item())

    # Spinner Activity Metrics
    spin_probs = torch.sigmoid(o1.spinner_activity_logit)[:, 0].cpu().numpy()
    spin_targets = gt_spin[:, 0].cpu().numpy()
    spin_preds = (spin_probs >= 0.50).astype(float)

    tp_spin = int(np.sum((spin_preds == 1) & (spin_targets == 1)))
    fp_spin = int(np.sum((spin_preds == 1) & (spin_targets == 0)))
    fn_spin = int(np.sum((spin_preds == 0) & (spin_targets == 1)))
    tn_spin = int(np.sum((spin_preds == 0) & (spin_targets == 0)))

    spin_sensitivity = float(tp_spin / (tp_spin + fn_spin + 1e-8))  # Recall
    spin_specificity = float(tn_spin / (tn_spin + fp_spin + 1e-8))
    spin_precision = float(tp_spin / (tp_spin + fp_spin + 1e-8)) if (tp_spin + fp_spin) > 0 else 1.0
    spin_f1 = (
        2.0 * spin_precision * spin_sensitivity / (spin_precision + spin_sensitivity + 1e-8)
    )
    spin_acc = float(np.mean(spin_preds == spin_targets)) * 100.0
    spin_auc = compute_roc_auc_binary(spin_targets, spin_probs)

    return {
        "heatmap": {
            "bce_loss": float(loss_change.item()),
            "mean_iou": mean_iou,
            "sensitivity_recall": pix_sensitivity,
            "specificity": pix_specificity,
            "precision": pix_precision,
            "f1_score": pix_f1,
            "accuracy": pix_accuracy,
            "tp": int(tp_pix),
            "fp": int(fp_pix),
            "fn": int(fn_pix),
            "tn": int(tn_pix),
        },
        "spinner": {
            "bce_loss": float(loss_spinner.item()),
            "accuracy_pct": spin_acc,
            "sensitivity_recall": spin_sensitivity,
            "specificity": spin_specificity,
            "precision": spin_precision,
            "f1_score": spin_f1,
            "roc_auc": spin_auc,
            "tp": tp_spin,
            "fp": fp_spin,
            "fn": fn_spin,
            "tn": tn_spin,
        },
        "total_head1_loss": float(total_h1.item()),
    }


def benchmark_cyclical_frequency_sweep(
    checkpoint_path: str = "runs/weights/staging/candidate_streaming_v2.pt",
    samples_per_freq: int = 25,
) -> Dict[str, Any]:
    """Test spinner detection across a spectrum of rotation frequencies and period lengths."""
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = StreamingVisualCortexV2().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prep = EdgeLuminancePreprocessor().to(device)
    prep.eval()

    periods = [2, 3, 4, 6, 8, 10, 12, 16]
    results_by_period = {}
    total_tp = 0
    total_pos = 0

    for p in periods:
        # Rotating Arc Spinner
        arc_hits = 0
        dots_hits = 0
        for i in range(samples_per_freq):
            r = random.Random(i + 100 * p)
            base_img = _render_base_desktop_canvas((128, 128), r)

            # Arc
            f_arc = []
            for t in range(2):
                img = base_img.copy()
                _draw_rotating_arc_spinner(ImageDraw.Draw(img), (64, 64), 14, t, p)
                f_arc.append(
                    torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                )
            f0 = prep(f_arc[0].unsqueeze(0).to(device))
            f1 = prep(f_arc[1].unsqueeze(0).to(device))
            with torch.no_grad():
                o0 = model.forward_step(f0, prev_state=None, timestamp=0.0)
                o1 = model.forward_step(f1, prev_state=o0.temporal_state, timestamp=0.1)
            prob_arc = float(torch.sigmoid(o1.spinner_activity_logit).item())
            if prob_arc >= 0.5:
                arc_hits += 1

            # Dots
            f_dots = []
            for t in range(2):
                img = base_img.copy()
                _draw_pulsating_dots_spinner(ImageDraw.Draw(img), (64, 64), 14, t, p)
                f_dots.append(
                    torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                )
            f0 = prep(f_dots[0].unsqueeze(0).to(device))
            f1 = prep(f_dots[1].unsqueeze(0).to(device))
            with torch.no_grad():
                o0 = model.forward_step(f0, prev_state=None, timestamp=0.0)
                o1 = model.forward_step(f1, prev_state=o0.temporal_state, timestamp=0.1)
            prob_dots = float(torch.sigmoid(o1.spinner_activity_logit).item())
            if prob_dots >= 0.5:
                dots_hits += 1

        total_tested = samples_per_freq * 2
        total_hits = arc_hits + dots_hits
        total_tp += total_hits
        total_pos += total_tested

        results_by_period[f"period_{p}_frames"] = {
            "period_frames": p,
            "arc_recall": arc_hits / samples_per_freq,
            "dots_recall": dots_hits / samples_per_freq,
            "combined_recall": total_hits / total_tested,
        }

    overall_freq_recall = total_tp / total_pos
    return {
        "results_by_period": results_by_period,
        "overall_cyclical_recall": overall_freq_recall,
        "total_cyclical_tested": total_pos,
        "total_cyclical_detected": total_tp,
    }


def benchmark_area_mt_v5_settling_thresholds(
    n_samples: int = 100,
    seed: int = 20261024,
) -> Dict[str, Any]:
    """Benchmark Area MT/V5 optic flow, motion flux, and visual settling thresholds (0.08 surprise, 0.05 energy)."""
    eye = HumanEye(source=SyntheticScreen())
    engine = eye.motion_engine

    # Generate 5 diverse scenario categories:
    # Class 1: Settled Screens (ground truth = SETTLED / Positive for settling)
    #   - Identical consecutive static canvas frames
    #   - Sub-threshold subtle micro-jitter / single-pixel line artifacts
    # Class 0: Unsettled / Dynamic Screens (ground truth = UNSETTLED / Negative for settling)
    #   - Active loading spinners
    #   - Moving cursor & element dragging
    #   - Sudden modal dialog appearance / scene transition
    scenarios = [
        ("static", True),
        ("micro_jitter", True),
        ("spinner", False),
        ("moving_cursor", False),
        ("modal_popup", False),
    ]

    records = []

    for name, is_settled_gt in scenarios:
        for i in range(n_samples // len(scenarios)):
            r = random.Random(seed + i * 17 + len(name) * 31)
            base = _render_base_desktop_canvas((128, 128), r)

            if name == "static":
                f0_pil = base.copy()
                f1_pil = base.copy()
            elif name == "micro_jitter":
                f0_pil = base.copy()
                f1_pil = base.copy()
                d = ImageDraw.Draw(f1_pil)
                # Sub-threshold micro-point
                d.point(
                    (r.randint(10, 110), r.randint(10, 110)),
                    fill=(r.randint(100, 200), r.randint(100, 200), r.randint(100, 200)),
                )
            elif name == "spinner":
                f0_pil = base.copy()
                f1_pil = base.copy()
                _draw_rotating_arc_spinner(ImageDraw.Draw(f0_pil), (64, 64), 14, 0, 4)
                _draw_rotating_arc_spinner(ImageDraw.Draw(f1_pil), (64, 64), 14, 1, 4)
            elif name == "moving_cursor":
                f0_pil = base.copy()
                f1_pil = base.copy()
                d0, d1 = ImageDraw.Draw(f0_pil), ImageDraw.Draw(f1_pil)
                d0.polygon([(30, 30), (38, 42), (33, 41), (30, 46)], fill=(20, 20, 20))
                d1.polygon([(45, 35), (53, 47), (48, 46), (45, 51)], fill=(20, 20, 20))
            elif name == "modal_popup":
                f0_pil = base.copy()
                f1_pil = base.copy()
                d1 = ImageDraw.Draw(f1_pil)
                d1.rectangle(
                    [20, 30, 108, 90], fill=(240, 240, 245), outline=(50, 50, 50), width=2
                )

            # Process through Area MT/V5 Magnocellular motion engine
            engine._prev_luminance = None
            arr0 = np.array(f0_pil)
            arr1 = np.array(f1_pil)
            engine.process(arr0)
            mf, mean_motion, moving_objs, surprise = engine.process(arr1)
            optical_flow_energy = float(engine.optical_flow_energy)

            # Decision rule for visual settling:
            # Screen is classified as SETTLED iff (surprise <= 0.08 AND optical_flow_energy <= 0.05)
            settled_pred = (surprise <= 0.08) and (optical_flow_energy <= 0.05)

            records.append({
                "scenario": name,
                "is_settled_gt": is_settled_gt,
                "settled_pred": settled_pred,
                "surprise": surprise,
                "optical_flow_energy": optical_flow_energy,
                "mean_motion": mean_motion,
                "num_moving_objs": len(moving_objs),
            })

    # Compute Confusion Matrix for Visual Settling:
    # Positive class = Settled (Static / settled screens)
    # Negative class = Unsettled (Dynamic motion, spinner, moving cursor, modal popup)
    tp = sum(1 for r in records if r["is_settled_gt"] and r["settled_pred"])
    fp = sum(1 for r in records if not r["is_settled_gt"] and r["settled_pred"])
    fn = sum(1 for r in records if r["is_settled_gt"] and not r["settled_pred"])
    tn = sum(1 for r in records if not r["is_settled_gt"] and not r["settled_pred"])

    settling_sensitivity = tp / (tp + fn + 1e-8)  # Settling Recall
    settling_specificity = tn / (tn + fp + 1e-8)
    settling_precision = tp / (tp + fp + 1e-8)
    settling_f1 = (
        2.0 * settling_precision * settling_sensitivity / (settling_precision + settling_sensitivity + 1e-8)
    )
    settling_accuracy = (tp + tn) / len(records) * 100.0

    # Also compute motion/unsettled detection metrics (where Positive = Unsettled):
    tp_motion = tn  # correctly identified as unsettled
    fp_motion = fn  # false alarm: settled screen called unsettled
    fn_motion = fp  # missed motion: unsettled screen called settled
    tn_motion = tp  # correctly identified as settled
    motion_sensitivity = tp_motion / (tp_motion + fn_motion + 1e-8)  # Motion Recall
    motion_specificity = tn_motion / (tn_motion + fp_motion + 1e-8)

    # Breakdown by scenario
    scenario_stats = {}
    for sc_name, _ in scenarios:
        sub = [r for r in records if r["scenario"] == sc_name]
        scenario_stats[sc_name] = {
            "count": len(sub),
            "mean_surprise": float(np.mean([r["surprise"] for r in sub])),
            "max_surprise": float(np.max([r["surprise"] for r in sub])),
            "mean_energy": float(np.mean([r["optical_flow_energy"] for r in sub])),
            "max_energy": float(np.max([r["optical_flow_energy"] for r in sub])),
            "settled_pred_rate": float(np.mean([1.0 if r["settled_pred"] else 0.0 for r in sub])),
        }

    return {
        "thresholds": {"surprise_max": 0.08, "energy_max": 0.05},
        "settling_detection": {
            "sensitivity_recall": settling_sensitivity,
            "specificity": settling_specificity,
            "precision": settling_precision,
            "f1_score": settling_f1,
            "accuracy_pct": settling_accuracy,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "motion_detection": {
            "sensitivity_recall": motion_sensitivity,
            "specificity": motion_specificity,
        },
        "scenario_breakdown": scenario_stats,
    }


def main():
    print("=" * 80)
    print("AREA MT/V5 OPTIC FLOW, MOTION FLUX & HEAD 1 BENCHMARK")
    print("=" * 80)

    # 1. Heatmap & Spinner Benchmark
    print("\n[1] Benchmarking Head 1 (TemporalChangeSpinnerHead)...")
    res1 = benchmark_heatmap_and_spinner()
    h = res1["heatmap"]
    s = res1["spinner"]
    print(f"  Head 1 Total Joint Loss: {res1['total_head1_loss']:.6f}")
    print(f"  Change Heatmap BCE Loss: {h['bce_loss']:.6f}")
    print(f"  Change Heatmap Mean IoU: {h['mean_iou']:.4f}")
    print(f"  Change Heatmap Sensitivity (Pixel Recall): {h['sensitivity_recall']:.4f} ({h['sensitivity_recall']*100:.2f}%)")
    print(f"  Change Heatmap Specificity: {h['specificity']:.4f} ({h['specificity']*100:.2f}%)")
    print(f"  Change Heatmap Precision:   {h['precision']:.4f}")
    print(f"  Change Heatmap F1 Score:    {h['f1_score']:.4f}")
    print(f"  Change Heatmap Pixel Accuracy: {h['accuracy']*100:.2f}%")
    print()
    print(f"  Spinner Activity BCE Loss: {s['bce_loss']:.6f}")
    print(f"  Spinner Activity Accuracy: {s['accuracy_pct']:.1f}%")
    print(f"  Spinner Sensitivity (Recall): {s['sensitivity_recall']:.4f} ({s['sensitivity_recall']*100:.1f}%)")
    print(f"  Spinner Specificity:          {s['specificity']:.4f} ({s['specificity']*100:.1f}%)")
    print(f"  Spinner Precision:            {s['precision']:.4f}")
    print(f"  Spinner F1 Score:             {s['f1_score']:.4f}")
    print(f"  Spinner ROC-AUC:              {s['roc_auc']:.4f}")

    # 2. Cyclical Frequency Sweep
    print("\n[2] Benchmarking Cyclical Frequency Detection (Periods 2..16)...")
    res2 = benchmark_cyclical_frequency_sweep()
    for period_k, p_res in res2["results_by_period"].items():
        print(f"  {period_k:18s}: Arc Recall={p_res['arc_recall']*100:.1f}%, Dots Recall={p_res['dots_recall']*100:.1f}%, Combined={p_res['combined_recall']*100:.1f}%")
    print(f"  Overall Cyclical Frequency Sensitivity/Recall: {res2['overall_cyclical_recall']:.4f} ({res2['overall_cyclical_recall']*100:.1f}%) ({res2['total_cyclical_detected']}/{res2['total_cyclical_tested']})")

    # 3. Area MT/V5 Optic Flow & Visual Settling
    print("\n[3] Benchmarking Area MT/V5 Motion Flux & Visual Settling Thresholds (0.08 surprise, 0.05 energy)...")
    res3 = benchmark_area_mt_v5_settling_thresholds(n_samples=200)
    sd = res3["settling_detection"]
    cm = sd["confusion_matrix"]
    print(f"  Decision Rule: SETTLED iff (surprise <= 0.08 AND optical_flow_energy <= 0.05)")
    print(f"  Visual Settling Sensitivity (Recall): {sd['sensitivity_recall']:.4f} ({sd['sensitivity_recall']*100:.1f}%)")
    print(f"  Visual Settling Specificity:          {sd['specificity']:.4f} ({sd['specificity']*100:.1f}%)")
    print(f"  Visual Settling Precision:            {sd['precision']:.4f} ({sd['precision']*100:.1f}%)")
    print(f"  Visual Settling F1 Score:             {sd['f1_score']:.4f}")
    print(f"  Visual Settling Overall Accuracy:     {sd['accuracy_pct']:.1f}%")
    print(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}")
    print("\n  Scenario Telemetry Breakdown:")
    for sc_name, sc_data in res3["scenario_breakdown"].items():
        print(f"    {sc_name:15s}: Mean Surprise={sc_data['mean_surprise']:.4f} (max={sc_data['max_surprise']:.4f}), "
              f"Mean Energy={sc_data['mean_energy']:.4f} (max={sc_data['max_energy']:.4f}), "
              f"Settled Pred Rate={sc_data['settled_pred_rate']*100:.1f}%")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
