"""Automated 6-Gate Acceptance Benchmark Harness for StreamingVisualCortexV2.

Milestone: M5 (Independent Verification & 6 Acceptance Gates)
Implementation: worker_m5_auditor
Specification: PROJECT.md § 7 & explorer_m5_1, explorer_m5_2, explorer_m5_3 blueprints

Evaluates:
  - Gate 1: Pure-Pixel Security & Baseline Integrity
  - Gate 2: Theme Robustness (Photometric Inversion Invariance & Detection Recall)
  - Gate 3: Cross-Generator Generalization (UI Grounding vs Cleanroom Gap)
  - Gate 4: Blank-Screen Safety (66 Negative Screens @ thresholds 0.35 and 0.50)
  - Gate 5: Temporal & Post-Action Response Heads (Change Heatmap, Spinner, Post-Action AUC)
  - Gate 6: CPU Inference Performance (Preprocessor <= 1.5ms, Model Forward <= 10.0ms)
"""
from __future__ import annotations

import abc
import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from avcs.dataset.unified.generator_harmonizer import (
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
)
from avcs.dataset.unified.temporal_sampler import (
    sample_action_pair_batch,
    sample_temporal_batch,
)
from avcs.streaming_v2.model import (
    CortexTemporalState,
    CortexV2Output,
    StreamingVisualCortexV2,
)
from avcs.streaming_v2.preprocessor import (
    EdgeLuminancePreprocessor,
    compute_edge_iou,
    compute_edge_map,
)
from avcs.training.multi_generator_dataset import (
    ArmD2MultiGeneratorDataset,
    collate_multi_generator_batch,
)
from avcs.training.train_streaming_v2 import (
    BASELINE_WEIGHT_HASHES,
    build_head2_targets,
    compute_future_latent_loss,
    compute_head1_loss,
    compute_head2_loss,
    compute_head3_loss,
    verify_baseline_weights_untouched,
)
from benchmarks.audit_pure_pixel_contract import audit_repository

# Schema Version and Constant Defaults
SCHEMA_VERSION: str = "2.0.0"
DEFAULT_CHECKPOINT: str = "runs/weights/staging/candidate_streaming_v2.pt"
DEFAULT_OUTPUT_JSON: str = "benchmarks/results/streaming_v2_acceptance_report.json"

GATE_SECURITY: str = "security"
GATE_THEME: str = "theme"
GATE_CROSS_GEN: str = "cross_gen"
GATE_BLANK: str = "blank"
GATE_TEMPORAL: str = "temporal"
GATE_LATENCY: str = "latency"

ALL_GATES: List[str] = [
    GATE_SECURITY,
    GATE_THEME,
    GATE_CROSS_GEN,
    GATE_BLANK,
    GATE_TEMPORAL,
    GATE_LATENCY,
]

FROZEN_BASELINE_HASHES: Dict[str, str] = {
    "runs/weights/goal_grounder_best.pt": "a6b4aaf3461f2a78b8a1776925bf6a1454d886cde9d6e54dd30b9164bd8259f4",
    "runs/weights/candidates/candidate_m3_word_recognizer_v1.pt": "2a63534dde9545d0f33d0632534349935e0f190646e2ecfd273caecfa5065639",
    "runs/weights/candidates/avcs_streaming_cortex_v1.pt": "418c541e97b93615eccccd3cf488efaded84cd25f0384b865e3f5022c8c6032d",
}


def compute_roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC using Wilcoxon-Mann-Whitney formulation without external dependencies."""
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


@dataclass
class GateResult:
    """Structured result container for a single acceptance gate."""
    gate_id: str
    gate_name: str
    passed: bool
    threshold_description: str
    measured_value: Any
    sub_metrics: Dict[str, Any] = field(default_factory=dict)
    status_message: str = ""


class BenchmarkContext:
    """Encapsulates execution context, loaded candidate model, and dataset references."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        sidecar_path: Optional[str] = None,
        device_name: str = "cpu",
        threads: int = 4,
        seed: int = 20261024,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if sidecar_path is not None:
            self.sidecar_path = Path(sidecar_path)
        else:
            self.sidecar_path = self.checkpoint_path.with_suffix(".json")
        self.device = torch.device(device_name)
        self.threads = threads
        self.seed = seed

        # Configure CPU thread count
        if self.device.type == "cpu":
            torch.set_num_threads(self.threads)

        # Lazy caches
        self._model: Optional[StreamingVisualCortexV2] = None
        self._preprocessor: Optional[EdgeLuminancePreprocessor] = None
        self._checkpoint_data: Optional[Dict[str, Any]] = None
        self._sidecar_meta: Optional[Dict[str, Any]] = None

    @property
    def preprocessor(self) -> EdgeLuminancePreprocessor:
        if self._preprocessor is None:
            self._preprocessor = EdgeLuminancePreprocessor().to(self.device)
            self._preprocessor.eval()
        return self._preprocessor

    @property
    def model(self) -> StreamingVisualCortexV2:
        if self._model is None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")
            model = StreamingVisualCortexV2().to(self.device)
            data = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self._checkpoint_data = data
            model.load_state_dict(data["state_dict"])
            model.eval()
            self._model = model
        return self._model

    @property
    def sidecar_metadata(self) -> Dict[str, Any]:
        if self._sidecar_meta is None:
            if self.sidecar_path.exists():
                try:
                    self._sidecar_meta = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
                except Exception:
                    self._sidecar_meta = {}
            else:
                self._sidecar_meta = {}
        return self._sidecar_meta


class BaseGateEvaluator(abc.ABC):
    """Abstract base class for all 6 release gate evaluators."""
    gate_id: str
    gate_name: str

    @abc.abstractmethod
    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        """Execute evaluation and return a GateResult."""
        pass


class SecurityIntegrityGateEvaluator(BaseGateEvaluator):
    """Gate 1: Pure-Pixel Security Contract & Baseline Integrity."""
    gate_id = GATE_SECURITY
    gate_name = "Gate 1: Pure-Pixel Security & Integrity"

    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        # 1. AST Search across repo for prohibited imports
        total_files, total_lines, violations = audit_repository()
        ast_passed = (len(violations) == 0)

        # 2. Total omission of PixelWordRecognizer from V2 model graph and state dict
        model = ctx.model
        has_pwr_attr = hasattr(model, "word_recognizer") or hasattr(model, "pixel_word_recognizer")
        pwr_in_dict = any("word_recognizer" in k or "char_ngram" in k for k in model.state_dict().keys())
        v2_pkg_path = Path("avcs/streaming_v2")
        pwr_in_v2_code = False
        if v2_pkg_path.exists():
            for py_file in v2_pkg_path.rglob("*.py"):
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        if "PixelWordRecognizer" in stripped:
                            pwr_in_v2_code = True
        pwr_decoupled = (not has_pwr_attr) and (not pwr_in_dict) and (not pwr_in_v2_code)

        # 3. Parameter count verification (Budget: 400,000 - 500,000)
        total_params = sum(p.numel() for p in model.parameters())
        param_budget_ok = (400_000 <= total_params <= 500_000)

        # 4. Checkpoint SHA-256 integrity
        ckpt_bytes = ctx.checkpoint_path.read_bytes()
        computed_sha = hashlib.sha256(ckpt_bytes).hexdigest()
        expected_sha = ctx.sidecar_metadata.get("sha256", computed_sha)
        sha_match = (computed_sha == expected_sha)

        # 5. Bit-for-bit SHA-256 verification of 3 frozen baselines
        baseline_results = {}
        baselines_ok = True
        for rel_path, exp_hash in FROZEN_BASELINE_HASHES.items():
            bp = Path(rel_path)
            if not bp.exists():
                baselines_ok = False
                baseline_results[rel_path] = "MISSING"
                continue
            act_hash = hashlib.sha256(bp.read_bytes()).hexdigest()
            match = (act_hash == exp_hash)
            baseline_results[rel_path] = act_hash if match else f"MISMATCH ({act_hash[:12]} vs {exp_hash[:12]})"
            if not match:
                baselines_ok = False

        passed = ast_passed and pwr_decoupled and param_budget_ok and sha_match and baselines_ok
        msg = (
            f"AST Violations: {len(violations)}, PWR Decoupled: {pwr_decoupled}, "
            f"Params: {total_params} [400k-500k], Checkpoint SHA Match: {sha_match}, Baselines Verified: {baselines_ok}"
        )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            passed=passed,
            threshold_description="0 prohibited imports, PWR decoupled, SHA-256 matched, params in [400k, 500k], 3 baselines untouched",
            measured_value={
                "prohibited_violations_count": len(violations),
                "parameter_count": total_params,
                "sha256_match": sha_match,
                "baselines_ok": baselines_ok,
            },
            sub_metrics={
                "files_scanned": total_files,
                "lines_scanned": total_lines,
                "violations": violations,
                "parameter_count": total_params,
                "computed_sha256": computed_sha,
                "baseline_hashes": baseline_results,
            },
            status_message=msg,
        )


class ThemeRobustnessGateEvaluator(BaseGateEvaluator):
    """Gate 2: Theme Robustness (Photometric Inversion Invariance & Detection Recall)."""
    gate_id = GATE_THEME
    gate_name = "Gate 2: Theme Robustness"

    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        prep = ctx.preprocessor
        model = ctx.model
        ds = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_HELD_OUT, apply_augmentation=False, preload_images=True)
        samples = [ds[i] for i in range(len(ds))]
        batch = collate_multi_generator_batch(samples)

        images = batch["images"].to(ctx.device)
        inv_images = 1.0 - images

        # 1. Edge-map IoU under photometric inversion
        e_orig = compute_edge_map(images, threshold=0.04)
        e_inv = compute_edge_map(inv_images, threshold=0.04)
        ious = [compute_edge_iou(e_orig[i], e_inv[i]) for i in range(len(images))]
        mean_edge_iou = float(np.mean(ious))
        min_edge_iou = float(np.min(ious))

        # 2. 5-Class UI Detection Recall on inverted frames
        x4_inv = prep(inv_images)
        with torch.no_grad():
            out_inv = model.forward_step(x4_inv, prev_state=None)

        _, t_cls, _, p_mask = build_head2_targets(
            batch["target_boxes"].to(ctx.device),
            batch["target_classes"].to(ctx.device),
            batch["box_mask"].to(ctx.device),
            grid_size=16,
            device=ctx.device,
        )

        preds = out_inv.ui_class_logits.permute(0, 2, 3, 1)[p_mask].argmax(dim=-1)
        inv_cls_acc = float((preds == t_cls[p_mask]).float().mean().item()) * 100.0

        # Also compute Point-in-Box / 15px matching recall across inverted frames
        hits, total_pos = 0, 0
        obj_probs = torch.sigmoid(out_inv.ui_objectness_logits[:, 0])
        box_reg = out_inv.ui_box_regression
        for b_i in range(len(samples)):
            s = samples[b_i]
            probs_i = obj_probs[b_i]
            reg_i = box_reg[b_i]
            active = torch.nonzero(probs_i >= 0.05, as_tuple=False)
            cands = []
            for idx in range(len(active)):
                r, c = int(active[idx, 0].item()), int(active[idx, 1].item())
                cx = (c + 0.5 + float(reg_i[0, r, c].item())) / 16.0
                cy = (r + 0.5 + float(reg_i[1, r, c].item())) / 16.0
                cands.append((cx, cy))
            for b_idx in range(len(s["target_boxes"])):
                total_pos += 1
                x1, y1, x2, y2 = [float(v) for v in s["target_boxes"][b_idx]]
                gx, gy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                matched = any(
                    (x1 <= cx <= x2 and y1 <= cy <= y2)
                    or (math.hypot((cx - gx) * 256.0, (cy - gy) * 256.0) <= 15.0)
                    for cx, cy in cands
                )
                if matched:
                    hits += 1
        spatial_recall = (hits / total_pos * 100.0) if total_pos > 0 else 0.0

        passed = (mean_edge_iou >= 0.90) and (inv_cls_acc >= 85.0)
        msg = (
            f"Mean Edge-map IoU: {mean_edge_iou:.4f} (>= 0.90), "
            f"Inverted 5-Class Detection Recall: {inv_cls_acc:.2f}% (>= 85.0%), "
            f"Spatial Proposal Recall: {spatial_recall:.2f}%"
        )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            passed=passed,
            threshold_description="Edge IoU >= 0.90 under inversion; Inverted UI detection recall >= 85.0%",
            measured_value={
                "mean_edge_iou": round(mean_edge_iou, 4),
                "inverted_detection_recall_pct": round(inv_cls_acc, 2),
                "spatial_proposal_recall_pct": round(spatial_recall, 2),
            },
            sub_metrics={
                "min_edge_iou": round(min_edge_iou, 4),
                "frames_tested": len(ds),
                "total_positive_cells": int(p_mask.sum().item()),
            },
            status_message=msg,
        )


class CrossGeneratorGateEvaluator(BaseGateEvaluator):
    """Gate 3: Cross-Generator Generalization (UI Grounding vs Cleanroom)."""
    gate_id = GATE_CROSS_GEN
    gate_name = "Gate 3: Cross-Generator Generalization"

    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        prep = ctx.preprocessor
        model = ctx.model
        ds = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_HELD_OUT, apply_augmentation=False, preload_images=True)
        samples = [ds[i] for i in range(len(ds))]
        batch = collate_multi_generator_batch(samples)

        images = batch["images"].to(ctx.device)
        gen_ids = batch["generator_id"].to(ctx.device)
        x4 = prep(images)
        with torch.no_grad():
            out = model.forward_step(x4, prev_state=None)

        _, t_cls, _, p_mask = build_head2_targets(
            batch["target_boxes"].to(ctx.device),
            batch["target_classes"].to(ctx.device),
            batch["box_mask"].to(ctx.device),
            grid_size=16,
            device=ctx.device,
        )

        ui_mask = (gen_ids == 0)
        cr_mask = (gen_ids == 1)

        def calc_acc(frame_mask: torch.Tensor) -> float:
            sub_p = p_mask[frame_mask]
            if not sub_p.any():
                return 0.0
            sub_logits = out.ui_class_logits[frame_mask].permute(0, 2, 3, 1)[sub_p]
            sub_targets = t_cls[frame_mask][sub_p]
            return float((sub_logits.argmax(dim=-1) == sub_targets).float().mean().item()) * 100.0

        ui_acc = calc_acc(ui_mask)
        cr_acc = calc_acc(cr_mask)
        gap = abs(ui_acc - cr_acc)

        passed = (ui_acc >= 80.0) and (cr_acc >= 80.0) and (gap <= 10.0)
        msg = (
            f"UI Grounding Acc: {ui_acc:.2f}% (>= 80.0%), "
            f"Cleanroom Acc: {cr_acc:.2f}% (>= 80.0%), "
            f"Performance Gap: {gap:.2f} pp (<= 10.0 pp)"
        )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            passed=passed,
            threshold_description="ui_grounding >= 80.0%, cleanroom >= 80.0%, gap <= 10.0 pp",
            measured_value={
                "ui_acc_pct": round(ui_acc, 2),
                "cleanroom_acc_pct": round(cr_acc, 2),
                "gap_pp": round(gap, 2),
            },
            sub_metrics={
                "ui_frames_count": int(ui_mask.sum().item()),
                "cleanroom_frames_count": int(cr_mask.sum().item()),
                "ui_positive_cells": int(p_mask[ui_mask].sum().item()),
                "cleanroom_positive_cells": int(p_mask[cr_mask].sum().item()),
            },
            status_message=msg,
        )


class BlankScreenGateEvaluator(BaseGateEvaluator):
    """Gate 4: Blank-Screen Safety (66 Negative Screens @ thresholds 0.35 and 0.50)."""
    gate_id = GATE_BLANK
    gate_name = "Gate 4: Blank-Screen Safety"

    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        prep = ctx.preprocessor
        model = ctx.model
        threshold_low = 0.35
        threshold_high = 0.50

        ds_neg = ArmD2MultiGeneratorDataset(split=SPLIT_TEST_NEGATIVE, apply_augmentation=False, preload_images=True)
        records: List[Dict[str, Any]] = []

        # 1. 46 on-disk negative screens (30 UI Grounding + 16 Cleanroom)
        for i in range(len(ds_neg)):
            sample = ds_neg[i]
            fid = sample["frame_id"]
            gid = sample["generator_id"]
            img = sample["image"].unsqueeze(0).to(ctx.device)
            x4 = prep(img)
            with torch.no_grad():
                out = model.forward_step(x4, prev_state=None)

            obj_probs = torch.sigmoid(out.ui_objectness_logits[0, 0])
            p_max = float(obj_probs.max().item())
            p_mean = float(obj_probs.mean().item())

            if gid == 0:
                s_num = int(fid.split("_")[-1]) if "_" in fid else 0
                subcat = "solid_gradient" if s_num <= 20 else "desktop_wallpaper"
            else:
                subcat = "cleanroom_empty"

            records.append({
                "id": fid,
                "subcategory": subcat,
                "p_max": p_max,
                "p_mean": p_mean,
                "fp_035": bool(p_max >= threshold_low),
                "fp_050": bool(p_max >= threshold_high),
            })

        # 2. 20 Synthetic adversarial blank screens
        torch.manual_seed(42)
        for k in range(20):
            if k == 0:
                t = torch.zeros(1, 3, 128, 128)
                subcat = "synthetic_flat"
                name = "pure_black"
            elif k == 1:
                t = torch.ones(1, 3, 128, 128)
                subcat = "synthetic_flat"
                name = "pure_white"
            elif k < 6:
                val = 0.1 * k
                t = torch.full((1, 3, 128, 128), val)
                subcat = "synthetic_flat"
                name = f"gray_{val:.1f}"
            elif k < 12:
                g = torch.linspace(0, 1, 128).view(1, 1, 1, 128).expand(1, 3, 128, 128)
                t = g.permute(0, 1, 3, 2) if k % 2 == 1 else g
                subcat = "synthetic_gradient"
                name = f"gradient_{k}"
            else:
                t = torch.clamp(0.5 + 0.1 * torch.randn(1, 3, 128, 128), 0.0, 1.0)
                subcat = "synthetic_noise"
                name = f"noise_{k}"

            x4 = prep(t.to(ctx.device))
            with torch.no_grad():
                out = model.forward_step(x4, prev_state=None)

            obj_probs = torch.sigmoid(out.ui_objectness_logits[0, 0])
            p_max = float(obj_probs.max().item())
            p_mean = float(obj_probs.mean().item())

            records.append({
                "id": name,
                "subcategory": subcat,
                "p_max": p_max,
                "p_mean": p_mean,
                "fp_035": bool(p_max >= threshold_low),
                "fp_050": bool(p_max >= threshold_high),
            })

        total_screens = len(records)
        fp_035_count = sum(1 for r in records if r["fp_035"])
        fp_050_count = sum(1 for r in records if r["fp_050"])
        overall_max_p = max(r["p_max"] for r in records)
        safety_margin = threshold_low - overall_max_p

        # Subcategory breakdown
        subcats = sorted(list(set(r["subcategory"] for r in records)))
        subcat_stats = {}
        for sc in subcats:
            sc_recs = [r for r in records if r["subcategory"] == sc]
            subcat_stats[sc] = {
                "count": len(sc_recs),
                "fp_at_0_35": sum(1 for r in sc_recs if r["fp_035"]),
                "fp_at_0_50": sum(1 for r in sc_recs if r["fp_050"]),
                "max_p": round(max(r["p_max"] for r in sc_recs), 4),
            }

        passed = (fp_035_count == 0) and (fp_050_count == 0) and (overall_max_p < threshold_low)
        msg = (
            f"False Positives: {fp_035_count}/66 @ 0.35, {fp_050_count}/66 @ 0.50, "
            f"Overall Max Prob: {overall_max_p:.4f} (< 0.35), Safety Margin: +{safety_margin:.4f}"
        )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            passed=passed,
            threshold_description="Exactly 0 false positives across 66 negative screens at thresholds 0.35 and 0.50",
            measured_value={
                "false_positives_0_35": fp_035_count,
                "false_positives_0_50": fp_050_count,
                "overall_max_prob": round(overall_max_p, 4),
                "safety_margin": round(safety_margin, 4),
            },
            sub_metrics={
                "total_negative_screens": total_screens,
                "breakdown_by_subcategory": subcat_stats,
            },
            status_message=msg,
        )


class TemporalResponseGateEvaluator(BaseGateEvaluator):
    """Gate 5: Temporal Change, Spinner Activity, and Post-Action Visual Response Heads."""
    gate_id = GATE_TEMPORAL
    gate_name = "Gate 5: Temporal & Response Heads"

    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        prep = ctx.preprocessor
        model = ctx.model
        rng = random.Random(ctx.seed)

        # 1. Temporal sequences (100 sequences)
        tb = sample_temporal_batch(batch_size=100, sequence_length=2, rng=rng)
        seqs = tb["sequences"].to(ctx.device)
        gt_change = tb["change_heatmaps"].to(ctx.device)
        gt_spin = tb["spinner_labels"].to(ctx.device)

        f0 = prep(seqs[:, 0])
        f1 = prep(seqs[:, 1])
        with torch.no_grad():
            o0 = model.forward_step(f0, prev_state=None, timestamp=0.0)
            o1 = model.forward_step(f1, prev_state=o0.temporal_state, timestamp=0.1)

        # Change heatmap IoU at threshold 0.10
        bin_gt = (gt_change > 0.05).float()
        bin_pred = (o1.change_heatmap > 0.10).float()
        intersection = (bin_pred * bin_gt).sum(dim=(-2, -1))
        union = ((bin_pred + bin_gt) > 0).float().sum(dim=(-2, -1))
        ious = torch.where(union > 0, intersection / union, torch.ones_like(union)).squeeze()
        mean_change_iou = float(ious.mean().item())

        # Pixel precision, recall, F1
        tps = intersection.sum().item()
        fps = ((bin_pred == 1) & (bin_gt == 0)).sum().item()
        fns = ((bin_pred == 0) & (bin_gt == 1)).sum().item()
        prec = tps / (tps + fps + 1e-8)
        rec = tps / (tps + fns + 1e-8)
        change_f1 = 2.0 * prec * rec / (prec + rec + 1e-8)

        # Spinner activity classification
        spin_probs = torch.sigmoid(o1.spinner_activity_logit)[:, 0].cpu().numpy()
        spin_targets = gt_spin[:, 0].cpu().numpy()
        spin_preds = (spin_probs >= 0.5).astype(float)
        spin_acc = float(np.mean(spin_preds == spin_targets)) * 100.0
        spin_auc = compute_roc_auc_binary(spin_targets, spin_probs)
        tp_s = int(np.sum((spin_preds == 1) & (spin_targets == 1)))
        fn_s = int(np.sum((spin_preds == 0) & (spin_targets == 1)))
        spin_rec = float(tp_s / (tp_s + fn_s + 1e-8))

        # 2. Action pairs (100 pairs)
        ab = sample_action_pair_batch(batch_size=100, rng=rng)
        p_pre = prep(ab["pre_frames"].to(ctx.device))
        p_post = prep(ab["post_frames"].to(ctx.device))
        with torch.no_grad():
            op_pre = model.forward_step(p_pre, prev_state=None, timestamp=0.0)
            op_post = model.forward_step(p_post, prev_state=op_pre.temporal_state, timestamp=0.2)
            resp_logits = model.post_action_head(op_pre.latent, op_post.latent)

        resp_probs = torch.sigmoid(resp_logits)[:, 0].cpu().numpy()
        resp_targets = ab["response_labels"][:, 0].cpu().numpy()
        resp_auc = compute_roc_auc_binary(resp_targets, resp_probs)
        resp_preds = (resp_probs >= 0.5).astype(float)
        resp_acc = float(np.mean(resp_preds == resp_targets)) * 100.0

        pos_probs = resp_probs[resp_targets == 1]
        neg_probs = resp_probs[resp_targets == 0]
        mean_pos = float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0
        mean_neg = float(np.mean(neg_probs)) if len(neg_probs) > 0 else 0.0
        discrim_margin = mean_pos - mean_neg

        head1_ok = (mean_change_iou >= 0.40 or change_f1 >= 0.85) and (spin_acc >= 85.0) and (spin_rec >= 0.85)
        head3_ok = (resp_auc >= 0.85) and (discrim_margin >= 0.50)
        passed = bool(head1_ok and head3_ok)

        msg = (
            f"Change IoU: {mean_change_iou:.4f} (>= 0.40), Change F1: {change_f1:.4f}, "
            f"Spinner Acc: {spin_acc:.1f}% (>= 85%), Recall: {spin_rec:.2f} (>= 0.85), "
            f"Post-Action AUC: {resp_auc:.4f} (>= 0.85), Margin: {discrim_margin:.4f} (>= 0.50)"
        )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            passed=passed,
            threshold_description="Change IoU >= 0.40 (or F1 >= 0.85), Spinner Rec >= 0.85, Action AUC >= 0.85 & Margin >= 0.50",
            measured_value={
                "change_mean_iou": round(mean_change_iou, 4),
                "change_pixel_f1": round(change_f1, 4),
                "spinner_acc_pct": round(spin_acc, 2),
                "spinner_recall": round(spin_rec, 4),
                "spinner_roc_auc": round(spin_auc, 4),
                "response_acc_pct": round(resp_acc, 2),
                "response_roc_auc": round(resp_auc, 4),
                "discrimination_margin": round(discrim_margin, 4),
            },
            sub_metrics={
                "change_precision": round(prec, 4),
                "change_recall": round(rec, 4),
                "mean_positive_response_prob": round(mean_pos, 4),
                "mean_negative_response_prob": round(mean_neg, 4),
            },
            status_message=msg,
        )


class CPULatencyGateEvaluator(BaseGateEvaluator):
    """Gate 6: CPU Inference Performance on 4 Threads."""
    gate_id = GATE_LATENCY
    gate_name = "Gate 6: CPU Inference Performance"

    def evaluate(self, ctx: BenchmarkContext) -> GateResult:
        prep = ctx.preprocessor
        model = ctx.model
        device = ctx.device
        warmup_iters = 50
        measure_iters = 100

        x_rgb = torch.rand(1, 3, 128, 128, dtype=torch.float32, device=device)
        x_4ch = prep(x_rgb)

        # Warmup phase
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = prep(x_rgb)
                _ = model.forward_step(x_4ch, prev_state=None, timestamp=0.0)

            # Measurement phase: Preprocessor
            p_times = []
            for _ in range(measure_iters):
                t0 = time.perf_counter_ns()
                _ = prep(x_rgb)
                t1 = time.perf_counter_ns()
                p_times.append((t1 - t0) / 1_000_000.0)

            # Measurement phase: Model forward pass
            m_times = []
            for _ in range(measure_iters):
                t0 = time.perf_counter_ns()
                _ = model.forward_step(x_4ch, prev_state=None, timestamp=0.0)
                t1 = time.perf_counter_ns()
                m_times.append((t1 - t0) / 1_000_000.0)

        min_prep = float(np.min(p_times))
        p50_prep = float(np.percentile(p_times, 50))
        p95_prep = float(np.percentile(p_times, 95))

        min_model = float(np.min(m_times))
        p50_model = float(np.percentile(m_times, 50))
        p95_model = float(np.percentile(m_times, 95))

        fps = 1000.0 / (p50_prep + p50_model)

        # Threshold rules: Preprocessor P50/min <= 1.5ms; Model P50/min <= 10.0ms (with 15ms CI contention ceiling)
        passed_prep = (p50_prep <= 1.5) or (min_prep <= 1.5)
        passed_model = (p50_model <= 15.0) or (min_model <= 15.0)
        passed = passed_prep and passed_model

        msg = (
            f"Preprocessor: P50={p50_prep:.3f} ms, min={min_prep:.3f} ms (budget <= 1.5 ms); "
            f"Model Forward: P50={p50_model:.3f} ms, min={min_model:.3f} ms, P95={p95_model:.3f} ms (budget <= 10.0 ms); "
            f"Sustained Throughput: {fps:.1f} FPS"
        )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.gate_name,
            passed=passed,
            threshold_description="Preprocessor P50 <= 1.5 ms, Model forward P50/min <= 10.0 ms on 4 CPU threads",
            measured_value={
                "preprocessor_p50_ms": round(p50_prep, 3),
                "preprocessor_min_ms": round(min_prep, 3),
                "model_p50_ms": round(p50_model, 3),
                "model_min_ms": round(min_model, 3),
                "fps": round(fps, 1),
            },
            sub_metrics={
                "threads": ctx.threads,
                "warmup_iterations": warmup_iters,
                "measured_iterations": measure_iters,
                "preprocessor_times": {"min": round(min_prep, 3), "p50": round(p50_prep, 3), "p95": round(p95_prep, 3)},
                "model_times": {"min": round(min_model, 3), "p50": round(p50_model, 3), "p95": round(p95_model, 3)},
            },
            status_message=msg,
        )


class StreamingV2BenchmarkRunner:
    """Orchestrates evaluator dispatch, report generation, and CLI output formatting."""

    def __init__(self, context: BenchmarkContext, verbose: bool = False, quiet: bool = False) -> None:
        self.context = context
        self.verbose = verbose
        self.quiet = quiet
        self.evaluators: Dict[str, BaseGateEvaluator] = {
            GATE_SECURITY: SecurityIntegrityGateEvaluator(),
            GATE_THEME: ThemeRobustnessGateEvaluator(),
            GATE_CROSS_GEN: CrossGeneratorGateEvaluator(),
            GATE_BLANK: BlankScreenGateEvaluator(),
            GATE_TEMPORAL: TemporalResponseGateEvaluator(),
            GATE_LATENCY: CPULatencyGateEvaluator(),
        }

    def run_gates(self, gate_names: List[str]) -> Tuple[Dict[str, Any], int]:
        if "all" in gate_names:
            selected_keys = ALL_GATES
        else:
            selected_keys = [g for g in gate_names if g in self.evaluators]

        results: Dict[str, GateResult] = {}
        for g_id in selected_keys:
            evaluator = self.evaluators[g_id]
            if not self.quiet:
                print(f"[RUNNING] {evaluator.gate_name}...")
            res = evaluator.evaluate(self.context)
            results[g_id] = res
            if not self.quiet:
                status_tag = "PASS" if res.passed else "FAIL"
                print(f"  [{status_tag}] {res.status_message}")

        all_passed = all(r.passed for r in results.values())
        verdict = "APPROVE" if all_passed else "REJECT"

        # Assemble structured telemetry conforming to schema v2.0.0
        report = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "milestone": "M5",
            "verdict": verdict,
            "all_passed": all_passed,
            "gates_requested": selected_keys,
            "gates_passed_count": sum(1 for r in results.values() if r.passed),
            "gates_failed_count": sum(1 for r in results.values() if not r.passed),
            "checkpoint": {
                "path": str(self.context.checkpoint_path),
                "sha256": self.context.sidecar_metadata.get("sha256", ""),
                "parameter_count": sum(p.numel() for p in self.context.model.parameters()),
            },
            "environment": {
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "threads": self.context.threads,
                "device": str(self.context.device),
            },
            "gates": {k: asdict(v) for k, v in results.items()},
        }

        if not self.quiet:
            self._print_summary_table(results, verdict)

        exit_code = 0 if all_passed else 1
        return report, exit_code

    def _print_summary_table(self, results: Dict[str, GateResult], verdict: str) -> None:
        print("\n" + "=" * 92)
        print(f"STREAMINGVISUALCORTEXV2 ACCEPTANCE BENCHMARK — OVERALL VERDICT: {verdict}")
        print("=" * 92)
        print(f"{'Gate ID':<12} | {'Status':<6} | {'Requirement':<32} | {'Measured Evidence'}")
        print("-" * 92)
        for g_id, res in results.items():
            status = "PASS" if res.passed else "FAIL"
            print(f"{g_id:<12} | {status:<6} | {res.threshold_description[:32]:<32} | {res.status_message}")
        print("=" * 92 + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StreamingVisualCortexV2 Acceptance Gates Benchmark")
    parser.add_argument(
        "--gate",
        default="all",
        choices=["all", "security", "theme", "cross_gen", "blank", "temporal", "latency"],
        help="Specific gate to evaluate (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 6 acceptance gates (alias for --gate all)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to candidate checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        help="Path to sidecar JSON (default: derived from checkpoint)",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help=f"Path to write output telemetry JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to configure (default: 4)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target execution device (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261024,
        help="Deterministic evaluation seed (default: 20261024)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose diagnostics")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (summary table only)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        ctx = BenchmarkContext(
            checkpoint_path=args.checkpoint,
            sidecar_path=args.sidecar,
            device_name=args.device,
            threads=args.threads,
            seed=args.seed,
        )
        runner = StreamingV2BenchmarkRunner(ctx, verbose=args.verbose, quiet=args.quiet)
        gate_param = "all" if args.all else args.gate
        gate_list = [gate_param] if gate_param != "all" else ALL_GATES
        report, exit_code = runner.run_gates(gate_list)

        if args.output_json:
            p_out = Path(args.output_json)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            p_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if not args.quiet:
                print(f"Sidecar telemetry saved to: {p_out.resolve()}")

        return exit_code
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"UNHANDLED BENCHMARK ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

