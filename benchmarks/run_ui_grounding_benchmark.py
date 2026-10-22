"""Held-out UI Grounding and Affordance Benchmark Runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from PIL import Image
import torch

from avcs.goal_grounder import MAX_QUERY_BYTES, encode_queries


@dataclass(frozen=True)
class GroundingBenchmarkResults:
    """Standardized results output for UI grounding benchmark evaluation."""
    total_positive_samples: int
    top_1_accuracy_pct: float
    top_1_hits: int
    median_centroid_error_px: float
    mean_centroid_error_px: float
    p95_centroid_error_px: float
    mean_iou: float
    iou_at_50_pct: float
    total_negative_queries: int
    negative_query_rejection_pct: float
    total_blank_screens: int
    blank_screen_fp_count: int
    blank_screen_false_alarm_rate: float
    center_collapse_detected: bool
    passed_all_gates: bool
    breakdown_by_category: Dict[str, Dict[str, float]]
    breakdown_by_element_kind: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        status = "PASSED ALL GATES" if self.passed_all_gates else "FAILED GATES"
        return (
            f"=== UI Grounding Benchmark Results [{status}] ===\n"
            f"Top-1 Grounding Accuracy:      {self.top_1_accuracy_pct:.2f}% (hits: {self.top_1_hits}/{self.total_positive_samples})\n"
            f"Median Centroid Error:         {self.median_centroid_error_px:.2f} px (mean: {self.mean_centroid_error_px:.2f} px, p95: {self.p95_centroid_error_px:.2f} px)\n"
            f"Mean Bounding Box IoU:         {self.mean_iou:.3f} (IoU@50: {self.iou_at_50_pct:.2f}%)\n"
            f"Ungroundable Query Rejection:  {self.negative_query_rejection_pct:.2f}% ({self.total_negative_queries} queries)\n"
            f"Negative Screen FP Count:      {self.blank_screen_fp_count} (rate: {self.blank_screen_false_alarm_rate:.2f}% across {self.total_blank_screens} screens)\n"
            f"Center-Collapse Detected:      {self.center_collapse_detected}\n"
        )


def _compute_box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Computes IoU between two [ymin, xmin, ymax, xmax] normalized boxes."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
        return 0.0

    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
    area1 = max(0.0, y1_max - y1_min) * max(0.0, x1_max - x1_min)
    area2 = max(0.0, y2_max - y2_min) * max(0.0, x2_max - x2_min)
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class UIGroundingBenchmarkRunner:
    """Independent benchmark evaluator for UI grounding models."""

    def __init__(
        self,
        manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
        device: str = "cpu",
        confidence_threshold: float = 0.45,
        tolerance_px: float = 15.0,
        image_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.tolerance_px = tolerance_px
        self.image_size = image_size

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def _infer_model(
        self,
        model: Any,
        image_tensor: torch.Tensor,
        query: str,
    ) -> Tuple[Tuple[float, float], float, Tuple[float, float, float, float]]:
        """Invokes model inference and extracts (pred_norm_xy, confidence, pred_bbox_norm).
        
        Fails closed with confidence 0.0 if query is invalid (empty, whitespace,
        oversized >96 UTF-8 bytes, non-alphanumeric) or model inference raises an exception.
        """
        # Upfront validation: empty, whitespace, non-string, oversized, or non-alphanumeric queries fail closed
        if not isinstance(query, str) or not query.strip():
            return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)
        if len(query.strip().encode("utf-8")) > MAX_QUERY_BYTES:
            return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)
        if not any(c.isalnum() for c in query):
            return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)

        if hasattr(model, "predict_point"):
            try:
                res = model.predict_point(image_tensor, query)
                pred_xy = res.get("point", (0.5, 0.5))
                conf = res.get("confidence", 1.0)
                bbox = res.get("bbox", (pred_xy[1] - 0.04, pred_xy[0] - 0.04, pred_xy[1] + 0.04, pred_xy[0] + 0.04))
                return pred_xy, conf, bbox
            except (ValueError, Exception):
                return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)

        if hasattr(model, "locate"):
            try:
                res = model.locate(query)
                pred_xy = getattr(res, "target_coord", (0.5, 0.5))
                conf = getattr(res, "confidence", 1.0)
                bbox = getattr(res, "bbox", (pred_xy[1] - 0.04, pred_xy[0] - 0.04, pred_xy[1] + 0.04, pred_xy[0] + 0.04))
                return pred_xy, conf, bbox
            except (ValueError, Exception):
                return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)

        # Standard GoalConditionedGrounder or PyTorch module
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.inference_mode():
                try:
                    q_tokens = encode_queries([query]).to(self.device)
                    img_in = image_tensor.unsqueeze(0).to(self.device)
                    logits = model(img_in, q_tokens)
                    probs = logits.flatten(1).softmax(dim=1).view_as(logits)
                    flat_idx = int(probs.flatten().argmax())
                    _, _, hm_h, hm_w = probs.shape
                    row, col = divmod(flat_idx, hm_w)
                    r_min, r_max = max(0, row - 2), min(hm_h, row + 3)
                    c_min, c_max = max(0, col - 2), min(hm_w, col + 3)
                    local_mass = float(probs[0, 0, r_min:r_max, c_min:c_max].sum())
                    conf = float(min(1.0, local_mass))
                    pred_x = (col + 0.5) / hm_w
                    pred_y = (row + 0.5) / hm_h
                    pred_box = (
                        max(0.0, pred_y - 0.04),
                        max(0.0, pred_x - 0.04),
                        min(1.0, pred_y + 0.04),
                        min(1.0, pred_x + 0.04),
                    )
                    return (pred_x, pred_y), conf, pred_box
                except (ValueError, Exception):
                    return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)

        # Fallback callable
        if callable(model):
            try:
                out = model(image_tensor, query)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    pred_xy = out[0]
                    conf = out[1]
                    pred_box = out[2] if len(out) >= 3 else (pred_xy[1] - 0.04, pred_xy[0] - 0.04, pred_xy[1] + 0.04, pred_xy[0] + 0.04)
                    return pred_xy, conf, pred_box
            except (ValueError, Exception):
                return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)

        return (0.5, 0.5), 0.0, (0.46, 0.46, 0.54, 0.54)

    def evaluate_model(
        self,
        model: Any,
        split: str = "test_held_out",
    ) -> GroundingBenchmarkResults:
        """Evaluates model against frozen test split and hostile negative controls."""
        frames = [fr for fr in self.manifest.get("frames", []) if fr.get("split") == split]
        neg_frames = [fr for fr in self.manifest.get("frames", []) if fr.get("split") == "test_negative"]

        pos_eval_samples: List[Dict[str, Any]] = []
        neg_queries_eval: List[Dict[str, Any]] = []

        for fr in frames:
            img_path = self.data_dir / fr["file_path"]
            with Image.open(img_path) as pil_img:
                if pil_img.size != self.image_size:
                    pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
                arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)

            for pt in fr.get("interactive_points", []):
                pos_eval_samples.append({
                    "image": img_t,
                    "point": pt,
                    "frame": fr,
                })

            for nq in fr.get("negative_queries", []):
                neg_queries_eval.append({
                    "image": img_t,
                    "query": nq,
                })

        # 1. Evaluate positive targets
        hits = 0
        pixel_errors: List[float] = []
        ious: List[float] = []
        pred_xs: List[float] = []
        pred_ys: List[float] = []

        cat_stats: Dict[str, Dict[str, Any]] = {}
        kind_stats: Dict[str, Dict[str, Any]] = {}

        for sample in pos_eval_samples:
            img_t = sample["image"]
            pt = sample["point"]
            fr = sample["frame"]
            query = pt.get("target_query", "click button")
            gt_x = float(pt["x"])
            gt_y = float(pt["y"])
            gt_box = pt.get("bbox", [gt_y - 0.04, gt_x - 0.04, gt_y + 0.04, gt_x + 0.04])
            w, h = fr.get("width", self.image_size[0]), fr.get("height", self.image_size[1])

            cat = fr.get("category", "unknown")
            kind = pt.get("element_kind", "unknown")

            pred_xy, conf, pred_box = self._infer_model(model, img_t, query)
            pred_x, pred_y = pred_xy
            pred_xs.append(pred_x)
            pred_ys.append(pred_y)

            # Error in physical screen pixels
            dx = (pred_x - gt_x) * w
            dy = (pred_y - gt_y) * h
            err_px = float(math.hypot(dx, dy))
            pixel_errors.append(err_px)

            # IoU
            iou = _compute_box_iou(tuple(pred_box), tuple(gt_box))
            ious.append(iou)

            # Top-1 Hit check: within bbox or tolerance radius and above threshold
            ymin, xmin, ymax, xmax = gt_box
            in_box = (xmin <= pred_x <= xmax) and (ymin <= pred_y <= ymax)
            in_radius = err_px <= self.tolerance_px
            is_hit = (in_box or in_radius) and (conf >= self.confidence_threshold)

            if is_hit:
                hits += 1

            # Accumulate category breakdown
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "hits": 0, "errors": []}
            cat_stats[cat]["total"] += 1
            if is_hit:
                cat_stats[cat]["hits"] += 1
            cat_stats[cat]["errors"].append(err_px)

            # Accumulate element kind breakdown
            if kind not in kind_stats:
                kind_stats[kind] = {"total": 0, "hits": 0, "errors": []}
            kind_stats[kind]["total"] += 1
            if is_hit:
                kind_stats[kind]["hits"] += 1
            kind_stats[kind]["errors"].append(err_px)

        total_pos = len(pos_eval_samples)
        top1_acc = (hits / total_pos * 100.0) if total_pos > 0 else 0.0
        med_err = float(np.median(pixel_errors)) if pixel_errors else 0.0
        mean_err = float(np.mean(pixel_errors)) if pixel_errors else 0.0
        p95_err = float(np.percentile(pixel_errors, 95)) if pixel_errors else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0
        iou_50 = (sum(1 for iou in ious if iou >= 0.50) / len(ious) * 100.0) if ious else 0.0

        # 2. Evaluate ungroundable negative queries
        neg_rejections = 0
        for nq_sample in neg_queries_eval:
            img_t = nq_sample["image"]
            query = nq_sample["query"]
            _, conf, _ = self._infer_model(model, img_t, query)
            if conf < self.confidence_threshold:
                neg_rejections += 1

        total_neg_q = len(neg_queries_eval)
        neg_rejection_pct = (neg_rejections / total_neg_q * 100.0) if total_neg_q > 0 else 100.0

        # 3. Evaluate blank screens & static wallpapers
        blank_fp = 0
        for bfr in neg_frames:
            img_path = self.data_dir / bfr["file_path"]
            with Image.open(img_path) as pil_img:
                if pil_img.size != self.image_size:
                    pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
                arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)

            for probe_q in ["click button", "search input", "close window"]:
                _, conf, _ = self._infer_model(model, img_t, probe_q)
                if conf >= self.confidence_threshold:
                    blank_fp += 1
                    break

        total_blanks = len(neg_frames)
        blank_fp_rate = (blank_fp / total_blanks * 100.0) if total_blanks > 0 else 0.0

        # 4. Center-collapse detection
        var_x = float(np.var(pred_xs)) if pred_xs else 0.0
        var_y = float(np.var(pred_ys)) if pred_ys else 0.0
        mean_x = float(np.mean(pred_xs)) if pred_xs else 0.5
        mean_y = float(np.mean(pred_ys)) if pred_ys else 0.5
        center_collapse = (
            (var_x < 0.02 and var_y < 0.02)
            and (abs(mean_x - 0.5) < 0.1 and abs(mean_y - 0.5) < 0.1)
            and total_pos > 10
        )

        # 5. Acceptance criteria gates
        passed_gates = (
            (top1_acc >= 85.0)
            and (med_err <= 5.0)
            and (neg_rejection_pct >= 90.0)
            and (blank_fp == 0)
            and (not center_collapse)
        )

        # Build breakdowns
        breakdown_cat = {
            c: {
                "accuracy_pct": round(stats["hits"] / max(1, stats["total"]) * 100.0, 2),
                "median_error_px": round(float(np.median(stats["errors"])), 2) if stats["errors"] else 0.0,
            }
            for c, stats in cat_stats.items()
        }
        breakdown_kind = {
            k: {
                "accuracy_pct": round(stats["hits"] / max(1, stats["total"]) * 100.0, 2),
                "median_error_px": round(float(np.median(stats["errors"])), 2) if stats["errors"] else 0.0,
            }
            for k, stats in kind_stats.items()
        }

        return GroundingBenchmarkResults(
            total_positive_samples=total_pos,
            top_1_accuracy_pct=round(top1_acc, 2),
            top_1_hits=hits,
            median_centroid_error_px=round(med_err, 2),
            mean_centroid_error_px=round(mean_err, 2),
            p95_centroid_error_px=round(p95_err, 2),
            mean_iou=round(mean_iou, 3),
            iou_at_50_pct=round(iou_50, 2),
            total_negative_queries=total_neg_q,
            negative_query_rejection_pct=round(neg_rejection_pct, 2),
            total_blank_screens=total_blanks,
            blank_screen_fp_count=blank_fp,
            blank_screen_false_alarm_rate=round(blank_fp_rate, 2),
            center_collapse_detected=center_collapse,
            passed_all_gates=passed_gates,
            breakdown_by_category=breakdown_cat,
            breakdown_by_element_kind=breakdown_kind,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UI Grounding Held-Out Benchmark")
    parser.add_argument("--manifest", type=str, default="datasets/ui_grounding/manifest.json")
    default_ckpt = (
        "runs/weights/goal_grounder_combined_best.pt"
        if Path("runs/weights/goal_grounder_combined_best.pt").is_file()
        else (
            "runs/weights/human_eye_weights.pt"
            if Path("runs/weights/human_eye_weights.pt").is_file()
            else "runs/weights/goal_grounder_best.pt"
        )
    )
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--split", type=str, default="test_held_out")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    runner = UIGroundingBenchmarkRunner(
        manifest_path=args.manifest,
        device=args.device,
        confidence_threshold=args.threshold,
    )
    print(f"Loaded benchmark manifest with {runner.manifest.get('total_frames')} frames.")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    from avcs.goal_grounder import (
        _select_grounder_architecture,
        load_grounding_checkpoint,
    )
    from avcs.weights import _extract_grounder_metadata

    try:
        model = load_grounding_checkpoint(ckpt_path, device=args.device)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        if "grounder" in ckpt:
            state = ckpt["grounder"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        if not any(k.startswith("visual_encoder.") for k in state.keys()) and any(
            k.startswith("grounder.") for k in state.keys()
        ):
            state = {k[9:] if k.startswith("grounder.") else k: v for k, v in state.items()}

        stem_weight = state.get("visual_encoder.stem.conv.weight")
        proj_weight = state.get("visual_encoder.proj.conv.weight")
        if proj_weight is not None:
            feature_dim = proj_weight.shape[0]
        elif stem_weight is not None and stem_weight.shape[0] >= 64:
            feature_dim = stem_weight.shape[0]
        else:
            hp = ckpt.get("hyperparameters", {}) if isinstance(ckpt.get("hyperparameters"), dict) else {}
            feature_dim = (
                ckpt.get("feature_dim")
                or hp.get("feature_dim")
                or (hp.get("backbone", {}).get("feature_dim") if isinstance(hp.get("backbone"), dict) else None)
                or 128
            )

        meta = _extract_grounder_metadata(ckpt)
        model = _select_grounder_architecture(state, feature_dim=feature_dim).to(args.device)
        model.load_state_dict(state, strict=False)
        if meta:
            model._mark_loaded_checkpoint(meta)
    model.eval()

    print(f"Evaluating {ckpt_path} on split '{args.split}'...")
    res = runner.evaluate_model(model, split=args.split)
    print(res.summary())

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report = res.to_dict()
        report["checkpoint"] = str(ckpt_path)
        report["checkpoint_sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        report["split"] = args.split
        report["confidence_threshold"] = args.threshold
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved evaluation report to {out_p}")


if __name__ == "__main__":
    main()
