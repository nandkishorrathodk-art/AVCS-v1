"""Evaluation Script: Fine-Tuned Qwen2.5-VL-3B + GUIActorPatchPointer on showdown-clicks-dev.

Evaluates the non-autoregressive single-pass GUIActorPatchPointer adapter head attached to
the frozen Qwen2.5-VL-3B backbone on the held-out General Agents Showdown-Clicks benchmark.

Primary Gate Metric: Strict inside-bounding-box (0.0px tolerance) >= 65.0%.
Secondary Diagnostic: hit_at_15px (Euclidean edge distance <= 15px).
Latency Metric: Single forward pass time in milliseconds (target: < 50ms).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image

# Ensure current directory and repository root are prioritized on sys.path
current_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from train_vla_gui_actor import QwenGUIActorModel
except ImportError:
    import importlib.util
    train_script = os.path.join(current_dir, "train_vla_gui_actor.py")
    spec = importlib.util.spec_from_file_location("train_vla_gui_actor", train_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_vla_gui_actor"] = mod
    spec.loader.exec_module(mod)
    QwenGUIActorModel = mod.QwenGUIActorModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [STEP-6.2-EVAL] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.gui_actor")

HF_RAW_BASE = "https://huggingface.co/datasets/generalagents/showdown-clicks/resolve/main/showdown-clicks-dev"
CSV_URL = f"{HF_RAW_BASE}/data.csv"


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for binomial proportions."""
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = p_hat + (z**2) / (2.0 * total)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)
    lower = max(0.0, (center - spread) / denom)
    upper = min(1.0, (center + spread) / denom)
    return round(lower * 100.0, 2), round(upper * 100.0, 2)


def compute_edge_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Computes Euclidean distance from a predicted point to the nearest bounding box edge.

    Returns 0.0 if the point is strictly inside the box.
    """
    bx1, bx2 = min(x1, x2), max(x1, x2)
    by1, by2 = min(y1, y2), max(y1, y2)
    dx = max(0.0, max(bx1 - px, px - bx2))
    dy = max(0.0, max(by1 - py, py - by2))
    return math.hypot(dx, dy)


def run_evaluation(
    checkpoint_path: str = "runs/checkpoints/gui_actor_step6_2_epoch_1.pt",
    limit: Optional[int] = None,
    output_path: str = "benchmarks/results/showdown_gui_actor_step6_2.json",
    cache_dir: str = "data/showdown-clicks-dev",
) -> Dict[str, Any]:
    print("=" * 80)
    print("  AVCS STEP 6.2: HELD-OUT BENCHMARK EVALUATION (showdown-clicks-dev)  ")
    print("  Model: Qwen2.5-VL-3B + GUIActorPatchPointer (Single Forward Pass)   ")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # 1. Download or load benchmark manifest
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_csv = cache_path / "data.csv"

    if not local_csv.exists():
        print(f"Downloading benchmark manifest from {CSV_URL}...")
        resp = requests.get(CSV_URL, timeout=20)
        resp.raise_for_status()
        local_csv.write_text(resp.text, encoding="utf-8")
        manifest_text = resp.text
    else:
        print(f"Loading cached manifest from {local_csv}...")
        manifest_text = local_csv.read_text(encoding="utf-8")

    reader = csv.DictReader(io.StringIO(manifest_text))
    samples = list(reader)
    if limit and limit > 0:
        samples = samples[:limit]
    total_samples = len(samples)
    print(f"Loaded {total_samples} held-out evaluation samples.")

    # 2. Instantiate Model and Load Trained Adapter Checkpoint
    print(f"Initializing QwenGUIActorModel on {device}...")
    model = QwenGUIActorModel(
        dry_run=not torch.cuda.is_available(),
        pointer_hidden_dim=512,
        query_dim=512,
    )

    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        print(f"Loading adapter weights from checkpoint: {ckpt_file}...")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded: {len(state_dict)} tensors restored (missing={len(missing)}, unexpected={len(unexpected)}).")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found; evaluating un-finetuned adapter head.")

    model.to(device)
    model.eval()

    # 3. Evaluation Loop
    strict_hits = 0
    hit_15px_count = 0
    total_latency_ms = 0.0
    latencies: List[float] = []
    results_detail: List[Dict[str, Any]] = []

    print(f"\nEvaluating {total_samples} samples...")
    t_start = time.time()

    for idx, row in enumerate(samples):
        sample_id = row.get("id", str(idx))
        img_rel = row.get("image") or row.get("image_rel_path") or row.get("image_path")
        if not img_rel:
            continue
        instruction = row["instruction"]
        x1, y1 = float(row["x1"]), float(row["y1"])
        x2, y2 = float(row["x2"]), float(row["y2"])
        raw_w, raw_h = int(row["width"]), int(row["height"])

        # Fetch / cache sample image
        img_file = cache_path / img_rel
        if not img_file.exists():
            img_file.parent.mkdir(parents=True, exist_ok=True)
            img_url = f"{HF_RAW_BASE}/{img_rel}"
            try:
                r_img = requests.get(img_url, timeout=15)
                r_img.raise_for_status()
                img_file.write_bytes(r_img.content)
            except Exception as e:
                logger.warning(f"Failed to fetch {img_url}: {e}")
                continue

        try:
            pil_img = Image.open(img_file).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open image {img_file}: {e}")
            continue

        # Single-pass model forward step (measure pure inference latency)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(images=pil_img, queries=[instruction])
            if device.type == "cuda":
                torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(infer_ms)
        total_latency_ms += infer_ms

        # Map normalized coordinates [0, 1] to native pixel space
        pred_norm_xy = out["target_coords"][0].cpu().numpy()
        pred_px = float(pred_norm_xy[0] * raw_w)
        pred_py = float(pred_norm_xy[1] * raw_h)

        # Evaluate bounding box inclusion
        bx1, bx2 = min(x1, x2), max(x1, x2)
        by1, by2 = min(y1, y2), max(y1, y2)
        is_strict = (bx1 <= pred_px <= bx2) and (by1 <= pred_py <= by2)
        dist_err = compute_edge_distance(pred_px, pred_py, bx1, by1, bx2, by2)
        is_15px = is_strict or (dist_err <= 15.0)

        if is_strict:
            strict_hits += 1
        if is_15px:
            hit_15px_count += 1

        results_detail.append({
            "id": sample_id,
            "instruction": instruction,
            "pred_px": [round(pred_px, 1), round(pred_py, 1)],
            "target_bbox": [x1, y1, x2, y2],
            "strict_hit": is_strict,
            "edge_dist": round(dist_err, 2),
            "latency_ms": round(infer_ms, 2),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
            cur_acc = (strict_hits / (idx + 1)) * 100.0
            cur_15 = (hit_15px_count / (idx + 1)) * 100.0
            avg_lat = total_latency_ms / (idx + 1)
            print(
                f"[{idx + 1:3d}/{total_samples}] Strict: {cur_acc:5.2f}% ({strict_hits}/{idx + 1}) | "
                f"15px: {cur_15:5.2f}% | Latency: {avg_lat:5.1f}ms/step"
            )

    elapsed_sec = time.time() - t_start
    evaluated_count = len(results_detail)
    strict_acc = (strict_hits / max(1, evaluated_count)) * 100.0
    hit_15_acc = (hit_15px_count / max(1, evaluated_count)) * 100.0
    wilson_strict = wilson_interval(strict_hits, evaluated_count)
    wilson_15 = wilson_interval(hit_15px_count, evaluated_count)

    latencies.sort()
    p50_lat = latencies[len(latencies) // 2] if latencies else 0.0
    p90_lat = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
    mean_lat = total_latency_ms / max(1, evaluated_count)

    summary = {
        "benchmark": "generalagents/showdown-clicks-dev",
        "checkpoint": str(checkpoint_path),
        "total_samples": evaluated_count,
        "elapsed_seconds": round(elapsed_sec, 2),
        "primary_metric": {
            "name": "strict_0.0px_in_bbox",
            "hits": strict_hits,
            "accuracy_percent": round(strict_acc, 2),
            "wilson_95_ci": wilson_strict,
            "pass_gate_65_pct": strict_acc >= 65.0,
        },
        "secondary_metric": {
            "name": "diagnostic_15.0px_edge_distance",
            "hits": hit_15px_count,
            "accuracy_percent": round(hit_15_acc, 2),
            "wilson_95_ci": wilson_15,
        },
        "latency_telemetry": {
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p90_ms": round(p90_lat, 2),
            "speedup_vs_autoregressive_baseline": round(3580.4 / max(0.1, mean_lat), 1),
        },
    }

    print("\n" + "=" * 80)
    print("  FINAL EVALUATION REPORT: STEP 6.2 GUI-ACTOR POINTER  ")
    print("=" * 80)
    print(f"Evaluated Samples:      {evaluated_count}/{total_samples}")
    print(f"Strict In-Bbox (0.0px): {strict_acc:.2f}% (Wilson 95% CI: [{wilson_strict[0]}%, {wilson_strict[1]}%])")
    print(f"Hit at 15.0px Edge:     {hit_15_acc:.2f}% (Wilson 95% CI: [{wilson_15[0]}%, {wilson_15[1]}%])")
    print(f"Latency per Step:       {mean_lat:.2f}ms (P50: {p50_lat:.2f}ms, P90: {p90_lat:.2f}ms)")
    print(f"Inference Acceleration: {3580.4 / max(0.1, mean_lat):.1f}x faster than autoregressive base model")
    gate_str = "PASS (>= 65.0%)" if strict_acc >= 65.0 else "PENDING (< 65.0%)"
    print(f"Milestone Gate:         {gate_str}")
    print("=" * 80)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation artifact to: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned GUI-Actor Pointer on showdown-clicks-dev")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/gui_actor_step6_2_epoch_1.pt", help="Path to adapter checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--output", type=str, default="benchmarks/results/showdown_gui_actor_step6_2.json", help="Output JSON path")
    parser.add_argument("--cache-dir", type=str, default="data/showdown-clicks-dev", help="Benchmark cache directory")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
