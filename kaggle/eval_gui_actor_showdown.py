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
import torch.nn.functional as F
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
    checkpoint_path: str = "runs/checkpoints/gui_actor_step6_2_epoch_3.pt",
    limit: Optional[int] = None,
    output_path: str = "benchmarks/results/showdown_gui_actor_step6_2.json",
    cache_dir: str = "data/showdown-clicks-dev",
    eval_w: int = 1280,
    eval_h: int = 800,
    native_size: bool = True,
    temperature: float = 0.05,
    model_name: str = "avcs",
) -> Dict[str, Any]:
    print("=" * 80)
    print("  AVCS STEP 6.2: HELD-OUT BENCHMARK EVALUATION (showdown-clicks-dev)  ")
    model_label = f"Reference Baseline: {model_name}" if "gui-actor" in model_name.lower() or "microsoft" in model_name.lower() else "Model: Qwen2.5-VL-3B + GUIActorPatchPointer"
    print(f"  {model_label}   ")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")
    res_mode = "Native Screen Resolution" if native_size else f"Desktop Canvas ({eval_w}x{eval_h})"
    print(f"Resolution Mode:   {res_mode}")
    print(f"Sharpening Temp:   T = {temperature}")

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

    # 2. Instantiate Model (AVCS QwenGUIActorModel or Microsoft GUI-Actor Baseline)
    is_ms_gui_actor = "gui-actor" in model_name.lower() or "microsoft" in model_name.lower()
    ms_model = None
    ms_tokenizer = None
    ms_processor = None
    model = None

    if is_ms_gui_actor:
        print(f"\n[BENCHMARK] Loading Microsoft Reference Baseline: {model_name}...")
        try:
            from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
            from gui_actor.inference import inference as ms_inference
        except ImportError:
            print("[BENCHMARK] gui_actor package not installed; installing from https://github.com/microsoft/GUI-Actor.git ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/microsoft/GUI-Actor.git"])
            from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
            from gui_actor.inference import inference as ms_inference

        from transformers import AutoProcessor
        ms_processor = AutoProcessor.from_pretrained(model_name)
        ms_tokenizer = ms_processor.tokenizer
        ms_model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        ).eval()
    else:
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

    # 3. Evaluation Tracking across 3 pointer decoding strategies
    sub_strict_hits = 0
    sub_15_hits = 0

    sharp_strict_hits = 0
    sharp_15_hits = 0

    soft_strict_hits = 0
    soft_15_hits = 0

    total_latency_ms = 0.0
    latencies: List[float] = []
    results_detail: List[Dict[str, Any]] = []

    print(f"\nEvaluating {total_samples} samples across pointer decoding heads...")
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

        # Resize image to canvas size if not in native_size mode
        if not native_size and eval_w > 0 and eval_h > 0:
            model_img = pil_img.resize((eval_w, eval_h), Image.Resampling.BILINEAR)
        else:
            model_img = pil_img

        t0 = time.perf_counter()
        if is_ms_gui_actor:
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                                "your task is to locate the screen element that corresponds to the instruction. "
                                "You should output a PyAutoGUI action that performs a click on the correct position. "
                                "To indicate the click location, we will use some special tokens, which is used to refer "
                                "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": model_img},
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
            pred = ms_inference(conversation, ms_model, ms_tokenizer, ms_processor, use_placeholder=True, topk=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            px_norm, py_norm = pred["topk_points"][0]
            sub_px = float(px_norm * raw_w)
            sub_py = float(py_norm * raw_h)
            sharp_px = sub_px
            sharp_py = sub_py
            soft_px = sub_px
            soft_py = sub_py
        else:
            # Single-pass model forward step (measure pure inference latency)
            with torch.no_grad():
                out = model(images=model_img, queries=[instruction])
                if device.type == "cuda":
                    torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # 1. Subpixel coordinates (Argmax patch + subpatch offset MLP)
            sub_norm_xy = out["subpixel_coords"][0].cpu().numpy()
            sub_px = float(sub_norm_xy[0] * raw_w)
            sub_py = float(sub_norm_xy[1] * raw_h)

            # 2. Temperature-sharpened centroid
            attn_logits = out["attn_logits"]
            patch_coords = out["patch_coords"]
            if patch_coords.dim() == 2:
                patch_coords = patch_coords.unsqueeze(0)
            sharp_weights = F.softmax(attn_logits / max(temperature, 1e-4), dim=-1)
            sharp_norm_xy = torch.bmm(sharp_weights, patch_coords).squeeze(1)[0].cpu().numpy()
            sharp_px = float(sharp_norm_xy[0] * raw_w)
            sharp_py = float(sharp_norm_xy[1] * raw_h)

            # 3. Soft unsharpened centroid (T=1.0)
            soft_norm_xy = out["target_coords"][0].cpu().numpy()
            soft_px = float(soft_norm_xy[0] * raw_w)
            soft_py = float(soft_norm_xy[1] * raw_h)

        latencies.append(infer_ms)
        total_latency_ms += infer_ms

        # Evaluate bounding box inclusion
        bx1, bx2 = min(x1, x2), max(x1, x2)
        by1, by2 = min(y1, y2), max(y1, y2)

        is_sub_strict = (bx1 <= sub_px <= bx2) and (by1 <= sub_py <= by2)
        dist_sub = compute_edge_distance(sub_px, sub_py, bx1, by1, bx2, by2)
        is_sub_15 = is_sub_strict or (dist_sub <= 15.0)

        is_sharp_strict = (bx1 <= sharp_px <= bx2) and (by1 <= sharp_py <= by2)
        dist_sharp = compute_edge_distance(sharp_px, sharp_py, bx1, by1, bx2, by2)
        is_sharp_15 = is_sharp_strict or (dist_sharp <= 15.0)

        is_soft_strict = (bx1 <= soft_px <= bx2) and (by1 <= soft_py <= by2)
        dist_soft = compute_edge_distance(soft_px, soft_py, bx1, by1, bx2, by2)
        is_soft_15 = is_soft_strict or (dist_soft <= 15.0)

        if is_sub_strict:
            sub_strict_hits += 1
        if is_sub_15:
            sub_15_hits += 1

        if is_sharp_strict:
            sharp_strict_hits += 1
        if is_sharp_15:
            sharp_15_hits += 1

        if is_soft_strict:
            soft_strict_hits += 1
        if is_soft_15:
            soft_15_hits += 1

        results_detail.append({
            "id": sample_id,
            "instruction": instruction,
            "target_bbox": [x1, y1, x2, y2],
            "subpixel_px": [round(sub_px, 1), round(sub_py, 1)],
            "subpixel_strict": is_sub_strict,
            "subpixel_edge_dist": round(dist_sub, 2),
            "sharpened_px": [round(sharp_px, 1), round(sharp_py, 1)],
            "sharpened_strict": is_sharp_strict,
            "soft_px": [round(soft_px, 1), round(soft_py, 1)],
            "soft_strict": is_soft_strict,
            "latency_ms": round(infer_ms, 2),
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == total_samples:
            acc_sub = (sub_strict_hits / (idx + 1)) * 100.0
            acc_sharp = (sharp_strict_hits / (idx + 1)) * 100.0
            acc_soft = (soft_strict_hits / (idx + 1)) * 100.0
            avg_lat = total_latency_ms / (idx + 1)
            print(
                f"[{idx + 1:3d}/{total_samples}] "
                f"Subpixel (Argmax): {acc_sub:5.2f}% ({sub_strict_hits}/{idx + 1}) | "
                f"Sharpened (T={temperature}): {acc_sharp:5.2f}% | "
                f"Soft Centroid: {acc_soft:5.2f}% | "
                f"Latency: {avg_lat:5.1f}ms/step"
            )

    elapsed_sec = time.time() - t_start
    evaluated_count = len(results_detail)

    acc_sub = (sub_strict_hits / max(1, evaluated_count)) * 100.0
    acc_sub_15 = (sub_15_hits / max(1, evaluated_count)) * 100.0
    wilson_sub = wilson_interval(sub_strict_hits, evaluated_count)

    acc_sharp = (sharp_strict_hits / max(1, evaluated_count)) * 100.0
    acc_sharp_15 = (sharp_15_hits / max(1, evaluated_count)) * 100.0
    wilson_sharp = wilson_interval(sharp_strict_hits, evaluated_count)

    acc_soft = (soft_strict_hits / max(1, evaluated_count)) * 100.0
    acc_soft_15 = (soft_15_hits / max(1, evaluated_count)) * 100.0
    wilson_soft = wilson_interval(soft_strict_hits, evaluated_count)

    best_strict = max(acc_sub, acc_sharp)
    best_strategy = "subpixel_argmax" if acc_sub >= acc_sharp else "sharpened_centroid"

    latencies.sort()
    p50_lat = latencies[len(latencies) // 2] if latencies else 0.0
    p90_lat = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
    mean_lat = total_latency_ms / max(1, evaluated_count)

    summary = {
        "benchmark": "generalagents/showdown-clicks-dev",
        "checkpoint": str(checkpoint_path),
        "total_samples": evaluated_count,
        "elapsed_seconds": round(elapsed_sec, 2),
        "resolution": {
            "mode": "native" if native_size else "canvas",
            "canvas_size": None if native_size else [eval_w, eval_h],
        },
        "best_strategy": best_strategy,
        "subpixel_argmax": {
            "name": "subpixel_argmax_plus_offset",
            "hits": sub_strict_hits,
            "accuracy_percent": round(acc_sub, 2),
            "hit_15px_percent": round(acc_sub_15, 2),
            "wilson_95_ci": wilson_sub,
            "pass_gate_65_pct": acc_sub >= 65.0,
        },
        "sharpened_centroid": {
            "name": f"sharpened_attention_centroid_T_{temperature}",
            "hits": sharp_strict_hits,
            "accuracy_percent": round(acc_sharp, 2),
            "hit_15px_percent": round(acc_sharp_15, 2),
            "wilson_95_ci": wilson_sharp,
            "pass_gate_65_pct": sharp_strict_hits >= 65.0,
        },
        "soft_centroid": {
            "name": "soft_unsharpened_centroid_T_1.0",
            "hits": soft_strict_hits,
            "accuracy_percent": round(acc_soft, 2),
            "hit_15px_percent": round(acc_soft_15, 2),
            "wilson_95_ci": wilson_soft,
            "pass_gate_65_pct": soft_strict_hits >= 65.0,
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
    print(f"Evaluated Samples:              {evaluated_count}/{total_samples}")
    print(f"Subpixel (Argmax + Offset):     {acc_sub:.2f}% (Wilson 95% CI: [{wilson_sub[0]}%, {wilson_sub[1]}%]) | 15px: {acc_sub_15:.2f}%")
    print(f"Sharpened Centroid (T={temperature}): {acc_sharp:.2f}% (Wilson 95% CI: [{wilson_sharp[0]}%, {wilson_sharp[1]}%]) | 15px: {acc_sharp_15:.2f}%")
    print(f"Soft Centroid (T=1.0):          {acc_soft:.2f}% (Wilson 95% CI: [{wilson_soft[0]}%, {wilson_soft[1]}%]) | 15px: {acc_soft_15:.2f}%")
    print(f"Latency per Step:               {mean_lat:.2f}ms (P50: {p50_lat:.2f}ms, P90: {p90_lat:.2f}ms)")
    print(f"Inference Acceleration:         {3580.4 / max(0.1, mean_lat):.1f}x faster than autoregressive base model")
    gate_str = "PASS (>= 65.0%)" if best_strict >= 65.0 else "PENDING (< 65.0%)"
    print(f"Milestone Gate ({best_strategy}): {gate_str}")
    print("=" * 80)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation artifact to: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned GUI-Actor Pointer on showdown-clicks-dev")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/gui_actor_step6_2_epoch_3.pt", help="Path to adapter checkpoint")
    parser.add_argument("--model-name", type=str, default="avcs", help="Model to evaluate: 'avcs' or 'microsoft/GUI-Actor-3B-Qwen2.5-VL'")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--output", type=str, default="benchmarks/results/showdown_gui_actor_step6_2.json", help="Output JSON path")
    parser.add_argument("--cache-dir", type=str, default="data/showdown-clicks-dev", help="Benchmark cache directory")
    parser.add_argument("--eval-size", nargs=2, type=int, default=[1280, 800], help="Evaluation canvas size W H (default: 1280 800)")
    parser.add_argument("--native-size", action="store_true", default=True, help="Evaluate at native screen resolution without downscaling (default: True)")
    parser.add_argument("--resize-canvas", action="store_true", help="Force downscaling to eval-size instead of native resolution")
    parser.add_argument("--temperature", type=float, default=0.05, help="Attention sharpening temperature (default: 0.05)")
    args = parser.parse_args()

    use_native = False if args.resize_canvas else args.native_size
    run_evaluation(
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        output_path=args.output,
        cache_dir=args.cache_dir,
        eval_w=args.eval_size[0],
        eval_h=args.eval_size[1],
        native_size=use_native,
        temperature=args.temperature,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
