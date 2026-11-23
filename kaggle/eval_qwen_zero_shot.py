"""Self-Contained Kaggle/Colab T4 Zero-Shot Evaluation Script: Qwen2.5-VL-3B.

Step 5.1 of Phase 5:
Runs un-finetuned Qwen2.5-VL-3B-Instruct on the held-out showdown-clicks-dev
benchmark (557 macOS clicks) on a single free T4 GPU.

Primary Gate Metric: Strict inside-bounding-box (0.0px tolerance).
Secondary Diagnostic Metric: hit_at_15px (Euclidean edge distance <= 15px).
Coordinate Space: Absolute screen pixels [0 to width, 0 to height].

Requirements (auto-installed in Colab/Kaggle):
pip install -q "transformers>=4.49.0" accelerate qwen-vl-utils pillow requests
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image

HF_RAW_BASE = "https://huggingface.co/datasets/generalagents/showdown-clicks/resolve/main/showdown-clicks-dev"
CSV_URL = f"{HF_RAW_BASE}/data.csv"


def wilson_score(hits: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = hits / total
    d = 1.0 + (z**2) / total
    c = p + (z**2) / (2.0 * total)
    s = z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)
    return round(max(0.0, (c - s) / d) * 100.0, 2), round(min(1.0, (c + s) / d) * 100.0, 2)


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Rescales the image dimensions to be divisible by factor and within [min_pixels, max_pixels].

    Matches Qwen2.5-VL / qwen_vl_utils smart_resize implementation.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def rescale_coordinates_to_native(
    coord: Optional[Tuple[int, int]],
    raw_w: int,
    raw_h: int,
    resized_w: int,
    resized_h: int,
) -> Optional[Tuple[int, int]]:
    """Maps predicted coordinates from Qwen2.5-VL resized pixel space back to raw native screen pixels."""
    if coord is None:
        return None
    if resized_w <= 0 or resized_h <= 0 or raw_w <= 0 or raw_h <= 0:
        return coord
    x, y = coord
    scale_x = raw_w / float(resized_w)
    scale_y = raw_h / float(resized_h)
    native_x = int(round(x * scale_x))
    native_y = int(round(y * scale_y))
    return max(0, min(raw_w - 1, native_x)), max(0, min(raw_h - 1, native_y))


def parse_prediction(
    text: str,
    w: int,
    h: int,
    raw_w: Optional[int] = None,
    raw_h: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Extracts coordinates from Qwen2.5-VL output, mapping from resized space to native screen pixels."""
    target_w = raw_w if raw_w is not None else w
    target_h = raw_h if raw_h is not None else h

    # 1. Canonical JSON coordinate (tolerates optional missing closing bracket ']')
    m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', text)
    if m:
        s_x, s_y = m.group(1), m.group(2)
        is_float = ("." in s_x) or ("." in s_y)
        x, y = float(s_x), float(s_y)
        return _normalize_and_bound(x, y, w, h, target_w, target_h, is_float=is_float)

    # 2. Qwen2.5-VL native box tokens in absolute pixels
    m_box = re.search(r'<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>', text)
    if m_box:
        y1, x1, y2, x2 = [float(v) for v in m_box.groups()]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return _normalize_and_bound(cx, cy, w, h, target_w, target_h, is_float=False)

    # 3. Plain numeric pair [x, y] (tolerates optional missing closing bracket ']')
    m_pair = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', text)
    if m_pair:
        s_x, s_y = m_pair.group(1), m_pair.group(2)
        is_float = ("." in s_x) or ("." in s_y)
        x, y = float(s_x), float(s_y)
        return _normalize_and_bound(x, y, w, h, target_w, target_h, is_float=is_float)

    return None


def _normalize_and_bound(
    x: float,
    y: float,
    canvas_w: int,
    canvas_h: int,
    target_w: int,
    target_h: int,
    is_float: bool = False,
) -> Tuple[int, int]:
    is_strictly_fractional = (0.0 < x < 1.0) or (0.0 < y < 1.0)
    is_normalized_float = is_float and (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    if (is_strictly_fractional or is_normalized_float) and (target_w > 1 and target_h > 1):
        real_x = int(round(x * target_w))
        real_y = int(round(y * target_h))
    else:
        if (canvas_w != target_w or canvas_h != target_h) and canvas_w > 0 and canvas_h > 0:
            scale_x = target_w / float(canvas_w)
            scale_y = target_h / float(canvas_h)
            real_x = int(round(x * scale_x))
            real_y = int(round(y * scale_y))
        else:
            real_x = int(round(x))
            real_y = int(round(y))

    return max(0, min(target_w - 1, real_x)), max(0, min(target_h - 1, real_y))


def run_benchmark(limit: Optional[int] = None, output_path: str = "showdown_qwen_zero_shot.json"):
    print("=" * 80)
    print("STEP 5.1: Qwen2.5-VL-3B Zero-Shot Baseline on showdown-clicks-dev (557 macOS clicks)")
    print("Primary Metric: Strict Inside-Bounding-Box (0.0px tolerance)")
    print("Secondary Metric: Diagnostic hit_at_15px (edge distance <= 15px)")
    print("=" * 80)

    # 1. Download benchmark CSV
    print(f"Downloading evaluation manifest from {CSV_URL}...")
    resp = requests.get(CSV_URL, timeout=15)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    samples = list(reader)
    if limit and limit > 0:
        samples = samples[:limit]
    total = len(samples)
    print(f"Loaded {total} held-out evaluation samples.")

    # 2. Load Model & Processor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Loading {model_id} onto device in fp16 with SDPA attention...")
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError:
        print("Installing transformers, accelerate, qwen-vl-utils...")
        os.system("pip install -q transformers>=4.49.0 accelerate qwen-vl-utils")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print(f"Model successfully loaded on {device}.")

    # 3. Evaluation Loop
    hits_strict = 0
    hits_15px = 0
    latencies = []
    results = []

    print("\nStarting inference loop with max_new_tokens=20...")
    for idx, row in enumerate(samples, start=1):
        sample_id = row["id"]
        img_rel = row["image"]
        instruction = row["instruction"]
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        w, h = int(row["width"]), int(row["height"])

        # Fetch image
        img_url = f"{HF_RAW_BASE}/{img_rel}"
        img_resp = requests.get(img_url, timeout=15)
        img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

        # Prepare prompt specifying absolute screen coordinates
        prompt_text = (
            f"Instruction: {instruction}\n"
            f"Screen Resolution: {w}x{h}\n"
            f"Coordinate Space: absolute screen pixels [0 to {w}, 0 to {h}]\n"
            "Output the physical click coordinate as a compact JSON with NO text reasoning:\n"
            '{"action": "left_click", "coordinate": [x, y]}'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,  # Expanded from 20 to 64 for zero-shot base model markdown compliance
                do_sample=False,
            )
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Rescale predicted coordinates from smart_resize space back to native w x h
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
            max_p = processor.image_processor.max_pixels
            min_p = getattr(processor.image_processor, "min_pixels", 56 * 56)
        else:
            max_p = 14 * 14 * 4 * 1280
            min_p = 56 * 56
        resized_h, resized_w = smart_resize(h, w, min_pixels=min_p, max_pixels=max_p)
        if "image_grid_thw" in inputs:
            grid_thw = inputs["image_grid_thw"]
            resized_h = int(grid_thw[0, 1].item()) * 28
            resized_w = int(grid_thw[0, 2].item()) * 28

        pred = parse_prediction(raw_output, canvas_w=resized_w, canvas_h=resized_h, raw_w=w, raw_h=h)

        # Debug print for initial samples to monitor raw token distribution
        if idx <= 10:
            print(f"[DEBUG Sample {idx}] Raw: {raw_output!r} | Pred: {pred} | Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

        # Distance calculation
        hit_strict = False
        hit_15px = False
        dist = -1.0
        if pred is not None:
            px, py = pred
            bx1, bx2 = min(x1, x2), max(x1, x2)
            by1, by2 = min(y1, y2), max(y1, y2)
            dx = max(0.0, float(bx1 - px), float(px - bx2))
            dy = max(0.0, float(by1 - py), float(py - by2))
            dist = math.hypot(dx, dy)
            in_bbox = (bx1 <= px <= bx2) and (by1 <= py <= by2)
            hit_strict = in_bbox
            hit_15px = in_bbox or (dist <= 15.0)

        if hit_strict:
            hits_strict += 1
        if hit_15px:
            hits_15px += 1

        results.append({
            "id": sample_id,
            "instruction": instruction,
            "target_bbox": [x1, y1, x2, y2],
            "predicted_coord": pred,
            "raw_output": raw_output,
            "hit_strict": hit_strict,
            "hit_at_15px": hit_15px,
            "dist_error": round(dist, 2),
            "latency_ms": round(latency_ms, 1),
        })

        if idx % 10 == 0 or idx == total:
            cur_strict = (hits_strict / idx) * 100.0
            cur_15px = (hits_15px / idx) * 100.0
            print(
                f"[{idx}/{total}] "
                f"Strict (0px): {cur_strict:.2f}% | "
                f"Diag (15px): {cur_15px:.2f}% | "
                f"Latency: {latency_ms:.1f}ms | Pred: {pred}"
            )

    acc_strict = (hits_strict / total) * 100.0
    w_low_strict, w_high_strict = wilson_score(hits_strict, total)

    acc_15px = (hits_15px / total) * 100.0
    w_low_15px, w_high_15px = wilson_score(hits_15px, total)

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(len(lat_sorted) * 0.50)]
    p90 = lat_sorted[int(len(lat_sorted) * 0.90)]

    report = {
        "benchmark": "generalagents/showdown-clicks-dev",
        "model": model_id,
        "sample_count": total,
        # PRIMARY METRIC
        "primary_metric": "strict_0px_in_bbox",
        "primary_hits": hits_strict,
        "primary_top1_accuracy": round(acc_strict, 2),
        "primary_wilson_95_ci": [w_low_strict, w_high_strict],
        # SECONDARY DIAGNOSTIC METRIC
        "secondary_metric": "diagnostic_15px_edge_distance",
        "secondary_hits_15px": hits_15px,
        "secondary_accuracy_15px": round(acc_15px, 2),
        "secondary_wilson_95_ci_15px": [w_low_15px, w_high_15px],
        # LATENCIES
        "latency_p50_ms": round(p50, 1),
        "latency_p90_ms": round(p90, 1),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1),
        "results": results,
    }

    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print("FINAL STEP 5.1 ZERO-SHOT BASELINE RESULT")
    print("=" * 80)
    print(f"Primary Top-1 Accuracy (Strict 0px): {report['primary_top1_accuracy']}%")
    print(f"Primary Wilson 95% CI: [{w_low_strict}%, {w_high_strict}%]")
    print(f"Secondary Diagnostic (15px): {report['secondary_accuracy_15px']}% [{w_low_15px}%, {w_high_15px}%]")
    print(f"P50 Latency: {report['latency_p50_ms']} ms | P90 Latency: {report['latency_p90_ms']} ms")
    print(f"Artifact written to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit sample count (default: all 557)")
    parser.add_argument("--output", type=str, default="showdown_qwen_zero_shot.json", help="Output path")
    args = parser.parse_args()
    run_benchmark(limit=args.limit, output_path=args.output)
