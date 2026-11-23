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


def parse_prediction(text: str, w: int, h: int) -> Optional[Tuple[int, int]]:
    """Extracts absolute pixel coordinates [0 to w, 0 to h] from Qwen2.5-VL output."""
    # 1. Canonical JSON coordinate
    m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 1:
            x, y = x * w, y * h
        return max(0, min(w - 1, int(round(x)))), max(0, min(h - 1, int(round(y))))

    # 2. Qwen2.5-VL native box tokens in absolute pixels
    m_box = re.search(r'<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>', text)
    if m_box:
        y1, x1, y2, x2 = [float(v) for v in m_box.groups()]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and w > 1:
            cx, cy = cx * w, cy * h
        return max(0, min(w - 1, int(round(cx)))), max(0, min(h - 1, int(round(cy))))

    # 3. Plain numeric pair [x, y]
    m_pair = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if m_pair:
        x, y = float(m_pair.group(1)), float(m_pair.group(2))
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 1:
            x, y = x * w, y * h
        return max(0, min(w - 1, int(round(x)))), max(0, min(h - 1, int(round(y))))

    return None


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

        pred = parse_prediction(raw_output, w, h)

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
