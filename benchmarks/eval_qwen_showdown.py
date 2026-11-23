"""Zero-Shot Benchmark Evaluator: Qwen2.5-VL-3B on Showdown-Clicks-Dev.

Step 5.1 of Phase 5: Measures un-finetuned foundation VLM grounding baseline
on the held-out General Agents Showdown-Clicks benchmark (557 macOS clicks).

Metrics:
1. Primary Metric: Strict Inside-Bounding-Box (0.0px tolerance).
2. Secondary Diagnostic Metric: hit_at_15px (Euclidean edge distance <= 15px).
3. Wilson 95% Confidence Interval for both metrics.
4. Strict token ceiling: max_new_tokens=20.
5. Native Absolute Pixel Coordinate Space: Qwen2.5-VL operates natively in
   absolute [0, W] x [0, H] screen pixels.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [STEP-5.1] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.qwen_showdown")

HF_DATASET_RAW_BASE = "https://huggingface.co/datasets/generalagents/showdown-clicks/resolve/main/showdown-clicks-dev"


@dataclass
class ShowdownSample:
    id: str
    image_rel_path: str
    instruction: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: int
    height: int


@dataclass
class EvalResult:
    sample_id: str
    instruction: str
    target_bbox: Tuple[float, float, float, float]
    predicted_coord: Optional[Tuple[int, int]]
    raw_response: str
    hit_strict: bool       # Primary Metric (0.0px inside bbox)
    hit_at_15px: bool      # Secondary Diagnostic Metric (<= 15px edge distance)
    dist_error: float
    latency_ms: float


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportions."""
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = p_hat + (z**2) / (2.0 * total)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)
    lower = max(0.0, (center - spread) / denom)
    upper = min(1.0, (center + spread) / denom)
    return round(lower * 100.0, 2), round(upper * 100.0, 2)


def parse_prediction(raw_text: str, canvas_w: int, canvas_h: int) -> Optional[Tuple[int, int]]:
    """Extracts (x, y) coordinates from model output text.

    Qwen2.5-VL natively operates in absolute pixel coordinate space [0, W] x [0, H].
    Supports:
    1. Canonical JSON: {"action": "left_click", "coordinate": [x, y]}
    2. Qwen2.5-VL native box tokens: <|box_start|>(y1, x1, y2, x2)<|box_end|> (absolute pixels)
    3. Bracket pair: [x, y] or (x, y)
    """
    if not raw_text:
        return None

    # Pattern 1: Canonical JSON coordinate key
    m_json = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', raw_text)
    if m_json:
        x, y = float(m_json.group(1)), float(m_json.group(2))
        return _normalize_and_bound(x, y, canvas_w, canvas_h)

    # Pattern 2: Qwen2.5-VL native bounding box (y1, x1, y2, x2 in absolute screen pixels)
    m_qwen_box = re.search(r'<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>', raw_text)
    if m_qwen_box:
        y1, x1, y2, x2 = [float(g) for g in m_qwen_box.groups()]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return _normalize_and_bound(cx, cy, canvas_w, canvas_h)

    # Pattern 3: Generic numeric coordinate pair [x, y]
    m_pair = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', raw_text)
    if m_pair:
        x, y = float(m_pair.group(1)), float(m_pair.group(2))
        return _normalize_and_bound(x, y, canvas_w, canvas_h)

    return None


def _normalize_and_bound(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    # Check if normalized in [0.0, 1.0] float space
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and (w > 1 and h > 1):
        x = x * w
        y = y * h
    # Otherwise, Qwen2.5-VL natively outputs absolute pixel values [0, W] x [0, H]
    real_x = int(round(x))
    real_y = int(round(y))
    return max(0, min(w - 1, real_x)), max(0, min(h - 1, real_y))


def load_showdown_samples(csv_path: str) -> List[ShowdownSample]:
    """Loads benchmark entries from data.csv."""
    samples: List[ShowdownSample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                ShowdownSample(
                    id=row["id"],
                    image_rel_path=row["image"],
                    instruction=row["instruction"],
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    return samples


def fetch_sample_image(sample: ShowdownSample, cache_dir: Path) -> Image.Image:
    """Fetches sample image from local cache or downloads from Hugging Face."""
    local_path = cache_dir / sample.image_rel_path
    if local_path.exists():
        return Image.open(local_path).convert("RGB")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_DATASET_RAW_BASE}/{sample.image_rel_path}"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {resp.status_code}")

    local_path.write_bytes(resp.content)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def evaluate_prediction(
    sample: ShowdownSample,
    pred_coord: Optional[Tuple[int, int]],
    diagnostic_tolerance_px: float = 15.0,
) -> Tuple[bool, bool, float]:
    """Evaluates prediction against target bounding box.

    Returns:
        (hit_strict, hit_at_15px, dist_error)
        - hit_strict (Primary): True strictly if inside bounding box (0.0px tolerance).
        - hit_at_15px (Secondary Diagnostic): True if inside bbox or within diagnostic_tolerance_px.
        - dist_error: Euclidean distance to nearest bounding box edge (0.0 if inside).
    """
    if pred_coord is None:
        return False, False, -1.0

    px, py = pred_coord
    bx1, bx2 = min(sample.x1, sample.x2), max(sample.x1, sample.x2)
    by1, by2 = min(sample.y1, sample.y2), max(sample.y1, sample.y2)

    in_bbox = (bx1 <= px <= bx2) and (by1 <= py <= by2)
    dx = max(0.0, float(bx1 - px), float(px - bx2))
    dy = max(0.0, float(by1 - py), float(py - by2))
    dist = math.hypot(dx, dy)

    hit_strict = in_bbox
    hit_at_15px = in_bbox or (dist <= diagnostic_tolerance_px)

    return hit_strict, hit_at_15px, round(dist, 2)


def run_evaluation(
    samples: List[ShowdownSample],
    inference_fn: Any,
    cache_dir: Path,
    diagnostic_tolerance_px: float = 15.0,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs zero-shot benchmark loop across samples."""
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    total = len(samples)
    logger.info(f"Starting Qwen2.5-VL-3B Zero-Shot Evaluation on {total} held-out samples...")

    results: List[EvalResult] = []
    latencies: List[float] = []
    hits_strict = 0
    hits_15px = 0

    for idx, sample in enumerate(samples, start=1):
        try:
            img = fetch_sample_image(sample, cache_dir)
        except Exception as e:
            logger.error(f"Failed to fetch image for sample {sample.id}: {e}")
            continue

        t0 = time.time()
        try:
            raw_output = inference_fn(img, sample.instruction, sample.width, sample.height)
            latency_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            logger.warning(f"Inference exception on {sample.id}: {e}")
            raw_output = f"ERROR: {e}"
            latency_ms = (time.time() - t0) * 1000.0

        latencies.append(latency_ms)
        pred_coord = parse_prediction(raw_output, sample.width, sample.height)
        hit_strict, hit_15px, dist = evaluate_prediction(sample, pred_coord, diagnostic_tolerance_px)
        if hit_strict:
            hits_strict += 1
        if hit_15px:
            hits_15px += 1

        results.append(
            EvalResult(
                sample_id=sample.id,
                instruction=sample.instruction,
                target_bbox=(sample.x1, sample.y1, sample.x2, sample.y2),
                predicted_coord=pred_coord,
                raw_response=raw_output.strip()[:200],
                hit_strict=hit_strict,
                hit_at_15px=hit_15px,
                dist_error=dist,
                latency_ms=round(latency_ms, 1),
            )
        )

        if idx % 10 == 0 or idx == total:
            cur_acc = (hits_strict / idx) * 100.0
            cur_15px = (hits_15px / idx) * 100.0
            avg_lat = sum(latencies) / len(latencies)
            logger.info(
                f"Progress: [{idx}/{total}] "
                f"Primary (Strict 0px): {cur_acc:.2f}% | "
                f"Secondary (15px): {cur_15px:.2f}% | "
                f"Latency: {avg_lat:.1f}ms"
            )

    acc_strict = (hits_strict / total) * 100.0 if total > 0 else 0.0
    w_low_strict, w_high_strict = wilson_interval(hits_strict, total)

    acc_15px = (hits_15px / total) * 100.0 if total > 0 else 0.0
    w_low_15px, w_high_15px = wilson_interval(hits_15px, total)

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0.0
    p90 = sorted_lat[int(len(sorted_lat) * 0.90)] if sorted_lat else 0.0

    report = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "benchmark": "generalagents/showdown-clicks-dev",
        "evaluation_type": "zero_shot_baseline",
        "total_samples": total,
        # PRIMARY METRIC: STRICT 0px IN-BBOX
        "primary_metric": "strict_0px_in_bbox",
        "primary_hits": hits_strict,
        "primary_accuracy_percent": round(acc_strict, 2),
        "primary_wilson_95_ci": [w_low_strict, w_high_strict],
        # SECONDARY DIAGNOSTIC METRIC: 15px
        "secondary_metric": "diagnostic_15px_edge_distance",
        "secondary_hits_15px": hits_15px,
        "secondary_accuracy_15px_percent": round(acc_15px, 2),
        "secondary_wilson_95_ci_15px": [w_low_15px, w_high_15px],
        # LATENCY TELEMETRY
        "latency_p50_ms": round(p50, 1),
        "latency_p90_ms": round(p90, 1),
        "latency_mean_ms": round(sum(latencies) / len(latencies) if latencies else 0.0, 1),
        "results": [asdict(r) for r in results],
    }

    return report
