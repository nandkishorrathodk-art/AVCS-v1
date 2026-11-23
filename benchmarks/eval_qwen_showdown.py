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


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Rescales the image so that dimensions are divisible by factor and within [min_pixels, max_pixels].

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
    raw_text: str,
    canvas_w: int,
    canvas_h: int,
    raw_w: Optional[int] = None,
    raw_h: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Extracts (x, y) coordinates from model output text.

    Qwen2.5-VL natively operates in absolute pixel coordinate space [0, W] x [0, H].
    Supports:
    1. Canonical JSON: {"action": "left_click", "coordinate": [x, y]}
       (also resilient to truncated closing bracket from token ceiling: [x, y)
    2. Qwen2.5-VL native box tokens: <|box_start|>(y1, x1, y2, x2)<|box_end|> (absolute pixels)
    3. Bracket pair: [x, y] or (x, y)

    If raw_w and raw_h are provided and differ from canvas_w and canvas_h,
    the coordinates are rescaled from resized space (canvas_w, canvas_h)
    back to the native screen resolution (raw_w, raw_h).
    """
    if not raw_text:
        return None

    target_w = raw_w if raw_w is not None else canvas_w
    target_h = raw_h if raw_h is not None else canvas_h

    # Pattern 1: Canonical JSON coordinate key (tolerates optional missing closing bracket ']')
    m_json = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', raw_text)
    if m_json:
        s_x, s_y = m_json.group(1), m_json.group(2)
        is_float = ("." in s_x) or ("." in s_y)
        x, y = float(s_x), float(s_y)
        return _normalize_and_bound(x, y, canvas_w, canvas_h, target_w, target_h, is_float=is_float)

    # Pattern 2: Qwen2.5-VL native bounding box (y1, x1, y2, x2 in absolute screen pixels)
    m_qwen_box = re.search(r'<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>', raw_text)
    if m_qwen_box:
        y1, x1, y2, x2 = [float(g) for g in m_qwen_box.groups()]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return _normalize_and_bound(cx, cy, canvas_w, canvas_h, target_w, target_h, is_float=False)

    # Pattern 3: Generic numeric coordinate pair [x, y] (tolerates optional missing closing bracket ']')
    m_pair = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', raw_text)
    if m_pair:
        s_x, s_y = m_pair.group(1), m_pair.group(2)
        is_float = ("." in s_x) or ("." in s_y)
        x, y = float(s_x), float(s_y)
        return _normalize_and_bound(x, y, canvas_w, canvas_h, target_w, target_h, is_float=is_float)

    return None


def _normalize_and_bound(
    x: float,
    y: float,
    w: int,
    h: int,
    raw_w: Optional[int] = None,
    raw_h: Optional[int] = None,
    is_float: bool = False,
) -> Tuple[int, int]:
    target_w = raw_w if raw_w is not None else w
    target_h = raw_h if raw_h is not None else h

    # A coordinate is considered normalized in [0.0, 1.0] if:
    # 1. At least one coordinate is strictly fractional (0.0 < coord < 1.0), OR
    # 2. Both coordinates are <= 1.0 AND were parsed from floating-point strings with decimal point (e.g. 1.0, 0.5)
    # Pure integer values 0 and 1 without decimals are absolute screen pixels (e.g. pixel row 1, col 1)
    is_strictly_fractional = (0.0 < x < 1.0) or (0.0 < y < 1.0)
    is_normalized_float = is_float and (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    if (is_strictly_fractional or is_normalized_float) and (target_w > 1 and target_h > 1):
        x = x * target_w
        y = y * target_h
        real_x = int(round(x))
        real_y = int(round(y))
    else:
        # Absolute pixel values in w x h space; map to target_w x target_h
        if (w != target_w or h != target_h) and w > 0 and h > 0:
            scale_x = target_w / float(w)
            scale_y = target_h / float(h)
            real_x = int(round(x * scale_x))
            real_y = int(round(y * scale_y))
        else:
            real_x = int(round(x))
            real_y = int(round(y))

    return max(0, min(target_w - 1, real_x)), max(0, min(target_h - 1, real_y))


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
        # Rescale smart_resize predicted coordinates back to native resolution
        resized_h, resized_w = smart_resize(sample.height, sample.width)
        pred_coord = parse_prediction(
            raw_output,
            canvas_w=resized_w,
            canvas_h=resized_h,
            raw_w=sample.width,
            raw_h=sample.height,
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on Showdown-Clicks-Dev Benchmark")
    parser.add_argument("--csv-path", type=str, default=r"data\showdown-clicks-dev\data.csv", help="Path to showdown data.csv")
    parser.add_argument("--cache-dir", type=str, default=r"data\showdown-clicks-dev", help="Directory where benchmark images are stored/cached")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of samples to evaluate (0 for all)")
    parser.add_argument("--tolerance-px", type=float, default=15.0, help="Diagnostic tolerance distance in pixels")
    parser.add_argument("--server-url", type=str, default="", help="Optional AVCS VLA server endpoint URL")
    parser.add_argument("--output", type=str, default="showdown_qwen_eval_results.json", help="Path to write JSON evaluation report")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        # Try finding relative to repo root
        repo_csv = Path(__file__).resolve().parent.parent / "data" / "showdown-clicks-dev" / "data.csv"
        if repo_csv.exists():
            csv_path = repo_csv
        else:
            logger.error(f"Cannot find benchmark manifest data.csv at {args.csv_path}")
            sys.exit(1)

    samples = load_showdown_samples(str(csv_path))
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        cache_dir = csv_path.parent

    # Setup inference callable
    if args.server_url:
        from avcs.runtime.vla_client import VLAClient
        client = VLAClient(server_url=args.server_url)

        def server_inference(image: Image.Image, instruction: str, w: int, h: int) -> str:
            res = client.act(image, instruction=instruction, width=w, height=h)
            return res.get("raw_response", "")

        inference_fn = server_inference
    else:
        # Standalone mock / baseline inference for test validation
        def mock_inference(image: Image.Image, instruction: str, w: int, h: int) -> str:
            resized_h, resized_w = smart_resize(h, w)
            return f'{{"action": "left_click", "coordinate": [{resized_w // 2}, {resized_h // 2}]}}'

        inference_fn = mock_inference

    max_samples = args.limit if args.limit > 0 else None
    report = run_evaluation(
        samples=samples,
        inference_fn=inference_fn,
        cache_dir=cache_dir,
        diagnostic_tolerance_px=args.tolerance_px,
        max_samples=max_samples,
    )

    out_file = Path(args.output)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Report written to {out_file.resolve()}")
    logger.info(
        f"Primary (Strict 0px): {report['primary_accuracy_percent']}% | "
        f"Secondary (15px): {report['secondary_accuracy_15px_percent']}%"
    )


if __name__ == "__main__":
    main()
