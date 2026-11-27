"""Showdown-Clicks Benchmark Runner for Ace-Control-Medium.

Evaluates GUI click accuracy and action prediction latency on the General Agents
Showdown-Clicks benchmark (557-click development subset of 5,679 total dataset).

Measures:
1. Click Top-1 Accuracy: Predicted click inside target bounding box or within distance threshold (15px).
2. Wilson 95% Confidence Intervals: [p_low, p_high] strictly per repo standards.
3. Action Prediction Latency: P50 (median), P90, P99, and mean latency in milliseconds.
"""
from __future__ import annotations

import argparse
import base64
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

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from avcs.protocol import Action, ActionKind  # noqa: E402
from avcs.runtime import AceAgent  # noqa: E402
from avcs.os_io import Observation  # noqa: E402
from avcs.models import LocalVAMClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [BENCHMARK] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmarks.showdown")


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson 95% score interval for binomial proportion."""
    if total <= 0:
        return 0.0, 0.0

    z = 1.95996  # 95% confidence z-score
    p_hat = float(successes) / float(total)

    denominator = 1.0 + (z ** 2) / float(total)
    center_adjusted_probability = p_hat + (z ** 2) / (2.0 * float(total))
    adjusted_standard_deviation = math.sqrt(
        (p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * float(total))) / float(total)
    )

    lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator

    return max(0.0, lower_bound), min(1.0, upper_bound)


def bootstrap_confidence_interval(
    outcomes: List[bool] | List[float] | List[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes seeded percentile bootstrap confidence interval for mean accuracy."""
    n = len(outcomes)
    if n <= 0:
        return 0.0, 0.0
    if n_resamples <= 0:
        mean_val = float(np.mean(outcomes))
        return mean_val, mean_val

    arr = np.asarray(outcomes, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[indices].mean(axis=1)
    alpha = (1.0 - confidence) * 50.0
    low = float(np.percentile(boot_means, alpha))
    high = float(np.percentile(boot_means, 100.0 - alpha))
    return max(0.0, low), min(1.0, high)


@dataclass
class BenchmarkSample:
    """A single Showdown-Clicks evaluation instance."""
    id: str
    instruction: str
    target_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in 1280x800 canonical coordinates
    target_center: Tuple[int, int]          # (cx, cy)
    domain: str = "desktop"
    image_bytes: Optional[bytes] = None


@dataclass
class SampleResult:
    """Evaluation result for a single sample."""
    id: str
    instruction: str
    target_center: Tuple[int, int]
    predicted_coord: Optional[Tuple[int, int]]
    action_kind: str
    hit: bool
    distance_error: float
    latency_ms: float
    strict_hit: bool = False


def load_showdown_clicks_dev(
    data_path: str = "data/showdown-clicks-dev/data.csv",
    limit: Optional[int] = None,
    canonical_size: Tuple[int, int] = (1280, 800),
) -> List[BenchmarkSample]:
    """Loads local Showdown-Clicks dev samples from CSV or JSONL into canonical BenchmarkSample items."""
    import csv

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Showdown-Clicks manifest not found: {path}")

    samples: List[BenchmarkSample] = []
    cw, ch = canonical_size
    base_dir = path.parent

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                img_bytes = base64.b64decode(item["image_b64"]) if "image_b64" in item else None
                samples.append(
                    BenchmarkSample(
                        id=str(item.get("id", f"showdown_{len(samples):04d}")),
                        instruction=str(item.get("instruction", item.get("goal", ""))),
                        target_bbox=tuple(item.get("target_bbox", (0, 0, 50, 50))),  # type: ignore[arg-type]
                        target_center=tuple(item.get("target_center", (25, 25))),    # type: ignore[arg-type]
                        domain=str(item.get("domain", "desktop")),
                        image_bytes=img_bytes,
                    )
                )
                if limit is not None and limit > 0 and len(samples) >= limit:
                    break
        return samples

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and limit > 0 and len(samples) >= limit:
                break
            rel_img = str(row.get("image") or row.get("image_path") or row.get("img") or "").strip()
            img_file = Path(rel_img) if Path(rel_img).is_absolute() else (base_dir / rel_img)
            img_bytes: Optional[bytes] = None
            nat_w = float(row.get("width") or cw)
            nat_h = float(row.get("height") or ch)
            if rel_img and img_file.exists():
                with Image.open(img_file) as pil_img:
                    nat_w, nat_h = float(pil_img.width), float(pil_img.height)
                    resized = pil_img.convert("RGB").resize((cw, ch), Image.Resampling.BILINEAR)
                    buf = io.BytesIO()
                    resized.save(buf, format="JPEG", quality=90)
                    img_bytes = buf.getvalue()

            x1 = float(row.get("x1", 0.0))
            y1 = float(row.get("y1", 0.0))
            x2 = float(row.get("x2", x1))
            y2 = float(row.get("y2", y1))
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
                sx1, sy1, sx2, sy2 = x1 * cw, y1 * ch, x2 * cw, y2 * ch
            else:
                sx1 = (x1 / max(nat_w, 1.0)) * cw
                sy1 = (y1 / max(nat_h, 1.0)) * ch
                sx2 = (x2 / max(nat_w, 1.0)) * cw
                sy2 = (y2 / max(nat_h, 1.0)) * ch

            bx1, bx2 = int(round(min(sx1, sx2))), int(round(max(sx1, sx2)))
            by1, by2 = int(round(min(sy1, sy2))), int(round(max(sy1, sy2)))
            samples.append(
                BenchmarkSample(
                    id=str(row.get("id") or f"showdown_{idx:04d}"),
                    instruction=str(row.get("instruction") or row.get("task") or ""),
                    target_bbox=(bx1, by1, bx2, by2),
                    target_center=((bx1 + bx2) // 2, (by1 + by2) // 2),
                    domain=str(row.get("domain") or "desktop"),
                    image_bytes=img_bytes,
                )
            )
    return samples


def generate_synthetic_showdown_suite(count: int = 50) -> List[BenchmarkSample]:
    """Generates a representative cleanroom synthetic Showdown suite for offline evaluation."""
    samples: List[BenchmarkSample] = []
    
    # 5 standard UI categories with distinct visual layouts
    templates = [
        ("Open Start Menu", (20, 760, 50, 795), "taskbar"),
        ("Click Search Bar", (60, 760, 220, 795), "taskbar"),
        ("Close Window", (1240, 5, 1275, 35), "window_chrome"),
        ("Save Document", (30, 45, 75, 75), "toolbar"),
        ("Toggle Dark Mode Switch", (800, 300, 850, 325), "settings"),
        ("Select File in Explorer", (300, 250, 450, 280), "explorer"),
        ("Submit Form Button", (600, 500, 700, 535), "browser"),
        ("Click Settings Gear Icon", (1235, 760, 1265, 795), "system_tray"),
        ("Click Tab 2", (200, 10, 320, 40), "browser_tab"),
        ("Select OK in Confirmation Dialog", (540, 420, 620, 455), "modal_dialog"),
    ]

    for i in range(count):
        tmpl = templates[i % len(templates)]
        instr = f"{tmpl[0]} (sample #{i+1})"
        bbox = tmpl[1]
        domain = tmpl[2]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        # Create 1280x800 canonical visual frame
        img = Image.new("RGB", (1280, 800), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)

        # Draw target element
        draw.rectangle(bbox, fill=(65, 105, 225), outline=(30, 60, 150), width=2)
        # Draw some background chrome
        draw.rectangle([(0, 750), (1280, 800)], fill=(32, 32, 36))  # Taskbar
        draw.rectangle([(0, 0), (1280, 40)], fill=(225, 225, 230))   # Title bar

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)

        samples.append(
            BenchmarkSample(
                id=f"synth_{i+1:04d}",
                instruction=instr,
                target_bbox=bbox,
                target_center=(cx, cy),
                domain=domain,
                image_bytes=buf.getvalue(),
            )
        )

    return samples


def evaluate_sample(
    agent: AceAgent,
    sample: BenchmarkSample,
    tolerance_px: float = 15.0,
) -> SampleResult:
    """Evaluates an agent on a single Showdown click sample."""
    img_bytes = sample.image_bytes or b""
    obs = Observation(
        image_bytes=img_bytes,
        format="jpeg",
        width=1280,
        height=800,
        native_width=1280,
        native_height=800,
    )

    t0 = time.time()
    try:
        action, agent_latency = agent.step(obs, goal=sample.instruction)
    except Exception as e:
        logger.warning(f"Agent failed on sample {sample.id}: {e}")
        action = Action(kind=ActionKind.WAIT, reason=str(e))
        agent_latency = (time.time() - t0) * 1000.0

    hit = False
    strict_hit = False
    dist_error = float("inf")
    pred_coord = action.coordinate

    if action.kind in (
        ActionKind.LEFT_CLICK,
        ActionKind.RIGHT_CLICK,
        ActionKind.DOUBLE_CLICK,
        ActionKind.TRIPLE_CLICK,
    ) and pred_coord is not None:
        px, py = pred_coord
        x1, y1, x2, y2 = sample.target_bbox
        bx1, bx2 = min(x1, x2), max(x1, x2)
        by1, by2 = min(y1, y2), max(y1, y2)

        # Measure distance to closest bounding box edge (0.0 when clicked inside bounding box)
        dx = max(0.0, float(bx1 - px), float(px - bx2))
        dy = max(0.0, float(by1 - py), float(py - by2))
        dist_error = math.hypot(dx, dy)

        in_bbox = (bx1 <= px <= bx2) and (by1 <= py <= by2)
        within_radius = dist_error <= tolerance_px

        strict_hit = in_bbox
        hit = in_bbox or within_radius

    return SampleResult(
        id=sample.id,
        instruction=sample.instruction,
        target_center=sample.target_center,
        predicted_coord=pred_coord,
        action_kind=action.kind.value,
        hit=hit,
        distance_error=dist_error if dist_error != float("inf") else -1.0,
        latency_ms=agent_latency,
        strict_hit=strict_hit,
    )


def run_benchmark(
    agent: AceAgent,
    samples: List[BenchmarkSample],
    tolerance_px: float = 15.0,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs the complete Showdown-Clicks benchmark suite and produces statistical report."""
    if limit is not None and limit > 0:
        samples = samples[:limit]

    total = len(samples)
    logger.info(f"Starting Showdown-Clicks evaluation on {total} samples...")

    results: List[SampleResult] = []
    latencies: List[float] = []
    hits = 0

    for idx, sample in enumerate(samples, start=1):
        res = evaluate_sample(agent, sample, tolerance_px=tolerance_px)
        results.append(res)
        latencies.append(res.latency_ms)
        if res.hit:
            hits += 1

        if idx % 10 == 0 or idx == total:
            cur_acc = (hits / idx) * 100.0
            logger.info(f"Progress: {idx}/{total} ({cur_acc:.2f}% click accuracy, mean latency {np.mean(latencies):.1f}ms)")

    acc_rate = (hits / total) if total > 0 else 0.0
    strict_hits = sum(1 for r in results if r.strict_hit)
    strict_rate = (strict_hits / total) if total > 0 else 0.0
    w_low, w_high = wilson_score_interval(hits, total, confidence=0.95)
    b_low, b_high = bootstrap_confidence_interval([r.hit for r in results], n_resamples=1000, confidence=0.95, seed=42)
    sb_low, sb_high = bootstrap_confidence_interval([r.strict_hit for r in results], n_resamples=1000, confidence=0.95, seed=42)

    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90_latency = float(np.percentile(latencies, 90)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    report = {
        "benchmark": "generalagents/showdown-clicks",
        "sample_count": total,
        "tolerance_px": tolerance_px,
        "hits": hits,
        "strict_hits": strict_hits,
        "accuracy": round(acc_rate * 100.0, 2),
        "strict_accuracy": round(strict_rate * 100.0, 2),
        "wilson_95_ci": [round(w_low * 100.0, 2), round(w_high * 100.0, 2)],
        "bootstrap_95_ci": [round(b_low * 100.0, 2), round(b_high * 100.0, 2)],
        "strict_bootstrap_95_ci": [round(sb_low * 100.0, 2), round(sb_high * 100.0, 2)],
        "latency_p50_ms": round(p50_latency, 1),
        "latency_p90_ms": round(p90_latency, 1),
        "latency_p95_ms": round(p95_latency, 1),
        "latency_p99_ms": round(p99_latency, 1),
        "latency_mean_ms": round(mean_latency, 1),
        "results": [asdict(r) for r in results],
    }

    return report


def main() -> None:
    """CLI launcher for Showdown benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Run Showdown-Clicks benchmark evaluation.")
    parser.add_argument("--split", type=str, default="synthetic", choices=["synthetic", "dev", "custom"], help="Dataset split to evaluate.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to local JSONL dataset file.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum samples to evaluate.")
    parser.add_argument("--tolerance", type=float, default=15.0, help="Click distance tolerance in pixels.")
    parser.add_argument("--local-vam", action="store_true", help="Use local edge VAM (default; cloud fallback is disabled).")
    parser.add_argument("--local-endpoint", type=str, default="http://127.0.0.1:8000/v1", help="Local VAM endpoint URL.")
    parser.add_argument("--endpoint", type=str, default=None, help="Local VAM endpoint URL override (loopback only).")
    parser.add_argument("--model", type=str, default=None, help="Local VLA model identifier.")
    parser.add_argument("--output", type=str, default="benchmarks/results/showdown_benchmark_report.json", help="Report output file.")

    args = parser.parse_args()

    # 1. Load dataset samples
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading custom dataset from {args.data_path}...")
        samples = load_showdown_clicks_dev(args.data_path, limit=args.limit)
    elif args.split == "dev":
        dev_csv = args.data_path or "data/showdown-clicks-dev/data.csv"
        logger.info(f"Loading Showdown-Clicks dev split from {dev_csv}...")
        samples = load_showdown_clicks_dev(dev_csv, limit=args.limit)
    else:
        logger.info(f"Generating synthetic {args.split} benchmark suite with {args.limit} samples...")
        samples = generate_synthetic_showdown_suite(count=args.limit)

    # 2. Initialize Agent (strictly local VLA; external cloud fallback disabled)
    from avcs.runtime.client import is_local_endpoint

    endpoint = args.endpoint or args.local_endpoint
    if endpoint and not is_local_endpoint(endpoint):
        raise ValueError(
            f"External cloud endpoint {endpoint!r} is disabled in 100% local VLA mode; "
            "use a loopback endpoint (127.0.0.1 / localhost)."
        )
    client = LocalVAMClient(
        endpoint=endpoint,
        model=args.model or "Qwen2.5-VL-3B-Instruct",
        allow_offline_mock=True,
    )
    agent = AceAgent(client=client)

    # 3. Run Benchmark
    report = run_benchmark(agent=agent, samples=samples, tolerance_px=args.tolerance, limit=args.limit)

    # 4. Print Summary Report Table
    print("\n" + "=" * 80)
    print("SHOWDOWN-CLICKS BENCHMARK EVALUATION REPORT")
    print("=" * 80)
    print(f"Model: {getattr(client, 'model', 'default')}")
    print(f"Samples Evaluated: {report['sample_count']}")
    print(f"Click Accuracy (15px): {report['accuracy']}% [Wilson {report['wilson_95_ci']}, Bootstrap {report['bootstrap_95_ci']}]")
    print(f"Strict Accuracy (0px): {report['strict_accuracy']}% [Bootstrap {report['strict_bootstrap_95_ci']}]")
    print(f"Prediction Latency (P50/P95/P99): {report['latency_p50_ms']} / {report['latency_p95_ms']} / {report['latency_p99_ms']} ms")
    print(f"Prediction Latency (Mean): {report['latency_mean_ms']} ms")
    print("=" * 80 + "\n")

    # 5. Save Artifact
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved benchmark report to {out_path}")


if __name__ == "__main__":
    main()

