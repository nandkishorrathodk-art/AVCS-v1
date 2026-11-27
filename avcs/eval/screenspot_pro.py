"""ScreenSpot-Pro Dense High-Resolution Grounding Harness (Pure-RGB).

Evaluates GUI grounding on dense professional desktop applications (CAD, IDEs,
creative tools, spreadsheets, OS chrome) at native 1080p/1440p resolutions,
reporting overall accuracy, strict 0px accuracy, small-icon (<24px) recall,
center L2 pixel error, Wilson/bootstrap 95% CIs, and P50/P95/P99 latencies.
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from avcs.eval.showdown import bootstrap_confidence_interval, wilson_score_interval
from avcs.os_io.computer import Observation
from avcs.protocol.actions import Action, ActionKind
from avcs.runtime.agent import AceAgent


@dataclasses.dataclass
class ScreenSpotProSample:
    """Single ScreenSpot-Pro high-resolution grounding instance."""

    id: str
    instruction: str
    app_category: str
    target_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in screen_size pixels
    target_center: Tuple[int, int]
    screen_size: Tuple[int, int] = (1920, 1080)
    is_small_icon: bool = False
    image_bytes: Optional[bytes] = None


@dataclasses.dataclass
class ScreenSpotProResult:
    """Evaluation outcome for one ScreenSpot-Pro sample."""

    id: str
    app_category: str
    is_small_icon: bool
    screen_size: Tuple[int, int]
    predicted_native_xy: Optional[Tuple[int, int]]
    action_kind: str
    hit: bool
    strict_hit: bool
    center_l2_px: float
    edge_distance_px: float
    latency_ms: float


def _is_small_target(bbox: Tuple[int, int, int, int], max_px: int = 24) -> bool:
    w = max(0, abs(bbox[2] - bbox[0]))
    h = max(0, abs(bbox[3] - bbox[1]))
    return max(w, h) <= max_px


def load_screenspot_pro(
    data_dir: str = "data/screenspot-pro",
    limit: Optional[int] = None,
    small_icon_max_px: int = 24,
) -> List[ScreenSpotProSample]:
    """Loads local ScreenSpot-Pro JSON/JSONL annotations with pure-RGB images."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"ScreenSpot-Pro directory not found: {root}")

    annotation_files = sorted(list(root.glob("*.json")) + list(root.glob("*.jsonl")))
    if not annotation_files:
        raise FileNotFoundError(f"No JSON/JSONL annotation manifests found inside {root}")

    records: List[Dict[str, Any]] = []
    for ann_file in annotation_files:
        if ann_file.suffix.lower() == ".jsonl":
            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        else:
            with open(ann_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    records.extend(loaded)
                elif isinstance(loaded, dict) and isinstance(loaded.get("samples"), list):
                    records.extend(loaded["samples"])

    samples: List[ScreenSpotProSample] = []
    for idx, rec in enumerate(records):
        if limit is not None and limit > 0 and len(samples) >= limit:
            break

        img_rel = str(rec.get("img_filename") or rec.get("image_path") or rec.get("image") or "").strip()
        img_path = Path(img_rel) if (img_rel and Path(img_rel).is_absolute()) else (root / img_rel)
        img_bytes: Optional[bytes] = None
        sw, sh = 1920, 1080
        if img_rel and img_path.exists():
            with Image.open(img_path) as pil_img:
                rgb = pil_img.convert("RGB")
                sw, sh = rgb.size
                buf = io.BytesIO()
                rgb.save(buf, format="JPEG", quality=90)
                img_bytes = buf.getvalue()
        elif isinstance(rec.get("img_size"), (list, tuple)) and len(rec["img_size"]) >= 2:
            sw, sh = int(rec["img_size"][0]), int(rec["img_size"][1])

        raw_bbox = rec.get("bbox") or rec.get("target_bbox") or [0, 0, 20, 20]
        vals = [float(v) for v in raw_bbox[:4]]
        if max(abs(v) for v in vals) <= 1.0:
            bx1, by1, bx2, by2 = vals[0] * sw, vals[1] * sh, vals[2] * sw, vals[3] * sh
        elif len(raw_bbox) == 4 and (vals[2] < vals[0] or vals[3] < vals[1]) and not rec.get("bbox_is_xyxy", False):
            # Treat ambiguous tiny third/fourth components as [x, y, w, h] if marked
            bx1, by1 = vals[0], vals[1]
            bx2, by2 = vals[0] + vals[2], vals[1] + vals[3]
        else:
            bx1, by1, bx2, by2 = vals[0], vals[1], vals[2], vals[3]

        bbox = (
            int(round(min(bx1, bx2))),
            int(round(min(by1, by2))),
            int(round(max(bx1, bx2))),
            int(round(max(by1, by2))),
        )
        samples.append(
            ScreenSpotProSample(
                id=str(rec.get("id") or f"sspro_{idx:04d}"),
                instruction=str(rec.get("instruction") or rec.get("query") or ""),
                app_category=str(rec.get("application") or rec.get("category") or rec.get("group") or "desktop"),
                target_bbox=bbox,
                target_center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                screen_size=(sw, sh),
                is_small_icon=_is_small_target(bbox, max_px=small_icon_max_px),
                image_bytes=img_bytes,
            )
        )
    return samples


def generate_synthetic_screenspot_pro_suite(
    count: int = 30,
    small_icon_max_px: int = 24,
) -> List[ScreenSpotProSample]:
    """Generates a deterministic cleanroom dense high-res ScreenSpot-Pro smoke suite."""
    templates = [
        ("CAD: Toggle Ortho Snap Icon", "cad", (1920, 1080), (412, 1044, 430, 1062)),
        ("IDE: Step Over Debug Toolbar Button", "ide", (1920, 1080), (968, 42, 988, 62)),
        ("Creative: Brush Hardness Eyedropper", "creative", (2560, 1440), (34, 318, 54, 338)),
        ("Spreadsheet: Freeze Top Row Menu Item", "spreadsheet", (1920, 1080), (620, 96, 740, 118)),
        ("OS Chrome: Minimize Window Control", "os_chrome", (2560, 1440), (2462, 6, 2482, 24)),
        ("IDE: Split Editor Right Micro-Icon", "ide", (2560, 1440), (2410, 44, 2428, 62)),
    ]

    samples: List[ScreenSpotProSample] = []
    for i in range(count):
        instr_base, app_cat, (sw, sh), bbox = templates[i % len(templates)]
        img = Image.new("RGB", (sw, sh), color=(28, 30, 34))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (sw, 72)], fill=(42, 45, 52))
        draw.rectangle([(0, 72), (68, sh)], fill=(36, 38, 44))
        draw.rectangle(bbox, fill=(72, 164, 242), outline=(220, 235, 255), width=2)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)

        samples.append(
            ScreenSpotProSample(
                id=f"sspro_synth_{i+1:04d}",
                instruction=f"{instr_base} #{i+1}",
                app_category=app_cat,
                target_bbox=bbox,
                target_center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                screen_size=(sw, sh),
                is_small_icon=_is_small_target(bbox, max_px=small_icon_max_px),
                image_bytes=buf.getvalue(),
            )
        )
    return samples


def evaluate_screenspot_sample(
    agent: AceAgent,
    sample: ScreenSpotProSample,
    tolerance_px: float = 15.0,
) -> ScreenSpotProResult:
    """Evaluates a pure-RGB agent on a single high-resolution ScreenSpot-Pro frame."""
    sw, sh = sample.screen_size
    obs = Observation(
        image_bytes=sample.image_bytes or b"",
        format="jpeg",
        width=AceAgent.CANONICAL_WIDTH,
        height=AceAgent.CANONICAL_HEIGHT,
        native_width=sw,
        native_height=sh,
    )

    t0 = time.perf_counter()
    try:
        action, latency_ms = agent.step(obs, goal=sample.instruction)
    except Exception as e:
        action = Action(kind=ActionKind.WAIT, reason=str(e))
        latency_ms = (time.perf_counter() - t0) * 1000.0

    hit = False
    strict_hit = False
    edge_dist = -1.0
    center_l2 = -1.0
    native_xy: Optional[Tuple[int, int]] = None

    if action.is_click() and action.coordinate is not None:
        cx_canon, cy_canon = action.coordinate
        px = int(round((float(cx_canon) / float(max(1, AceAgent.CANONICAL_WIDTH))) * sw))
        py = int(round((float(cy_canon) / float(max(1, AceAgent.CANONICAL_HEIGHT))) * sh))
        native_xy = (px, py)

        bx1, by1, bx2, by2 = sample.target_bbox
        dx = max(0.0, float(bx1 - px), float(px - bx2))
        dy = max(0.0, float(by1 - py), float(py - by2))
        edge_dist = math.hypot(dx, dy)
        center_l2 = math.hypot(float(px - sample.target_center[0]), float(py - sample.target_center[1]))
        strict_hit = (bx1 <= px <= bx2) and (by1 <= py <= by2)
        hit = strict_hit or (edge_dist <= tolerance_px)

    return ScreenSpotProResult(
        id=sample.id,
        app_category=sample.app_category,
        is_small_icon=sample.is_small_icon,
        screen_size=sample.screen_size,
        predicted_native_xy=native_xy,
        action_kind=action.kind.value,
        hit=hit,
        strict_hit=strict_hit,
        center_l2_px=center_l2,
        edge_distance_px=edge_dist,
        latency_ms=float(latency_ms),
    )


def run_screenspot_pro(
    agent: AceAgent,
    samples: List[ScreenSpotProSample],
    tolerance_px: float = 15.0,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs ScreenSpot-Pro grounding evaluation and returns metrics + per-category breakdown."""
    if limit is not None and limit > 0:
        samples = samples[:limit]

    results = [evaluate_screenspot_sample(agent, s, tolerance_px=tolerance_px) for s in samples]
    total = len(results)
    hits = sum(1 for r in results if r.hit)
    strict_hits = sum(1 for r in results if r.strict_hit)

    small_results = [r for r in results if r.is_small_icon]
    small_hits = sum(1 for r in small_results if r.hit)
    small_strict = sum(1 for r in small_results if r.strict_hit)

    l2_vals = [r.center_l2_px for r in results if r.center_l2_px >= 0.0]
    latencies = [r.latency_ms for r in results]

    w_low, w_high = wilson_score_interval(hits, total, confidence=0.95)
    b_low, b_high = bootstrap_confidence_interval([r.hit for r in results], n_resamples=1000, confidence=0.95, seed=42)

    by_category: Dict[str, Dict[str, Any]] = {}
    for r in results:
        bucket = by_category.setdefault(r.app_category, {"count": 0, "hits": 0, "strict_hits": 0})
        bucket["count"] += 1
        if r.hit:
            bucket["hits"] += 1
        if r.strict_hit:
            bucket["strict_hits"] += 1
    for cat_stats in by_category.values():
        cnt = max(1, cat_stats["count"])
        cat_stats["accuracy"] = round((cat_stats["hits"] / cnt) * 100.0, 2)
        cat_stats["strict_accuracy"] = round((cat_stats["strict_hits"] / cnt) * 100.0, 2)

    return {
        "benchmark": "screenspot-pro",
        "observation_mode": "pure_rgb",
        "sample_count": total,
        "tolerance_px": tolerance_px,
        "hits": hits,
        "strict_hits": strict_hits,
        "accuracy": round((hits / max(1, total)) * 100.0, 2),
        "strict_accuracy": round((strict_hits / max(1, total)) * 100.0, 2),
        "small_icon_count": len(small_results),
        "small_icon_recall": round((small_hits / max(1, len(small_results))) * 100.0, 2) if small_results else 0.0,
        "small_icon_strict_recall": round((small_strict / max(1, len(small_results))) * 100.0, 2) if small_results else 0.0,
        "mean_center_l2_px": round(float(np.mean(l2_vals)), 1) if l2_vals else -1.0,
        "median_center_l2_px": round(float(np.median(l2_vals)), 1) if l2_vals else -1.0,
        "wilson_95_ci": [round(w_low * 100.0, 2), round(w_high * 100.0, 2)],
        "bootstrap_95_ci": [round(b_low * 100.0, 2), round(b_high * 100.0, 2)],
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0.0,
        "by_category": by_category,
        "results": [dataclasses.asdict(r) for r in results],
    }
