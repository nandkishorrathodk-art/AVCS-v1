"""AVCS System 1 V1 — Standalone Neural Visual Perception CLI & Demo.

Executes end-to-end pixel perception on an image or screenshot without external LLMs,
OCR engines, accessibility APIs, or motor dispatch.

Usage:
    # Run perception on an image with a grounding query:
    python -m avcs.demo_system1 --image datasets/ui_grounding/screens/test_held_out_0001.png --query "Save button"

    # Run unconditioned scene perception (detect UI regions, categories, and text):
    python -m avcs.demo_system1 --image path/to/screenshot.png

    # Output structured machine-readable JSON:
    python -m avcs.demo_system1 --image path/to/screenshot.png --query "Terminal icon" --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs import AVCSSystem1V1, VisualStateObservation, load_system1


def format_observation_text(obs: VisualStateObservation, latency_ms: float) -> str:
    """Formats a VisualStateObservation into a clean, human-readable terminal dashboard."""
    lines = []
    lines.append("=" * 68)
    lines.append("   PARTH LABS — AVCS SYSTEM 1 V1 (NEURAL VISUAL CORTEX PREVIEW)")
    lines.append("=" * 68)
    lines.append(f"  Frame Sequence     : {obs.frame_sequence}")
    lines.append(f"  Screen Resolution  : {obs.screen_dimensions[0]}x{obs.screen_dimensions[1]} px")
    lines.append(f"  Application Type   : {obs.primary_category} ({obs.category_confidence * 100:.1f}% confidence)")
    lines.append(f"  Inference Latency  : {latency_ms:.2f} ms")
    lines.append("-" * 68)

    # UI Elements & Proposals
    regions = obs.detected_regions
    lines.append(f"  [Interactive Elements Detected]: {len(regions)}")
    if regions:
        # Tally kinds
        kinds_tally: dict[str, int] = {}
        for r in regions:
            kinds_tally[r.element_kind] = kinds_tally.get(r.element_kind, 0) + 1
        tally_str = ", ".join(f"{k}: {v}" for k, v in sorted(kinds_tally.items()))
        lines.append(f"    Breakdown        : {tally_str}")

        lines.append("    Top Detected Regions:")
        for r in regions[:6]:
            txt = f" '{r.detected_text}' ({r.text_confidence:.2f})" if r.detected_text else " [no text]"
            bbox = f"({r.bbox_px[0]:.0f}, {r.bbox_px[1]:.0f}) - ({r.bbox_px[2]:.0f}, {r.bbox_px[3]:.0f})"
            lines.append(f"      - {r.element_kind:12s} @ {bbox:24s} conf={r.confidence:.2f}{txt}")
        if len(regions) > 6:
            lines.append(f"      ... and {len(regions) - 6} more regions")
    lines.append("-" * 68)

    # Query Grounding
    if obs.active_query:
        lines.append(f"  [Active Query Grounding]: \"{obs.active_query}\"")
        lines.append(f"    Presence State   : {obs.target_presence} (conf={obs.presence_confidence:.3f})")
        if obs.selected_target_px:
            lines.append(f"    Grounded Center  : X={obs.selected_target_px[0]:.1f} px, Y={obs.selected_target_px[1]:.1f} px (norm: {obs.selected_target_norm})")
        else:
            lines.append("    Grounded Center  : None (Target not localized or sub-threshold)")

        lines.append(f"    Is Ambiguous     : {obs.is_ambiguous}")
        if obs.is_ambiguous:
            lines.append(f"    Candidate Margin : {obs.target_margin:.3f} (< 0.20 threshold)")
        if obs.uncertainty_flags:
            lines.append(f"    Uncertainty Flags: {', '.join(obs.uncertainty_flags)}")

        lines.append(f"    Action Authorized: {obs.action_authorized}")
        if not obs.action_authorized:
            reasons = []
            if obs.target_presence != "PRESENT":
                reasons.append(f"target is {obs.target_presence}")
            if obs.is_ambiguous:
                reasons.append("spatial target ambiguity")
            if obs.uncertainty_flags:
                reasons.append(f"flags={obs.uncertainty_flags}")
            lines.append(f"    Safety Status    : FAIL-CLOSED INHIBITED ({', '.join(reasons) if reasons else 'precondition failed'})")
        else:
            lines.append("    Safety Status    : ELIGIBLE FOR GUARDED MOTOR INTERACTION")
    else:
        lines.append("  [Query Grounding]  : No active query provided (unconditioned perception pass)")

    lines.append("-" * 68)
    tc = obs.temporal_change
    lines.append(f"  [Temporal State]   : {tc.stability_state} | Mean Motion: {tc.mean_motion:.4f} | Surprise: {tc.surprise_score:.4f}")
    lines.append("=" * 68)
    return "\n".join(lines)


def run_demo(
    image_path: Optional[str] = None,
    query: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    output_json: bool = False,
) -> None:
    # 1. Load Model
    model = load_system1(checkpoint_path=checkpoint, device=device)

    # 2. Acquire Image
    if image_path:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
    else:
        # Fall back to sample held-out dataset image or synthetic frame
        sample_path = os.path.join(_repo_root, "datasets", "ui_grounding", "screens", "test_held_out_0001.png")
        if os.path.isfile(sample_path):
            print(f"[Demo] No --image supplied, using sample frame: {sample_path}")
            img = Image.open(sample_path).convert("RGB")
        else:
            print("[Demo] Generating synthetic canvas for demonstration...")
            arr = np.full((720, 1280, 3), 40, dtype=np.uint8)
            img = Image.fromarray(arr)

    img_np = np.array(img, dtype=np.uint8)

    # 3. Execute Unified Perception
    t0 = time.perf_counter()
    obs = model.observe_frame(
        img_np,
        active_query=query,
        run_word_recognition=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Output Results
    if output_json:
        data = obs.to_dict()
        data["inference_latency_ms"] = round(dt_ms, 2)
        print(json.dumps(data, indent=2))
    else:
        print(format_observation_text(obs, latency_ms=dt_ms))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVCS System 1 V1 Standalone Neural Visual Perception Demo & CLI"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to screen image or screenshot (PNG, JPG, BMP).",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Natural language UI target query (e.g. 'Save button', 'Terminal icon').",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Optional path to custom AVCSSystem1V1 .pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Torch computation device ('cpu', 'cuda'). Default: 'cpu'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw machine-readable JSON observation instead of formatted dashboard.",
    )
    args = parser.parse_args()

    try:
        run_demo(
            image_path=args.image,
            query=args.query,
            checkpoint=args.checkpoint,
            device=args.device,
            output_json=args.json,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    main()
