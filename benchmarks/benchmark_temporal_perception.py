"""AVCS Phase 3 Temporal Visual Perception Benchmark.

Evaluates temporal state perception across consecutive frames:
1. Scenario A: Static Idle Screen (zero false motion/overlay alerts).
2. Scenario B: Cursor Movement Only (rejecting cursor motion as false modal overlay).
3. Scenario C: Modal Dialog Emergence (surprise pulse trigger & overlay_suspected detection).
4. Scenario D: Multi-Step Window Transition (icon click -> ripple -> window launch -> category transition).
5. Scenario E: Target Identity Persistence (maintaining grounded target identity across frames).

Emits populated TemporalChangeInfo and VisualStateObservation objects.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.human_eye import HumanEye, MagnocellularMotionEngine
from avcs.perception.visual_pilot import VisualUnderstandingPilot
from avcs.perception.visual_state import (
    SpatialCandidateInfo,
    TemporalChangeInfo,
    VisualRegion,
    VisualStateObservation,
)
from avcs.verifier import VisualRippleVerifier


class TemporalSceneGenerator:
    """Generates controlled, pixel-accurate desktop frame sequences."""

    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height

    def make_base_desktop(self) -> np.ndarray:
        img = Image.new("RGB", (self.width, self.height), color=(30, 50, 85))
        draw = ImageDraw.Draw(img)
        # Wallpaper accent
        draw.polygon([(0, 0), (self.width, 0), (self.width, 80), (0, 180)], fill=(40, 70, 110))
        # Taskbar
        draw.rectangle([0, self.height - 32, self.width, self.height], fill=(20, 20, 22))
        draw.rectangle([8, self.height - 26, 32, self.height - 6], fill=(0, 120, 215)) # Start

        # Desktop icon 1: Terminal
        draw.rectangle([40, 40, 80, 80], fill=(20, 20, 20), outline=(80, 80, 80))
        draw.text((48, 52), ">_", fill=(50, 220, 100))

        # Desktop icon 2: Browser
        draw.ellipse([40, 110, 80, 150], fill=(220, 60, 50), outline=(240, 180, 10))

        return np.array(img)

    def add_cursor(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img)
        # Arrow cursor
        draw.polygon([(x, y), (x, y + 14), (x + 10, y + 10)], fill=(255, 255, 255), outline=(0, 0, 0))
        return np.array(img)

    def add_click_ripple(self, frame: np.ndarray, x: int, y: int, radius: int = 20) -> np.ndarray:
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img)
        draw.rectangle([x - radius, y - radius, x + radius, y + radius], outline=(180, 220, 255), width=3)
        return np.array(img)

    def add_modal_dialog(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img)
        # Scrim dim
        # Modal box in center
        cx, cy = self.width // 2, self.height // 2
        w, h = 260, 140
        draw.rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], fill=(240, 240, 245), outline=(80, 80, 90), width=2)
        # Title bar
        draw.rectangle([cx - w//2, cy - h//2, cx + w//2, cy - h//2 + 26], fill=(0, 100, 200))
        draw.text((cx - w//2 + 10, cy - h//2 + 6), "Confirm Action", fill=(255, 255, 255))
        # OK Button
        draw.rectangle([cx + 20, cy + 30, cx + 100, cy + 55], fill=(0, 120, 215))
        draw.text((cx + 45, cy + 36), "OK", fill=(255, 255, 255))
        return np.array(img)

    def add_terminal_window(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img)
        # Terminal window
        draw.rectangle([120, 40, 520, 300], fill=(12, 12, 12), outline=(100, 100, 100), width=2)
        # Title bar
        draw.rectangle([120, 40, 520, 66], fill=(30, 30, 30))
        draw.text((130, 46), "Command Prompt - Python 3.14", fill=(220, 220, 220))
        # Terminal prompt
        draw.text((130, 75), "C:\\Users\\admin> py -3.14 -m avcs.demo", fill=(200, 200, 200))
        draw.text((130, 95), "[AVCS System 1 Active] Neural Perception Ready.", fill=(50, 230, 120))
        return np.array(img)


def run_temporal_benchmark() -> Dict[str, Any]:
    print("=" * 60)
    print("AVCS PHASE 3 — TEMPORAL VISUAL PERCEPTION BENCHMARK")
    print("=" * 60)

    gen = TemporalSceneGenerator(width=640, height=360)
    motion_engine = MagnocellularMotionEngine()

    # Load visual pilot model
    pilot_ckpt = "runs/weights/candidates/candidate_m3_visual_pilot_v1.pt"
    pilot_model = VisualUnderstandingPilot()
    if os.path.exists(pilot_ckpt):
        ckpt_data = torch.load(pilot_ckpt, map_location="cpu", weights_only=False)
        pilot_model.load_state_dict(ckpt_data["model_state_dict"])
        print(f"Loaded visual pilot checkpoint: {pilot_ckpt}")
    pilot_model.eval()

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------
    def reset_engine(eng: MagnocellularMotionEngine) -> None:
        eng._prev_luminance = None
        eng._prev_time = time.perf_counter()

    # -------------------------------------------------------------
    # Scenario A: Static Idle Desktop (5 frames)
    # -------------------------------------------------------------
    print("\n[Scenario A] Testing Static Idle Screen (zero change)...")
    base_frame = gen.make_base_desktop()
    reset_engine(motion_engine)

    a_motions: List[float] = []
    a_surprises: List[float] = []
    a_overlays: List[bool] = []

    for f_idx in range(5):
        m_field, m_mag, objs, surprise = motion_engine.process_frame(base_frame)
        overlay = motion_engine.overlay_suspected
        a_motions.append(m_mag)
        a_surprises.append(surprise)
        a_overlays.append(overlay)

    max_idle_motion = max(a_motions)
    max_idle_surprise = max(a_surprises)
    any_idle_overlay = any(a_overlays)

    pass_a = (max_idle_motion < 0.001) and (max_idle_surprise < 0.05) and (not any_idle_overlay)
    print(f"  Max Motion: {max_idle_motion:.5f} | Max Surprise: {max_idle_surprise:.4f} | Overlay FPs: {sum(a_overlays)}")
    print(f"  Scenario A Verdict: {'PASS' if pass_a else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario B: Cursor Movement Only (5 frames)
    # -------------------------------------------------------------
    print("\n[Scenario B] Testing Cursor Movement Discrimination...")
    reset_engine(motion_engine)
    motion_engine.process_frame(base_frame) # prime

    b_motions: List[float] = []
    b_surprises: List[float] = []
    b_overlays: List[bool] = []

    # Cursor moves from (100, 100) to (180, 140)
    cursor_traj = [(100 + i * 20, 100 + i * 10) for i in range(1, 5)]
    for cx, cy in cursor_traj:
        c_frame = gen.add_cursor(base_frame, cx, cy)
        m_field, m_mag, objs, surprise = motion_engine.process_frame(c_frame)
        overlay = motion_engine.overlay_suspected
        b_motions.append(m_mag)
        b_surprises.append(surprise)
        b_overlays.append(overlay)

    mean_cursor_motion = float(np.mean(b_motions))
    max_cursor_surprise = max(b_surprises)
    cursor_overlay_fp = any(b_overlays)

    pass_b = (mean_cursor_motion > 0.0001) and (not cursor_overlay_fp)
    print(f"  Mean Motion: {mean_cursor_motion:.5f} | Max Surprise: {max_cursor_surprise:.4f} | False Overlays: {sum(b_overlays)}")
    print(f"  Scenario B Verdict: {'PASS' if pass_b else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario C: Modal Dialog Emergence
    # -------------------------------------------------------------
    print("\n[Scenario C] Testing Modal Dialog Emergence...")
    reset_engine(motion_engine)
    motion_engine.process_frame(base_frame) # Frame t-1

    modal_frame = gen.add_modal_dialog(base_frame) # Frame t
    m_field, m_mag, objs, modal_surprise = motion_engine.process_frame(modal_frame)
    modal_overlay = motion_engine.overlay_suspected

    # Classify visual structure with pilot
    cat_prev, _, reg_prev = pilot_model.predict_visual_structure(base_frame)
    cat_curr, _, reg_curr = pilot_model.predict_visual_structure(modal_frame)

    appeared_regions = max(0, len(reg_curr) - len(reg_prev))

    pass_c = (modal_surprise > 0.40) and modal_overlay and (cat_curr == "modal_dialog" or appeared_regions > 0)
    print(f"  Modal Surprise Score: {modal_surprise:.4f} (>0.40 expected)")
    print(f"  Modal Overlay Suspected: {modal_overlay} (True expected)")
    print(f"  Category Shift: {cat_prev} -> {cat_curr}")
    print(f"  Appeared Interactive Regions: {appeared_regions}")
    print(f"  Scenario C Verdict: {'PASS' if pass_c else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario D: Window Launch Transition & Ripple Verification
    # -------------------------------------------------------------
    print("\n[Scenario D] Testing Icon Click & Window Transition Tracking...")
    # 1. Click Terminal Icon at (60, 60)
    terminal_icon_coord = (60.0 / 640.0, 60.0 / 360.0)
    clicked_frame = gen.add_click_ripple(base_frame, 60, 60, radius=22)

    class MemoryCapture:
        def __init__(self, sequence: List[np.ndarray]) -> None:
            self.seq = sequence
            self.idx = 0
        def grab(self) -> np.ndarray:
            f = self.seq[min(self.idx, len(self.seq) - 1)]
            self.idx += 1
            return f

    cap_click = MemoryCapture([base_frame, clicked_frame, clicked_frame])
    eye_click = HumanEye(source=cap_click, threaded_capture=False, load_weights=False)
    verifier = VisualRippleVerifier(eye=eye_click, ripple_radius=0.08)

    pre_state = eye_click.observe()
    verify_res = verifier.verify_action("click_terminal_icon", terminal_icon_coord, pre_state)

    # 2. Window expands and terminal shell appears
    term_img = Image.open("datasets/ui_grounding/screens/test_held_out_0016.png").convert("RGB").resize((640, 360))
    term_frame = np.array(term_img)
    cat_term, cat_term_conf, reg_term = pilot_model.predict_visual_structure(term_frame)

    pass_d = (verify_res.is_successful) and (cat_term == "terminal_shell")
    print(f"  Click Ripple Verified: {verify_res.is_successful} (ripple_energy={verify_res.optical_ripple_energy:.5f}, status={verify_res.status.value})")
    print(f"  Terminal Category Prediction: {cat_term} (conf={cat_term_conf:.3f})")
    print(f"  Scenario D Verdict: {'PASS' if pass_d else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario E: Target Identity Persistence Across Transitions
    # -------------------------------------------------------------
    print("\n[Scenario E] Testing Target Identity Persistence Across Frames...")
    # Ground target "Terminal" on base_frame
    # Check if target location remains consistent across frames when cursor moves
    t_id = "icon_terminal"
    t_target_box = (40.0 / 360.0, 40.0 / 640.0, 80.0 / 360.0, 80.0 / 640.0) # [ymin, xmin, ymax, xmax]

    tracked_locations = []
    for f in [base_frame, c_frame, clicked_frame]:
        _, _, regs = pilot_model.predict_visual_structure(f)
        # Find best matching region for terminal icon
        best_r = None
        best_d = 1.0
        for r in regs:
            # Distance from expected icon center
            cx = (r.bbox_norm[1] + r.bbox_norm[3]) / 2.0
            cy = (r.bbox_norm[0] + r.bbox_norm[2]) / 2.0
            d = np.hypot(cx - (60.0 / 640.0), cy - (60.0 / 360.0))
            if d < best_d:
                best_d = d
                best_r = r
        if best_r and best_d < 0.10:
            tracked_locations.append(best_r.bbox_norm)

    persistence_rate = len(tracked_locations) / 3.0 * 100.0
    pass_e = (len(tracked_locations) >= 2)
    print(f"  Tracked Target Frames: {len(tracked_locations)}/3 ({persistence_rate:.1f}%)")
    print(f"  Scenario E Verdict: {'PASS' if pass_e else 'FAIL'}")

    # -------------------------------------------------------------
    # Integrated VisualStateObservation Emission Test
    # -------------------------------------------------------------
    print("\n[Integrated VisualStateObservation Generation]...")
    m_field, m_mag, objs, surprise = motion_engine.process_frame(term_frame)
    overlay = motion_engine.overlay_suspected
    temp_info = TemporalChangeInfo(
        stability_state="overlay" if overlay else ("changed" if surprise > 0.3 else "stable"),
        mean_motion=round(m_mag, 5),
        optical_flow_energy=round(m_mag ** 2, 6),
        surprise_score=round(surprise, 4),
        overlay_suspected=overlay,
        appeared_regions_count=len(reg_term),
        disappeared_regions_count=0,
    )

    obs = VisualStateObservation(
        frame_sequence=142,
        timestamp=time.time(),
        screen_dimensions=(640, 360),
        coordinate_convention="normalized_0_1_origin_top_left",
        primary_category=cat_term,
        category_confidence=cat_term_conf,
        detected_regions=reg_term,
        target_presence="PRESENT",
        presence_confidence=0.92,
        active_query="Terminal icon",
        target_candidates=[
            SpatialCandidateInfo(
                candidate_id=0,
                coord_norm=(round(60.0/640.0, 4), round(60.0/360.0, 4)),
                coord_px=(60.0, 60.0),
                coarse_logit=3.82,
                rerank_score=0.94,
                rerank_prob=0.96,
            )
        ],
        selected_target_norm=(round(60.0/640.0, 4), round(60.0/360.0, 4)),
        selected_target_px=(60.0, 60.0),
        target_margin=0.28,
        is_ambiguous=False,
        action_authorized=False, # Fail-closed invariant
        temporal_change=temp_info,
        uncertainty_flags=[],
    )

    obs_json = obs.to_dict()
    print("Sample Generated VisualStateObservation (JSON extract):")
    print(json.dumps({
        "frame_sequence": obs_json["frame_sequence"],
        "primary_category": obs_json["primary_category"],
        "category_confidence": obs_json["category_confidence"],
        "num_detected_regions": len(obs_json["detected_regions"]),
        "target_presence": obs_json["target_presence"],
        "action_authorized": obs_json["action_authorized"],
        "temporal_change": obs_json["temporal_change"],
    }, indent=2))

    results = {
        "scenario_a_static_idle": {"pass": pass_a, "max_motion": max_idle_motion, "max_surprise": max_idle_surprise},
        "scenario_b_cursor_motion": {"pass": pass_b, "mean_motion": mean_cursor_motion, "overlay_fps": sum(b_overlays)},
        "scenario_c_modal_dialog": {"pass": pass_c, "surprise": modal_surprise, "overlay_suspected": modal_overlay, "category": cat_curr},
        "scenario_d_window_transition": {"pass": pass_d, "ripple_verified": verify_res.is_successful, "optical_delta": verify_res.optical_ripple_energy, "category": cat_term},
        "scenario_e_identity_persistence": {"pass": pass_e, "persistence_rate_pct": persistence_rate},
        "all_scenarios_passed": bool(pass_a and pass_b and pass_c and pass_d and pass_e),
    }

    out_file = "benchmarks/results/temporal_perception_benchmark.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll Scenarios Passed: {results['all_scenarios_passed']}")
    print(f"Results saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_temporal_benchmark()
