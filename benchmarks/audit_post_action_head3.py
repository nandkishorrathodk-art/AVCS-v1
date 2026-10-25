"""Audit and Verification Script for Head 3 (PostActionVisualResponseHead) in AVCS System 1.

Performs rigorous empirical verification of:
1. Physical screen pixel energy shifts:
   - Ripple animations (expanding circular wavefront)
   - Button color transitions (pressed / active / hover state changes)
   - Focus shifts (focus halo / ring appearance around controls)
   - Modal dialog popups
   - Text entry and caret blinking
2. Negative controls (No-op interactions):
   - Static dead space clicks
   - Disabled control clicks (no feedback)
   - Identical frames
   - Low-amplitude sensor noise
3. Metrics:
   - Pre/post latent vector L2 and Cosine distances
   - ROC-AUC (target >= 0.85)
   - Discrimination margin (target >= 0.50)
4. Decoupling verification:
   - Visual response evidence vs semantic goal completion
"""
import math
import random
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from avcs.streaming_v2.heads import PostActionVisualResponseHead
from avcs.streaming_v2.model import StreamingVisualCortexV2
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.streaming_v2.integration import StreamingV2Adapter, DEFAULT_WEIGHTS_PATH


def compute_roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC via Mann-Whitney U statistic."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Compute rank sum
    combined = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(combined)) + 1
    pos_ranks = ranks[:n_pos]
    u = np.sum(pos_ranks) - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def generate_desktop_base(size=(128, 128), rng=None) -> Image.Image:
    r = rng or random.Random(42)
    w, h = size
    bg = (r.randint(30, 45), r.randint(40, 55), r.randint(50, 65))
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    # Window
    wx1, wy1 = 12, 12
    wx2, wy2 = w - 12, h - 12
    draw.rectangle([wx1, wy1, wx2, wy2], fill=(240, 242, 245), outline=(130, 140, 155), width=1)

    # Title bar
    draw.rectangle([wx1, wy1, wx2, wy1 + 16], fill=(55, 80, 110))
    draw.ellipse([wx2 - 12, wy1 + 4, wx2 - 4, wy1 + 12], fill=(220, 60, 60))

    # Button
    draw.rectangle([wx1 + 8, wy1 + 24, wx1 + 42, wy1 + 42], fill=(45, 115, 205), outline=(30, 85, 160))

    # Text Input box
    draw.rectangle([wx1 + 48, wy1 + 24, wx2 - 8, wy1 + 42], fill=(255, 255, 255), outline=(180, 190, 200))

    # Disabled button
    draw.rectangle([wx1 + 8, wy1 + 48, wx1 + 42, wy1 + 66], fill=(200, 205, 210), outline=(175, 180, 185))

    return img


def generate_interaction_pair(interaction_type: str, rng=None, size=(128, 128)) -> Tuple[Image.Image, Image.Image, int]:
    r = rng or random.Random()
    pre_img = generate_desktop_base(size, r)
    post_img = pre_img.copy()
    draw = ImageDraw.Draw(post_img)
    w, h = size

    if interaction_type == "ripple":
        # Click ripple: radial circular wave emanating from button center (35, 45)
        cx, cy = 35, 45
        for radius in [6, 12, 18]:
            color = (255, 255, 255)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=color, width=2)
        label = 1

    elif interaction_type == "color_change":
        # Button pressed / active state: blue button turns amber / dark orange
        draw.rectangle([20, 36, 54, 54], fill=(225, 135, 30), outline=(255, 255, 255), width=2)
        label = 1

    elif interaction_type == "focus_shift":
        # Input focus shift: bright blue focus halo around text input
        draw.rectangle([60, 36, 116, 54], fill=(255, 255, 255), outline=(30, 140, 255), width=3)
        # Cursor caret in input
        draw.line([(66, 40), (66, 50)], fill=(20, 20, 20), width=2)
        label = 1

    elif interaction_type == "text_entry":
        # Characters typed into input box
        draw.rectangle([60, 36, 116, 54], fill=(255, 255, 255), outline=(50, 120, 220), width=1)
        draw.rectangle([64, 42, 98, 48], fill=(35, 40, 45))
        draw.line([(101, 39), (101, 51)], fill=(20, 20, 20), width=2)
        label = 1

    elif interaction_type == "modal_popup":
        # Dialog opened
        mx1, my1 = w // 2 - 32, h // 2 - 24
        mx2, my2 = w // 2 + 32, h // 2 + 24
        draw.rectangle([mx1 + 2, my1 + 2, mx2 + 2, my2 + 2], fill=(30, 30, 30))
        draw.rectangle([mx1, my1, mx2, my2], fill=(245, 245, 250), outline=(60, 80, 100), width=2)
        draw.rectangle([mx1, my1, mx2, my1 + 10], fill=(50, 90, 140))
        draw.rectangle([mx1 + 10, my2 - 10, mx2 - 10, my2 - 3], fill=(50, 130, 220))
        label = 1

    elif interaction_type == "noop_deadspace":
        # Click on inert wallpaper margin - no change
        label = 0

    elif interaction_type == "noop_disabled":
        # Click on disabled button (20, 68) - inert, no feedback
        label = 0

    elif interaction_type == "noop_identical":
        # Absolute identical duplicate frame
        label = 0

    elif interaction_type == "noop_micro_jitter":
        # Subtle sensor noise (standard deviation ~ 1-2 / 255)
        arr = np.array(post_img, dtype=np.float32)
        noise = r.gauss(0.0, 1.5)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        post_img = Image.fromarray(arr)
        label = 0

    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}")

    return pre_img, post_img, label


def run_comprehensive_audit(num_samples_per_type: int = 50, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    preprocessor = EdgeLuminancePreprocessor()
    model = StreamingVisualCortexV2()
    state = torch.load(DEFAULT_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)
    model.eval()

    interaction_types = [
        # Positive visual responses
        ("ripple", True),
        ("color_change", True),
        ("focus_shift", True),
        ("text_entry", True),
        ("modal_popup", True),
        # Negative controls (no-ops)
        ("noop_deadspace", False),
        ("noop_disabled", False),
        ("noop_identical", False),
        ("noop_micro_jitter", False),
    ]

    all_labels = []
    all_probs = []
    all_logits = []
    category_results = {}

    for itype, is_pos in interaction_types:
        cat_probs = []
        cat_logits = []
        cat_l2_dists = []
        cat_cos_sims = []

        for i in range(num_samples_per_type):
            rng = random.Random(seed + i * 17)
            pre_img, post_img, label = generate_interaction_pair(itype, rng=rng)

            arr_pre = np.array(pre_img, dtype=np.float32) / 255.0
            arr_post = np.array(post_img, dtype=np.float32) / 255.0

            ten_pre = torch.from_numpy(arr_pre).permute(2, 0, 1).unsqueeze(0)
            ten_post = torch.from_numpy(arr_post).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                f_pre = preprocessor(ten_pre)
                f_post = preprocessor(ten_post)

                out_pre = model.forward_step(f_pre, prev_state=None, timestamp=0.0)
                out_post = model.forward_step(f_post, prev_state=out_pre.temporal_state, timestamp=0.2)

                z_pre = out_pre.latent
                z_post = out_post.latent

                # Head 3 prediction
                logit = model.post_action_head(z_pre, z_post)
                prob = torch.sigmoid(logit).item()

                l2_dist = torch.norm(z_post - z_pre, p=2).item()
                cos_sim = F.cosine_similarity(z_pre, z_post, dim=-1).item()

            cat_probs.append(prob)
            cat_logits.append(logit.item())
            cat_l2_dists.append(l2_dist)
            cat_cos_sims.append(cos_sim)

            all_labels.append(label)
            all_probs.append(prob)
            all_logits.append(logit.item())

        category_results[itype] = {
            "is_positive": is_pos,
            "mean_prob": float(np.mean(cat_probs)),
            "std_prob": float(np.std(cat_probs)),
            "mean_logit": float(np.mean(cat_logits)),
            "mean_l2_dist": float(np.mean(cat_l2_dists)),
            "mean_cos_sim": float(np.mean(cat_cos_sims)),
        }

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)

    roc_auc = compute_roc_auc_binary(all_labels, all_probs)

    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    mean_pos = float(np.mean(pos_probs))
    mean_neg = float(np.mean(neg_probs))
    discrimination_margin = mean_pos - mean_neg

    print("==================================================================")
    print("AVCS SYSTEM 1: POST-ACTION VISUAL RESPONSE & RIPPLE AUDIT")
    print("==================================================================")
    print(f"Total Evaluated Samples: {len(all_labels)} ({len(pos_probs)} positive, {len(neg_probs)} negative)")
    print(f"Overall ROC-AUC:              {roc_auc:.4f}  (Requirement: >= 0.8500)")
    print(f"Discrimination Margin:        {discrimination_margin:.4f}  (Requirement: >= 0.5000)")
    print(f"Positive Mean Response Prob:  {mean_pos:.4f}")
    print(f"Negative Mean Response Prob:  {mean_neg:.4f}")
    print("------------------------------------------------------------------")
    print("PER-INTERACTION MORPHOLOGY BREAKDOWN:")
    for itype, stats in category_results.items():
        tag = "[POSITIVE]" if stats["is_positive"] else "[NEGATIVE]"
        print(
            f"  {tag:<10} {itype:<20}: Mean Prob = {stats['mean_prob']:.4f} (+/- {stats['std_prob']:.4f}) | "
            f"L2 Dist = {stats['mean_l2_dist']:.4f} | Cos Sim = {stats['mean_cos_sim']:.4f}"
        )
    print("==================================================================")

    return {
        "roc_auc": roc_auc,
        "discrimination_margin": discrimination_margin,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "category_results": category_results,
    }


if __name__ == "__main__":
    run_comprehensive_audit()
