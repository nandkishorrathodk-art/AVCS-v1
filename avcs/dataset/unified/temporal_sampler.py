"""Procedural Temporal Sequence and Action Response Pair Sampler for AVCS Streaming V2.

Part of StreamingVisualCortexV2 Milestone M2: Multi-Generator Dataset Pipeline.

Provides high-throughput procedural dataset generation for:
1. Head 1 (Temporal Change & Spinner Activity):
   - Cyclical loading spinner sequences (rotating arc, pulsating dots ring).
   - Moving UI elements (moving cursors, dragging windows, tooltips).
   - Static sequences (zero visual delta).
   - Ground truth spatial change heatmaps (128x128) and binary spinner activity labels.
2. Head 3 (Post-Action Visual Response):
   - Pre-action observation frame.
   - Action location (cx, cy) in normalized coordinates.
   - Post-action observation frame with visible visual response (1.0) or no response (0.0).
   - Scenarios: button state transition, dialog appearance, text entry, static click miss.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import torch


# =========================================================================
# Head 1: Temporal Sequence Generation
# =========================================================================
@dataclass
class TemporalSequence:
    """A multi-frame temporal sequence with ground truth change heatmap and spinner label."""

    frames: torch.Tensor  # [T, 3, H, W] float32 in [0.0, 1.0]
    change_heatmap: torch.Tensor  # [1, H, W] float32 in [0.0, 1.0]
    spinner_label: float  # 1.0 if cyclical spinner active, 0.0 otherwise
    mode: str  # "spinner" | "moving" | "static"
    timestamps: torch.Tensor  # [T] float32


def _draw_rotating_arc_spinner(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    radius: int,
    step_idx: int,
    num_steps: int,
    color: Tuple[int, int, int] = (60, 140, 240),
    width: int = 3,
) -> None:
    """Draw a rotating arc loading spinner at center."""
    cx, cy = center
    start_angle = (step_idx * (360.0 / num_steps)) % 360.0
    end_angle = start_angle + 270.0
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=width)


def _draw_pulsating_dots_spinner(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    radius: int,
    step_idx: int,
    num_steps: int,
    num_dots: int = 8,
    base_color: Tuple[int, int, int] = (80, 160, 250),
) -> None:
    """Draw a ring of pulsating dots simulating a classic loading spinner."""
    cx, cy = center
    active_idx = step_idx % num_dots
    for i in range(num_dots):
        angle = i * (2.0 * math.pi / num_dots)
        dx = int(radius * math.cos(angle))
        dy = int(radius * math.sin(angle))
        dot_center = (cx + dx, cy + dy)
        dot_r = 3 if i == active_idx else 2
        # Alpha/color fade based on distance to active
        fade = 1.0 - (abs(i - active_idx) % num_dots) / float(num_dots)
        color = tuple(int(c * (0.3 + 0.7 * fade)) for c in base_color)
        draw.ellipse(
            [dot_center[0] - dot_r, dot_center[1] - dot_r, dot_center[0] + dot_r, dot_center[1] + dot_r],
            fill=color,
        )


def _render_base_desktop_canvas(
    size: Tuple[int, int] = (128, 128),
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Render a realistic synthetic desktop canvas (window, toolbar, buttons)."""
    r = rng or random.Random()
    w, h = size
    bg_color = (r.randint(25, 45), r.randint(35, 55), r.randint(45, 65))
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    # Window frame
    wx1, wy1 = r.randint(8, 16), r.randint(8, 16)
    wx2, wy2 = w - r.randint(8, 16), h - r.randint(8, 16)
    win_color = (r.randint(230, 245), r.randint(230, 245), r.randint(235, 250))
    draw.rectangle([wx1, wy1, wx2, wy2], fill=win_color, outline=(140, 150, 165), width=1)

    # Titlebar
    tb_h = 16
    draw.rectangle([wx1, wy1, wx2, wy1 + tb_h], fill=(70, 95, 125))
    # Close button dot
    draw.ellipse([wx2 - 12, wy1 + 4, wx2 - 4, wy1 + 12], fill=(220, 70, 70))

    # Internal controls
    draw.rectangle([wx1 + 6, wy1 + tb_h + 8, wx1 + 40, wy1 + tb_h + 24], fill=(50, 120, 210))
    draw.rectangle([wx1 + 46, wy1 + tb_h + 8, wx2 - 8, wy1 + tb_h + 24], fill=(255, 255, 255), outline=(180, 190, 200))

    return img


def generate_temporal_sequence(
    mode: str = "spinner",
    sequence_length: int = 4,
    image_size: Tuple[int, int] = (128, 128),
    rng: Optional[random.Random] = None,
) -> TemporalSequence:
    """Generate a single procedural temporal sequence with ground truth change heatmap.

    Args:
        mode: "spinner" | "moving" | "static"
        sequence_length: number of frames (T >= 2)
        image_size: (H, W) frame resolution
        rng: optional random.Random instance for determinism
    """
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")

    r = rng or random.Random()
    w, h = image_size
    base_img = _render_base_desktop_canvas(image_size, r)

    pil_frames: List[Image.Image] = []
    spinner_active: bool = False

    if mode == "spinner":
        spinner_active = True
        spinner_type = r.choice(["arc", "dots"])
        scx = w // 2 + r.randint(-15, 15)
        scy = h // 2 + r.randint(-10, 10)
        radius = r.randint(12, 18)
        color = (r.randint(40, 90), r.randint(110, 180), r.randint(210, 255))

        for t in range(sequence_length):
            frame = base_img.copy()
            draw = ImageDraw.Draw(frame)
            if spinner_type == "arc":
                _draw_rotating_arc_spinner(draw, (scx, scy), radius, t, sequence_length, color)
            else:
                _draw_pulsating_dots_spinner(draw, (scx, scy), radius, t, sequence_length, base_color=color)
            pil_frames.append(frame)

    elif mode == "moving":
        spinner_active = False
        # A moving element (e.g. cursor or floating badge) traversing the screen
        start_x = r.randint(20, 40)
        start_y = r.randint(30, 80)
        vx = r.randint(10, 20)
        vy = r.randint(-5, 10)

        for t in range(sequence_length):
            frame = base_img.copy()
            draw = ImageDraw.Draw(frame)
            cx = start_x + t * vx
            cy = start_y + t * vy
            # Draw cursor arrow / small rectangle
            draw.polygon([(cx, cy), (cx + 8, cy + 12), (cx + 3, cy + 11), (cx, cy + 16)], fill=(20, 20, 20))
            draw.polygon([(cx, cy), (cx + 6, cy + 10), (cx + 2, cy + 9), (cx, cy + 13)], fill=(255, 255, 255))
            pil_frames.append(frame)

    elif mode == "static":
        spinner_active = False
        # Exactly identical frames
        for _ in range(sequence_length):
            pil_frames.append(base_img.copy())
    else:
        raise ValueError(f"Unknown mode {mode!r}; must be 'spinner', 'moving', or 'static'")

    # Convert frames to torch tensors [T, 3, H, W] in [0.0, 1.0]
    frame_tensors = []
    for f in pil_frames:
        arr = np.array(f, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        frame_tensors.append(ten)

    seq_tensor = torch.stack(frame_tensors, dim=0)  # [T, 3, H, W]

    # Compute ground truth spatial change heatmap across the sequence
    # Difference between consecutive frames: max over time of inter-frame delta
    inter_frame_diffs = []
    for t in range(1, sequence_length):
        diff = (seq_tensor[t] - seq_tensor[t - 1]).abs().max(dim=0)[0]  # [H, W]
        inter_frame_diffs.append(diff)

    max_diff = torch.stack(inter_frame_diffs, dim=0).max(dim=0)[0]  # [H, W]
    # Binary / soft threshold: pixels with perceptible optical change (> 0.04)
    change_heatmap = (max_diff > 0.04).float().unsqueeze(0)  # [1, H, W]

    timestamps = torch.tensor([float(i * 0.1) for i in range(sequence_length)], dtype=torch.float32)

    return TemporalSequence(
        frames=seq_tensor,
        change_heatmap=change_heatmap,
        spinner_label=1.0 if spinner_active else 0.0,
        mode=mode,
        timestamps=timestamps,
    )


def sample_temporal_batch(
    batch_size: int = 8,
    sequence_length: int = 4,
    image_size: Tuple[int, int] = (128, 128),
    spinner_ratio: float = 0.35,
    moving_ratio: float = 0.35,
    rng: Optional[random.Random] = None,
) -> Dict[str, torch.Tensor]:
    """Sample a batch of temporal sequences balanced across spinner, moving, and static modes.

    Returns dict with:
        sequences: [B, T, 3, H, W] float32
        change_heatmaps: [B, 1, H, W] float32
        spinner_labels: [B, 1] float32
    """
    r = rng or random.Random()
    seq_list = []
    heat_list = []
    spin_list = []

    for _ in range(batch_size):
        roll = r.random()
        if roll < spinner_ratio:
            mode = "spinner"
        elif roll < spinner_ratio + moving_ratio:
            mode = "moving"
        else:
            mode = "static"

        t_seq = generate_temporal_sequence(
            mode=mode,
            sequence_length=sequence_length,
            image_size=image_size,
            rng=r,
        )
        seq_list.append(t_seq.frames)
        heat_list.append(t_seq.change_heatmap)
        spin_list.append(t_seq.spinner_label)

    return {
        "sequences": torch.stack(seq_list, dim=0),
        "change_heatmaps": torch.stack(heat_list, dim=0),
        "spinner_labels": torch.tensor(spin_list, dtype=torch.float32).unsqueeze(1),
    }


# =========================================================================
# Head 3: Post-Action Observation Pair Generation
# =========================================================================
@dataclass
class ActionObservationPair:
    """Pre/post action pair with action location and visual response outcome."""

    pre_frame: torch.Tensor  # [3, H, W] float32 in [0.0, 1.0]
    action_location_norm: Tuple[float, float]  # (cx, cy) in [0.0, 1.0]
    action_type: str  # "click" | "type"
    post_frame: torch.Tensor  # [3, H, W] float32 in [0.0, 1.0]
    response_label: float  # 1.0 if visible response occurred, 0.0 if no response
    scenario: str  # e.g. "button_pressed", "dialog_opened", "text_typed", "no_response"


def generate_action_pair(
    has_response: Optional[bool] = None,
    image_size: Tuple[int, int] = (128, 128),
    rng: Optional[random.Random] = None,
) -> ActionObservationPair:
    """Generate a pre/post action observation pair.

    Args:
        has_response: True for visible response (1.0), False for no response (0.0), None for 50/50.
        image_size: (H, W) frame resolution.
        rng: optional random.Random instance.
    """
    r = rng or random.Random()
    w, h = image_size

    if has_response is None:
        has_response = r.random() < 0.5

    pre_img = _render_base_desktop_canvas(image_size, r)
    post_img = pre_img.copy()

    if has_response:
        scenario = r.choice(["button_pressed", "dialog_opened", "text_typed"])
        if scenario == "button_pressed":
            # Action on button: location inside button at (25, 45)
            action_x, action_y = r.randint(20, 35), r.randint(35, 45)
            # Post-action: button color turns dark/active (blue -> dark navy/amber highlight)
            draw_post = ImageDraw.Draw(post_img)
            draw_post.rectangle([14, 32, 48, 48], fill=(220, 130, 30), outline=(255, 255, 255), width=2)
            act_type = "click"

        elif scenario == "dialog_opened":
            # Click action in toolbar triggers a modal popup dialog
            action_x, action_y = r.randint(15, 40), r.randint(15, 25)
            draw_post = ImageDraw.Draw(post_img)
            # Modal dialog pops up in center
            mx1, my1 = w // 2 - 35, h // 2 - 25
            mx2, my2 = w // 2 + 35, h // 2 + 25
            draw_post.rectangle([mx1 + 2, my1 + 2, mx2 + 2, my2 + 2], fill=(20, 20, 20))  # shadow
            draw_post.rectangle([mx1, my1, mx2, my2], fill=(245, 245, 250), outline=(50, 70, 90), width=2)
            draw_post.rectangle([mx1, my1, mx2, my1 + 12], fill=(45, 85, 140))
            # Modal ok button
            draw_post.rectangle([mx1 + 10, my2 - 12, mx2 - 10, my2 - 4], fill=(50, 130, 220))
            act_type = "click"

        else:  # "text_typed"
            # Action inside input box: typed characters appear
            action_x, action_y = r.randint(60, 95), r.randint(35, 45)
            draw_post = ImageDraw.Draw(post_img)
            # Text entered inside input box
            draw_post.rectangle([54, 32, 112, 48], fill=(255, 255, 255), outline=(50, 120, 220), width=1)
            # Text block inside input
            draw_post.rectangle([58, 38, 92, 42], fill=(30, 30, 35))
            draw_post.line([(94, 35), (94, 45)], fill=(20, 20, 20), width=1)  # Caret
            act_type = "type"

        resp_label = 1.0

    else:
        # No response: action clicked static background or missed target
        scenario = "no_response"
        action_x = r.randint(2, 10)  # On background margin
        action_y = r.randint(2, 10)
        # Post frame is exactly identical to pre frame
        post_img = pre_img.copy()
        resp_label = 0.0
        act_type = r.choice(["click", "type"])

    # Convert to tensors
    arr_pre = np.array(pre_img, dtype=np.float32) / 255.0
    arr_post = np.array(post_img, dtype=np.float32) / 255.0

    ten_pre = torch.from_numpy(arr_pre).permute(2, 0, 1)  # [3, H, W]
    ten_post = torch.from_numpy(arr_post).permute(2, 0, 1)  # [3, H, W]

    cx_norm = round(action_x / float(w), 6)
    cy_norm = round(action_y / float(h), 6)

    return ActionObservationPair(
        pre_frame=ten_pre,
        action_location_norm=(cx_norm, cy_norm),
        action_type=act_type,
        post_frame=ten_post,
        response_label=resp_label,
        scenario=scenario,
    )


def sample_action_pair_batch(
    batch_size: int = 8,
    image_size: Tuple[int, int] = (128, 128),
    positive_ratio: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Dict[str, torch.Tensor]:
    """Sample a batch of pre/post action observation pairs balanced between response (1.0) and no response (0.0).

    Returns dict with:
        pre_frames: [B, 3, H, W] float32
        post_frames: [B, 3, H, W] float32
        action_locations: [B, 2] float32 (cx, cy)
        response_labels: [B, 1] float32
    """
    r = rng or random.Random()
    pre_list = []
    post_list = []
    loc_list = []
    resp_list = []

    for _ in range(batch_size):
        has_resp = r.random() < positive_ratio
        pair = generate_action_pair(
            has_response=has_resp,
            image_size=image_size,
            rng=r,
        )
        pre_list.append(pair.pre_frame)
        post_list.append(pair.post_frame)
        loc_list.append(list(pair.action_location_norm))
        resp_list.append(pair.response_label)

    return {
        "pre_frames": torch.stack(pre_list, dim=0),
        "post_frames": torch.stack(post_list, dim=0),
        "action_locations": torch.tensor(loc_list, dtype=torch.float32),
        "response_labels": torch.tensor(resp_list, dtype=torch.float32).unsqueeze(1),
    }
