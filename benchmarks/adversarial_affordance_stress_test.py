"""Adversarial Stress Test and Hard Reality Check for FastAffordanceHead.

Evaluates:
1. Real, complex, uncurated desktop screens:
   - Messy multi-window desktops (overlapping cascaded/tiled windows, desktop icons, taskbar)
   - Terminal sessions (dense code text, compiler logs, git status, htop meters, shell prompts)
   - Browser tabs (dense multi-tab headers, omnibox, toolbar buttons, web cards, hyperlinks, inputs)
   - Media playback (video player with photographic scenery in viewport, scrubber, controls, volume)
   - Live Windows desktop captures (real desktop screen grabs)
2. Hostile negative images:
   - Pure blank screens (solid black, solid white, solid greys)
   - Smooth gradients (horizontal, vertical, diagonal, radial)
   - Photographic wallpapers & natural textures (landscapes, Perlin noise, fractals)
   - Checkerboards (1px, 2px, 4px, 8px, 16px, 32px, 64px)
   - Gaussian noise fields (sigma from 5 to 100)
   - Salt-and-pepper textures (density from 0.005 to 0.50)
3. Precision, recall, false positive rates, F1 scores, failure modes, and foveal survival rates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.human_eye import FastAffordanceHead, HumanEye, RealTimeCorticalBackbone
from forge.vision.capture import open_capture


# ==============================================================================
# 1. Hostile Negative Generator
# ==============================================================================

class HostileNegativeGenerator:
    """Generates guaranteed non-interactive hostile negative test images."""

    def __init__(self, width: int = 1280, height: int = 800) -> None:
        self.W = width
        self.H = height

    def generate_suite(self, seed: int = 42) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        py_rng = random.Random(seed)
        samples: List[Dict[str, Any]] = []

        # --- A. Pure Blank Screens (15 samples) ---
        blank_specs = [
            ("blank_solid_black", (0, 0, 0)),
            ("blank_solid_white", (255, 255, 255)),
            ("blank_solid_gray_16", (16, 16, 16)),
            ("blank_solid_gray_32", (32, 32, 32)),
            ("blank_solid_gray_64", (64, 64, 64)),
            ("blank_solid_gray_96", (96, 96, 96)),
            ("blank_solid_gray_128", (128, 128, 128)),
            ("blank_solid_gray_160", (160, 160, 160)),
            ("blank_solid_gray_192", (192, 192, 192)),
            ("blank_solid_gray_224", (224, 224, 224)),
            ("blank_solid_gray_240", (240, 240, 240)),
            ("blank_solid_dark_blue", (15, 23, 42)),
            ("blank_solid_slate", (30, 41, 59)),
            ("blank_solid_wine", (40, 10, 15)),
            ("blank_solid_forest", (10, 35, 20)),
        ]
        for name, color in blank_specs:
            img = np.full((self.H, self.W, 3), color, dtype=np.uint8)
            samples.append({
                "category": "pure_blank",
                "subtype": name,
                "image": img,
                "gt_points": [],
                "description": f"Solid uniform screen RGB={color}",
            })

        # --- B. Smooth Gradients (25 samples) ---
        yy, xx = np.mgrid[:self.H, :self.W]
        norm_x = xx.astype(np.float32) / (self.W - 1)
        norm_y = yy.astype(np.float32) / (self.H - 1)

        # 1. Linear Horizontal
        for ramp_name, c1, c2 in [
            ("grad_h_bw", (0, 0, 0), (255, 255, 255)),
            ("grad_h_wb", (255, 255, 255), (0, 0, 0)),
            ("grad_h_blue_cyan", (10, 20, 60), (30, 180, 220)),
            ("grad_h_purple_pink", (40, 10, 60), (220, 80, 140)),
            ("grad_h_dark_subtle", (20, 22, 28), (45, 50, 60)),
        ]:
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            for ch in range(3):
                arr[:, :, ch] = np.clip(c1[ch] * (1.0 - norm_x) + c2[ch] * norm_x, 0, 255).astype(np.uint8)
            samples.append({"category": "smooth_gradient", "subtype": ramp_name, "image": arr, "gt_points": [], "description": "Horizontal linear gradient"})

        # 2. Linear Vertical
        for ramp_name, c1, c2 in [
            ("grad_v_bw", (0, 0, 0), (255, 255, 255)),
            ("grad_v_wb", (255, 255, 255), (0, 0, 0)),
            ("grad_v_sunset", (240, 100, 40), (20, 10, 40)),
            ("grad_v_sky", (100, 160, 240), (240, 245, 255)),
            ("grad_v_dark_subtle", (15, 17, 22), (50, 55, 68)),
        ]:
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            for ch in range(3):
                arr[:, :, ch] = np.clip(c1[ch] * (1.0 - norm_y) + c2[ch] * norm_y, 0, 255).astype(np.uint8)
            samples.append({"category": "smooth_gradient", "subtype": ramp_name, "image": arr, "gt_points": [], "description": "Vertical linear gradient"})

        # 3. Diagonal & Bilinear
        norm_diag1 = (norm_x + norm_y) / 2.0
        norm_diag2 = (norm_x + (1.0 - norm_y)) / 2.0
        for ramp_name, metric, c1, c2 in [
            ("grad_diag_tl_br", norm_diag1, (0, 20, 50), (200, 220, 255)),
            ("grad_diag_bl_tr", norm_diag2, (10, 50, 30), (220, 240, 180)),
            ("grad_diag_bw", norm_diag1, (0, 0, 0), (255, 255, 255)),
            ("grad_diag_wb", norm_diag1, (255, 255, 255), (0, 0, 0)),
            ("grad_bilinear_subtle", (norm_x * norm_y), (30, 35, 45), (70, 85, 110)),
        ]:
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            for ch in range(3):
                arr[:, :, ch] = np.clip(c1[ch] * (1.0 - metric) + c2[ch] * metric, 0, 255).astype(np.uint8)
            samples.append({"category": "smooth_gradient", "subtype": ramp_name, "image": arr, "gt_points": [], "description": "Diagonal/Bilinear gradient"})

        # 4. Radial Gradients
        cx, cy = self.W / 2.0, self.H / 2.0
        max_r = math.hypot(cx, cy)
        dist_r = np.clip(np.hypot(xx - cx, yy - cy) / max_r, 0.0, 1.0)
        for r_name, d_metric, c1, c2 in [
            ("grad_radial_center_bright", 1.0 - dist_r, (20, 25, 35), (220, 230, 255)),
            ("grad_radial_center_dark", dist_r, (240, 240, 255), (20, 25, 35)),
            ("grad_radial_vignette", dist_r ** 1.8, (60, 70, 90), (10, 12, 15)),
            ("grad_radial_glow_orange", (1.0 - dist_r) ** 2, (15, 15, 20), (255, 140, 40)),
            ("grad_radial_bw", dist_r, (255, 255, 255), (0, 0, 0)),
            ("grad_radial_bw_inv", 1.0 - dist_r, (0, 0, 0), (255, 255, 255)),
            ("grad_radial_soft_blue", 1.0 - dist_r, (15, 20, 30), (70, 130, 200)),
            ("grad_radial_deep_purple", 1.0 - dist_r, (10, 8, 15), (110, 50, 140)),
            ("grad_radial_corner_glow", np.clip(np.hypot(xx, yy) / max_r, 0, 1), (255, 255, 255), (20, 20, 30)),
            ("grad_radial_corner_glow_br", np.clip(np.hypot(xx - self.W, yy - self.H) / max_r, 0, 1), (255, 240, 220), (25, 25, 35)),
        ]:
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            for ch in range(3):
                arr[:, :, ch] = np.clip(c1[ch] * (1.0 - d_metric) + c2[ch] * d_metric, 0, 255).astype(np.uint8)
            samples.append({"category": "smooth_gradient", "subtype": r_name, "image": arr, "gt_points": [], "description": "Radial gradient field"})

        # --- C. Photographic Wallpapers & Natural Textures (25 samples) ---
        for idx in range(25):
            sub_seed = seed + 1000 + idx
            arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            wall_type = idx % 5

            if wall_type == 0:
                # Mountain Dusk / Horizon
                horizon = int(self.H * (0.50 + 0.15 * math.sin(idx)))
                sky_grad = np.linspace(20, 180, horizon)[:, None]
                sky = np.zeros((horizon, self.W, 3), dtype=np.float32)
                sky[:, :, 0] = sky_grad * 1.2
                sky[:, :, 1] = sky_grad * 0.7
                sky[:, :, 2] = sky_grad * 0.9 + 40
                arr[:horizon] = np.clip(sky, 0, 255).astype(np.uint8)
                # Mountain ridge silhouette
                ridge_x = np.linspace(0, 4 * math.pi, self.W)
                ridge_h = (horizon - 60) + (np.sin(ridge_x * 2.3 + idx) * 35 + np.sin(ridge_x * 5.1) * 15).astype(int)
                for c in range(self.W):
                    rh = max(0, min(self.H - 1, ridge_h[c]))
                    arr[rh:, c] = (25, 30, 45)
                desc = "Dusk mountain landscape wallpaper"
                sub_name = f"wallpaper_mountain_dusk_{idx}"
            elif wall_type == 1:
                # Multi-harmonic smooth waves (abstract wallpaper)
                w1 = np.sin(xx / 60.0 + idx) * np.cos(yy / 80.0)
                w2 = np.sin((xx + yy) / 100.0) * 0.5
                wave = ((w1 + w2 + 1.5) / 3.0)
                arr[:, :, 0] = np.clip(30 + wave * 180, 0, 255).astype(np.uint8)
                arr[:, :, 1] = np.clip(20 + wave * 100, 0, 255).astype(np.uint8)
                arr[:, :, 2] = np.clip(60 + wave * 190, 0, 255).astype(np.uint8)
                desc = "Multi-harmonic smooth wave abstract wallpaper"
                sub_name = f"wallpaper_harmonic_waves_{idx}"
            elif wall_type == 2:
                # Abstract low-poly faceted triangles
                poly_canvas = Image.new("RGB", (self.W, self.H), (30, 32, 40))
                draw = ImageDraw.Draw(poly_canvas)
                grid_step = 80
                pts: Dict[Tuple[int, int], Tuple[int, int]] = {}
                local_rng = np.random.RandomState(sub_seed)
                for gy in range(-grid_step, self.H + grid_step * 2, grid_step):
                    for gx in range(-grid_step, self.W + grid_step * 2, grid_step):
                        jx = gx + local_rng.randint(-grid_step // 3, grid_step // 3)
                        jy = gy + local_rng.randint(-grid_step // 3, grid_step // 3)
                        pts[(gx, gy)] = (jx, jy)
                for gy in range(0, self.H, grid_step):
                    for gx in range(0, self.W, grid_step):
                        p1 = pts[(gx, gy)]
                        p2 = pts[(gx + grid_step, gy)]
                        p3 = pts[(gx, gy + grid_step)]
                        p4 = pts[(gx + grid_step, gy + grid_step)]
                        c_val1 = local_rng.randint(30, 110)
                        draw.polygon([p1, p2, p3], fill=(c_val1, c_val1 + 10, c_val1 + 25))
                        c_val2 = local_rng.randint(30, 110)
                        draw.polygon([p2, p4, p3], fill=(c_val2 + 5, c_val2 + 15, c_val2 + 30))
                arr = np.array(poly_canvas)
                desc = "Low-poly faceted geometric wallpaper"
                sub_name = f"wallpaper_lowpoly_facets_{idx}"
            elif wall_type == 3:
                # Fractal/Perlin noise-like cloud cover
                cloud = np.zeros((self.H, self.W), dtype=np.float32)
                for oct_idx, freq in enumerate([0.005, 0.012, 0.025, 0.05]):
                    phase = sub_seed * 0.1 + oct_idx
                    cloud += (np.sin(xx * freq + phase) * np.cos(yy * freq - phase)) * (0.5 ** oct_idx)
                cloud_norm = (cloud - cloud.min()) / (cloud.max() - cloud.min() + 1e-6)
                arr[:, :, 0] = (40 + cloud_norm * 180).astype(np.uint8)
                arr[:, :, 1] = (50 + cloud_norm * 190).astype(np.uint8)
                arr[:, :, 2] = (70 + cloud_norm * 185).astype(np.uint8)
                desc = "Multi-octave cloud / nebula texture wallpaper"
                sub_name = f"wallpaper_cloud_nebula_{idx}"
            else:
                # Organic cellular / Voronoi texture
                local_rng = np.random.RandomState(sub_seed)
                n_centers = 18
                vcx = local_rng.randint(0, self.W, size=n_centers)
                vcy = local_rng.randint(0, self.H, size=n_centers)
                # Compute distance map downsampled then resized
                sub_h, sub_w = self.H // 4, self.W // 4
                syy, sxx = np.mgrid[:sub_h, :sub_w]
                sxx = sxx * 4
                syy = syy * 4
                d_min = np.full((sub_h, sub_w), 1e6, dtype=np.float32)
                for ci in range(n_centers):
                    d = np.hypot(sxx - vcx[ci], syy - vcy[ci])
                    d_min = np.minimum(d_min, d)
                d_norm = (d_min / np.max(d_min)) * 255.0
                d_full = np.array(Image.fromarray(d_norm.astype(np.uint8)).resize((self.W, self.H), Image.BILINEAR))
                arr[:, :, 0] = np.clip(25 + d_full * 0.8, 0, 255).astype(np.uint8)
                arr[:, :, 1] = np.clip(35 + d_full * 0.6, 0, 255).astype(np.uint8)
                arr[:, :, 2] = np.clip(60 + d_full * 0.7, 0, 255).astype(np.uint8)
                desc = "Voronoi cellular texture wallpaper"
                sub_name = f"wallpaper_voronoi_texture_{idx}"

            samples.append({
                "category": "photographic_wallpaper",
                "subtype": sub_name,
                "image": arr,
                "gt_points": [],
                "description": desc,
            })

        # --- D. Checkerboards (25 samples) ---
        checker_sizes = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200, 256]
        for sz in checker_sizes:
            grid_c = ((xx // sz + yy // sz) % 2).astype(np.uint8)
            if sz % 3 == 0:
                # Color checkerboard (accent blue & dark gray)
                arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
                arr[grid_c == 1] = (0, 120, 215)
                arr[grid_c == 0] = (40, 44, 52)
                desc = f"Color checkerboard tile={sz}px (Blue/Gray)"
            elif sz % 3 == 1:
                # High-contrast black & white
                arr = np.stack([grid_c * 255] * 3, axis=-1)
                desc = f"Binary B/W checkerboard tile={sz}px"
            else:
                # Subtle contrast gray
                arr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
                arr[grid_c == 1] = (140, 140, 140)
                arr[grid_c == 0] = (100, 100, 100)
                desc = f"Low-contrast gray checkerboard tile={sz}px"

            samples.append({
                "category": "checkerboard",
                "subtype": f"checker_{sz}px",
                "image": arr,
                "gt_points": [],
                "description": desc,
            })

        # --- E. Gaussian Noise Fields (30 samples) ---
        noise_configs = [
            # (name, mean, sigma)
            ("gauss_dark_s5", 20, 5),
            ("gauss_dark_s10", 20, 10),
            ("gauss_dark_s25", 25, 25),
            ("gauss_dark_s50", 40, 50),
            ("gauss_mid_s5", 128, 5),
            ("gauss_mid_s10", 128, 10),
            ("gauss_mid_s15", 128, 15),
            ("gauss_mid_s20", 128, 20),
            ("gauss_mid_s25", 128, 25),
            ("gauss_mid_s30", 128, 30),
            ("gauss_mid_s40", 128, 40),
            ("gauss_mid_s50", 128, 50),
            ("gauss_mid_s60", 128, 60),
            ("gauss_mid_s75", 128, 75),
            ("gauss_mid_s90", 128, 90),
            ("gauss_mid_s100", 128, 100),
            ("gauss_bright_s5", 230, 5),
            ("gauss_bright_s10", 230, 10),
            ("gauss_bright_s20", 220, 20),
            ("gauss_bright_s35", 210, 35),
            ("gauss_bright_s50", 200, 50),
            ("gauss_color_uncorr_s20", 128, 20),
            ("gauss_color_uncorr_s40", 128, 40),
            ("gauss_color_uncorr_s60", 128, 60),
            ("gauss_color_uncorr_s80", 128, 80),
            ("gauss_lowfreq_s30", 128, 30),
            ("gauss_lowfreq_s50", 128, 50),
            ("gauss_highpass_s40", 128, 40),
            ("gauss_structured_bands_s30", 128, 30),
            ("gauss_extreme_s120", 128, 120),
        ]
        for name, mean_val, sigma_val in noise_configs:
            local_rng = np.random.RandomState(seed + len(samples))
            if "color_uncorr" in name:
                raw = local_rng.normal(mean_val, sigma_val, (self.H, self.W, 3))
            elif "lowfreq" in name:
                # Downsampled noise interpolated up (correlated spatial blob noise)
                small = local_rng.normal(mean_val, sigma_val, (self.H // 8, self.W // 8, 3))
                small_img = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8))
                raw = np.array(small_img.resize((self.W, self.H), Image.BILINEAR)).astype(np.float32)
            else:
                mono = local_rng.normal(mean_val, sigma_val, (self.H, self.W, 1))
                raw = np.repeat(mono, 3, axis=-1)

            arr = np.clip(raw, 0, 255).astype(np.uint8)
            samples.append({
                "category": "gaussian_noise",
                "subtype": name,
                "image": arr,
                "gt_points": [],
                "description": f"Gaussian noise mean={mean_val}, sigma={sigma_val}",
            })

        # --- F. Salt-and-Pepper Textures (20 samples) ---
        sp_probs = [0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]
        for p in sp_probs:
            local_rng = np.random.RandomState(seed + int(p * 10000))
            base_gray = 128 if p < 0.25 else 40
            arr = np.full((self.H, self.W, 3), base_gray, dtype=np.uint8)
            rnd = local_rng.uniform(0.0, 1.0, (self.H, self.W))
            arr[rnd < (p / 2.0)] = 0
            arr[(rnd >= (p / 2.0)) & (rnd < p)] = 255

            samples.append({
                "category": "salt_and_pepper",
                "subtype": f"salt_pepper_p{int(p * 1000)}permil",
                "image": arr,
                "gt_points": [],
                "description": f"Salt-and-pepper impulsive noise p={p * 100:.1f}%",
            })

        return samples


# ==============================================================================
# 2. Real & Complex Uncurated Screen Generator
# ==============================================================================

class ComplexScreenGenerator:
    """Generates dense, messy, multi-window uncurated desktop screens with strict ground-truth."""

    def __init__(self, width: int = 1280, height: int = 800) -> None:
        self.W = width
        self.H = height

    def _draw_button(
        self,
        draw: ImageDraw.ImageDraw,
        points: List[Tuple[float, float, str, str]],
        x1: int, y1: int, x2: int, y2: int,
        label: str,
        btn_type: str = "primary",
        action: str = "clickable",
        kind: str = "button",
    ) -> None:
        if btn_type == "primary":
            fill = (0, 120, 215)
            outline = (60, 150, 240)
            text_fill = (255, 255, 255)
        elif btn_type == "danger":
            fill = (200, 35, 35)
            outline = (240, 70, 70)
            text_fill = (255, 255, 255)
        elif btn_type == "tab":
            fill = (45, 48, 58)
            outline = (70, 75, 88)
            text_fill = (210, 215, 225)
        else:
            fill = (55, 58, 68)
            outline = (85, 90, 105)
            text_fill = (220, 220, 225)

        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=1)
        # Simplified text representation
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        draw.text((x1 + 6, y1 + max(1, (y2 - y1 - 12) // 2)), label[:max(1, (x2 - x1) // 8)], fill=text_fill)

        # Ground truth centroid
        norm_cx = round(cx / self.W, 4)
        norm_cy = round(cy / self.H, 4)
        points.append((norm_cx, norm_cy, action, kind))

    def _draw_input(
        self,
        draw: ImageDraw.ImageDraw,
        points: List[Tuple[float, float, str, str]],
        x1: int, y1: int, x2: int, y2: int,
        placeholder: str = "",
        dark: bool = True,
    ) -> None:
        bg = (24, 26, 32) if dark else (255, 255, 255)
        border = (0, 120, 215) if dark else (180, 185, 195)
        draw.rectangle([x1, y1, x2, y2], fill=bg, outline=border, width=1)
        if placeholder:
            draw.text((x1 + 6, y1 + max(1, (y2 - y1 - 12) // 2)), placeholder[:max(1, (x2 - x1) // 8)], fill=(120, 125, 135))
        # Text cursor
        draw.line([(x1 + 8, y1 + 3), (x1 + 8, y2 - 3)], fill=(255, 255, 255) if dark else (0, 0, 0), width=1)

        norm_cx = round(((x1 + x2) / 2.0) / self.W, 4)
        norm_cy = round(((y1 + y2) / 2.0) / self.H, 4)
        points.append((norm_cx, norm_cy, "typeable", "input_field"))

    def generate_messy_multi_window(self, seed: int) -> Dict[str, Any]:
        """Messy desktop with overlapping windows, desktop wallpaper, and taskbar."""
        rng = random.Random(seed)
        im = Image.new("RGB", (self.W, self.H), (20, 24, 33))
        draw = ImageDraw.Draw(im)
        points: List[Tuple[float, float, str, str]] = []

        # Desktop wallpaper gradient background
        for y in range(self.H - 40):
            r = int(25 + (y / self.H) * 20)
            g = int(30 + (y / self.H) * 35)
            b = int(45 + (y / self.H) * 60)
            draw.line([(0, y), (self.W, y)], fill=(r, g, b))

        # Desktop Icons on the left
        for row in range(5):
            ix, iy = 24, 30 + row * 80
            draw.rectangle([ix, iy, ix + 44, iy + 44], fill=(40, 80, 150), outline=(100, 150, 220))
            draw.text((ix - 4, iy + 48), f"Icon_{row+1}", fill=(240, 240, 245))
            points.append((round((ix + 22) / self.W, 4), round((iy + 22) / self.H, 4), "clickable", "icon"))

        # Generate 3 to 5 overlapping windows
        num_windows = rng.randint(3, 5)
        for w_idx in range(num_windows):
            ww = rng.randint(450, 700)
            wh = rng.randint(320, 500)
            wx = rng.randint(60, max(70, self.W - ww - 40))
            wy = rng.randint(30, max(40, self.H - wh - 60))

            # Window drop shadow
            draw.rectangle([wx + 4, wy + 4, wx + ww + 4, wy + wh + 4], fill=(10, 12, 16))
            # Window body
            is_dark = (w_idx % 2 == 0)
            body_c = (32, 34, 42) if is_dark else (245, 246, 250)
            title_c = (42, 45, 55) if is_dark else (225, 228, 235)
            border_c = (70, 75, 90) if is_dark else (190, 195, 205)

            draw.rectangle([wx, wy, wx + ww, wy + wh], fill=body_c, outline=border_c, width=1)
            # Titlebar
            draw.rectangle([wx, wy, wx + ww, wy + 28], fill=title_c)
            draw.text((wx + 10, wy + 7), f"Application Window #{w_idx+1}", fill=(220, 225, 235) if is_dark else (30, 35, 45))

            # Window Control Buttons (Min, Max, Close)
            close_x = wx + ww - 24
            max_x = close_x - 26
            min_x = max_x - 26
            self._draw_button(draw, points, min_x, wy + 4, min_x + 20, wy + 24, "-", btn_type="secondary", kind="button")
            self._draw_button(draw, points, max_x, wy + 4, max_x + 20, wy + 24, "[]", btn_type="secondary", kind="button")
            self._draw_button(draw, points, close_x, wy + 4, close_x + 20, wy + 24, "X", btn_type="danger", kind="button")

            # Window content items
            if w_idx == num_windows - 1:
                # Top-most active window has prominent input & buttons
                self._draw_input(draw, points, wx + 30, wy + 55, wx + ww - 140, wy + 85, placeholder="Search or input query...", dark=is_dark)
                self._draw_button(draw, points, wx + ww - 120, wy + 55, wx + ww - 30, wy + 85, "Submit", btn_type="primary", kind="button")
                self._draw_button(draw, points, wx + 30, wy + wh - 45, wx + 130, wy + wh - 15, "Apply", btn_type="primary", kind="button")
                self._draw_button(draw, points, wx + 145, wy + wh - 45, wx + 235, wy + wh - 15, "Cancel", btn_type="secondary", kind="button")
            else:
                # Background windows have minor buttons
                self._draw_button(draw, points, wx + 25, wy + 45, wx + 95, wy + 70, "Action", btn_type="secondary", kind="button")

        # Windows Taskbar at bottom
        ty = self.H - 40
        draw.rectangle([0, ty, self.W, self.H], fill=(22, 24, 30), outline=(45, 48, 58), width=1)
        # Start button
        self._draw_button(draw, points, 6, ty + 5, 46, self.H - 5, "Win", btn_type="primary", kind="button")
        # Taskbar search bar
        self._draw_input(draw, points, 55, ty + 6, 220, self.H - 6, placeholder="Type to search", dark=True)
        # Pinned app icons
        for a_idx in range(4):
            ax = 235 + a_idx * 40
            self._draw_button(draw, points, ax, ty + 6, ax + 30, self.H - 6, f"A{a_idx+1}", btn_type="secondary", kind="icon")

        return {
            "category": "messy_multi_window",
            "subtype": f"multi_window_seed{seed}",
            "image": np.array(im),
            "gt_points": points,
            "description": f"Overlapping multi-window desktop with {num_windows} windows, icons, and taskbar",
        }

    def generate_complex_terminal(self, seed: int) -> Dict[str, Any]:
        """Complex terminal session with dense text, tabs, compiler logs, and command prompt."""
        rng = random.Random(seed)
        im = Image.new("RGB", (self.W, self.H), (14, 16, 20))
        draw = ImageDraw.Draw(im)
        points: List[Tuple[float, float, str, str]] = []

        # Terminal Titlebar with Tabs
        draw.rectangle([0, 0, self.W, 36], fill=(24, 26, 34))
        # Window controls
        self._draw_button(draw, points, self.W - 28, 6, self.W - 6, 28, "X", btn_type="danger", kind="button")
        self._draw_button(draw, points, self.W - 54, 6, self.W - 32, 28, "[]", btn_type="secondary", kind="button")
        self._draw_button(draw, points, self.W - 80, 6, self.W - 58, 28, "-", btn_type="secondary", kind="button")

        # 4 Terminal Tabs
        for t_idx, title in enumerate(["1: bash (main)", "2: python server", "3: htop", "4: git log"]):
            tx1 = 10 + t_idx * 145
            tx2 = tx1 + 135
            b_type = "primary" if t_idx == 0 else "tab"
            self._draw_button(draw, points, tx1, 5, tx2, 32, title, btn_type=b_type, kind="tab")
        # New Tab (+) button
        self._draw_button(draw, points, 600, 7, 626, 30, "+", btn_type="secondary", kind="button")

        # Dense Terminal Text Buffer
        y_pos = 50
        fake_terminal_lines = [
            ("user@forge-node-01:~/workspace$ git status", (0, 220, 110)),
            ("On branch main - Your branch is up to date with 'origin/main'.", (200, 200, 205)),
            ("Changes not staged for commit:", (220, 180, 60)),
            ("  modified: avcs/human_eye.py", (220, 80, 80)),
            ("  modified: tests/test_affordance_and_focus.py", (220, 80, 80)),
            ("user@forge-node-01:~/workspace$ py -3.14 -m pytest tests/ -q", (0, 220, 110)),
            ("running 231 tests across 12 test suites...", (160, 165, 175)),
            ("[PASS] tests/test_cortical_backbone.py (14ms)", (80, 200, 100)),
            ("[PASS] tests/test_ventral_stream.py (9ms)", (80, 200, 100)),
            ("[PASS] tests/test_affordance_and_focus.py (18ms)", (80, 200, 100)),
            ("[WARN] Zero-padding edge activation detected on solid white boundaries", (240, 160, 40)),
            ("[INFO] CPU memory footprint: 2.14 GB / 15.00 GB (4 threads active)", (120, 180, 240)),
            ("user@forge-node-01:~/workspace$ htop --delay 10", (0, 220, 110)),
            ("  CPU[|||||||||||||||||||||||||||||||| 68.4%]   Tasks: 42, 118 thr", (0, 200, 240)),
            ("  Mem[||||||||||||||||                 2.1G/15G] Swp[                    0K/0K]", (180, 100, 220)),
        ]
        for line_text, color in fake_terminal_lines:
            if y_pos > self.H - 90:
                break
            draw.text((20, y_pos), line_text, fill=color)
            y_pos += 22

        # Active terminal prompt typing area (Typeable input field)
        prompt_y = self.H - 55
        draw.text((20, prompt_y + 4), "user@forge-node-01:~/workspace$ ", fill=(0, 220, 110))
        self._draw_input(draw, points, 270, prompt_y, self.W - 30, prompt_y + 30, placeholder="pytest -k test_adversarial_affordance", dark=True)

        return {
            "category": "complex_terminal",
            "subtype": f"terminal_seed{seed}",
            "image": np.array(im),
            "gt_points": points,
            "description": "Terminal session with multiple tabs, compiler logs, htop meter, and active input line",
        }

    def generate_complex_browser(self, seed: int) -> Dict[str, Any]:
        """Complex browser window with multi-tabs, omnibox, extensions, and web page content."""
        rng = random.Random(seed)
        im = Image.new("RGB", (self.W, self.H), (242, 244, 248))
        draw = ImageDraw.Draw(im)
        points: List[Tuple[float, float, str, str]] = []

        # Browser Chrome Header (Dark/Light)
        draw.rectangle([0, 0, self.W, 75], fill=(35, 38, 46))

        # 6 Browser Tabs
        for t_idx, tab_name in enumerate(["GitHub - AVCS", "Google Search", "StackOverflow", "Arxiv: 2603.0411", "Documentation", "Dashboard"]):
            tx1 = 12 + t_idx * 135
            tx2 = tx1 + 128
            is_active = (t_idx == 0)
            fill_c = (48, 52, 64) if is_active else (30, 32, 38)
            draw.rectangle([tx1, 8, tx2, 38], fill=fill_c, outline=(60, 65, 78))
            draw.text((tx1 + 8, 16), tab_name[:14], fill=(240, 240, 245) if is_active else (160, 165, 175))
            points.append((round(((tx1 + tx2) / 2.0) / self.W, 4), round(23 / self.H, 4), "clickable", "tab"))

        # Window Controls
        self._draw_button(draw, points, self.W - 28, 8, self.W - 8, 30, "X", btn_type="danger", kind="button")
        self._draw_button(draw, points, self.W - 52, 8, self.W - 32, 30, "[]", btn_type="secondary", kind="button")
        self._draw_button(draw, points, self.W - 76, 8, self.W - 56, 30, "-", btn_type="secondary", kind="button")

        # Navigation Bar (Back, Forward, Refresh)
        self._draw_button(draw, points, 10, 42, 36, 68, "<", btn_type="secondary", kind="button")
        self._draw_button(draw, points, 42, 42, 68, 68, ">", btn_type="secondary", kind="button")
        self._draw_button(draw, points, 74, 42, 100, 68, "R", btn_type="secondary", kind="button")

        # Omnibox / URL Bar (Typeable)
        self._draw_input(draw, points, 110, 42, self.W - 140, 68, placeholder="https://github.com/forge/avcs-cortex", dark=True)

        # Action / Extension Buttons on Right
        self._draw_button(draw, points, self.W - 130, 42, self.W - 95, 68, "Ext1", btn_type="secondary", kind="icon")
        self._draw_button(draw, points, self.W - 90, 42, self.W - 55, 68, "Ext2", btn_type="secondary", kind="icon")
        self._draw_button(draw, points, self.W - 50, 42, self.W - 15, 68, "User", btn_type="primary", kind="button")

        # Web Page Body Content
        # Hero Section
        draw.rectangle([40, 95, self.W - 40, 210], fill=(255, 255, 255), outline=(220, 225, 235))
        draw.text((60, 115), "AVCS Visual Cortex Documentation & API", fill=(20, 25, 35))
        draw.text((60, 140), "Ultra-fast zero-delay conscious visual perception on commodity CPU.", fill=(80, 85, 95))
        self._draw_button(draw, points, 60, 168, 170, 198, "Get Started", btn_type="primary", kind="button")
        self._draw_button(draw, points, 185, 168, 280, 198, "GitHub Repo", btn_type="secondary", kind="button")

        # Two content cards
        card_w = (self.W - 100) // 2
        # Card 1
        draw.rectangle([40, 230, 40 + card_w, 420], fill=(255, 255, 255), outline=(220, 225, 235))
        draw.text((60, 245), "Search Documentation", fill=(30, 35, 45))
        self._draw_input(draw, points, 60, 275, 40 + card_w - 30, 310, placeholder="Filter by keyword (e.g. fovea)...", dark=False)
        self._draw_button(draw, points, 60, 325, 150, 355, "Search", btn_type="primary", kind="button")

        # Card 2
        c2_x = 60 + card_w
        draw.rectangle([c2_x, 230, c2_x + card_w, 420], fill=(255, 255, 255), outline=(220, 225, 235))
        draw.text((c2_x + 20, 245), "Account Settings & Preferences", fill=(30, 35, 45))
        self._draw_input(draw, points, c2_x + 20, 275, c2_x + card_w - 30, 310, placeholder="User email address", dark=False)
        self._draw_button(draw, points, c2_x + 20, 325, c2_x + 130, 355, "Save Settings", btn_type="primary", kind="button")
        self._draw_button(draw, points, c2_x + 145, 325, c2_x + 235, 355, "Discard", btn_type="secondary", kind="button")

        return {
            "category": "complex_browser",
            "subtype": f"browser_seed{seed}",
            "image": np.array(im),
            "gt_points": points,
            "description": "Multi-tab browser with omnibox, hero CTA, card forms, and navigation buttons",
        }

    def generate_rich_media_playback(self, seed: int) -> Dict[str, Any]:
        """Media player with photographic/natural video scene inside viewport and rich player UI chrome."""
        rng = np.random.RandomState(seed)
        im = Image.new("RGB", (self.W, self.H), (18, 20, 26))
        draw = ImageDraw.Draw(im)
        points: List[Tuple[float, float, str, str]] = []

        # Header / Titlebar
        draw.rectangle([0, 0, self.W, 35], fill=(28, 30, 38))
        draw.text((15, 10), "VLC Media Player - Documentaries_Nature_4K.mp4", fill=(210, 215, 225))
        self._draw_button(draw, points, self.W - 26, 6, self.W - 6, 28, "X", btn_type="danger", kind="button")

        # Video Viewport Dimensions
        vx1, vy1 = 30, 50
        vx2, vy2 = self.W - 280, self.H - 120

        # Render realistic photographic nature texture inside video viewport
        # Gradient sunset sky + silhouette hills
        vw, vh = vx2 - vx1, vy2 - vy1
        v_img = np.zeros((vh, vw, 3), dtype=np.uint8)
        vy_grid, vx_grid = np.mgrid[:vh, :vw]
        sky_factor = vy_grid / float(vh)
        v_img[:, :, 0] = np.clip(240 - sky_factor * 160, 0, 255).astype(np.uint8)  # Sunset red/gold
        v_img[:, :, 1] = np.clip(140 - sky_factor * 90, 0, 255).astype(np.uint8)
        v_img[:, :, 2] = np.clip(50 + sky_factor * 120, 0, 255).astype(np.uint8)
        # Add mountain hill silhouette
        ridge = (vh * 0.65 + np.sin(vx_grid[0] / 50.0 + seed) * 35).astype(int)
        for col_idx in range(vw):
            r_val = max(0, min(vh - 1, ridge[col_idx]))
            v_img[r_val:, col_idx] = (25, 32, 45)
        # Paste video viewport into main frame
        im.paste(Image.fromarray(v_img), (vx1, vy1))
        # Viewport border
        draw.rectangle([vx1, vy1, vx2, vy2], outline=(60, 65, 80), width=1)

        # Video Timeline Scrubber Bar
        scrub_y = vy2 + 15
        draw.rectangle([vx1, scrub_y, vx2, scrub_y + 8], fill=(50, 54, 66))
        # Progress filled
        fill_w = int((vx2 - vx1) * 0.42)
        draw.rectangle([vx1, scrub_y, vx1 + fill_w, scrub_y + 8], fill=(220, 50, 50))
        # Scrubber thumb handle
        thumb_x = vx1 + fill_w
        self._draw_button(draw, points, thumb_x - 6, scrub_y - 4, thumb_x + 6, scrub_y + 12, "O", btn_type="danger", kind="icon")

        # Media Control Buttons
        ctrl_y = scrub_y + 20
        self._draw_button(draw, points, vx1 + 10, ctrl_y, vx1 + 55, ctrl_y + 35, "|<", btn_type="secondary", kind="button")
        self._draw_button(draw, points, vx1 + 65, ctrl_y, vx1 + 120, ctrl_y + 35, "PLAY", btn_type="primary", kind="button")
        self._draw_button(draw, points, vx1 + 130, ctrl_y, vx1 + 175, ctrl_y + 35, ">|", btn_type="secondary", kind="button")
        self._draw_button(draw, points, vx1 + 185, ctrl_y, vx1 + 230, ctrl_y + 35, "STOP", btn_type="secondary", kind="button")

        # Volume Slider
        draw.text((vx1 + 250, ctrl_y + 10), "VOL:", fill=(180, 185, 195))
        draw.rectangle([vx1 + 290, ctrl_y + 14, vx1 + 420, ctrl_y + 22], fill=(60, 65, 78))
        draw.rectangle([vx1 + 290, ctrl_y + 14, vx1 + 380, ctrl_y + 22], fill=(0, 120, 215))
        self._draw_button(draw, points, vx1 + 375, ctrl_y + 8, vx1 + 387, ctrl_y + 28, "v", btn_type="primary", kind="icon")

        # Right-hand Playlist Sidebar
        px1, py1 = vx2 + 20, vy1
        px2, py2 = self.W - 20, self.H - 50
        draw.rectangle([px1, py1, px2, py2], fill=(25, 27, 34), outline=(50, 54, 66))
        draw.text((px1 + 12, py1 + 10), "Playlist (12 items)", fill=(220, 225, 235))
        self._draw_input(draw, points, px1 + 10, py1 + 32, px2 - 10, py1 + 58, placeholder="Filter playlist...", dark=True)

        for pl_idx in range(5):
            item_y = py1 + 70 + pl_idx * 42
            self._draw_button(draw, points, px1 + 8, item_y, px2 - 8, item_y + 34, f"Track 0{pl_idx+1}: Clip", btn_type="secondary", kind="button")

        return {
            "category": "rich_media_playback",
            "subtype": f"media_seed{seed}",
            "image": np.array(im),
            "gt_points": points,
            "description": "Rich media player with natural sunset video scene, timeline scrubber, transport controls, and playlist",
        }


# ==============================================================================
# 3. Stress Test Runner & Evaluator
# ==============================================================================

class AdversarialStressTestRunner:
    """Executes the adversarial stress testing suite against FastAffordanceHead."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu") -> None:
        self.device = device
        self.eye = HumanEye(device=device, load_weights=True, weights_path=weights_path)
        self.backbone = self.eye.backbone
        self.affordance_head = self.eye.affordance_head
        self.backbone.eval()
        self.affordance_head.eval()

    @torch.no_grad()
    def evaluate_frame(
        self,
        frame: np.ndarray,
        threshold: float = 0.35,
        margin: float = 0.05,
        top_k: int = 5,
        action_type: str = "all",
    ) -> List[Tuple[float, float, str, float]]:
        """Processes frame through backbone + FastAffordanceHead with custom parameters."""
        H, W = frame.shape[:2]
        sy = max(1, H // 128)
        sx = max(1, W // 128)
        frame_sub = frame[::sy, ::sx]
        if frame_sub.ndim == 2:
            frame_rgb = np.stack([frame_sub] * 3, axis=-1)
        else:
            frame_rgb = frame_sub[..., :3] if frame_sub.shape[-1] > 3 else frame_sub

        t_in = torch.from_numpy(np.ascontiguousarray(frame_rgb)).float() / 255.0
        t_in = t_in.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if t_in.shape[-2:] != (128, 128):
            t_resized = F.interpolate(t_in, size=(128, 128), mode="bilinear", align_corners=False)
        else:
            t_resized = t_in

        feat = self.backbone(t_resized)
        return self.affordance_head.find_hotspots(
            feat,
            action_type=action_type,
            top_k=top_k,
            threshold=threshold,
            margin=margin,
        )

    def run_hostile_negatives_benchmark(
        self,
        samples: List[Dict[str, Any]],
        threshold: float = 0.35,
        margin: float = 0.05,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Measures False Positives and failure modes on hostile negative images."""
        category_stats: Dict[str, Dict[str, Any]] = {}
        total_samples = len(samples)
        total_fps = 0
        screens_with_fps = 0
        all_confidences: List[float] = []
        action_counts = {"clickable": 0, "typeable": 0}
        foveal_survived_count = 0
        corner_fp_count = 0  # within 0.1 of boundary

        for sample in samples:
            cat = sample["category"]
            if cat not in category_stats:
                category_stats[cat] = {
                    "total_screens": 0,
                    "screens_with_fps": 0,
                    "total_fps": 0,
                    "max_confidence": 0.0,
                    "mean_confidence": 0.0,
                    "all_confidences": [],
                    "foveal_survived": 0,
                    "failures": [],
                }

            cat_dict = category_stats[cat]
            cat_dict["total_screens"] += 1

            img = sample["image"]
            hotspots = self.evaluate_frame(img, threshold=threshold, margin=margin, top_k=top_k)

            if len(hotspots) > 0:
                screens_with_fps += 1
                cat_dict["screens_with_fps"] += 1
                cat_dict["total_fps"] += len(hotspots)
                total_fps += len(hotspots)

                for hx, hy, act, conf in hotspots:
                    all_confidences.append(conf)
                    cat_dict["all_confidences"].append(conf)
                    cat_dict["max_confidence"] = max(cat_dict["max_confidence"], conf)
                    action_counts[act] = action_counts.get(act, 0) + 1

                    # Check if corner / zero-padding edge
                    if hx <= 0.08 or hx >= 0.92 or hy <= 0.08 or hy >= 0.92:
                        corner_fp_count += 1

                    # Check foveal refinement
                    refined = self.eye.refine_affordance_with_fovea((hx, hy, act, conf), frame=img)
                    if refined is not None:
                        foveal_survived_count += 1
                        cat_dict["foveal_survived"] += 1

                cat_dict["failures"].append({
                    "subtype": sample["subtype"],
                    "hotspots": hotspots,
                })

        for cat, c_data in category_stats.items():
            if c_data["all_confidences"]:
                c_data["mean_confidence"] = round(float(np.mean(c_data["all_confidences"])), 3)
            else:
                c_data["mean_confidence"] = 0.0
            del c_data["all_confidences"]

        return {
            "total_negative_screens": total_samples,
            "screens_with_fps": screens_with_fps,
            "screen_false_alarm_rate": round(screens_with_fps / max(1, total_samples), 4),
            "total_false_positive_hotspots": total_fps,
            "mean_fps_per_screen": round(total_fps / max(1, total_samples), 3),
            "max_confidence": round(float(max(all_confidences)), 4) if all_confidences else 0.0,
            "mean_confidence": round(float(np.mean(all_confidences)), 4) if all_confidences else 0.0,
            "foveal_survival_count": foveal_survived_count,
            "foveal_survival_rate": round(foveal_survived_count / max(1, total_fps), 4) if total_fps else 0.0,
            "corner_edge_fps": corner_fp_count,
            "corner_edge_fp_rate": round(corner_fp_count / max(1, total_fps), 4) if total_fps else 0.0,
            "action_breakdown": action_counts,
            "category_breakdown": category_stats,
        }

    def run_complex_screens_benchmark(
        self,
        samples: List[Dict[str, Any]],
        match_radius: float = 0.15,
        threshold: float = 0.35,
        margin: float = 0.05,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Measures TP, FP, FN, Precision, Recall, F1 on complex uncurated desktop screens."""
        total_screens = len(samples)
        total_gt = 0
        total_pred = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        action_match_correct = 0
        localization_errors: List[float] = []

        category_stats: Dict[str, Dict[str, Any]] = {}

        for sample in samples:
            cat = sample["category"]
            if cat not in category_stats:
                category_stats[cat] = {
                    "screens": 0,
                    "gt_count": 0,
                    "pred_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "loc_errors": [],
                }

            cat_dict = category_stats[cat]
            cat_dict["screens"] += 1

            img = sample["image"]
            gt_points = sample["gt_points"]
            cat_dict["gt_count"] += len(gt_points)
            total_gt += len(gt_points)

            hotspots = self.evaluate_frame(img, threshold=threshold, margin=margin, top_k=top_k)
            cat_dict["pred_count"] += len(hotspots)
            total_pred += len(hotspots)

            matched_gt_indices = set()
            for hx, hy, act, conf in hotspots:
                best_dist = 1e9
                best_gt_idx = -1
                for idx, gt in enumerate(gt_points):
                    if idx in matched_gt_indices:
                        continue
                    gx, gy, gact = gt[0], gt[1], gt[2]
                    dist = math.hypot(hx - gx, hy - gy)
                    if dist <= match_radius and dist < best_dist:
                        best_dist = dist
                        best_gt_idx = idx

                if best_gt_idx >= 0:
                    total_tp += 1
                    cat_dict["tp"] += 1
                    matched_gt_indices.add(best_gt_idx)
                    localization_errors.append(best_dist)
                    cat_dict["loc_errors"].append(best_dist)
                    if act == gt_points[best_gt_idx][2]:
                        action_match_correct += 1
                else:
                    total_fp += 1
                    cat_dict["fp"] += 1

            fn_sample = len(gt_points) - len(matched_gt_indices)
            total_fn += fn_sample
            cat_dict["fn"] += fn_sample

        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = (2 * precision * recall) / max(1e-6, precision + recall)

        for cat, c_data in category_stats.items():
            c_tp = c_data["tp"]
            c_fp = c_data["fp"]
            c_fn = c_data["fn"]
            c_prec = c_tp / max(1, c_tp + c_fp)
            c_rec = c_tp / max(1, c_tp + c_fn)
            c_f1 = (2 * c_prec * c_rec) / max(1e-6, c_prec + c_rec)
            c_data["precision"] = round(c_prec, 4)
            c_data["recall"] = round(c_rec, 4)
            c_data["f1_score"] = round(c_f1, 4)
            c_data["mean_loc_error"] = round(float(np.mean(c_data["loc_errors"])), 4) if c_data["loc_errors"] else 0.0
            del c_data["loc_errors"]

        return {
            "total_screens": total_screens,
            "total_ground_truth": total_gt,
            "total_predictions": total_pred,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "action_type_accuracy_on_tp": round(action_match_correct / max(1, total_tp), 4),
            "mean_localization_error": round(float(np.mean(localization_errors)), 4) if localization_errors else 0.0,
            "category_breakdown": category_stats,
        }


# ==============================================================================
# 4. Main Entry Point & Command Line Runner
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial Stress Test for FastAffordanceHead")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--out-dir", type=str, default="runs/adversarial_affordance", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      FAST AFFORDANCE HEAD ADVERSARIAL STRESS TEST BENCHMARK     ")
    print("=================================================================")
    print(f"Device: {args.device} | Master Seed: {args.seed}")

    runner = AdversarialStressTestRunner(device=args.device)

    # 1. Build Hostile Negative Dataset
    print("\n[1/4] Generating Hostile Negative Dataset...")
    neg_gen = HostileNegativeGenerator(width=1280, height=800)
    neg_samples = neg_gen.generate_suite(seed=args.seed)
    print(f"  Generated {len(neg_samples)} hostile negative images across 6 distinct categories.")

    # 2. Build Complex Uncurated Screens Dataset
    print("\n[2/4] Generating Complex Uncurated Screens Dataset...")
    complex_gen = ComplexScreenGenerator(width=1280, height=800)
    complex_samples: List[Dict[str, Any]] = []

    # A. 25 Messy Multi-window desktops
    for i in range(25):
        complex_samples.append(complex_gen.generate_messy_multi_window(seed=args.seed + 100 + i))

    # B. 25 Complex terminal sessions
    for i in range(25):
        complex_samples.append(complex_gen.generate_complex_terminal(seed=args.seed + 200 + i))

    # C. 25 Complex browser tabs
    for i in range(25):
        complex_samples.append(complex_gen.generate_complex_browser(seed=args.seed + 300 + i))

    # D. 25 Rich media playback screens
    for i in range(25):
        complex_samples.append(complex_gen.generate_rich_media_playback(seed=args.seed + 400 + i))

    # E. Live Windows desktop screen grabs (if available)
    try:
        cap = open_capture(prefer_real=True)
        live_frame = cap.grab()
        if live_frame is not None:
            # Create a few variations/crops of live desktop
            complex_samples.append({
                "category": "live_desktop",
                "subtype": "live_win32_full",
                "image": live_frame,
                "gt_points": [(0.02, 0.97, "clickable", "button")],  # Start button
                "description": "Live Windows desktop capture",
            })
            print("  Successfully integrated live Windows desktop capture.")
    except Exception as exc:
        print(f"  Notice: Live screen capture fallback ({exc}).")

    print(f"  Generated {len(complex_samples)} complex uncurated desktop screens.")

    # 3. Execute Hostile Negatives Stress Test
    print("\n[3/4] Running Hostile Negatives Stress Test...")
    neg_results_default = runner.run_hostile_negatives_benchmark(neg_samples, threshold=0.35, margin=0.05, top_k=5)
    print(f"  Hostile Negatives Tested: {neg_results_default['total_negative_screens']}")
    print(f"  Screens with False Positives: {neg_results_default['screens_with_fps']} ({neg_results_default['screen_false_alarm_rate']*100:.1f}%)")
    print(f"  Total False Positive Hotspots: {neg_results_default['total_false_positive_hotspots']}")
    print(f"  Max False Alarm Confidence: {neg_results_default['max_confidence']:.4f}")
    print(f"  Foveal Survival (Unfiltered Hallucinations): {neg_results_default['foveal_survival_count']} ({neg_results_default['foveal_survival_rate']*100:.1f}%)")

    # 4. Execute Complex Screens Stress Test
    print("\n[4/4] Running Complex Desktop Screens Stress Test...")
    comp_results_default = runner.run_complex_screens_benchmark(complex_samples, match_radius=0.15, threshold=0.35, margin=0.05, top_k=5)
    print(f"  Complex Screens Tested: {comp_results_default['total_screens']}")
    print(f"  Total Ground Truth Elements: {comp_results_default['total_ground_truth']}")
    print(f"  Total Hotspot Predictions: {comp_results_default['total_predictions']}")
    print(f"  True Positives: {comp_results_default['true_positives']}")
    print(f"  False Positives: {comp_results_default['false_positives']}")
    print(f"  False Negatives: {comp_results_default['false_negatives']}")
    print(f"  Precision: {comp_results_default['precision']*100:.2f}%")
    print(f"  Recall: {comp_results_default['recall']*100:.2f}%")
    print(f"  F1 Score: {comp_results_default['f1_score']*100:.2f}%")

    # 5. Sensitivity Sweep Across Thresholds
    print("\n[Threshold Sensitivity Sweep]")
    sweep_results = {}
    for th, mg in [(0.20, 0.02), (0.35, 0.05), (0.50, 0.10), (0.70, 0.15)]:
        n_res = runner.run_hostile_negatives_benchmark(neg_samples, threshold=th, margin=mg, top_k=5)
        c_res = runner.run_complex_screens_benchmark(complex_samples, threshold=th, margin=mg, top_k=5)
        sweep_results[f"th_{th:.2f}_mg_{mg:.2f}"] = {
            "threshold": th,
            "margin": mg,
            "neg_screens_with_fp": n_res["screens_with_fps"],
            "neg_total_fps": n_res["total_false_positive_hotspots"],
            "neg_max_conf": n_res["max_confidence"],
            "comp_tp": c_res["true_positives"],
            "comp_fp": c_res["false_positives"],
            "comp_fn": c_res["false_negatives"],
            "comp_precision": c_res["precision"],
            "comp_recall": c_res["recall"],
            "comp_f1": c_res["f1_score"],
        }
        print(f"  th={th:.2f}, mg={mg:.2f} -> Neg FPs: {n_res['total_false_positive_hotspots']} ({n_res['screens_with_fps']}/{len(neg_samples)} screens) | Comp Prec: {c_res['precision']*100:.1f}%, Rec: {c_res['recall']*100:.1f}%, F1: {c_res['f1_score']*100:.1f}%")

    # 6. Top-K Sweep (top_k=5 vs top_k=15 vs unconstrained top_k=50)
    print("\n[Top-K Capacity Sweep]")
    topk_results = {}
    for tk in [3, 5, 10, 20]:
        n_res = runner.run_hostile_negatives_benchmark(neg_samples, threshold=0.35, margin=0.05, top_k=tk)
        c_res = runner.run_complex_screens_benchmark(complex_samples, threshold=0.35, margin=0.05, top_k=tk)
        topk_results[f"topk_{tk}"] = {
            "top_k": tk,
            "neg_total_fps": n_res["total_false_positive_hotspots"],
            "comp_tp": c_res["true_positives"],
            "comp_fp": c_res["false_positives"],
            "comp_fn": c_res["false_negatives"],
            "comp_precision": c_res["precision"],
            "comp_recall": c_res["recall"],
            "comp_f1": c_res["f1_score"],
        }
        print(f"  top_k={tk} -> Neg FPs: {n_res['total_false_positive_hotspots']} | Comp Prec: {c_res['precision']*100:.1f}%, Rec: {c_res['recall']*100:.1f}%, F1: {c_res['f1_score']*100:.1f}%")

    # Assemble Full Report Data
    full_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "master_seed": args.seed,
        "hostile_negatives_summary": neg_results_default,
        "complex_screens_summary": comp_results_default,
        "threshold_sensitivity_sweep": sweep_results,
        "top_k_sweep": topk_results,
    }

    report_path = out_dir / "adversarial_affordance_results.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Saved Full JSON Results] -> {report_path}")


if __name__ == "__main__":
    main()
