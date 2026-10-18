"""AVCS System 1 Visual Understanding Pilot (Candidate M3-VisualPilot-V1).

Extracts pixel-native visual structure from desktop frames:
1. High-level UI Category Classification (8 application types).
2. Spatial Affordance Heatmap (16x16 grid: static background, clickable, typeable).
3. Region Proposal & Boundary Regression (element kinds and bounding boxes).
4. Emits typed VisualRegion objects conforming to VisualStateObservation contract.

Pure-pixel neural architecture: zero OCR, zero UI Automation, zero DOM scraping.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.perception.visual_state import VisualRegion


UI_CATEGORIES: List[str] = [
    "desktop_canvas",
    "editor_code",
    "browser_webpage",
    "terminal_shell",
    "modal_dialog",
    "form_input",
    "media_playback",
    "settings_panel",
]

ELEMENT_KINDS: List[str] = [
    "button",
    "input_field",
    "icon",
    "tab",
]


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.dw(x)))
        out = self.act(self.bn2(self.pw(out)))
        return out


class VisualUnderstandingPilot(nn.Module):
    """Candidate M3 Visual Understanding Pilot Network.

    Transforms (B, 3, 256, 256) desktop images into:
    1. category_logits: (B, 8) - UI application category
    2. affordance_logits: (B, 3, 16, 16) - 0: static, 1: clickable, 2: typeable
    3. region_box: (B, 4, 16, 16) - [ymin, xmin, ymax, xmax] relative to cell
    4. region_kind_logits: (B, 4, 16, 16) - element kind classification
    """

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Backbone: 256x256 -> 16x16 (stride 16 total: 2 * 2 * 2 * 2)
        self.stem = ConvBNAct(3, 32, stride=2)              # 256 -> 128
        self.stage1 = DepthwiseSeparable(32, 64, stride=2)   # 128 -> 64
        self.stage2 = DepthwiseSeparable(64, 96, stride=2)   # 64 -> 32
        self.stage3 = DepthwiseSeparable(96, feature_dim, stride=2) # 32 -> 16
        self.refine = ConvBNAct(feature_dim, feature_dim, stride=1) # 16 -> 16

        # Category Classification Stream
        self.cat_pool = nn.AdaptiveAvgPool2d(1)
        self.cat_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(UI_CATEGORIES)),
        )

        # Spatial Affordance Stream (16x16 grid)
        self.affordance_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        # Region Bounding Box & Kind Stream
        self.region_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 4 + len(ELEMENT_KINDS), kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass on image tensor (B, 3, 256, 256) normalized to [0, 1]."""
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.refine(feat) # (B, 128, 16, 16)

        # 1. Category logits
        pooled = self.cat_pool(feat).flatten(1)
        cat_logits = self.cat_head(pooled) # (B, 8)

        # 2. Affordance logits (3 classes)
        aff_logits = self.affordance_head(feat) # (B, 3, 16, 16)

        # 3. Region predictions
        reg_out = self.region_head(feat) # (B, 8, 16, 16)
        box_out = torch.sigmoid(reg_out[:, :4, :, :]) # [ymin, xmin, ymax, xmax] in [0, 1]
        kind_logits = reg_out[:, 4:, :, :] # (B, 4, 16, 16)

        return {
            "features": feat,
            "category_logits": cat_logits,
            "affordance_logits": aff_logits,
            "box_pred": box_out,
            "kind_logits": kind_logits,
        }

    @torch.inference_mode()
    def predict_visual_structure(
        self,
        image_np: np.ndarray,
        threshold: float = 0.35,
        margin: float = 0.05,
        top_k: int = 12,
    ) -> Tuple[str, float, List[VisualRegion]]:
        """Extracts structured UI category and visual regions from a raw RGB frame."""
        H, W = image_np.shape[:2]

        # Preprocess to (1, 3, 256, 256)
        img_t = torch.from_numpy(image_np).float() / 255.0
        if img_t.ndim == 2:
            img_t = torch.stack([img_t] * 3, dim=-1)
        elif img_t.shape[-1] > 3:
            img_t = img_t[..., :3]

        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        if (H, W) != (256, 256):
            img_t = F.interpolate(img_t, size=(256, 256), mode="bilinear", align_corners=False)

        device = next(self.parameters()).device
        img_t = img_t.to(device)

        out = self.forward(img_t)

        # 1. Primary Category Prediction
        cat_probs = F.softmax(out["category_logits"], dim=-1)[0]
        cat_idx = int(torch.argmax(cat_probs).item())
        pred_cat = UI_CATEGORIES[cat_idx]
        cat_conf = float(cat_probs[cat_idx].item())

        # 2. Affordance & Region Extraction
        aff_probs = F.softmax(out["affordance_logits"][0], dim=0) # (3, 16, 16)
        static_heat = aff_probs[0] # (16, 16)
        interactive_heat = aff_probs[1] + aff_probs[2] # clickable + typeable

        # 2D Spatial NMS using 3x3 MaxPool
        heat_4d = interactive_heat.unsqueeze(0).unsqueeze(0)
        max_pool = F.max_pool2d(heat_4d, kernel_size=3, stride=1, padding=1).squeeze()

        is_peak = (interactive_heat == max_pool) & (interactive_heat >= threshold)
        if margin > 0.0:
            is_peak = is_peak & ((interactive_heat - static_heat) >= margin)

        peak_coords = torch.nonzero(is_peak, as_tuple=False)
        if len(peak_coords) == 0:
            return pred_cat, cat_conf, []

        vals = interactive_heat[peak_coords[:, 0], peak_coords[:, 1]]
        if len(vals) > top_k:
            top_order = torch.argsort(vals, descending=True)[:top_k]
            peak_coords = peak_coords[top_order]
            vals = vals[top_order]

        box_pred = out["box_pred"][0] # (4, 16, 16)
        kind_probs = F.softmax(out["kind_logits"][0], dim=0) # (4, 16, 16)

        detected_regions: List[VisualRegion] = []
        grid_sz = 16

        for i in range(len(peak_coords)):
            py = int(peak_coords[i, 0].item())
            px = int(peak_coords[i, 1].item())
            conf = float(vals[i].item())

            # Cell center in normalized coords
            cell_cx = (px + 0.5) / float(grid_sz)
            cell_cy = (py + 0.5) / float(grid_sz)

            # Element Kind
            k_probs = kind_probs[:, py, px]
            k_idx = int(torch.argmax(k_probs).item())
            kind_str = ELEMENT_KINDS[k_idx] if k_idx < len(ELEMENT_KINDS) else "button"

            # Bounding Box relative to cell
            # [ymin, xmin, ymax, xmax]
            b = box_pred[:, py, px]
            ymin = float(b[0].item())
            xmin = float(b[1].item())
            ymax = float(b[2].item())
            xmax = float(b[3].item())

            # Ensure proper ordering and span
            if ymax <= ymin:
                ymax = ymin + 0.05
            if xmax <= xmin:
                xmax = xmin + 0.05

            # Global normalized coords
            # Span around cell center
            span_w = (xmax - xmin) * 0.5
            span_h = (ymax - ymin) * 0.5
            g_xmin = max(0.0, min(1.0, cell_cx - span_w))
            g_xmax = max(0.0, min(1.0, cell_cx + span_w))
            g_ymin = max(0.0, min(1.0, cell_cy - span_h))
            g_ymax = max(0.0, min(1.0, cell_cy + span_h))

            bbox_norm = (round(g_ymin, 4), round(g_xmin, 4), round(g_ymax, 4), round(g_xmax, 4))
            bbox_px = (round(g_ymin * H, 1), round(g_xmin * W, 1), round(g_ymax * H, 1), round(g_xmax * W, 1))

            is_clickable = aff_probs[1, py, px] > aff_probs[2, py, px]
            is_typeable = not is_clickable

            detected_regions.append(
                VisualRegion(
                    region_id=f"region_{i}",
                    element_kind=kind_str,
                    bbox_norm=bbox_norm,
                    bbox_px=bbox_px,
                    confidence=round(conf, 3),
                    is_interactive=True,
                    is_focused=False,
                )
            )

        return pred_cat, cat_conf, detected_regions
