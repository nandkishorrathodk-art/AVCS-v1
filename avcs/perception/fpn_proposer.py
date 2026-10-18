"""AVCS Path 1: High-Resolution Feature-Pyramid Proposal Head (FPN).

Extracts spatial proposals across multiple pyramid levels:
- P2 (Stride 4, 64x64 grid): Small targets (<24 px)
- P3 (Stride 8, 32x32 grid): Medium targets (24-64 px)
- P4 (Stride 16, 16x16 grid): Large targets (>64 px) & Global UI category

Employs top-down lateral feature fusion to prevent spatial collapse of adjacent dense controls.
Zero OCR, zero accessibility APIs, zero DOM scraping.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.perception.visual_pilot import ELEMENT_KINDS, UI_CATEGORIES
from avcs.perception.visual_state import VisualRegion


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
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class FPNRegionProposer(nn.Module):
    """Multi-Scale Feature Pyramid Region Proposer.

    Resolves the stride-16 quantization bottleneck by providing stride-4 (P2)
    and stride-8 (P3) dense spatial proposal heads alongside stride-16 (P4).
    """

    def __init__(self, pyramid_dim: int = 48) -> None:
        super().__init__()
        self.pyramid_dim = pyramid_dim

        # Bottom-up feature extractor
        # 256x256 -> C1 (128x128) -> C2 (64x64) -> C3 (32x32) -> C4 (16x16)
        self.stem = ConvBNAct(3, 24, stride=2)               # 128x128
        self.stage1 = DepthwiseSeparable(24, 48, stride=2)   # C2: 64x64 (stride 4)
        self.stage2 = DepthwiseSeparable(48, 64, stride=2)   # C3: 32x32 (stride 8)
        self.stage3 = DepthwiseSeparable(64, 96, stride=2)   # C4: 16x16 (stride 16)

        # Lateral 1x1 convolutions
        self.lat4 = nn.Conv2d(96, pyramid_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(64, pyramid_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(48, pyramid_dim, kernel_size=1)

        # Smooth 3x3 convolutions
        self.smooth3 = nn.Conv2d(pyramid_dim, pyramid_dim, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(pyramid_dim, pyramid_dim, kernel_size=3, padding=1)

        # Global Application Category Head (from P4)
        self.cat_pool = nn.AdaptiveAvgPool2d(1)
        self.cat_head = nn.Sequential(
            nn.Linear(pyramid_dim, 64),
            nn.GELU(),
            nn.Linear(64, len(UI_CATEGORIES)),
        )

        # Multi-scale proposal heads
        # Outputs: 2 classes (static=0, interactive=1) + 4 bbox offsets [ymin, xmin, ymax, xmax] + 4 kind logits
        num_outputs = 2 + 4 + len(ELEMENT_KINDS)

        self.head_p2 = nn.Conv2d(pyramid_dim, num_outputs, kernel_size=1) # 64x64
        self.head_p3 = nn.Conv2d(pyramid_dim, num_outputs, kernel_size=1) # 32x32
        self.head_p4 = nn.Conv2d(pyramid_dim, num_outputs, kernel_size=1) # 16x16

        self.word_recognizer: Optional[Any] = None

    def attach_word_recognizer(self, recognizer: Any) -> None:
        """Attaches a PixelWordRecognizer to predict text on extracted spatial proposals."""
        self.word_recognizer = recognizer

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # Bottom-up
        c1 = self.stem(x)
        c2 = self.stage1(c1) # 64x64
        c3 = self.stage2(c2) # 32x32
        c4 = self.stage3(c3) # 16x16

        # Top-down FPN pathway
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)

        # Category prediction from global P4
        pooled = self.cat_pool(p4).flatten(1)
        cat_logits = self.cat_head(pooled)

        # Multi-scale proposals
        out_p2 = self.head_p2(p2) # (B, 10, 64, 64)
        out_p3 = self.head_p3(p3) # (B, 10, 32, 32)
        out_p4 = self.head_p4(p4) # (B, 10, 16, 16)

        return {
            "cat_logits": cat_logits,
            "p2": out_p2,
            "p3": out_p3,
            "p4": out_p4,
        }

    @torch.inference_mode()
    def propose_regions(
        self,
        image_np: np.ndarray,
        threshold: float = 0.40,
        margin: float = 0.05,
        max_proposals: int = 36,
        nms_iou_threshold: float = 0.35,
    ) -> Tuple[str, float, List[VisualRegion]]:
        """Extracts multi-scale UI proposals across stride-4, stride-8, and stride-16 grids."""
        H, W = image_np.shape[:2]

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

        # 1. Primary Category
        cat_probs = F.softmax(out["cat_logits"][0], dim=-1)
        cat_idx = int(torch.argmax(cat_probs).item())
        pred_cat = UI_CATEGORIES[cat_idx]
        cat_conf = float(cat_probs[cat_idx].item())

        candidate_regions: List[Tuple[float, Tuple[float, float, float, float], str]] = []

        # Process pyramid levels: (level_tensor, grid_size, max_level_peaks, default_span)
        levels = [
            (out["p2"][0], 64, 20, 0.04), # Stride 4 (small controls)
            (out["p3"][0], 32, 14, 0.08), # Stride 8 (medium controls)
            (out["p4"][0], 16, 8,  0.18), # Stride 16 (large controls)
        ]

        for head_out, grid_sz, max_peaks, default_span in levels:
            # head_out: (10, grid_sz, grid_sz)
            aff_logits = head_out[:2] # 0: static, 1: interactive
            aff_probs = F.softmax(aff_logits, dim=0)
            static_heat = aff_probs[0]
            interactive_heat = aff_probs[1]

            # 2D spatial NMS on this level
            heat_4d = interactive_heat.unsqueeze(0).unsqueeze(0)
            max_pool = F.max_pool2d(heat_4d, kernel_size=3, stride=1, padding=1).squeeze()

            is_peak = (interactive_heat == max_pool) & (interactive_heat >= threshold)
            if margin > 0.0:
                is_peak = is_peak & ((interactive_heat - static_heat) >= margin)

            peak_coords = torch.nonzero(is_peak, as_tuple=False)
            if len(peak_coords) == 0:
                continue

            vals = interactive_heat[peak_coords[:, 0], peak_coords[:, 1]]
            if len(vals) > max_peaks:
                top_order = torch.argsort(vals, descending=True)[:max_peaks]
                peak_coords = peak_coords[top_order]
                vals = vals[top_order]

            box_preds = torch.sigmoid(head_out[2:6]) # (4, grid_sz, grid_sz)
            kind_probs = F.softmax(head_out[6:], dim=0) # (4, grid_sz, grid_sz)

            for i in range(len(peak_coords)):
                py = int(peak_coords[i, 0].item())
                px = int(peak_coords[i, 1].item())
                conf = float(vals[i].item())

                cx = (px + 0.5) / float(grid_sz)
                cy = (py + 0.5) / float(grid_sz)

                # Regressed box offsets
                b = box_preds[:, py, px]
                ymin = float(b[0].item())
                xmin = float(b[1].item())
                ymax = float(b[2].item())
                xmax = float(b[3].item())

                span_w = max(0.015, (xmax - xmin) * 0.5)
                span_h = max(0.015, (ymax - ymin) * 0.5)

                g_xmin = max(0.0, min(1.0, cx - span_w))
                g_xmax = max(0.0, min(1.0, cx + span_w))
                g_ymin = max(0.0, min(1.0, cy - span_h))
                g_ymax = max(0.0, min(1.0, cy + span_h))

                k_idx = int(torch.argmax(kind_probs[:, py, px]).item())
                kind_str = ELEMENT_KINDS[k_idx] if k_idx < len(ELEMENT_KINDS) else "button"

                candidate_regions.append((conf, (g_ymin, g_xmin, g_ymax, g_xmax), kind_str))

        # Cross-level Non-Maximum Suppression (NMS)
        candidate_regions.sort(key=lambda x: x[0], reverse=True)
        kept_proposals: List[VisualRegion] = []

        def calc_iou(bA, bB):
            yA, xA = max(bA[0], bB[0]), max(bA[1], bB[1])
            yB, xB = min(bA[2], bB[2]), min(bA[3], bB[3])
            inter = max(0.0, yB - yA) * max(0.0, xB - xA)
            areaA = max(0.0, bA[2] - bA[0]) * max(0.0, bA[3] - bA[1])
            areaB = max(0.0, bB[2] - bB[0]) * max(0.0, bB[3] - bB[1])
            union = areaA + areaB - inter
            return inter / union if union > 0 else 0.0

        for conf, box, kind_str in candidate_regions:
            if len(kept_proposals) >= max_proposals:
                break
            # Check overlap against already kept
            suppressed = False
            for kp in kept_proposals:
                if calc_iou(box, kp.bbox_norm) > nms_iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                b_px = (round(box[0] * H, 1), round(box[1] * W, 1), round(box[2] * H, 1), round(box[3] * W, 1))
                b_norm = (round(box[0], 4), round(box[1], 4), round(box[2], 4), round(box[3], 4))
                kept_proposals.append(
                    VisualRegion(
                        region_id=f"fpn_reg_{len(kept_proposals)}",
                        element_kind=kind_str,
                        bbox_norm=b_norm,
                        bbox_px=b_px,
                        confidence=round(conf, 3),
                        is_interactive=True,
                        is_focused=False,
                    )
                )

        # Attach text recognition if word_recognizer is present
        if self.word_recognizer is not None and len(kept_proposals) > 0:
            crops: List[np.ndarray] = []
            for kp in kept_proposals:
                y1 = max(0, min(H - 1, int(round(kp.bbox_px[0]))))
                x1 = max(0, min(W - 1, int(round(kp.bbox_px[1]))))
                y2 = max(y1 + 4, min(H, int(round(kp.bbox_px[2]))))
                x2 = max(x1 + 4, min(W, int(round(kp.bbox_px[3]))))
                crops.append(image_np[y1:y2, x1:x2])

            word_preds = self.word_recognizer.predict_crops_batch(crops)
            proposals_with_text: List[VisualRegion] = []
            for kp, (w_str, w_conf) in zip(kept_proposals, word_preds):
                det_text = w_str if w_str != "<none>" else None
                proposals_with_text.append(
                    VisualRegion(
                        region_id=kp.region_id,
                        element_kind=kp.element_kind,
                        bbox_norm=kp.bbox_norm,
                        bbox_px=kp.bbox_px,
                        confidence=kp.confidence,
                        is_interactive=kp.is_interactive,
                        is_focused=kp.is_focused,
                        detected_text=det_text,
                        text_confidence=round(w_conf, 3) if det_text else 0.0,
                    )
                )
            kept_proposals = proposals_with_text

        return pred_cat, cat_conf, kept_proposals

