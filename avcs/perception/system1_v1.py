"""AVCS System 1 V1 — Unified Multi-Task Pixel-Native Visual Perception Model.

Unifies in a single versioned architecture:
1. Multi-Scale Visual Region Detection (small, medium, large UI elements across Stride 4/8/16).
2. Region Classification (button, input_field, icon, tab, link, text_label, container, unknown)
   and Global UI Application Category Classification (8 application types).
3. Query-Conditioned Target Grounding (Byte-level text encoder, cross-modal modulation,
   128x128 dense spatial heatmap, subpixel peak extraction, and calibrated presence classification).
4. Pixel-Based Patch Word Recognition (131 distinct functional UI terms + class 0 <none>).
5. Temporal Motion Perception (Magnocellular optical flow energy, motion magnitude, surprise,
   scene stability, and overlay detection across consecutive frames).

Outputs typed, non-hallucinated VisualStateObservation objects conforming to the System 1 contract.
Strict Pure-Pixel Mandate: ZERO external OCR libraries, ZERO UI Automation, ZERO DOM inspection.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.torch_version import TorchVersion
import torch.nn as nn
import torch.nn.functional as F

from avcs.perception.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    ByteQueryEncoder,
    GoalConditionedGrounder,
    encode_queries,
    extract_subpixel_peak,
)
from avcs.perception.human_eye import (
    ConvBNAct,
    MagnocellularMotionEngine,
    MovingObject,
)
from avcs.perception.visual_pilot import ELEMENT_KINDS, UI_CATEGORIES
from avcs.perception.visual_state import (
    SpatialCandidateInfo,
    TemporalChangeInfo,
    VisualRegion,
    VisualStateObservation,
)
from avcs.perception.word_recognizer import (
    CANONICAL_CROP_HEIGHT,
    CANONICAL_CROP_WIDTH,
    ID_TO_WORD,
    PixelWordRecognizer,
    UI_WORD_VOCABULARY,
    WORD_TO_ID,
)
from avcs.query_limits import MAX_QUERY_BYTES

GENERIC_QUERY_WORDS = {
    "button", "icon", "input", "field", "box", "menu", "tab", "window",
    "panel", "link", "slider", "control", "click", "the", "a", "an", "on", "in", "to", "or", "of", "and",
}
VOCAB_LOWER_SET = {w.lower() for w in UI_WORD_VOCABULARY if w != "<none>"}


def is_out_of_vocabulary_query(query: str) -> bool:
    """Checks if query contains unknown semantic tokens outside the closed UI vocabulary."""
    clean_q = query.lower().strip()
    if not clean_q:
        return False
    # Check if any multi-character vocabulary phrase/word is in query
    for v in VOCAB_LOWER_SET:
        if len(v) > 1 and v in clean_q:
            return False
    # Check individual words
    words = [w.strip(".,:;!?\"'") for w in clean_q.split()]
    non_generic = [w for w in words if w and w not in GENERIC_QUERY_WORDS]
    if not non_generic:
        return False
    for w in non_generic:
        if w not in VOCAB_LOWER_SET:
            return True
    return False


SYSTEM1_V1_SCHEMA_VERSION: int = 2
SYSTEM1_V1_CHECKPOINT_NAME: str = "avcs_system1_v1.pt"

# Supported element kind classes
SYSTEM1_ELEMENT_KINDS: List[str] = [
    "button",
    "input_field",
    "icon",
    "tab",
]
KIND_TO_IDX: Dict[str, int] = {k: i for i, k in enumerate(SYSTEM1_ELEMENT_KINDS)}
IDX_TO_KIND: Dict[int, str] = {i: k for i, k in enumerate(SYSTEM1_ELEMENT_KINDS)}


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


class AVCSSystem1V1(nn.Module):
    """AVCS System 1 V1: Unified Multi-Task Neural Visual Cortex for Desktop Perception.

    Encapsulates:
    - Multi-Scale FPN Visual Proposer (Stride 4/8/16)
    - Goal-Conditioned Language Gaze Grounder
    - Pixel-Native Patch Word Recognizer
    - Temporal Magnocellular Motion & Stability Engine
    """

    def __init__(
        self,
        pyramid_dim: int = 48,
        grounder_dim: int = 192,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.pyramid_dim = pyramid_dim
        self.grounder_dim = grounder_dim

        # -------------------------------------------------------------
        # 1. Multi-Scale Feature Pyramid Proposal Backbone
        # -------------------------------------------------------------
        self.stem = ConvBNAct(3, 24, stride=2)               # 128x128
        self.stage1 = DepthwiseSeparable(24, 48, stride=2)   # C2: 64x64 (stride 4)
        self.stage2 = DepthwiseSeparable(48, 64, stride=2)   # C3: 32x32 (stride 8)
        self.stage3 = DepthwiseSeparable(64, 96, stride=2)   # C4: 16x16 (stride 16)

        self.lat4 = nn.Conv2d(96, pyramid_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(64, pyramid_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(48, pyramid_dim, kernel_size=1)

        self.smooth3 = nn.Conv2d(pyramid_dim, pyramid_dim, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(pyramid_dim, pyramid_dim, kernel_size=3, padding=1)

        # Global Application Category Head (from P4)
        self.cat_pool = nn.AdaptiveAvgPool2d(1)
        self.cat_head = nn.Sequential(
            nn.Linear(pyramid_dim, 64),
            nn.GELU(),
            nn.Linear(64, len(UI_CATEGORIES)),
        )

        # Multi-scale proposal heads: 2 affordance logits + 4 bbox offsets + 8 element kind logits
        num_prop_outputs = 2 + 4 + len(SYSTEM1_ELEMENT_KINDS)
        self.head_p2 = nn.Conv2d(pyramid_dim, num_prop_outputs, kernel_size=1) # 64x64 (small)
        self.head_p3 = nn.Conv2d(pyramid_dim, num_prop_outputs, kernel_size=1) # 32x32 (medium)
        self.head_p4 = nn.Conv2d(pyramid_dim, num_prop_outputs, kernel_size=1) # 16x16 (large)

        # -------------------------------------------------------------
        # 2. Query-Conditioned Visual Grounding Head
        # -------------------------------------------------------------
        self.grounder = GoalConditionedGrounder(feature_dim=grounder_dim, input_size=input_size)

        # -------------------------------------------------------------
        # 3. Pixel-Native Patch Word Recognizer Head
        # -------------------------------------------------------------
        self.word_recognizer = PixelWordRecognizer(num_classes=len(UI_WORD_VOCABULARY))

        # -------------------------------------------------------------
        # 4. Temporal Motion Perception Engine
        # -------------------------------------------------------------
        self.temporal_engine = MagnocellularMotionEngine(grid_rows=16, grid_cols=24)

        # Runtime State Tracking
        self._frame_sequence: int = 0
        self._prev_regions: List[VisualRegion] = []
        self._checkpoint_metadata: Dict[str, Any] = {}

    @property
    def total_parameters(self) -> int:
        """Returns total trainable parameter count across all heads."""
        return sum(p.numel() for p in self.parameters())

    def forward_proposals(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes multi-scale feature pyramid proposals and application category."""
        c1 = self.stem(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)

        pooled = self.cat_pool(p4).flatten(1)
        cat_logits = self.cat_head(pooled)

        out_p2 = self.head_p2(p2)
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)

        return {
            "cat_logits": cat_logits,
            "p2": out_p2,
            "p3": out_p3,
            "p4": out_p4,
        }

    def forward_grounding(self, x: torch.Tensor, query_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs query-conditioned grounding. Returns (spatial_heatmap [B, 1, 128, 128], presence_logits [B])."""
        return self.grounder.forward_with_presence(x, query_tokens)

    def forward_word_recognition(self, crops_tensor: torch.Tensor) -> torch.Tensor:
        """Runs patch word classification on (B, 3, 32, 96) crops. Returns logits [B, 132]."""
        return self.word_recognizer(crops_tensor)

    # -----------------------------------------------------------------
    # Unified Inference Pipeline
    # -----------------------------------------------------------------
    @torch.inference_mode()
    def observe_frame(
        self,
        frame: Union[np.ndarray, Image.Image],
        active_query: Optional[str] = None,
        frame_sequence: Optional[int] = None,
        timestamp: Optional[float] = None,
        proposal_threshold: float = 0.05,
        proposal_margin: float = 0.0,
        max_proposals: int = 32,
        presence_threshold: float = 0.70,
        run_word_recognition: bool = True,
    ) -> VisualStateObservation:
        """Executes full unified visual perception on a desktop RGB frame."""
        self.eval()
        t_start = time.perf_counter()
        now_ts = timestamp if timestamp is not None else time.time()
        seq = frame_sequence if frame_sequence is not None else (self._frame_sequence + 1)
        self._frame_sequence = seq

        # 1. Image preprocessing and format validation
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame.convert("RGB"), dtype=np.uint8)
        elif isinstance(frame, np.ndarray):
            frame_np = frame
            if frame_np.ndim == 2:
                frame_np = np.stack([frame_np] * 3, axis=-1)
            elif frame_np.ndim == 3 and frame_np.shape[-1] == 1:
                frame_np = np.repeat(frame_np, 3, axis=-1)
            elif frame_np.ndim == 3 and frame_np.shape[-1] == 4:
                frame_np = frame_np[..., :3]
            elif frame_np.ndim != 3 or frame_np.shape[-1] != 3:
                raise ValueError(
                    "NumPy frames must have shape (height, width), "
                    "(height, width, 1), (height, width, 3), or "
                    "(height, width, 4)"
                )
            if not np.issubdtype(frame_np.dtype, np.integer):
                raise TypeError("NumPy frame pixels must use an integer dtype in [0, 255]")
            if frame_np.size == 0 or int(frame_np.min()) < 0 or int(frame_np.max()) > 255:
                raise ValueError("NumPy frame pixels must be non-empty values in [0, 255]")
            if frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        H, W = frame_np.shape[:2]
        if H == 0 or W == 0:
            raise ValueError("Frame dimensions must be non-zero")
        if not frame_np.flags.writeable:
            frame_np = frame_np.copy()
        device = next(self.parameters()).device

        # 2. Temporal Perception (Optical Flow, Stability, Surprise)
        motion_field, mean_motion, moving_objects, surprise_score = self.temporal_engine.process(frame_np)
        temporal_info = TemporalChangeInfo(
            stability_state=self.temporal_engine.stability_state,
            mean_motion=round(float(mean_motion), 4),
            optical_flow_energy=round(float(self.temporal_engine.optical_flow_energy), 4),
            surprise_score=round(float(surprise_score), 4),
            overlay_suspected=self.temporal_engine.overlay_suspected,
            appeared_regions_count=0,
            disappeared_regions_count=0,
        )

        # 3. Spatial Tensor Conversion (256x256 normalized)
        img_t = torch.from_numpy(frame_np).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        if (H, W) != (self.input_size, self.input_size):
            img_t = F.interpolate(img_t, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        img_t = img_t.to(device)

        # 4. Multi-Scale Region Proposals & Category Classification
        prop_out = self.forward_proposals(img_t)
        cat_probs = F.softmax(prop_out["cat_logits"][0], dim=-1)
        cat_idx = int(torch.argmax(cat_probs).item())
        pred_cat = UI_CATEGORIES[cat_idx] if cat_idx < len(UI_CATEGORIES) else "desktop_canvas"
        cat_conf = float(cat_probs[cat_idx].item())

        # Extract proposals from pyramid levels (P2, P3, P4)
        candidate_proposals: List[Tuple[float, Tuple[float, float, float, float], str]] = []
        levels = [
            (prop_out["p2"][0], 64, 18), # Stride 4 (small controls)
            (prop_out["p3"][0], 32, 12), # Stride 8 (medium controls)
            (prop_out["p4"][0], 16, 8),  # Stride 16 (large controls)
        ]

        for head_out, grid_sz, max_level_peaks in levels:
            aff_logits = head_out[:2]
            aff_probs = F.softmax(aff_logits, dim=0)
            static_heat = aff_probs[0]
            inter_heat = aff_probs[1]

            heat_4d = inter_heat.unsqueeze(0).unsqueeze(0)
            max_pool = F.max_pool2d(heat_4d, kernel_size=3, stride=1, padding=1).squeeze()

            is_peak = (inter_heat == max_pool) & (inter_heat >= proposal_threshold)
            if proposal_margin > 0.0:
                is_peak = is_peak & ((inter_heat - static_heat) >= proposal_margin)

            peak_coords = torch.nonzero(is_peak, as_tuple=False)
            if len(peak_coords) == 0:
                continue

            vals = inter_heat[peak_coords[:, 0], peak_coords[:, 1]]
            if len(vals) > max_level_peaks:
                top_order = torch.argsort(vals, descending=True)[:max_level_peaks]
                peak_coords = peak_coords[top_order]
                vals = vals[top_order]

            box_preds = torch.sigmoid(head_out[2:6])
            kind_probs = F.softmax(head_out[6:], dim=0)

            for i in range(len(peak_coords)):
                py = int(peak_coords[i, 0].item())
                px = int(peak_coords[i, 1].item())
                conf = float(vals[i].item())

                cx = (px + 0.5) / float(grid_sz)
                cy = (py + 0.5) / float(grid_sz)

                b = box_preds[:, py, px]
                span_w = max(0.015, (float(b[3].item()) - float(b[1].item())) * 0.5)
                span_h = max(0.015, (float(b[2].item()) - float(b[0].item())) * 0.5)

                g_xmin = max(0.0, min(1.0, cx - span_w))
                g_xmax = max(0.0, min(1.0, cx + span_w))
                g_ymin = max(0.0, min(1.0, cy - span_h))
                g_ymax = max(0.0, min(1.0, cy + span_h))

                k_idx = int(torch.argmax(kind_probs[:, py, px]).item())
                kind_str = SYSTEM1_ELEMENT_KINDS[k_idx] if k_idx < len(SYSTEM1_ELEMENT_KINDS) else "button"
                candidate_proposals.append((conf, (g_ymin, g_xmin, g_ymax, g_xmax), kind_str))

        # Spatial NMS cross-pyramid suppression
        candidate_proposals.sort(key=lambda x: x[0], reverse=True)
        kept_regions: List[VisualRegion] = []

        def calc_iou(bA, bB):
            yA, xA = max(bA[0], bB[0]), max(bA[1], bB[1])
            yB, xB = min(bA[2], bB[2]), min(bA[3], bB[3])
            inter = max(0.0, yB - yA) * max(0.0, xB - xA)
            areaA = max(0.0, bA[2] - bA[0]) * max(0.0, bA[3] - bA[1])
            areaB = max(0.0, bB[2] - bB[0]) * max(0.0, bB[3] - bB[1])
            union = areaA + areaB - inter
            return inter / union if union > 0 else 0.0

        for conf, box, kind_str in candidate_proposals:
            if len(kept_regions) >= max_proposals:
                break
            suppressed = False
            for kr in kept_regions:
                if calc_iou(box, kr.bbox_norm) > 0.35:
                    suppressed = True
                    break
            if not suppressed:
                b_px = (round(box[0] * H, 1), round(box[1] * W, 1), round(box[2] * H, 1), round(box[3] * W, 1))
                b_norm = (round(box[0], 4), round(box[1], 4), round(box[2], 4), round(box[3], 4))
                kept_regions.append(
                    VisualRegion(
                        region_id=f"reg_{len(kept_regions)}",
                        element_kind=kind_str,
                        bbox_norm=b_norm,
                        bbox_px=b_px,
                        confidence=round(conf, 3),
                        is_interactive=True,
                        is_focused=False,
                    )
                )

        # 5. Patch Word Recognition on Detected Interactive Regions
        if run_word_recognition and len(kept_regions) > 0:
            crops: List[np.ndarray] = []
            for r in kept_regions:
                y1 = max(0, min(H - 1, int(round(r.bbox_px[0]))))
                x1 = max(0, min(W - 1, int(round(r.bbox_px[1]))))
                y2 = max(y1 + 4, min(H, int(round(r.bbox_px[2]))))
                x2 = max(x1 + 4, min(W, int(round(r.bbox_px[3]))))
                crop = frame_np[y1:y2, x1:x2]
                crops.append(crop)

            word_preds = self.word_recognizer.predict_crops_batch(crops, threshold=0.25)
            regions_with_text: List[VisualRegion] = []
            for r, (w_str, w_conf) in zip(kept_regions, word_preds):
                det_text = w_str if w_str != "<none>" else None
                regions_with_text.append(
                    VisualRegion(
                        region_id=r.region_id,
                        element_kind=r.element_kind,
                        bbox_norm=r.bbox_norm,
                        bbox_px=r.bbox_px,
                        confidence=r.confidence,
                        is_interactive=r.is_interactive,
                        is_focused=r.is_focused,
                        detected_text=det_text,
                        text_confidence=round(w_conf, 3) if det_text else 0.0,
                    )
                )
            kept_regions = regions_with_text

        # 6. Query-Conditioned Target Grounding (if query provided)
        target_presence: str = "ABSENT"
        presence_conf: float = 0.0
        selected_target_norm: Optional[Tuple[float, float]] = None
        selected_target_px: Optional[Tuple[float, float]] = None
        spatial_candidates: List[SpatialCandidateInfo] = []
        target_margin: float = 0.0
        is_ambiguous: bool = False
        uncertainty_flags: List[str] = []

        if active_query:
            query_str = active_query.strip()
            encoded_q = encode_queries([query_str]).to(device)
            heatmap, presence_logit = self.forward_grounding(img_t, encoded_q)

            raw_presence = float(torch.sigmoid(presence_logit[0] if presence_logit.ndim > 0 else presence_logit).item())

            # Sub-pixel peak extraction
            coords, peak_conf = extract_subpixel_peak(heatmap)
            cx = float(coords[0, 0].item())
            cy = float(coords[0, 1].item())
            peak_conf_val = float(peak_conf[0].item())

            # Calibrate presence by combining existence head and spatial heatmap peak
            calib_presence = min(raw_presence, peak_conf_val)

            # Adversarial out-of-vocabulary check (fail closed to UNCERTAIN on unknown terms)
            if is_out_of_vocabulary_query(query_str):
                calib_presence = min(calib_presence, 0.45)
                uncertainty_flags.append("out_of_vocabulary_query")

            presence_conf = round(calib_presence, 3)

            # Multi-candidate spatial peak clustering with spatial exclusion radius
            hm_2d = heatmap[0, 0]
            min_radius = 8
            heat_4d = hm_2d.unsqueeze(0).unsqueeze(0)
            pooled = F.max_pool2d(heat_4d, kernel_size=2 * min_radius + 1, stride=1, padding=min_radius).squeeze()
            is_peak = (hm_2d == pooled) & (hm_2d > 0.0)
            peak_indices = torch.nonzero(is_peak, as_tuple=False)

            hm_w = hm_2d.shape[1]
            hm_h = hm_2d.shape[0]

            if len(peak_indices) > 0:
                vals = hm_2d[peak_indices[:, 0], peak_indices[:, 1]]
                k = min(4, len(vals))
                top_vals, top_order = torch.topk(vals, k=k)
                cand_probs = F.softmax(top_vals, dim=0)

                for c_id in range(k):
                    p_idx = top_order[c_id]
                    c_row = int(peak_indices[p_idx, 0].item())
                    c_col = int(peak_indices[p_idx, 1].item())
                    cand_cx = round((c_col + 0.5) / float(hm_w), 4)
                    cand_cy = round((c_row + 0.5) / float(hm_h), 4)
                    val = float(top_vals[c_id].item())
                    prob = round(float(cand_probs[c_id].item()), 3)
                    spatial_candidates.append(
                        SpatialCandidateInfo(
                            candidate_id=c_id,
                            coord_norm=(cand_cx, cand_cy),
                            coord_px=(round(cand_cx * W, 1), round(cand_cy * H, 1)),
                            coarse_logit=val,
                            rerank_score=val,
                            rerank_prob=prob,
                        )
                    )

                if len(spatial_candidates) >= 2:
                    target_margin = round(spatial_candidates[0].rerank_prob - spatial_candidates[1].rerank_prob, 3)
                    if target_margin < 0.20:
                        is_ambiguous = True
                        uncertainty_flags.append("multimodal_target_ambiguity")
                else:
                    target_margin = 1.0
            else:
                target_margin = 0.0

            if calib_presence >= presence_threshold:
                target_presence = "PRESENT"
                selected_target_norm = (round(cx, 4), round(cy, 4))
                selected_target_px = (round(cx * W, 1), round(cy * H, 1))
            elif calib_presence < 0.35:
                target_presence = "ABSENT"
            else:
                target_presence = "UNCERTAIN"
                uncertainty_flags.append("low_presence_confidence")

        # Temporal delta tracking against previous observation
        if self._prev_regions:
            temporal_info = dataclasses.replace(
                temporal_info,
                appeared_regions_count=max(0, len(kept_regions) - len(self._prev_regions)),
                disappeared_regions_count=max(0, len(self._prev_regions) - len(kept_regions)),
            )
        self._prev_regions = kept_regions

        if temporal_info.stability_state in ("busy", "surprise"):
            uncertainty_flags.append("frame_motion_unstable")

        # Fail-closed action authorization policy:
        # action_authorized is FALSE by default.
        # Eligible ONLY when target is PRESENT (>= 0.70), unambiguous, and frame is stable.
        action_authorized = (
            active_query is not None
            and target_presence == "PRESENT"
            and presence_conf >= presence_threshold
            and not is_ambiguous
            and temporal_info.stability_state in ("stable", "changed")
            and len(uncertainty_flags) == 0
        )

        return VisualStateObservation(
            frame_sequence=seq,
            timestamp=round(now_ts, 4),
            screen_dimensions=(W, H),
            coordinate_convention="normalized_0_1_origin_top_left",
            primary_category=pred_cat,
            category_confidence=round(cat_conf, 3),
            detected_regions=kept_regions,
            target_presence=target_presence,
            presence_confidence=presence_conf,
            active_query=active_query,
            target_candidates=spatial_candidates,
            selected_target_norm=selected_target_norm,
            selected_target_px=selected_target_px,
            target_margin=target_margin,
            is_ambiguous=is_ambiguous,
            action_authorized=action_authorized,
            temporal_change=temporal_info,
            uncertainty_flags=uncertainty_flags,
        )

    # -----------------------------------------------------------------
    # Checkpoint Serialization & Integrity
    # -----------------------------------------------------------------
    def save_checkpoint(
        self,
        output_path: str,
        training_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """Saves unified AVCSSystem1V1 weights and provenance metadata."""
        if checkpoint_name is not None and Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("checkpoint_name must be a filename, not a path")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        meta = training_metadata or {}
        payload = {
            "schema_version": SYSTEM1_V1_SCHEMA_VERSION,
            "architecture": "AVCSSystem1V1",
            "checkpoint_name": checkpoint_name or SYSTEM1_V1_CHECKPOINT_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameter_count": self.total_parameters,
            "input_size": self.input_size,
            "pyramid_dim": self.pyramid_dim,
            "grounder_dim": self.grounder_dim,
            "element_kinds": SYSTEM1_ELEMENT_KINDS,
            "ui_categories": UI_CATEGORIES,
            "word_vocabulary": UI_WORD_VOCABULARY,
            "training_metadata": meta,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, output_path)

        with open(output_path, "rb") as f:
            sha256_hash = hashlib.sha256(f.read()).hexdigest()

        sidecar_path = os.path.splitext(output_path)[0] + ".json"
        sidecar_meta = {
            "schema_version": SYSTEM1_V1_SCHEMA_VERSION,
            "checkpoint_file": output_path,
            "sha256": sha256_hash,
            "parameter_count": self.total_parameters,
            "architecture": "AVCSSystem1V1",
            "created_at": payload["created_at"],
            "training_metadata": meta,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_meta, f, indent=2)

        return sha256_hash

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> AVCSSystem1V1:
        """Loads AVCSSystem1V1 checkpoint with strict schema validation."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        sidecar_path = Path(checkpoint_path).with_suffix(".json")
        sidecar_parameters = None
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid checkpoint sidecar {sidecar_path}: {exc}") from exc
            if not isinstance(sidecar, dict):
                raise ValueError(f"Invalid checkpoint sidecar {sidecar_path}: expected a JSON object")
            sidecar_architecture = sidecar.get("architecture")
            if sidecar_architecture not in (None, "AVCSSystem1V1"):
                raise ValueError(
                    f"Checkpoint sidecar architecture {sidecar_architecture!r} "
                    "is not AVCSSystem1V1"
                )
            sidecar_parameters = sidecar.get("parameter_count")
            if sidecar_parameters is not None and (
                not isinstance(sidecar_parameters, int)
                or isinstance(sidecar_parameters, bool)
                or sidecar_parameters <= 0
            ):
                raise ValueError(
                    f"Invalid checkpoint sidecar parameter_count {sidecar_parameters!r}"
                )
            expected_hash = sidecar.get("sha256")
            if expected_hash is not None:
                if (
                    not isinstance(expected_hash, str)
                    or len(expected_hash) != 64
                    or any(ch not in "0123456789abcdefABCDEF" for ch in expected_hash)
                ):
                    raise ValueError(f"Invalid checkpoint SHA-256 in sidecar {sidecar_path}")
                digest = hashlib.sha256()
                with open(checkpoint_path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest().lower() != expected_hash.lower():
                    raise ValueError(
                        f"Checkpoint SHA-256 mismatch for {checkpoint_path}: "
                        f"expected {expected_hash}, got {digest.hexdigest()}"
                    )

        try:
            with torch.serialization.safe_globals([TorchVersion]):
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as exc:
            raise ValueError(f"Invalid or incompatible checkpoint {checkpoint_path}: {exc}") from exc
        if not isinstance(ckpt, dict):
            raise ValueError("Checkpoint must contain a state dictionary or metadata object")
        schema = ckpt.get("schema_version")
        if schema is not None and schema != SYSTEM1_V1_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible schema version: {schema} != {SYSTEM1_V1_SCHEMA_VERSION}"
            )
        architecture = ckpt.get("architecture")
        if architecture is not None and architecture != "AVCSSystem1V1":
            raise ValueError(f"Incompatible checkpoint architecture: {architecture!r}")

        state_dict = ckpt.get("state_dict", ckpt)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint state_dict must be a mapping")

        model = cls(
            pyramid_dim=ckpt.get("pyramid_dim", 48),
            grounder_dim=ckpt.get("grounder_dim", 192),
            input_size=ckpt.get("input_size", DEFAULT_INPUT_SIZE),
        )

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint weights do not match AVCSSystem1V1: {exc}") from exc
        for source_name, expected_parameters in (
            ("checkpoint", ckpt.get("parameter_count")),
            ("sidecar", sidecar_parameters),
        ):
            if expected_parameters is not None and expected_parameters != model.total_parameters:
                raise ValueError(
                    f"{source_name} parameter_count {expected_parameters!r} does not "
                    f"match loaded architecture ({model.total_parameters})"
                )
        model.to(device)
        model.eval()
        model._checkpoint_metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return model


def load_system1(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> AVCSSystem1V1:
    """Convenience factory to load the canonical AVCSSystem1V1 standalone model."""
    if checkpoint_path is None:
        repo_root = str(Path(__file__).resolve().parents[2])
        candidate_paths = [
            os.path.join(repo_root, "runs", "weights", "candidates", "avcs_system1_v1.pt"),
            os.path.join(repo_root, "runs", "weights", "avcs_system1_v1.pt"),
            "runs/weights/candidates/avcs_system1_v1.pt",
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                checkpoint_path = p
                break
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Could not find default AVCSSystem1V1 checkpoint. Checked: "
                + ", ".join(candidate_paths)
            )
    return AVCSSystem1V1.load_from_checkpoint(checkpoint_path, device=device)
