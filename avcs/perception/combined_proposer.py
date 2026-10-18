"""AVCS Path 3: Combined Global-Context and Local-Detail Proposal Pipeline.

Integrates:
1. Global FPN Proposer: Wide-field application categorization and medium/large UI container layout.
2. Native Tiled Proposer: 1:1 original resolution patch scanning for small controls (<24 px) lost during downsampling.
3. Multi-Scale Spatial Fusion & IoU NMS: Joint deduplication producing high-recall, calibrated visual state.

Pure-pixel neural architecture: zero OCR, zero accessibility APIs, zero DOM scraping.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from avcs.perception.fpn_proposer import FPNRegionProposer
from avcs.perception.tiled_proposer import TiledNativeProposer, _calc_iou
from avcs.perception.visual_pilot import VisualUnderstandingPilot
from avcs.perception.visual_state import VisualRegion


class CombinedVisualProposer:
    """Fuses global multi-scale context with local native-resolution detail scanning."""

    def __init__(
        self,
        fpn_proposer: Optional[FPNRegionProposer] = None,
        tiled_proposer: Optional[TiledNativeProposer] = None,
        fusion_iou_threshold: float = 0.35,
        max_total_proposals: int = 48,
        word_recognizer: Optional[Any] = None,
    ) -> None:
        self.fpn = fpn_proposer or FPNRegionProposer()
        self.tiled = tiled_proposer or TiledNativeProposer()
        self.fusion_iou_thresh = fusion_iou_threshold
        self.max_total_proposals = max_total_proposals
        self.word_recognizer = word_recognizer
        if word_recognizer is not None:
            self.fpn.attach_word_recognizer(word_recognizer)

    def attach_word_recognizer(self, recognizer: Any) -> None:
        """Attaches a PixelWordRecognizer to predict text on fused spatial proposals."""
        self.word_recognizer = recognizer
        self.fpn.attach_word_recognizer(recognizer)

    def propose_all(
        self,
        image_np: np.ndarray,
        fpn_threshold: float = 0.40,
        fpn_margin: float = 0.05,
    ) -> Tuple[str, float, List[VisualRegion], Dict[str, float]]:
        """Executes combined global-context and native-detail visual proposal pipeline."""
        t0 = time.perf_counter()

        # 1. Global Context Pass (FPN)
        t_global0 = time.perf_counter()
        pred_cat, cat_conf, global_proposals = self.fpn.propose_regions(
            image_np, threshold=fpn_threshold, margin=fpn_margin, max_proposals=28
        )
        t_global_ms = (time.perf_counter() - t_global0) * 1000.0

        # 2. Local Native-Detail Pass (Tiled Native Resolution)
        t_local0 = time.perf_counter()
        local_proposals = self.tiled.extract_tile_proposals(
            image_np, max_proposals=24, nms_iou_threshold=0.30
        )
        t_local_ms = (time.perf_counter() - t_local0) * 1000.0

        # 3. Spatial Fusion & Deduplication
        t_fuse0 = time.perf_counter()
        all_proposals = list(global_proposals) + list(local_proposals)
        # Sort by confidence descending
        all_proposals.sort(key=lambda r: r.confidence, reverse=True)

        fused_proposals: List[VisualRegion] = []
        for prop in all_proposals:
            if len(fused_proposals) >= self.max_total_proposals:
                break
            suppressed = False
            for fp in fused_proposals:
                if _calc_iou(prop.bbox_norm, fp.bbox_norm) > self.fusion_iou_thresh:
                    suppressed = True
                    break
            if not suppressed:
                fused_proposals.append(
                    VisualRegion(
                        region_id=f"fused_reg_{len(fused_proposals)}",
                        element_kind=prop.element_kind,
                        bbox_norm=prop.bbox_norm,
                        bbox_px=prop.bbox_px,
                        confidence=prop.confidence,
                        is_interactive=True,
                        is_focused=prop.is_focused,
                        detected_text=prop.detected_text,
                        text_confidence=prop.text_confidence,
                    )
                )

        t_fuse_ms = (time.perf_counter() - t_fuse0) * 1000.0
        t_total_ms = (time.perf_counter() - t0) * 1000.0

        timings = {
            "global_ms": round(t_global_ms, 2),
            "local_ms": round(t_local_ms, 2),
            "fusion_ms": round(t_fuse_ms, 2),
            "total_ms": round(t_total_ms, 2),
        }

        return pred_cat, cat_conf, fused_proposals, timings

