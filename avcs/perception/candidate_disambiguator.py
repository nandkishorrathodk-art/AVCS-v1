"""Global Candidate Disambiguation and Neural Reranking for AVCS System 1.

Extends coarse grounding from a single argmax peak to a bounded set of spatial
candidates (Top-K peaks) evaluated with surrounding visual context and query semantics:
1. Spatial Local Maxima Extraction: NMS-style peak detection with minimum separation.
2. Contextual Feature Pooling: Multi-scale local RoI features + global screen representation.
3. Neural Candidate Reranker: Lightweight cross-modal scoring network that ranks candidates
   based on query compatibility, spatial context, and appearance.
4. Calibrated Margin Gate: Evaluates candidate competition margin (s_1 - s_2) and spatial
   separation; marks conflicting targets as UNCERTAIN rather than guessing.
5. Runtime-Only Evaluation: Never uses ground-truth labels or test annotations during inference.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.goal_grounder import (
    ByteQueryEncoder,
    GoalConditionedGrounder,
    MAX_QUERY_BYTES,
    encode_queries,
    extract_subpixel_peak,
)


@dataclasses.dataclass(frozen=True)
class SpatialCandidate:
    """Represents a spatial candidate peak on the screen."""
    candidate_id: int
    coord_norm: Tuple[float, float]       # (cx, cy) in [0, 1] range
    coord_px: Tuple[float, float]         # (x, y) in physical screen pixels
    coarse_logit: float                   # Raw heatmap logit at peak
    coarse_confidence: float              # Sigmoid peak probability
    rerank_score: float                   # Score from neural reranker
    rerank_prob: float                    # Softmax probability across Top-K
    local_feature: Optional[torch.Tensor] = None


@dataclasses.dataclass(frozen=True)
class DisambiguationResult:
    """Structured output of global candidate disambiguation."""
    query: str
    selected_coord_norm: Tuple[float, float]
    selected_coord_px: Tuple[float, float]
    confidence: float
    uncertainty: float
    margin: float                         # Score difference between Top-1 and Top-2
    is_ambiguous: bool                    # True if top competing candidates disagree
    decision: str                         # 'CONFIDENT', 'UNCERTAIN', 'ABSENT'
    candidates: List[SpatialCandidate]
    coarse_top1_norm: Tuple[float, float]
    reranked_top1_changed: bool           # True if reranking picked a different candidate than coarse argmax
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "margin": round(self.margin, 4),
            "is_ambiguous": self.is_ambiguous,
            "selected_coord_norm": (round(self.selected_coord_norm[0], 4), round(self.selected_coord_norm[1], 4)),
            "reranked_top1_changed": self.reranked_top1_changed,
            "num_candidates": len(self.candidates),
            "latency_ms": round(self.latency_ms, 2),
        }


def extract_spatial_peaks_nms(
    heatmap_logits: torch.Tensor,
    k: int = 4,
    min_separation_cells: int = 4,
    min_logit_threshold: float = -6.0,
) -> List[List[Tuple[int, int, float]]]:
    """Extracts Top-K local spatial peaks per batch element with NMS separation.

    Args:
        heatmap_logits: [B, 1, H, W] tensor of click logits.
        k: Maximum number of candidates to extract per image.
        min_separation_cells: Minimum spatial distance (in feature grid cells) between peaks.
        min_logit_threshold: Minimum logit to consider a valid candidate.

    Returns:
        List of lists of (row, col, logit) per batch item.
    """
    B, _, H, W = heatmap_logits.shape
    kernel_size = min_separation_cells * 2 + 1
    padding = min_separation_cells
    
    # Max pooling over local neighborhood
    pooled = F.max_pool2d(heatmap_logits, kernel_size=kernel_size, stride=1, padding=padding)
    is_local_max = (heatmap_logits == pooled) & (heatmap_logits >= min_logit_threshold)

    batch_peaks = []
    for b in range(B):
        peaks = []
        # Find coordinates of local maxima
        locs = torch.nonzero(is_local_max[b, 0])  # [N, 2] -> (r, c)
        if len(locs) == 0:
            # Fall back to single global argmax if no local peak passed threshold
            flat_idx = heatmap_logits[b, 0].flatten().argmax().item()
            r = flat_idx // W
            c = flat_idx % W
            peaks.append((r, c, float(heatmap_logits[b, 0, r, c].item())))
        else:
            vals = heatmap_logits[b, 0, locs[:, 0], locs[:, 1]]
            sorted_indices = torch.argsort(vals, descending=True)
            for idx in sorted_indices[:k]:
                r = int(locs[idx, 0].item())
                c = int(locs[idx, 1].item())
                l = float(vals[idx].item())
                peaks.append((r, c, l))
        batch_peaks.append(peaks)
    return batch_peaks


class CandidateReranker(nn.Module):
    """Neural reranker evaluating cross-modal compatibility of spatial candidates.

    Considers:
    1. Local visual features at candidate RoI (3x3 grid = 64 * 9 = 576-d).
    2. Global visual context from conscious screen representation (64-d).
    3. Natural language query representation from ByteQueryEncoder (192-d).
    4. Normalized spatial coordinates and boundary distances (4-d).
    5. Coarse heatmap activation logit (1-d).
    """

    def __init__(
        self,
        visual_channels: int = 64,
        query_dim: int = 192,
        hidden_dim: int = 128,
        roi_grid_size: int = 3,
    ) -> None:
        super().__init__()
        self.visual_channels = visual_channels
        self.query_dim = query_dim
        self.roi_grid_size = roi_grid_size

        local_dim = visual_channels * (roi_grid_size * roi_grid_size)
        global_dim = visual_channels
        coord_dim = 4  # cx, cy, dist_from_center, dist_from_edge
        coarse_dim = 1

        total_input_dim = local_dim + global_dim + query_dim + coord_dim + coarse_dim

        self.scoring_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def pool_local_context(
        self,
        decoded_features: torch.Tensor,
        row_col_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Pools 3x3 local patch around candidate feature coordinates."""
        B, C, H, W = decoded_features.shape
        half = self.roi_grid_size // 2
        
        # Pad features so borders don't crash
        padded = F.pad(decoded_features, (half, half, half, half), mode="replicate")
        
        local_patches = []
        for r, c in row_col_pairs:
            # Shift by padding
            rp = r + half
            cp = c + half
            patch = padded[:, :, rp - half : rp + half + 1, cp - half : cp + half + 1]  # [B, C, 3, 3]
            local_patches.append(patch.flatten(1))  # [B, C * 9]

        if not local_patches:
            return torch.zeros((B, C * self.roi_grid_size * self.roi_grid_size), device=decoded_features.device)
        return torch.stack(local_patches, dim=1)  # [B, K, C * 9]

    def forward(
        self,
        local_features: torch.Tensor,     # [B, K, 576]
        global_features: torch.Tensor,    # [B, 64]
        query_emb: torch.Tensor,          # [B, 192]
        coords_norm: torch.Tensor,        # [B, K, 2]
        coarse_logits: torch.Tensor,      # [B, K, 1]
    ) -> torch.Tensor:
        """Computes reranking compatibility scores for K candidates.

        Returns:
            rerank_scores: [B, K] compatibility logits.
        """
        B, K, _ = local_features.shape
        
        # Broadcast global and query features to each candidate K
        global_expanded = global_features.unsqueeze(1).expand(B, K, -1)  # [B, K, 64]
        query_expanded = query_emb.unsqueeze(1).expand(B, K, -1)          # [B, K, 192]

        # Spatial geometry features
        cx = coords_norm[:, :, 0:1]
        cy = coords_norm[:, :, 1:2]
        dist_center = torch.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        dist_edge = torch.minimum(torch.minimum(cx, 1.0 - cx), torch.minimum(cy, 1.0 - cy))
        spatial_feats = torch.cat([cx, cy, dist_center, dist_edge], dim=-1)  # [B, K, 4]

        combined = torch.cat(
            [local_features, global_expanded, query_expanded, spatial_feats, coarse_logits],
            dim=-1,
        )  # [B, K, total_input_dim]

        scores = self.scoring_mlp(combined).squeeze(-1)  # [B, K]
        return scores


class GlobalCandidateDisambiguator(nn.Module):
    """Integrated Global Candidate Disambiguator & Reranking Engine."""

    def __init__(
        self,
        coarse_model: GoalConditionedGrounder,
        reranker: Optional[CandidateReranker] = None,
        k_candidates: int = 4,
        margin_threshold: float = 0.25,
        conflict_distance_px: float = 12.0,
    ) -> None:
        super().__init__()
        self.coarse_model = coarse_model
        self.reranker = reranker if reranker is not None else CandidateReranker()
        self.k_candidates = k_candidates
        self.margin_threshold = margin_threshold
        self.conflict_distance_px = conflict_distance_px
        self._checkpoint_metadata: Dict[str, Any] = {}

    @torch.inference_mode()
    def disambiguate(
        self,
        image_tensor: torch.Tensor,
        query: str,
        screen_size: Tuple[int, int] = (256, 256),
    ) -> DisambiguationResult:
        """Performs global candidate extraction, neural reranking, and margin validation."""
        t0 = time.perf_counter()
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        device = next(self.coarse_model.parameters()).device
        img = image_tensor.to(device)

        W_scr, H_scr = screen_size
        clean_q = query.strip() if isinstance(query, str) else ""
        if not clean_q or not any(c.isalnum() for c in clean_q) or len(clean_q.encode("utf-8")) > MAX_QUERY_BYTES:
            dt = (time.perf_counter() - t0) * 1000.0
            return DisambiguationResult(
                query=query,
                selected_coord_norm=(0.5, 0.5),
                selected_coord_px=(W_scr * 0.5, H_scr * 0.5),
                confidence=0.0,
                uncertainty=1.0,
                margin=0.0,
                is_ambiguous=True,
                decision="ABSENT",
                candidates=[],
                coarse_top1_norm=(0.5, 0.5),
                reranked_top1_changed=False,
                latency_ms=dt,
            )

        q_tokens = encode_queries([clean_q]).to(device)

        # 1. Forward coarse pyramid decoder
        decoded = self.coarse_model._decode(img, q_tokens)  # [1, 64, 128, 128]
        heatmap_logits = self.coarse_model.output(decoded)  # [1, 1, 128, 128]
        presence_pooled = F.adaptive_avg_pool2d(decoded, output_size=1).flatten(1)
        presence_logit = self.coarse_model.target_presence(presence_pooled).squeeze(1)
        presence_prob = float(torch.sigmoid(presence_logit[0]).item())

        # Presence check against coarse calibrated threshold
        coarse_threshold = float(self.coarse_model._checkpoint_metadata.get("target_presence_threshold", 0.55))
        if presence_prob < coarse_threshold:
            dt = (time.perf_counter() - t0) * 1000.0
            # Target absent or sub-threshold
            coords_top1, peak_conf_top1 = extract_subpixel_peak(heatmap_logits)
            cx = float(coords_top1[0, 0].item())
            cy = float(coords_top1[0, 1].item())
            return DisambiguationResult(
                query=clean_q,
                selected_coord_norm=(cx, cy),
                selected_coord_px=(cx * (W_scr - 1), cy * (H_scr - 1)),
                confidence=presence_prob,
                uncertainty=1.0 - presence_prob,
                margin=0.0,
                is_ambiguous=False,
                decision="ABSENT",
                candidates=[],
                coarse_top1_norm=(cx, cy),
                reranked_top1_changed=False,
                latency_ms=dt,
            )

        # 2. Extract Top-K spatial candidate peaks
        B, _, H_feat, W_feat = heatmap_logits.shape
        batch_peaks = extract_spatial_peaks_nms(
            heatmap_logits,
            k=self.k_candidates,
            min_separation_cells=4,  # ~8 physical pixels separation
            min_logit_threshold=-5.0,
        )
        peaks = batch_peaks[0]

        # Single argmax peak as coarse baseline reference
        coords_top1, peak_conf_top1 = extract_subpixel_peak(heatmap_logits)
        coarse_top1_cx = float(coords_top1[0, 0].item())
        coarse_top1_cy = float(coords_top1[0, 1].item())

        # 3. Contextual feature extraction for each candidate
        row_col_pairs = [(r, c) for r, c, _ in peaks]
        local_feats = self.reranker.pool_local_context(decoded, row_col_pairs)  # [1, K, 576]
        global_feat = presence_pooled  # [1, 64]
        query_emb = self.coarse_model.query_encoder(q_tokens)  # [1, 192]

        coords_list = []
        logits_list = []
        for r, c, l in peaks:
            norm_x = (float(c) + 0.5) / float(W_feat)
            norm_y = (float(r) + 0.5) / float(H_feat)
            coords_list.append([norm_x, norm_y])
            logits_list.append([l])

        coords_t = torch.tensor([coords_list], device=device, dtype=torch.float32)  # [1, K, 2]
        logits_t = torch.tensor([logits_list], device=device, dtype=torch.float32)  # [1, K, 1]

        # 4. Neural Reranker scoring pass
        rerank_scores = self.reranker(
            local_features=local_feats,
            global_features=global_feat,
            query_emb=query_emb,
            coords_norm=coords_t,
            coarse_logits=logits_t,
        )  # [1, K]

        rerank_probs = F.softmax(rerank_scores[0], dim=-1)

        # Construct spatial candidates
        candidates: List[SpatialCandidate] = []
        for idx, (r, c, logit) in enumerate(peaks):
            cx = coords_list[idx][0]
            cy = coords_list[idx][1]
            px_x = cx * (W_scr - 1)
            px_y = cy * (H_scr - 1)
            c_conf = float(torch.sigmoid(torch.tensor(logit)).item())
            r_score = float(rerank_scores[0, idx].item())
            r_prob = float(rerank_probs[idx].item())
            candidates.append(
                SpatialCandidate(
                    candidate_id=idx,
                    coord_norm=(cx, cy),
                    coord_px=(px_x, px_y),
                    coarse_logit=logit,
                    coarse_confidence=c_conf,
                    rerank_score=r_score,
                    rerank_prob=r_prob,
                )
            )

        # Rank candidates by rerank score
        candidates.sort(key=lambda cand: cand.rerank_score, reverse=True)
        top1_cand = candidates[0]
        coarse_argmax_cand = candidates[0] if not candidates else None
        for cand in candidates:
            if math.hypot(cand.coord_norm[0] - coarse_top1_cx, cand.coord_norm[1] - coarse_top1_cy) < 0.02:
                coarse_argmax_cand = cand
                break

        # 5. Margin and Ambiguity Analysis
        if len(candidates) >= 2:
            top2_cand = candidates[1]
            margin = top1_cand.rerank_score - top2_cand.rerank_score
            dist_12_px = math.hypot(
                top1_cand.coord_px[0] - top2_cand.coord_px[0],
                top1_cand.coord_px[1] - top2_cand.coord_px[1],
            )
            # Competition ambiguity flag: close scores but geographically separated elements
            is_ambiguous = (margin < self.margin_threshold) and (dist_12_px > self.conflict_distance_px)
        else:
            margin = 1.0
            is_ambiguous = False

        reranked_top1_changed = bool(
            coarse_argmax_cand is not None and top1_cand.candidate_id != coarse_argmax_cand.candidate_id
        )

        final_conf = presence_prob * top1_cand.rerank_prob
        uncertainty = 1.0 - final_conf

        if is_ambiguous or final_conf < 0.40:
            decision = "UNCERTAIN"
        else:
            decision = "CONFIDENT"

        dt = (time.perf_counter() - t0) * 1000.0

        return DisambiguationResult(
            query=clean_q,
            selected_coord_norm=top1_cand.coord_norm,
            selected_coord_px=top1_cand.coord_px,
            confidence=final_conf,
            uncertainty=uncertainty,
            margin=margin,
            is_ambiguous=is_ambiguous,
            decision=decision,
            candidates=candidates,
            coarse_top1_norm=(coarse_top1_cx, coarse_top1_cy),
            reranked_top1_changed=reranked_top1_changed,
            latency_ms=dt,
        )


def save_disambiguator_checkpoint(
    disambiguator: GlobalCandidateDisambiguator,
    save_path: Union[str, Path],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves candidate reranker weights and metadata."""
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "avcs_candidate_disambiguator_v1",
        "schema_version": 1,
        "reranker_state_dict": disambiguator.reranker.state_dict(),
        "k_candidates": disambiguator.k_candidates,
        "margin_threshold": disambiguator.margin_threshold,
        "conflict_distance_px": disambiguator.conflict_distance_px,
        **(extra_metadata or {}),
    }
    torch.save(payload, str(save_p))


def load_disambiguator_checkpoint(
    coarse_model: GoalConditionedGrounder,
    reranker_checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> GlobalCandidateDisambiguator:
    """Loads a trained candidate disambiguator checkpoint."""
    ckpt = torch.load(reranker_checkpoint_path, map_location=device, weights_only=False)
    reranker = CandidateReranker().to(device)
    if "reranker_state_dict" in ckpt:
        reranker.load_state_dict(ckpt["reranker_state_dict"])
    elif "state_dict" in ckpt:
        reranker.load_state_dict(ckpt["state_dict"])
    reranker.eval()

    disambiguator = GlobalCandidateDisambiguator(
        coarse_model=coarse_model,
        reranker=reranker,
        k_candidates=ckpt.get("k_candidates", 4),
        margin_threshold=ckpt.get("margin_threshold", 0.25),
        conflict_distance_px=ckpt.get("conflict_distance_px", 12.0),
    )
    disambiguator._checkpoint_metadata = dict(ckpt)
    return disambiguator

