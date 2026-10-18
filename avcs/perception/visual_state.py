"""AVCS System 1 Visual State Observation Contract.

Defines the typed, non-hallucinated structured contract representing what AVCS System 1
empirically infers from raw desktop pixels.

Enforces:
1. Complete provenance (sequence numbers, capture timestamps, coordinate conventions).
2. Explicit visual regions with bounding boxes and predicted element kinds.
3. Grounded target candidates with calibrated presence and uncertainty margins.
4. Temporal state delta (motion energy, stability state, surprise scores).
5. Explicit unknown/uncertainty flags (no fabricated labels or content).
6. Local visual privacy: zero raw images or visual embeddings exported across System 2 boundary.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class VisualRegion:
    """A detected distinct visual region on the desktop screen."""
    region_id: str
    element_kind: str                        # "button", "input_field", "icon", "tab", "window", "container"
    bbox_norm: Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax] in [0.0, 1.0]
    bbox_px: Tuple[float, float, float, float]    # [ymin, xmin, ymax, xmax] in physical screen px
    confidence: float                        # Neural detection confidence in [0.0, 1.0]
    is_interactive: bool                     # True if clickable or typeable
    is_focused: bool = False                 # True if caret blink / focus glow detected
    detected_text: Optional[str] = None      # Pixel-native recognized text label if present
    text_confidence: float = 0.0             # Neural word classification confidence in [0.0, 1.0]


@dataclasses.dataclass(frozen=True)
class SpatialCandidateInfo:
    """Information about a query-conditioned spatial peak candidate."""
    candidate_id: int
    coord_norm: Tuple[float, float]          # (cx, cy) in [0.0, 1.0]
    coord_px: Tuple[float, float]            # (x, y) in screen pixels
    coarse_logit: float
    rerank_score: float
    rerank_prob: float


@dataclasses.dataclass(frozen=True)
class TemporalChangeInfo:
    """Frame-to-frame temporal visual perception metrics."""
    stability_state: str                     # "stable", "busy", "changed", "surprise", "overlay"
    mean_motion: float                       # Photometric motion magnitude
    optical_flow_energy: float               # Mean velocity energy ||v||^2
    surprise_score: float                    # Normalized prediction error [0.0, 1.0]
    overlay_suspected: bool                  # True if localized modal / notification emerged
    appeared_regions_count: int = 0
    disappeared_regions_count: int = 0


@dataclasses.dataclass(frozen=True)
class VisualStateObservation:
    """Structured, evidence-backed visual perception state for a single desktop frame."""
    frame_sequence: int
    timestamp: float
    screen_dimensions: Tuple[int, int]       # (width, height)
    coordinate_convention: str               # "normalized_0_1_origin_top_left"
    primary_category: str                    # Screen application class
    category_confidence: float
    detected_regions: List[VisualRegion]
    target_presence: str                     # "PRESENT", "ABSENT", "UNCERTAIN"
    presence_confidence: float
    active_query: Optional[str] = None
    target_candidates: List[SpatialCandidateInfo] = dataclasses.field(default_factory=list)
    selected_target_norm: Optional[Tuple[float, float]] = None
    selected_target_px: Optional[Tuple[float, float]] = None
    target_margin: float = 0.0
    is_ambiguous: bool = False
    action_authorized: bool = False          # Fail-closed invariant
    temporal_change: Optional[TemporalChangeInfo] = None
    uncertainty_flags: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts observation to a System-2 consumable JSON dictionary with zero pixel data."""
        return {
            "frame_sequence": self.frame_sequence,
            "timestamp": round(self.timestamp, 4),
            "screen_dimensions": list(self.screen_dimensions),
            "coordinate_convention": self.coordinate_convention,
            "primary_category": self.primary_category,
            "category_confidence": round(self.category_confidence, 3),
            "detected_regions": [
                {
                    "region_id": r.region_id,
                    "element_kind": r.element_kind,
                    "bbox_norm": [round(c, 4) for c in r.bbox_norm],
                    "bbox_px": [round(c, 1) for c in r.bbox_px],
                    "confidence": round(r.confidence, 3),
                    "is_interactive": r.is_interactive,
                    "is_focused": r.is_focused,
                    "detected_text": r.detected_text,
                    "text_confidence": round(r.text_confidence, 3) if r.detected_text else 0.0,
                }
                for r in self.detected_regions
            ],
            "target_presence": self.target_presence,
            "presence_confidence": round(self.presence_confidence, 3),
            "active_query": self.active_query,
            "target_candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "coord_norm": [round(c.coord_norm[0], 4), round(c.coord_norm[1], 4)],
                    "coord_px": [round(c.coord_px[0], 1), round(c.coord_px[1], 1)],
                    "coarse_logit": round(c.coarse_logit, 2),
                    "rerank_score": round(c.rerank_score, 3),
                }
                for c in self.target_candidates
            ],
            "selected_target_norm": [round(self.selected_target_norm[0], 4), round(self.selected_target_norm[1], 4)] if self.selected_target_norm else None,
            "selected_target_px": [round(self.selected_target_px[0], 1), round(self.selected_target_px[1], 1)] if self.selected_target_px else None,
            "target_margin": round(self.target_margin, 3),
            "is_ambiguous": self.is_ambiguous,
            "action_authorized": self.action_authorized,
            "temporal_change": {
                "stability_state": self.temporal_change.stability_state,
                "mean_motion": round(self.temporal_change.mean_motion, 4),
                "optical_flow_energy": round(self.temporal_change.optical_flow_energy, 4),
                "surprise_score": round(self.temporal_change.surprise_score, 4),
                "overlay_suspected": self.temporal_change.overlay_suspected,
            } if self.temporal_change else None,
            "uncertainty_flags": list(self.uncertainty_flags),
        }
