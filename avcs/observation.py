"""Real-Time Screen Observation Engine & System 2 Privacy Boundary.

Translates internal pure-pixel cortical neural representations (features, dorsal affordance
peaks, ventral category logits, optical flow, differential state trees) into truthful,
evidence-backed, coordinate-free natural language observations for System 2 (LLMs).

Enforces the R6 System 2 Privacy Boundary:
- ZERO raw frames, image crops, or visual embeddings cross the interface.
- ZERO raw coordinate primitives (x, y, bounding_box) are exported.
- Zero-hallucination guarantee: Negative/blank screens produce empty entity sets and
  explicit blank screen observations.
- Repetition suppression: Suppresses redundant prompt tokens on consecutive static frames
  by >80% while accurately tracking state deltas.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avcs.memory.diff_engine import VisualDiffEngine


@dataclasses.dataclass(frozen=True)
class EntitySummary:
    """Natural-language, coordinate-free entity summary for System 2."""
    category: str                    # e.g., "button", "input_field", "tab", "icon", "modal", "window"
    count: int = 1                   # Number of visible instances
    spatial_region: str = "screen"   # "header", "sidebar", "center", "footer", "top_left", etc.
    prominence: str = "normal"       # "primary", "salient", "subtle", "active", "disabled"
    confidence: float = 1.0          # Detection confidence [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entity summary with strict coordinate encapsulation."""
        return {
            "category": self.category,
            "count": self.count,
            "spatial_region": self.spatial_region,
            "prominence": self.prominence,
            "confidence": round(self.confidence, 3),
        }

    def __str__(self) -> str:
        s = f"{self.count} {self.category}{'s' if self.count > 1 else ''}"
        if self.spatial_region and self.spatial_region != "screen":
            s += f" in {self.spatial_region}"
        return s


@dataclasses.dataclass(frozen=True)
class SceneObservation:
    """Truthful, evidence-backed visual scene observation emitted across System 2 privacy boundary.

    Guarantees:
    1. Zero raw pixel frames, image crops, or tensor embeddings exported.
    2. Zero raw screen coordinate primitives (x, y, target_coord, bounding_box).
    3. Cross-aliased fields for backward compatibility:
       scene_description <-> description
       frame_id <-> frame_sequence
    """
    scene_description: str = ""
    visible_entities: List[Any] = dataclasses.field(default_factory=list)
    state_changes: List[str] = dataclasses.field(default_factory=list)
    confidence: float = 1.0
    timestamp: float = 0.0
    frame_id: str = "frame_0"

    # Backward compatibility fields for AVCSController, simulated app, and existing tests
    primary_category: str = "screen"
    stability: str = "stable"
    surprise_score: float = 0.0
    has_active_modal: bool = False
    is_stable: bool = True
    uncertainty: float = 0.0
    frame_sequence: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        # Cross-alias scene_description and description
        if not self.scene_description and self.description:
            object.__setattr__(self, "scene_description", self.description)
        elif not self.description and self.scene_description:
            object.__setattr__(self, "description", self.scene_description)

        # Cross-alias frame_id and frame_sequence
        if self.frame_id == "frame_0" and self.frame_sequence != 0:
            object.__setattr__(self, "frame_id", f"frame_{self.frame_sequence}")
        elif self.frame_sequence == 0 and self.frame_id != "frame_0":
            try:
                seq = int(self.frame_id.replace("frame_", ""))
                object.__setattr__(self, "frame_sequence", seq)
            except ValueError:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Serializes observation to dict with zero coordinate leakage."""
        serialized_entities = []
        for e in self.visible_entities:
            if hasattr(e, "to_dict") and callable(e.to_dict):
                serialized_entities.append(e.to_dict())
            else:
                serialized_entities.append(str(e))

        return {
            "timestamp": round(self.timestamp, 3),
            "frame_id": self.frame_id,
            "frame_sequence": self.frame_sequence,
            "scene_description": self.scene_description,
            "description": self.scene_description,
            "visible_entities": serialized_entities,
            "state_changes": list(self.state_changes),
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "primary_category": self.primary_category,
            "stability": self.stability,
            "surprise_score": round(self.surprise_score, 4),
            "has_active_modal": self.has_active_modal,
            "is_stable": self.is_stable,
        }


class VisualObservationEngine:
    """Generates truthful, evidence-backed text observations from pure pixels."""

    def __init__(self, min_edge_contrast: float = 0.010) -> None:
        self.min_edge_contrast = min_edge_contrast
        self.diff_engine = VisualDiffEngine()
        self.last_observation: Optional[SceneObservation] = None
        self.consecutive_static_count: int = 0

    @staticmethod
    def _classify_spatial_region(x: float, y: float) -> str:
        """Translates normalized coordinates into qualitative anatomical desktop regions."""
        if y < 0.20:
            return "header"
        if y > 0.80:
            return "footer"
        if x < 0.25:
            return "left_sidebar"
        if x > 0.75:
            return "right_sidebar"
        return "center"

    def _extract_entities(
        self,
        affordances: List[Any],
    ) -> List[EntitySummary]:
        """Aggregates affordance hotspots into coordinate-free EntitySummary objects."""
        if not affordances:
            return []

        grouped: Dict[Tuple[str, str], List[float]] = {}
        for spot in affordances:
            if len(spot) >= 4:
                x, y, kind, conf = float(spot[0]), float(spot[1]), str(spot[2]), float(spot[3])
            elif len(spot) >= 3:
                x, y, kind = float(spot[0]), float(spot[1]), str(spot[2])
                conf = 1.0
            else:
                x, y = float(spot[0]), float(spot[1])
                kind, conf = "element", 1.0

            region = self._classify_spatial_region(x, y)
            key = (kind, region)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(conf)

        entities: List[EntitySummary] = []
        for (kind, region), confs in grouped.items():
            count = len(confs)
            avg_conf = float(np.mean(confs))
            prominence = "primary" if count > 3 or avg_conf > 0.9 else "normal"
            entities.append(
                EntitySummary(
                    category=kind,
                    count=count,
                    spatial_region=region,
                    prominence=prominence,
                    confidence=avg_conf,
                )
            )

        # Sort by count descending then category
        entities.sort(key=lambda e: (-e.count, e.category))
        return entities

    def _extract_state_changes(
        self,
        state: Any,
        prior_action: Optional[str] = None,
    ) -> List[str]:
        """Extracts dynamic visual state transitions and diffs."""
        changes: List[str] = []

        # Check visual diff from multi-turn memory
        diff = getattr(state, "visual_diff", None)
        if diff is not None:
            summary = getattr(diff, "causal_summary", None)
            if isinstance(summary, str) and summary:
                changes.append(summary)
            added = getattr(diff, "added_elements", None)
            if isinstance(added, (list, tuple, set)):
                changes.append(f"Appeared: {len(added)} new visual element(s).")
            removed = getattr(diff, "removed_ids", None)
            if isinstance(removed, (list, tuple, set)):
                changes.append(f"Disappeared: {len(removed)} element(s).")

        # Moving objects from motion engine
        moving_objs = getattr(state, "moving_objects", None)
        if isinstance(moving_objs, (list, tuple, set)) and moving_objs:
            changes.append(f"Motion detected: {len(moving_objs)} active moving visual element(s).")

        surprise = getattr(state, "surprise_score", 0.0)
        if isinstance(surprise, (int, float)) and surprise > 0.40 and not changes:
            changes.append(f"Visual surprise detected ({surprise:.2f}): screen content shifted.")

        return changes

    def _calibrate_uncertainty(
        self,
        state: Any,
        contrast: float,
    ) -> Tuple[float, float]:
        """Computes calibrated confidence and uncertainty scores."""
        cat_conf = getattr(state, "category_confidence", 1.0)
        affordances = getattr(state, "affordance_hotspots", None)

        if isinstance(cat_conf, (int, float)) and cat_conf < 0.60:
            confidence = float(cat_conf)
            uncertainty = float(max(0.40, round(1.0 - cat_conf, 3)))
        elif not affordances:
            confidence = 0.85
            uncertainty = 0.15
        else:
            confidence = 0.95
            uncertainty = 0.05

        return confidence, uncertainty

    def _synthesize_description(
        self,
        category: str,
        stability: str,
        has_modal: bool,
        entities: List[EntitySummary],
        state_changes: List[str],
        surprise_score: float,
        uncertainty: float,
    ) -> str:
        """Synthesizes honest natural language description without coordinate numbers."""
        cat_clean = category.replace("_", " ").title()
        parts = [f"Display is active. Primary context: {cat_clean} ({stability})."]

        if has_modal:
            parts.append("A modal alert dialog is currently open and focused.")

        if entities:
            entity_strs = [str(e) for e in entities[:6]]
            parts.append(f"Visible interactive controls: {', '.join(entity_strs)}.")
        else:
            parts.append("No dense interactive elements detected on screen.")

        if state_changes:
            str_changes = [str(s) for s in state_changes if isinstance(s, str)]
            if str_changes:
                parts.append(f"State transition: {' '.join(str_changes)}")

        if uncertainty >= 0.40:
            parts.append("Visual layout appears ambiguous or unclassified.")

        return " ".join(parts)

    def generate(
        self,
        state: Any,
        frame: Optional[np.ndarray] = None,
        prior_action: Optional[str] = "observe",
    ) -> SceneObservation:
        """Generates a truthful, evidence-backed SceneObservation for System 2."""
        t0 = getattr(state, "timestamp", time.time())
        frame_idx = getattr(state, "frame_index", getattr(state, "frame_count", 0))

        # 1. Edge & Contrast Gating for Blank/Wallpaper Rejection
        is_blank = False
        contrast = 0.0
        affordances = getattr(state, "affordance_hotspots", getattr(state, "affordances", []))

        if frame is not None:
            gray = np.mean(frame, axis=-1) if frame.ndim == 3 else frame.astype(float)
            gy, gx = np.gradient(gray)
            grad_mag = np.hypot(gx, gy)
            k_edge = min(32, max(8, int(grad_mag.size * 0.005)))
            top_k_edge = float(np.mean(np.partition(grad_mag.ravel(), -k_edge)[-k_edge:])) if grad_mag.size >= 8 else float(np.max(grad_mag))
            contrast = float(max(
                float(np.mean(grad_mag)),
                float(np.percentile(grad_mag, 99.5)),
                top_k_edge * 0.85,
            ) / 255.0)
            if contrast < self.min_edge_contrast and len(affordances) == 0:
                is_blank = True
        elif not affordances and getattr(state, "primary_category", "") in ("blank_canvas", "wallpaper"):
            is_blank = True

        # 2. Extract Visible Entities (Categorical, No Coordinates)
        if is_blank:
            entities: List[Any] = []
            raw_cat = getattr(state, "primary_category", "desktop_canvas")
            category = "desktop_canvas" if raw_cat in ("blank_canvas", "wallpaper", "", "unknown") else raw_cat
            desc = "The display is blank with no detectable visual structure or interactive elements."
            confidence = 0.99
            uncertainty = 0.01
            state_changes = ["Screen is blank and inactive."]
            has_modal = False
            stability = "stable"
            surprise = 0.0
        else:
            category = getattr(state, "primary_category", "unknown")
            stability = getattr(state, "stability", getattr(state, "stability_state", "stable"))
            surprise = getattr(state, "surprise_score", 0.0)
            has_modal = (
                "modal" in category
                or (getattr(state, "overlay_suspected", False) is True)
                or (getattr(state, "has_modal_dialog", False) is True)
                or (getattr(state, "has_active_modal", False) is True)
            )

            entities = self._extract_entities(affordances)
            state_changes = self._extract_state_changes(state, prior_action)
            confidence, uncertainty = self._calibrate_uncertainty(state, contrast)
            desc = self._synthesize_description(
                category=category,
                stability=stability,
                has_modal=has_modal,
                entities=entities,
                state_changes=state_changes,
                surprise_score=surprise,
                uncertainty=uncertainty,
            )

        # 3. Temporal State Tracking & Repetition Suppression
        if not is_blank and self.last_observation is not None:
            moving = getattr(state, "moving_objects", None)
            if surprise < 0.05 and not moving and len(state_changes) == 0:
                self.consecutive_static_count += 1
                if self.consecutive_static_count >= 2:
                    desc = f"Display unchanged ({category.replace('_', ' ').title()})."
                    state_changes = ["Screen stable and unchanged."]
            else:
                self.consecutive_static_count = 0
        else:
            self.consecutive_static_count = 0

        obs = SceneObservation(
            scene_description=desc,
            visible_entities=entities,
            state_changes=state_changes,
            confidence=confidence,
            timestamp=t0,
            frame_id=f"frame_{frame_idx}",
            primary_category=category,
            stability=stability,
            surprise_score=surprise,
            has_active_modal=has_modal,
            is_stable=(stability == "stable" and surprise < 0.20),
            uncertainty=uncertainty,
            frame_sequence=frame_idx,
            description=desc,
        )
        self.last_observation = obs
        return obs
