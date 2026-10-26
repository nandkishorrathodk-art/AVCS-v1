"""Typed Sensory Telemetry API for AVCS / AVC Perception.

Exposes pure scalar metrics, spatial affordance coordinates, and typed enums.
Eliminates mock conversational templates and natural language prose in favor of
measurable, structured sensory telemetry for System 2 AI Agents.
"""
from __future__ import annotations

try:
    from avc.telemetry import (
        AffordanceKind,
        RecommendedAction,
        SceneCategory,
        SensoryTelemetry,
        StabilityState,
        TelemetryHotspot,
    )
except ImportError:
    import json
    from dataclasses import dataclass, field
    from enum import Enum
    from typing import Any, Dict, List, Optional, Tuple, Union

    class SceneCategory(str, Enum):
        """Categorical classification of the visual scene."""
        DESKTOP_CANVAS = "desktop_canvas"
        EDITOR_CODE = "editor_code"
        BROWSER_WEBPAGE = "browser_webpage"
        TERMINAL_SHELL = "terminal_shell"
        MODAL_DIALOG = "modal_dialog"
        FORM_INPUT = "form_input"
        MEDIA_PLAYBACK = "media_playback"
        SETTINGS_PANEL = "settings_panel"
        INTERACTIVE_CONTROL = "interactive_control"
        TOGGLE_ICON = "toggle_icon"
        TEXT_INPUT = "text_input"
        CONTENT_REGION = "content_region"
        UNKNOWN = "unknown"

        @classmethod
        def _missing_(cls, value: object) -> "SceneCategory":
            val_str = str(value).lower().strip().replace(" ", "_").replace("-", "_")
            for member in cls:
                if member.value == val_str:
                    return member
            alias_map = {
                "dialog": cls.MODAL_DIALOG,
                "modal": cls.MODAL_DIALOG,
                "code": cls.EDITOR_CODE,
                "editor": cls.EDITOR_CODE,
                "browser": cls.BROWSER_WEBPAGE,
                "terminal": cls.TERMINAL_SHELL,
                "shell": cls.TERMINAL_SHELL,
                "input": cls.FORM_INPUT,
                "desktop": cls.DESKTOP_CANVAS,
            }
            return alias_map.get(val_str, cls.UNKNOWN)

        def to_dict(self) -> Dict[str, Any]:
            return {"name": self.name, "value": self.value}

        def to_json(self) -> str:
            return json.dumps(self.value)

    class StabilityState(str, Enum):
        """Temporal stability assessment of the visual display."""
        STABLE = "stable"
        ANIMATING = "animating"
        TRANSITIONING = "transitioning"
        SURPRISE_EVENT = "surprise_event"
        SETTLED = "settled"
        UNSTABLE = "unstable"

        @classmethod
        def _missing_(cls, value: object) -> "StabilityState":
            val_str = str(value).lower().strip().replace(" ", "_").replace("-", "_")
            for member in cls:
                if member.value == val_str:
                    return member
            if "animat" in val_str or "spin" in val_str:
                return cls.ANIMATING
            if "transit" in val_str:
                return cls.TRANSITIONING
            if "surpris" in val_str or "popup" in val_str or "modal" in val_str:
                return cls.SURPRISE_EVENT
            if "settle" in val_str:
                return cls.SETTLED
            return cls.STABLE

        def to_dict(self) -> Dict[str, Any]:
            return {"name": self.name, "value": self.value}

        def to_json(self) -> str:
            return json.dumps(self.value)

    class RecommendedAction(str, Enum):
        """Policy action emitted from sensory inference."""
        NOOP = "noop"
        MOVE = "move"
        CLICK = "click"
        TYPE = "type"
        SCROLL = "scroll"
        DRAG = "drag"
        WAIT = "wait"
        FINISH = "finish"
        ABORT = "abort"

        @classmethod
        def _missing_(cls, value: object) -> "RecommendedAction":
            val_str = str(value).lower().strip().replace(" ", "_").replace("-", "_")
            for member in cls:
                if member.value == val_str:
                    return member
            if "click" in val_str:
                return cls.CLICK
            if "type" in val_str:
                return cls.TYPE
            if "scroll" in val_str:
                return cls.SCROLL
            if "wait" in val_str or "hold" in val_str:
                return cls.WAIT
            return cls.NOOP

        def to_dict(self) -> Dict[str, Any]:
            return {"name": self.name, "value": self.value}

        def to_json(self) -> str:
            return json.dumps(self.value)

    class AffordanceKind(str, Enum):
        """Morphological interaction affordance classes."""
        STATIC = "static"
        CLICKABLE = "clickable"
        TYPEABLE = "typeable"
        SCROLLABLE = "scrollable"
        DRAGGABLE = "draggable"
        BUTTON = "button"
        INPUT = "input"
        TOGGLE = "toggle"
        DIALOG = "dialog"
        UNKNOWN = "unknown"

        @classmethod
        def _missing_(cls, value: object) -> "AffordanceKind":
            val_str = str(value).lower().strip().replace(" ", "_").replace("-", "_")
            for member in cls:
                if member.value == val_str:
                    return member
            if "click" in val_str or "btn" in val_str or "button" in val_str or "interactive" in val_str:
                return cls.CLICKABLE
            if "type" in val_str or "text" in val_str or "input" in val_str:
                return cls.TYPEABLE
            if "scroll" in val_str:
                return cls.SCROLLABLE
            if "drag" in val_str:
                return cls.DRAGGABLE
            return cls.UNKNOWN

        def to_dict(self) -> Dict[str, Any]:
            return {"name": self.name, "value": self.value}

        def to_json(self) -> str:
            return json.dumps(self.value)

    @dataclass
    class TelemetryHotspot:
        """Typed spatial affordance hotspot with normalized coordinates."""
        x: float
        y: float
        kind: Union[AffordanceKind, str] = AffordanceKind.CLICKABLE
        confidence: float = 1.0

        def __post_init__(self) -> None:
            self.x = float(self.x)
            self.y = float(self.y)
            self.confidence = float(self.confidence)
            if isinstance(self.kind, str):
                self.kind = AffordanceKind(self.kind)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "kind": self.kind.value if isinstance(self.kind, Enum) else str(self.kind),
                "confidence": round(self.confidence, 4),
            }

        def to_json(self, indent: Optional[int] = None) -> str:
            return json.dumps(self.to_dict(), indent=indent)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "TelemetryHotspot":
            return cls(
                x=float(data["x"]),
                y=float(data["y"]),
                kind=AffordanceKind(data.get("kind", AffordanceKind.CLICKABLE)),
                confidence=float(data.get("confidence", 1.0)),
            )

        @classmethod
        def from_tuple(cls, item: Union[Tuple, List, "TelemetryHotspot"]) -> "TelemetryHotspot":
            if isinstance(item, TelemetryHotspot):
                return item
            x = float(item[0])
            y = float(item[1])
            kind = item[2] if len(item) > 2 else AffordanceKind.CLICKABLE
            conf = float(item[3]) if len(item) > 3 else 1.0
            return cls(x=x, y=y, kind=AffordanceKind(kind) if isinstance(kind, str) else kind, confidence=conf)

        def to_tuple(self) -> Tuple[float, float, str, float]:
            return (self.x, self.y, self.kind.value if isinstance(self.kind, Enum) else str(self.kind), self.confidence)

        def __iter__(self):
            yield self.x
            yield self.y
            yield self.kind.value if isinstance(self.kind, Enum) else str(self.kind)
            yield self.confidence

        def __getitem__(self, idx: int) -> Any:
            return (self.x, self.y, self.kind.value if isinstance(self.kind, Enum) else str(self.kind), self.confidence)[idx]

        def __len__(self) -> int:
            return 4

    @dataclass
    class SensoryTelemetry:
        """Pure scalar sensory telemetry emitted by the Visual Cortex / Eye Stream.

        Replaces mock narrative commentary with typed, measurable perception metrics.
        Zero natural language templates or mock conversational prose.
        """
        category: Union[SceneCategory, str] = SceneCategory.DESKTOP_CANVAS
        category_confidence: float = 0.0
        optical_motion_pct: float = 0.0
        is_settled: bool = True
        stability_state: Union[StabilityState, str] = StabilityState.STABLE
        surprise_score: float = 0.0
        has_modal: bool = False
        hotspots: List[TelemetryHotspot] = field(default_factory=list)
        recommended_action: Union[RecommendedAction, str] = RecommendedAction.NOOP
        latency_ms: float = 0.0

        def __post_init__(self) -> None:
            if isinstance(self.category, str):
                self.category = SceneCategory(self.category)
            if isinstance(self.stability_state, str):
                self.stability_state = StabilityState(self.stability_state)
            if isinstance(self.recommended_action, str):
                self.recommended_action = RecommendedAction(self.recommended_action)

            self.category_confidence = float(self.category_confidence)
            self.optical_motion_pct = float(self.optical_motion_pct)
            self.is_settled = bool(self.is_settled)
            self.surprise_score = float(self.surprise_score)
            self.has_modal = bool(self.has_modal)
            self.latency_ms = float(self.latency_ms)

            if self.hotspots:
                normalized = []
                for h in self.hotspots:
                    if isinstance(h, TelemetryHotspot):
                        normalized.append(h)
                    elif isinstance(h, (tuple, list)):
                        normalized.append(TelemetryHotspot.from_tuple(h))
                    elif isinstance(h, dict):
                        normalized.append(TelemetryHotspot.from_dict(h))
                    else:
                        normalized.append(h)
                self.hotspots = normalized

        def to_dict(self) -> Dict[str, Any]:
            """Serializes telemetry metrics to dictionary."""
            return {
                "category": self.category.value if isinstance(self.category, Enum) else str(self.category),
                "category_confidence": round(self.category_confidence, 4),
                "optical_motion_pct": round(self.optical_motion_pct, 4),
                "is_settled": self.is_settled,
                "stability_state": self.stability_state.value if isinstance(self.stability_state, Enum) else str(self.stability_state),
                "surprise_score": round(self.surprise_score, 4),
                "has_modal": self.has_modal,
                "hotspots": [h.to_dict() if hasattr(h, "to_dict") else h for h in self.hotspots],
                "recommended_action": self.recommended_action.value if isinstance(self.recommended_action, Enum) else str(self.recommended_action),
                "latency_ms": round(self.latency_ms, 3),
            }

        def to_json(self, indent: Optional[int] = None) -> str:
            """Serializes telemetry metrics to JSON string."""
            return json.dumps(self.to_dict(), indent=indent)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "SensoryTelemetry":
            """Reconstructs SensoryTelemetry from a dictionary."""
            hotspots = [
                TelemetryHotspot.from_dict(h) if isinstance(h, dict)
                else (TelemetryHotspot.from_tuple(h) if isinstance(h, (list, tuple)) else h)
                for h in data.get("hotspots", [])
            ]
            return cls(
                category=SceneCategory(data.get("category", SceneCategory.DESKTOP_CANVAS)),
                category_confidence=float(data.get("category_confidence", 0.0)),
                optical_motion_pct=float(data.get("optical_motion_pct", 0.0)),
                is_settled=bool(data.get("is_settled", True)),
                stability_state=StabilityState(data.get("stability_state", StabilityState.STABLE)),
                surprise_score=float(data.get("surprise_score", 0.0)),
                has_modal=bool(data.get("has_modal", False)),
                hotspots=hotspots,
                recommended_action=RecommendedAction(data.get("recommended_action", RecommendedAction.NOOP)),
                latency_ms=float(data.get("latency_ms", 0.0)),
            )

        @classmethod
        def from_json(cls, json_str: str) -> "SensoryTelemetry":
            """Reconstructs SensoryTelemetry from a JSON string."""
            return cls.from_dict(json.loads(json_str))

        @classmethod
        def from_eye_frame(
            cls,
            eye_frame: Any,
            active_window: str = "",
            foveal_elements: Optional[List[Any]] = None,
            ripple_verified: Optional[bool] = None,
            prior_action: Optional[str] = None,
        ) -> "SensoryTelemetry":
            """Constructs SensoryTelemetry directly from an EyeFrame observation."""
            has_popup = bool(getattr(eye_frame, "has_popup", False))
            surprise = float(getattr(eye_frame, "surprise_score", 0.0))
            is_stable = bool(getattr(eye_frame, "is_stable", True))
            latency = float(getattr(eye_frame, "latency_ms", 0.0))
            hotspots_raw = getattr(eye_frame, "hotspots", None) or []

            if has_popup:
                cat = SceneCategory.MODAL_DIALOG
            elif foveal_elements and len(foveal_elements) > 0:
                cat = SceneCategory(getattr(foveal_elements[0], "category", "interactive_control"))
            elif hotspots_raw:
                cat = SceneCategory(hotspots_raw[0][2])
            else:
                cat = SceneCategory.DESKTOP_CANVAS

            if foveal_elements and len(foveal_elements) > 0:
                cat_conf = float(getattr(foveal_elements[0], "confidence", 0.95))
            elif hotspots_raw:
                cat_conf = float(hotspots_raw[0][3])
            else:
                cat_conf = 0.95

            if has_popup or surprise > 0.45:
                stab = StabilityState.SURPRISE_EVENT
            elif not is_stable:
                stab = StabilityState.ANIMATING
            else:
                stab = StabilityState.STABLE

            if not is_stable:
                act = RecommendedAction.WAIT
            elif has_popup or hotspots_raw or ripple_verified is not None:
                act = RecommendedAction.CLICK
            else:
                act = RecommendedAction.NOOP

            hotspots = [TelemetryHotspot.from_tuple(h) for h in hotspots_raw]

            return cls(
                category=cat,
                category_confidence=cat_conf,
                optical_motion_pct=round(surprise * 100.0, 2),
                is_settled=is_stable,
                stability_state=stab,
                surprise_score=surprise,
                has_modal=has_popup,
                hotspots=hotspots,
                recommended_action=act,
                latency_ms=latency,
            )

        @classmethod
        def from_cortex_state(cls, state: Any) -> "SensoryTelemetry":
            """Constructs SensoryTelemetry directly from a CortexState object."""
            primary_cat = getattr(state, "primary_category", "desktop_canvas")
            conf = float(getattr(state, "category_confidence", 0.95))
            stab_raw = getattr(state, "stability_state", "stable")
            surprise = float(getattr(state, "surprise_score", 0.0))
            act_raw = getattr(state, "action", "noop")
            latency = float(getattr(state, "inference_ms", 0.0))
            hotspots_raw = getattr(state, "action_hotspots", None) or []

            has_modal = primary_cat == "modal_dialog"
            is_settled = stab_raw == "stable"
            hotspots = [TelemetryHotspot.from_tuple(h) for h in hotspots_raw]

            return cls(
                category=SceneCategory(primary_cat),
                category_confidence=conf,
                optical_motion_pct=round(surprise * 100.0, 2),
                is_settled=is_settled,
                stability_state=StabilityState(stab_raw),
                surprise_score=surprise,
                has_modal=has_modal,
                hotspots=hotspots,
                recommended_action=RecommendedAction(act_raw),
                latency_ms=latency,
            )


__all__ = [
    "SceneCategory",
    "StabilityState",
    "RecommendedAction",
    "AffordanceKind",
    "TelemetryHotspot",
    "SensoryTelemetry",
]
