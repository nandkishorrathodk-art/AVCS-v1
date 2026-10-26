"""Comprehensive Unit & Regression Tests for Sensory Telemetry API.

Validates:
1. Clean imports for avc.telemetry and avcs_sdk.telemetry.
2. Dataclass and Enum definitions: SceneCategory, StabilityState,
   RecommendedAction, AffordanceKind, TelemetryHotspot, SensoryTelemetry.
3. Accurate serialization via to_dict() and to_json() for all dataclasses/enums.
4. Pure numeric/typed metrics exposure on SensoryTelemetry.
5. Integration with LiveCommentary and LiveNarrator.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from avc.telemetry import (
    AffordanceKind,
    RecommendedAction,
    SceneCategory,
    SensoryTelemetry,
    StabilityState,
    TelemetryHotspot,
)
import avcs_sdk.telemetry as sdk_telemetry
from avcs_sdk.eye_stream import EyeFrame
from avcs_sdk.fovea_reader import FovealElement
from avcs_sdk.narrator import LiveCommentary, LiveNarrator


def test_clean_imports_and_parity():
    """Verify that both avc.telemetry and avcs_sdk.telemetry export identical symbols."""
    symbols = [
        "SceneCategory",
        "StabilityState",
        "RecommendedAction",
        "AffordanceKind",
        "TelemetryHotspot",
        "SensoryTelemetry",
    ]
    import avc.telemetry as avc_tel
    for s in symbols:
        assert hasattr(avc_tel, s), f"Missing {s} in avc.telemetry"
        assert hasattr(sdk_telemetry, s), f"Missing {s} in avcs_sdk.telemetry"


def test_enums_serialization():
    """Verify to_dict and to_json on all enums."""
    for member in SceneCategory:
        d = member.to_dict()
        assert "name" in d and "value" in d
        j = json.loads(member.to_json())
        assert j == member.value

    for member in StabilityState:
        assert member.to_dict()["value"] == member.value
        assert json.loads(member.to_json()) == member.value

    for member in RecommendedAction:
        assert member.to_dict()["value"] == member.value
        assert json.loads(member.to_json()) == member.value

    for member in AffordanceKind:
        assert member.to_dict()["value"] == member.value
        assert json.loads(member.to_json()) == member.value


def test_telemetry_hotspot_dataclass():
    """Verify TelemetryHotspot coordinates, indexing, iteration, to_dict, to_json."""
    h = TelemetryHotspot(x=0.45, y=0.55, kind=AffordanceKind.CLICKABLE, confidence=0.92)
    assert h.x == 0.45
    assert h.y == 0.55
    assert h.kind == AffordanceKind.CLICKABLE
    assert h.confidence == 0.92

    # Indexing & iteration support
    assert h[0] == 0.45
    assert h[1] == 0.55
    assert h[2] == "clickable"
    assert h[3] == 0.92
    x, y, kind, conf = h
    assert (x, y, kind, conf) == (0.45, 0.55, "clickable", 0.92)

    # Serialization
    d = h.to_dict()
    assert d == {"x": 0.45, "y": 0.55, "kind": "clickable", "confidence": 0.92}
    j = json.loads(h.to_json())
    assert j["kind"] == "clickable"

    # From tuple / dict round-trip
    from_t = TelemetryHotspot.from_tuple((0.1, 0.2, "typeable", 0.8))
    assert from_t.kind == AffordanceKind.TYPEABLE
    from_d = TelemetryHotspot.from_dict(d)
    assert from_d == h


def test_sensory_telemetry_pure_metrics_contract():
    """Verify all 10 required metrics are exposed as pure numeric/typed attributes."""
    hotspots = [
        TelemetryHotspot(x=0.2, y=0.3, kind=AffordanceKind.CLICKABLE, confidence=0.95),
        TelemetryHotspot(x=0.7, y=0.8, kind=AffordanceKind.TYPEABLE, confidence=0.88),
    ]
    telemetry = SensoryTelemetry(
        category=SceneCategory.EDITOR_CODE,
        category_confidence=0.98,
        optical_motion_pct=1.45,
        is_settled=True,
        stability_state=StabilityState.STABLE,
        surprise_score=0.0145,
        has_modal=False,
        hotspots=hotspots,
        recommended_action=RecommendedAction.CLICK,
        latency_ms=3.21,
    )

    # 10 pure metrics
    assert telemetry.category == SceneCategory.EDITOR_CODE
    assert isinstance(telemetry.category_confidence, float)
    assert telemetry.category_confidence == 0.98
    assert isinstance(telemetry.optical_motion_pct, float)
    assert telemetry.optical_motion_pct == 1.45
    assert isinstance(telemetry.is_settled, bool)
    assert telemetry.is_settled is True
    assert telemetry.stability_state == StabilityState.STABLE
    assert isinstance(telemetry.surprise_score, float)
    assert telemetry.surprise_score == 0.0145
    assert isinstance(telemetry.has_modal, bool)
    assert telemetry.has_modal is False
    assert len(telemetry.hotspots) == 2
    assert telemetry.recommended_action == RecommendedAction.CLICK
    assert isinstance(telemetry.latency_ms, float)
    assert telemetry.latency_ms == 3.21

    # Zero natural language templates or mock prose in dictionary
    d = telemetry.to_dict()
    assert d["category"] == "editor_code"
    assert d["category_confidence"] == 0.98
    assert d["optical_motion_pct"] == 1.45
    assert d["is_settled"] is True
    assert d["stability_state"] == "stable"
    assert d["surprise_score"] == 0.0145
    assert d["has_modal"] is False
    assert len(d["hotspots"]) == 2
    assert d["recommended_action"] == "click"
    assert d["latency_ms"] == 3.21

    # Round trip JSON serialization
    json_str = telemetry.to_json()
    reconstructed = SensoryTelemetry.from_json(json_str)
    assert reconstructed.category == telemetry.category
    assert reconstructed.category_confidence == telemetry.category_confidence
    assert reconstructed.optical_motion_pct == telemetry.optical_motion_pct
    assert reconstructed.is_settled == telemetry.is_settled
    assert reconstructed.stability_state == telemetry.stability_state
    assert reconstructed.surprise_score == telemetry.surprise_score
    assert reconstructed.has_modal == telemetry.has_modal
    assert len(reconstructed.hotspots) == 2
    assert reconstructed.recommended_action == telemetry.recommended_action
    assert reconstructed.latency_ms == telemetry.latency_ms


def test_live_commentary_telemetry_wiring():
    """Verify LiveCommentary wires and exposes structured telemetry."""
    frame = EyeFrame(
        raw_frame=np.zeros((64, 64, 3), dtype=np.uint8),
        surprise_score=0.05,
        is_stable=True,
        has_popup=False,
        hotspots=[(0.5, 0.5, "button", 0.92)],
        latency_ms=2.1,
    )
    narrator = LiveNarrator()
    comm = narrator.narrate(frame, active_window="Test Workspace")

    assert comm is not None
    assert comm.telemetry is not None
    assert isinstance(comm.telemetry, SensoryTelemetry)
    assert comm.telemetry.is_settled is True
    assert comm.telemetry.surprise_score == 0.05
    assert comm.telemetry.latency_ms == 2.1
    assert len(comm.telemetry.hotspots) == 1

    # comm.to_dict() and comm.to_json() delegates directly to telemetry
    t_dict = comm.to_dict()
    assert "category" in t_dict
    assert "optical_motion_pct" in t_dict
    assert "is_settled" in t_dict
    assert "stability_state" in t_dict
    assert "surprise_score" in t_dict
    assert "has_modal" in t_dict
    assert "hotspots" in t_dict
    assert "recommended_action" in t_dict
    assert "latency_ms" in t_dict

    t_json = comm.to_json()
    parsed = json.loads(t_json)
    assert parsed["surprise_score"] == 0.05


def test_narrate_telemetry_direct():
    """Verify LiveNarrator.narrate_telemetry returns pure structured telemetry without prose."""
    frame = EyeFrame(
        raw_frame=np.zeros((64, 64, 3), dtype=np.uint8),
        surprise_score=0.72,
        is_stable=False,
        has_popup=True,
        hotspots=[(0.5, 0.5, "dialog", 0.99)],
        latency_ms=4.5,
    )
    narrator = LiveNarrator()
    tel = narrator.narrate_telemetry(frame, active_window="Alert Window")

    assert tel is not None
    assert isinstance(tel, SensoryTelemetry)
    assert tel.has_modal is True
    assert tel.category == SceneCategory.MODAL_DIALOG
    assert tel.stability_state == StabilityState.SURPRISE_EVENT
    assert tel.recommended_action == RecommendedAction.CLICK
    assert tel.is_settled is False
    assert tel.optical_motion_pct == 72.0
