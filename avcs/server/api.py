"""AVCS Cloud Inference API Server (FastAPI).

Hosts the proprietary AVCS neural model weights in a secure cloud environment,
preventing model extraction, unauthorized weight copying, or local weights leakage.

Clients send screen frames over HTTPS; the cloud server computes visual perception,
affordances, and foveal acuity, returning structured JSON action coordinates.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.grounding import FovealAcuityEngine
from avcs.focus import VisualFocusDetector
from avcs.servoing import NeuromuscularServoingController
from forge.vision.cortex.motor.driver import VirtualCursorDriver

app = FastAPI(
    title="AVCS Cloud Neural Inference Service",
    description="Zero-leakage cloud-hosted biological visual cortex for screen AI agents",
    version="1.0.0",
)

# Global lazy-loaded cloud neural engine
_eye: Optional[HumanEye] = None
_fovea: Optional[FovealAcuityEngine] = None
_focus_detector: Optional[VisualFocusDetector] = None
_servoing: Optional[NeuromuscularServoingController] = None


def get_engine() -> Tuple[HumanEye, FovealAcuityEngine, VisualFocusDetector, NeuromuscularServoingController]:
    global _eye, _fovea, _focus_detector, _servoing
    if _eye is None:
        _eye = HumanEye(source="synthetic", load_weights=True)
        _fovea = FovealAcuityEngine(device="cpu", load_weights=True)
        _focus_detector = VisualFocusDetector(history_maxlen=35)
        _servoing = NeuromuscularServoingController(driver=VirtualCursorDriver())
    return _eye, _fovea, _focus_detector, _servoing


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    master_key = os.environ.get("AVCS_API_KEY")
    if master_key:
        if not x_api_key or x_api_key != master_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header",
            )
    return x_api_key or "anon"


def decode_image(b64_str: str) -> np.ndarray:
    try:
        raw_bytes = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.asarray(pil_img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}")


# Pydantic Schemas
class PerceiveRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded JPEG or PNG screen frame")
    action_type: str = Field("all", description="'clickable', 'typeable', or 'all'")
    top_k: int = Field(5, description="Max affordances to extract")


class PerceiveResponse(BaseModel):
    category: str
    category_confidence: float
    stability: str
    surprise_score: float
    latency_ms: float
    affordances: List[Tuple[float, float, str, float]]


class FoveateRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded screen frame")
    target: Tuple[float, float] = Field(..., description="Normalized (x, y) target coordinate")


class FoveateResponse(BaseModel):
    element_kind: str
    confidence: float
    contrast_score: float
    bbox: Tuple[float, float, float, float]
    latency_ms: float


class VerifyFocusRequest(BaseModel):
    current_image_b64: str
    reference_image_b64: Optional[str] = None
    target: Tuple[float, float]


class VerifyFocusResponse(BaseModel):
    is_focused: bool
    confidence: float
    caret_detected: bool
    caret_blinking: bool


class PlanTrajectoryRequest(BaseModel):
    start: Tuple[float, float]
    target: Tuple[float, float]
    duration_s: Optional[float] = None


class PlanTrajectoryResponse(BaseModel):
    duration_s: float
    waypoints_count: int
    waypoints: List[Dict[str, Any]]


@app.get("/v1/health")
def health_check():
    eye, fovea, _, _ = get_engine()
    return {
        "status": "healthy",
        "service": "AVCS Cloud Neural Inference",
        "weights_loaded": eye.weights_loaded,
        "fovea_weights_loaded": fovea.weights_loaded,
        "device": "cpu",
    }


@app.post("/v1/perceive", response_model=PerceiveResponse)
def perceive_screen(req: PerceiveRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    frame = decode_image(req.image_b64)
    eye, _, _, _ = get_engine()

    state: HumanEyeState = eye.observe(frame=frame)

    affordances = [
        spot for spot in state.affordance_hotspots
        if req.action_type == "all" or spot[2] == req.action_type
    ][:req.top_k]

    return PerceiveResponse(
        category=state.primary_category,
        category_confidence=round(state.category_confidence, 3),
        stability=state.stability,
        surprise_score=round(state.surprise_score, 4),
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        affordances=affordances,
    )


@app.post("/v1/foveate", response_model=FoveateResponse)
def foveate_target(req: FoveateRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    frame = decode_image(req.image_b64)
    _, fovea, _, _ = get_engine()

    insp = fovea.inspect(frame=frame, centroid=req.target)

    return FoveateResponse(
        element_kind=insp.element_kind,
        confidence=round(insp.confidence, 3),
        contrast_score=round(insp.contrast_score, 4),
        bbox=insp.bbox,
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )


@app.post("/v1/verify_focus", response_model=VerifyFocusResponse)
def verify_focus(req: VerifyFocusRequest, api_key: str = Depends(verify_api_key)):
    cur_frame = decode_image(req.current_image_b64)
    ref_frame = decode_image(req.reference_image_b64) if req.reference_image_b64 else None
    _, _, focus_detector, _ = get_engine()

    st = focus_detector.verify_focus(cur_frame, req.target, reference_frame=ref_frame)

    return VerifyFocusResponse(
        is_focused=st.is_focused,
        confidence=round(st.confidence, 3),
        caret_detected=st.caret_detected,
        caret_blinking=st.caret_blinking,
    )


@app.post("/v1/plan_trajectory", response_model=PlanTrajectoryResponse)
def plan_trajectory(req: PlanTrajectoryRequest, api_key: str = Depends(verify_api_key)):
    _, _, _, servoing = get_engine()

    dx = req.target[0] - req.start[0]
    dy = req.target[1] - req.start[1]
    dist = float(np.hypot(dx, dy))

    dur = req.duration_s or servoing.calculate_fitts_duration(distance=dist)
    pts = servoing.generate_minimum_jerk_profile(start=req.start, target=req.target, duration_s=dur)

    waypoints_json = [
        {
            "timestamp": round(p.timestamp, 4),
            "x": round(p.x, 4),
            "y": round(p.y, 4),
            "vx": round(p.vx, 4),
            "vy": round(p.vy, 4),
        }
        for p in pts
    ]

    return PlanTrajectoryResponse(
        duration_s=round(dur, 3),
        waypoints_count=len(pts),
        waypoints=waypoints_json,
    )

