"""AVCS Cloud Inference API Server (FastAPI).

Hosts AVCS model weights on the server. Clients send screen frames to this service
for inference; HTTPS protects transport but does not prevent server-side access.
"""
from __future__ import annotations

import base64
import hmac
import io
import os
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from avcs.human_eye import HumanEye, HumanEyeState
from avcs.grounding import FovealAcuityEngine
from avcs.focus import VisualFocusDetector
from avcs.goal_grounder import GoalConditionedGrounder, load_grounding_checkpoint
from avcs.query_limits import MAX_QUERY_BYTES
from avcs.servoing import NeuromuscularServoingController
from forge.vision.cortex.motor.driver import VirtualCursorDriver

app = FastAPI(
    title="AVCS Cloud Neural Inference Service",
    description="Remote AVCS visual inference; submitted screen frames are processed by this service.",
    version="1.0.0",
)

# Global lazy-loaded cloud neural engine
import threading

_init_lock = threading.Lock()
_eye: Optional[HumanEye] = None
_fovea: Optional[FovealAcuityEngine] = None
_focus_detector: Optional[VisualFocusDetector] = None
_servoing: Optional[NeuromuscularServoingController] = None
_grounder: Optional[GoalConditionedGrounder] = None
_grounder_loaded = False
_grounder_load_attempted = False


def get_engine() -> Tuple[HumanEye, FovealAcuityEngine, VisualFocusDetector, NeuromuscularServoingController]:
    global _eye, _fovea, _focus_detector, _servoing
    if _eye is None or _servoing is None:
        with _init_lock:
            if _eye is None:
                eye = HumanEye(source="synthetic", load_weights=True)
                fovea = FovealAcuityEngine(device="cpu", load_weights=True)
                focus_detector = VisualFocusDetector(history_maxlen=35)
                servoing = NeuromuscularServoingController(driver=VirtualCursorDriver())
                _fovea = fovea
                _focus_detector = focus_detector
                _servoing = servoing
                _eye = eye
    return _eye, _fovea, _focus_detector, _servoing


def _grounding_checkpoint_path() -> str:
    configured = os.environ.get("AVCS_GROUNDING_WEIGHTS")
    if configured:
        return configured
    return os.path.join(os.path.dirname(__file__), "..", "..", "runs", "weights", "goal_grounder_best.pt")


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    master_key = (
        os.environ.get("AVCS_INFERENCE_API_KEY") or os.environ.get("AVCS_API_KEY") or ""
    ).strip()
    if not master_key or len(master_key) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service authentication is not configured.",
        )
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    if not hmac.compare_digest(x_api_key.encode("utf-8"), master_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return "authenticated"


def decode_image(b64_str: str) -> np.ndarray:
    try:
        raw_bytes = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.asarray(pil_img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}")


def get_grounder() -> Optional[GoalConditionedGrounder]:
    global _grounder, _grounder_loaded, _grounder_load_attempted
    if _grounder_load_attempted:
        return _grounder

    with _init_lock:
        if _grounder_load_attempted:
            return _grounder
        checkpoint = _grounding_checkpoint_path()
        try:
            g = load_grounding_checkpoint(checkpoint, device="cpu")
            _grounder = g
            _grounder_loaded = g.is_trained
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[AVCS] Goal-grounding checkpoint unavailable: {exc}", file=sys.stderr)
            _grounder = None
            _grounder_loaded = False
        finally:
            _grounder_load_attempted = True
    return _grounder


# Pydantic Schemas
class PerceiveRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded JPEG or PNG screen frame")
    action_type: Literal["clickable", "typeable", "all"] = Field(
        "all", description="'clickable', 'typeable', or 'all'"
    )
    top_k: int = Field(5, ge=0, le=50, description="Max affordances to extract")


class PerceiveResponse(BaseModel):
    category: str
    category_confidence: float
    stability: str
    surprise_score: float
    latency_ms: float
    affordances: List[Tuple[float, float, str, float]]
    weights_loaded: bool


class FoveateRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded screen frame")
    target: Tuple[float, float] = Field(..., description="Normalized (x, y) target coordinate")


class FoveateResponse(BaseModel):
    element_kind: str
    confidence: float
    contrast_score: float
    bbox: Tuple[float, float, float, float]
    latency_ms: float


class GroundRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded JPEG or PNG screen frame")
    query: str = Field(..., min_length=1, max_length=256, description="Natural-language target")


class GroundResponse(BaseModel):
    query: str
    target_coord: Optional[Tuple[float, float]] = Field(
        ..., description="Candidate normalized (x, y) coordinates; null when outside the frame"
    )
    confidence: float
    target_present: bool
    target_presence_probability: float
    frame_width: int
    frame_height: int
    checkpoint: str
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
def health_check(api_key: str = Depends(verify_api_key)):
    eye, fovea, _, _ = get_engine()
    grounder = get_grounder()
    perception_ready = bool(eye.weights_loaded)
    fovea_ready = bool(fovea.weights_loaded)
    grounding_ready = bool(grounder and _grounder_loaded and grounder.is_trained)
    service_ready = perception_ready or fovea_ready or grounding_ready
    return {
        "status": "healthy" if service_ready else "degraded",
        "service": "AVCS Cloud Neural Inference",
        "weights_loaded": eye.weights_loaded,
        "fovea_weights_loaded": fovea.weights_loaded,
        "grounder_weights_loaded": grounding_ready,
        "perceive_ready": perception_ready,
        "foveate_ready": fovea_ready,
        "ground_ready": grounding_ready,
        "device": "cpu",
    }


@app.post("/v1/perceive", response_model=PerceiveResponse)
def perceive_screen(req: PerceiveRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    frame = decode_image(req.image_b64)
    eye, _, _, _ = get_engine()
    if not eye.weights_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A trained visual-perception checkpoint is not loaded.",
        )

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
        weights_loaded=eye.weights_loaded,
    )


@app.post("/v1/ground", response_model=GroundResponse)
def ground_target(req: GroundRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    query = req.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Grounding query cannot be empty.",
        )
    if len(query.encode("utf-8")) > MAX_QUERY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Grounding query exceeds {MAX_QUERY_BYTES} UTF-8 bytes.",
        )
    frame = decode_image(req.image_b64)
    grounder = get_grounder()
    if grounder is None or not grounder.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A demonstration-trained grounding checkpoint is not loaded.",
        )
    try:
            prediction = grounder.predict(frame, query)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return GroundResponse(
        query=query,
        target_coord=prediction.candidate_point_norm,
        confidence=round(prediction.peak_probability, 4),
        target_present=prediction.target_present,
        target_presence_probability=round(
            prediction.target_present_probability, 4
        ),
        frame_width=int(frame.shape[1]),
        frame_height=int(frame.shape[0]),
        checkpoint="demonstration_trained",
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
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
