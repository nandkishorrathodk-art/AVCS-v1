from .adapters import RepositoryAVCSBackend
from .backend import AVCSBackend
from .client import AVCSClient
from .errors import (
    AVCSException, BackendActionError, BackendContractError,
    ClientNotConnectedError, DispatchState, DuplicateActionIdError,
    ObservationNotKnownError, StaleTargetError,
)
from .models import (
    ActionClass, ActionKind, ActionRequest, ActionResult, ActionStatus,
    BoundingBox, NormalizedPoint, Observation, RiskLevel, Target,
    TargetProvenance, VerificationEvidence,
)
from .eye_stream import EyeStream, EyeObservation
from .fovea_reader import FoveaReader, FovealElement
from .narrator import LiveNarrator, LiveCommentary
from .telemetry import (
    AffordanceKind,
    RecommendedAction,
    SceneCategory,
    SensoryTelemetry,
    StabilityState,
    TelemetryHotspot,
)
from .hand_motor import HandMotor, RippleResult
from .llm_bridge import LLMBridge, LLMCommand, describe_screen
from .run_live_agent import run_agent, run_monitor

__version__ = "0.3.0"

__all__ = [
    "AVCSBackend", "AVCSClient", "AVCSException", "BackendActionError",
    "BackendContractError", "ClientNotConnectedError", "DispatchState",
    "DuplicateActionIdError", "ObservationNotKnownError", "StaleTargetError",
    "ActionClass", "ActionKind", "ActionRequest", "ActionResult", "ActionStatus",
    "BoundingBox", "NormalizedPoint", "Observation", "RiskLevel", "Target",
    "TargetProvenance", "VerificationEvidence", "RepositoryAVCSBackend",
    "EyeStream", "EyeObservation", "FoveaReader", "FovealElement",
    "LiveNarrator", "LiveCommentary", "HandMotor", "RippleResult",
    "SceneCategory", "StabilityState", "RecommendedAction", "AffordanceKind",
    "TelemetryHotspot", "SensoryTelemetry",
    "LLMBridge", "LLMCommand", "describe_screen", "run_agent", "run_monitor",
    "__version__",
]
