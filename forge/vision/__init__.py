"""Real-time, pixel-only screen vision.

Pure visual awareness of a desktop: capture -> reflex perception -> typed scene
state -> optional description.  No OCR, no accessibility tree, no structured UI
detection -- only the pixels a human eye would receive.
"""
from forge.vision.capture import (
    CaptureError,
    FrameSource,
    ScreenCapture,
    SyntheticScreen,
    open_capture,
)
from forge.vision.cortex import (
    ArtificialVisualCortex,
    CortexState,
    DorsalAffordanceStream,
    LanguageModulation,
    PredictiveCodingEngine,
    RetinalFoveation,
    VentralSemanticStream,
)
from forge.vision.describer import HeuristicDescriber, ModelDescriber, VLMDescriber
from forge.vision.desktop_context import (
    DesktopContextProvider,
    DesktopWindow,
    Win32DesktopContextProvider,
)
from forge.vision.encoder import (
    SceneUnderstandingModel,
    Understanding,
    train_understanding,
    word_report,
)
from forge.vision.pipeline import ScreenVision
from forge.vision.reflex import ReflexPerception, events_from_states
from forge.vision.scene import Region, SceneEvent, SceneState
from forge.vision.scenes import SCENE_CATEGORIES, SceneGenerator, sample_states
from forge.vision.vocab import VOCAB, render_tokens, tokens_for_state

__all__ = [
    "SCENE_CATEGORIES",
    "VOCAB",
    "ArtificialVisualCortex",
    "CaptureError",
    "CortexState",
    "DesktopContextProvider",
    "DesktopWindow",
    "DorsalAffordanceStream",
    "FrameSource",
    "HeuristicDescriber",
    "LanguageModulation",
    "ModelDescriber",
    "PredictiveCodingEngine",
    "ReflexPerception",
    "Region",
    "RetinalFoveation",
    "SceneEvent",
    "SceneGenerator",
    "SceneState",
    "SceneUnderstandingModel",
    "ScreenCapture",
    "ScreenVision",
    "SyntheticScreen",
    "Understanding",
    "VLMDescriber",
    "VentralSemanticStream",
    "Win32DesktopContextProvider",
    "events_from_states",
    "open_capture",
    "render_tokens",
    "sample_states",
    "tokens_for_state",
    "train_understanding",
    "word_report",
]
