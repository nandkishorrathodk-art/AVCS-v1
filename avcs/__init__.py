"""AVCS: Artificial Visual Cortex for Screens (Official Developer SDK).

Real-Time Non-Generative Biological Visual Perception Foundation for Screen AI Agents.
"""
from __future__ import annotations

from forge.vision.cortex.sdk import ScreenEye
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState
from forge.vision.cortex.config import CortexConfig
from forge.vision.cortex.core.servoing import (
    VisualServoingEngine as VisualServoingController,
    VisualMotorState as MotorServoState,
)
from forge.vision.cortex.core import (
    AFFORDANCE_CLASSES,
    UI_CATEGORIES,
    STABILITY_STATES,
    RetinalFoveation,
    CorticalBackbone,
    DorsalAffordanceStream,
    VentralSemanticStream,
    PredictiveCodingEngine,
)
from forge.vision.capture import open_capture, ScreenCapture, SyntheticScreen
from avcs.weights import (
    download_weights,
    get_weights_path,
    load_avcs_weights,
)

__version__ = "0.2.0"
__author__ = "Praxis Neural"
__license__ = "Apache-2.0"

__all__ = [
    # Developer SDK Class
    "ScreenEye",
    # Core Neural Model & State
    "ArtificialVisualCortex",
    "CortexConfig",
    "CortexState",
    # Motor Servoing
    "VisualServoingController",
    "MotorServoState",
    # Ontologies & Streams
    "AFFORDANCE_CLASSES",
    "UI_CATEGORIES",
    "STABILITY_STATES",
    "RetinalFoveation",
    "CorticalBackbone",
    "DorsalAffordanceStream",
    "VentralSemanticStream",
    "PredictiveCodingEngine",
    # Capture Utilities
    "open_capture",
    "ScreenCapture",
    "SyntheticScreen",
    # Checkpoint & Weights Management
    "get_weights_path",
    "load_avcs_weights",
    "download_weights",
    # Native Desktop GUI
    "launch_gui",
    "AVCSDesktopApp",
    # Metadata
    "__version__",
]

def launch_gui() -> None:
    """Launches the AVCS Desktop Station GUI application."""
    from avcs.gui import launch_gui as _launch
    _launch()


def __getattr__(name: str):
    if name == "AVCSDesktopApp":
        from avcs.gui import AVCSDesktopApp
        return AVCSDesktopApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

