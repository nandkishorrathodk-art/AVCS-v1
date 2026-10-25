"""AVC — Artificial Visual Cortex Neural Network.

A biomimetic visual cortex for screens and computer-use AI agents.
Combines Retinal Foveation, Dual-path MBConv Backbone, Dorsal Stream (Affordances),
Ventral Stream (Semantics), Language Modulation, and Recurrent GRU Predictive Coding.

Quickstart:
-----------
>>> import avc
>>> cortex = avc.load_cortex()
>>> state = cortex.perceive(screen_frame)
>>> print(state.primary_category, state.stability_state, state.action_hotspots)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from avc.config import CortexConfig, ModelConfig, RetinaConfig, TrainingConfig, realtime_config
from avc.core.dorsal import AFFORDANCE_CLASSES, INTERACTIVE_CLASSES
from avc.core.predictive import STABILITY_STATES
from avc.core.ventral import UI_CATEGORIES
from avc.eye import ScreenEye, default_weights_path
from avc.model import ArtificialVisualCortex, CortexState
from avc.vault import InvalidKeyError, ModelVault, VaultSecurityError

__version__ = "1.0.0"


def load_cortex(
    weights_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    config: Optional[Union[ModelConfig, CortexConfig]] = None,
    api_key: Optional[str] = None,
) -> ArtificialVisualCortex:
    """Instantiate and load the pretrained Artificial Visual Cortex.

    Args:
        weights_path: Optional path to .pt weights checkpoint or .avcs vault.
                      Defaults to the bundled pretrained weights.
        device: Device to map model to ("cpu", "cuda", etc.).
        config: Optional CortexConfig or ModelConfig.
        api_key: Optional authorization key for encrypted .avcs vaults.

    Returns:
        ArtificialVisualCortex loaded in eval() mode.
    """
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cortex = ArtificialVisualCortex(config=config).to(target_device).eval()

    target_path = Path(weights_path) if weights_path else default_weights_path()
    if target_path.exists():
        cortex.load_weights(target_path, strict=False, api_key=api_key)

    return cortex


__all__ = [
    "ArtificialVisualCortex",
    "CortexState",
    "ScreenEye",
    "CortexConfig",
    "ModelConfig",
    "RetinaConfig",
    "TrainingConfig",
    "realtime_config",
    "load_cortex",
    "default_weights_path",
    "AFFORDANCE_CLASSES",
    "INTERACTIVE_CLASSES",
    "UI_CATEGORIES",
    "STABILITY_STATES",
    "ModelVault",
    "VaultSecurityError",
    "InvalidKeyError",
    "__version__",
]
