"""ScreenEye: Developer-facing Sensory Vision SDK for Computer-Use AI Agents."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from avc.config import CortexConfig, realtime_config
from avc.model import ArtificialVisualCortex, CortexState


def default_weights_path() -> Path:
    key = os.environ.get("AVC_API_KEY") or os.environ.get("AVCS_API_KEY") or os.environ.get("AVCS_KEY")
    vault_path = Path(__file__).parent / "weights" / "avcs_cortex_weights.avcs"
    pt_path = Path(__file__).parent / "weights" / "avcs_cortex_weights.pt"
    if vault_path.exists() and (key or not pt_path.exists()):
        return vault_path
    if pt_path.exists():
        return pt_path
    return vault_path


class ScreenEye:
    """The developer-facing Sensory Vision SDK for Computer-Use AI Agents.

    Usage:
    ------
    >>> import avc
    >>> eye = avc.ScreenEye()
    >>> state = eye.capture_and_perceive()
    >>> print("Category:", state.primary_category)
    >>> print("Interactive Hotspots:", state.action_hotspots)
    """

    def __init__(
        self,
        fps: int = 25,
        device: Optional[str] = None,
        weights_path: Optional[Union[str, Path]] = None,
        load_pretrained: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        self.fps = fps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cortex = ArtificialVisualCortex().to(self.device).eval()
        self.weights_loaded = False

        if load_pretrained:
            target_path = Path(weights_path) if weights_path else default_weights_path()
            if target_path.exists():
                self.cortex.load_weights(target_path, strict=False, api_key=api_key)
                self.weights_loaded = True

        self._sct = None

    def _init_capture(self) -> None:
        if self._sct is None:
            try:
                import mss
                self._sct = mss.mss()
            except ImportError:
                self._sct = None

    def grab_frame(self) -> np.ndarray:
        """Capture the primary monitor as a (H, W, 3) uint8 RGB array."""
        self._init_capture()
        if self._sct is not None:
            monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]
            sct_img = self._sct.grab(monitor)
            # BGRA -> RGB
            frame = np.array(sct_img, dtype=np.uint8)[:, :, :3][:, :, ::-1]
            return np.ascontiguousarray(frame)

        # Fallback if mss is unavailable or headless
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def perceive(
        self,
        frame: Optional[np.ndarray] = None,
        cursor_position: Optional[Tuple[float, float]] = None,
        task_query: Optional[str] = None,
    ) -> CortexState:
        """Run real-time neural perception on the provided or captured screen frame."""
        target_frame = frame if frame is not None else self.grab_frame()
        return self.cortex.perceive(
            frame=target_frame,
            cursor_position=cursor_position,
            text_query=task_query,
            device=self.device,
        )

    def capture_and_perceive(self, task_query: Optional[str] = None) -> CortexState:
        """Capture primary display and execute full cortical perception pipeline."""
        return self.perceive(task_query=task_query)

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

    def __enter__(self) -> ScreenEye:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
