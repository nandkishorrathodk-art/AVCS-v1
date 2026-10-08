"""AVCS Developer SDK (ScreenEye).

Provides a clean, intuitive, 4-line developer experience for AI agent builders,
RPA engineers, and computer-use automation systems.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from forge.vision.capture import open_capture
from forge.vision.cortex.inference.engine import AVCSInferenceEngine
from forge.vision.cortex.model import ArtificialVisualCortex, CortexState


class ScreenEye:
    """The developer-facing Sensory Vision SDK for Computer-Use AI Agents.

    Usage:
    ------
    >>> from forge.vision.cortex import ScreenEye
    >>> eye = ScreenEye(fps=25)
    >>> eye.start()
    >>>
    >>> # 1. Check if screen is stable or still loading
    >>> if eye.wait_until_stable():
    ...     # 2. Get coordinates for clickable elements
    ...     buttons = eye.clickable_hotspots()
    ...     # 3. Locate target without sending screenshots to cloud LLM
    ...     x, y = eye.locate("Search bar")
    """

    def __init__(
        self,
        fps: int = 25,
        device: Optional[str] = None,
        source: Optional[object] = None,
    ) -> None:
        self.fps = fps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.source = source or open_capture()
        self.cortex = ArtificialVisualCortex().to(self.device).eval()

        self._change_callbacks: List[Callable[[CortexState], None]] = []
        self._surprise_callbacks: List[Callable[[CortexState], None]] = []

        self.engine = AVCSInferenceEngine(
            cortex=self.cortex,
            source=self.source,
            device=self.device,
            fps_target=fps,
            on_state=self._dispatch_state,
            on_surprise=self._dispatch_surprise,
        )

    def start(self) -> ScreenEye:
        """Starts real-time background perception stream."""
        self.engine.start()
        return self

    def stop(self) -> None:
        """Stops background perception stream."""
        self.engine.stop()

    def perceive_now(self, frame: Optional[np.ndarray] = None) -> CortexState:
        """Forces an instantaneous perception pass and returns current state."""
        return self.engine.step(frame=frame)

    @property
    def current_state(self) -> Optional[CortexState]:
        """Returns the most recent perceived screen state."""
        return self.engine.latest_state

    def is_stable(self) -> bool:
        """Returns True if the screen is settled and not animating or transitioning."""
        state = self.current_state or self.perceive_now()
        return state.stability_state == "stable"

    def wait_until_stable(self, timeout: float = 4.0, poll_interval: float = 0.05) -> bool:
        """Blocks until animations/page loads finish, or until timeout expires."""
        t_start = time.time()
        while time.time() - t_start < timeout:
            state = self.perceive_now()
            if state.stability_state == "stable" and state.surprise_score < 0.2:
                return True
            time.sleep(poll_interval)
        return False

    def look_at(self, query: str) -> Tuple[float, float]:
        """Directs the foveal gaze to lock onto the target element."""
        dummy_text = torch.randn(1, self.cortex.cfg.latent_dim, device=self.device)
        frame = self.source.grab()
        state = self.cortex.perceive(frame, text_embedding=dummy_text, device=self.device)
        target = state.query_target or state.fixation
        # Update retinal gaze fixation
        self.cortex.retina.update_fixation(target_fixation=target)
        return target

    def steer_hand(
        self,
        current_cursor: Tuple[float, float],
        target_gaze: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[float, float], bool]:
        """Biological Hand-Eye coordination step.

        Returns:
            (motor_velocity: (vx, vy), contact_acquired: bool)
        Steers the hand toward the gaze point without calculating absolute pixel jumps.
        """
        gaze = target_gaze or self.cortex.retina.current_fixation
        motor_state = self.cortex.servoing.compute_motor_flow(
            gaze_target=gaze,
            current_cursor=current_cursor,
        )
        return motor_state.motor_velocity, motor_state.contact_acquired

    def is_aligned(self, current_cursor: Tuple[float, float], target_gaze: Optional[Tuple[float, float]] = None) -> bool:
        """Returns True if the cursor is locked onto the foveal target basin."""
        _, aligned = self.steer_hand(current_cursor, target_gaze=target_gaze)
        return aligned

    def get_affordances(self, action_type: str = "clickable") -> List[Tuple[float, float, float]]:
        """Returns list of normalized (x, y, confidence) points for interactive controls."""
        state = self.current_state or self.perceive_now()
        spots = []
        for x, y, act, conf in (state.action_hotspots or []):
            if act == action_type:
                spots.append((x, y, conf))
        return spots

    def clickable_hotspots(self) -> List[Tuple[float, float, float]]:
        """Convenience helper: returns top clickable focus points."""
        return self.get_affordances(action_type="clickable")

    def typeable_hotspots(self) -> List[Tuple[float, float, float]]:
        """Convenience helper: returns top input/typeable focus points."""
        return self.get_affordances(action_type="typeable")

    def on_change(self, callback: Callable[[CortexState], None]) -> Callable[[CortexState], None]:
        """Decorator or method to register a screen state change listener."""
        self._change_callbacks.append(callback)
        return callback

    def on_surprise(self, callback: Callable[[CortexState], None]) -> Callable[[CortexState], None]:
        """Decorator or method to register a surprise/anomaly event listener."""
        self._surprise_callbacks.append(callback)
        return callback

    def _dispatch_state(self, state: CortexState) -> None:
        for cb in self._change_callbacks:
            try:
                cb(state)
            except Exception as e:
                print(f"[ScreenEye] Callback error: {e}")

    def _dispatch_surprise(self, state: CortexState) -> None:
        for cb in self._surprise_callbacks:
            try:
                cb(state)
            except Exception as e:
                print(f"[ScreenEye] Surprise callback error: {e}")

    def __enter__(self) -> ScreenEye:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

