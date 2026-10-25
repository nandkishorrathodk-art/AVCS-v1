"""Integration adapters for StreamingVisualCortexV2 into AVCS System 1 runtime.

Provides a unified interface for streaming desktop perception, safe checkpoint loading,
and clean separation of internal spatial geometries from System 2 text dialogue.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor
from avcs.streaming_v2.model import (
    StreamingVisualCortexV2,
    CortexV2Output,
    CortexTemporalState,
    System2SymbolicSummary,
)

DEFAULT_WEIGHTS_PATH = os.path.join("runs", "weights", "avcs_v2_weights.pt")
STAGING_WEIGHTS_PATH = os.path.join("runs", "weights", "staging", "candidate_streaming_v2.pt")

CLASS_NAMES = {
    0: "Interactive_Control",  # Clickable element / action control
    1: "Text_Input_Field",     # Text entry / search bar / field
    2: "Navigation_Toggle",    # Toggle switch / icon / menu handle / checkbox
    3: "Modal_Dialog",         # Dialog window / popup alert / modal card
    4: "Content_Region",       # Content block / paragraph / structured text area
}


@dataclasses.dataclass
class StreamingV2Observation:
    """Safe, structured perception observation from StreamingVisualCortexV2.
    
    Coordinates remain internal for motor servoing; text descriptions
    expose only symbolic visual states to System 2.
    """
    spinner_active: bool
    spinner_probability: float
    detected_ui_count: int
    detected_classes: List[str]
    candidate_boxes: List[Tuple[int, float, List[float]]]  # (class_id, conf, [cx, cy, w, h])
    change_detected: bool
    mean_change_energy: float
    visual_response_verified: bool
    visual_response_confidence: float
    forward_latency_ms: float

    def to_system2_summary(self) -> str:
        """Renders pure symbolic summary of the desktop screen state for System 2 (zero coordinates)."""
        parts = []
        if self.spinner_active:
            parts.append(f"dynamic loading motion detected (confidence: {self.spinner_probability:.2f})")
        if self.change_detected:
            parts.append(f"visual transition observed across desktop (energy: {self.mean_change_energy:.2f})")
        if self.detected_ui_count > 0:
            counts: Dict[str, int] = {}
            for cls_name in self.detected_classes:
                counts[cls_name] = counts.get(cls_name, 0) + 1
            breakdown = ", ".join(f"{cnt} {cls}" for cls, cnt in counts.items())
            parts.append(f"active screen topology: {breakdown}")
        else:
            parts.append("static non-interactive canvas")
        return "; ".join(parts).capitalize() + "."


class StreamingV2Adapter:
    """Production runtime adapter for StreamingVisualCortexV2."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        presence_threshold: float = 0.35,
        target_size: Tuple[int, int] = (128, 128),
    ) -> None:
        self.device = torch.device(device)
        self.presence_threshold = presence_threshold
        self.target_size = target_size

        self.preprocessor = EdgeLuminancePreprocessor().to(self.device)
        self.model = StreamingVisualCortexV2().to(self.device)
        self.model.eval()
        self.temporal_state: Optional[CortexTemporalState] = None
        self.last_output: Optional[CortexV2Output] = None
        self.last_observation: Optional[StreamingV2Observation] = None

        self.weights_loaded = False
        resolved_path = weights_path or (
            DEFAULT_WEIGHTS_PATH if os.path.exists(DEFAULT_WEIGHTS_PATH) else STAGING_WEIGHTS_PATH
        )
        if os.path.exists(resolved_path):
            self.load_weights(resolved_path)

    def load_weights(self, path: str) -> bool:
        """Loads weights safely into the V2 model."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
        self.model.load_state_dict(state_dict, strict=True)
        self.weights_loaded = True
        return True

    def reset_temporal_state(self) -> None:
        """Explicitly wipes temporal memory."""
        self.temporal_state = None
        self.last_output = None
        self.last_observation = None
        if hasattr(self.model, "reset_hidden_state"):
            self.model.reset_hidden_state()

    @torch.no_grad()
    def process_frame(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        timestamp: Optional[float] = None,
        dropped_frame: bool = False,
    ) -> StreamingV2Observation:
        """Processes an incoming screen frame at 128x128 through the V2 cortex."""
        import time
        start_t = time.perf_counter()
        ts = timestamp if timestamp is not None else time.time()

        if isinstance(frame, np.ndarray):
            # Frame is H, W, 3 or H, W
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            H, W = frame.shape[:2]
            if (H, W) != self.target_size:
                sy = max(1.0, float(H) / float(self.target_size[0]))
                sx = max(1.0, float(W) / float(self.target_size[1]))
                y_idx = np.minimum((np.arange(self.target_size[0]) * sy).astype(int), H - 1)
                x_idx = np.minimum((np.arange(self.target_size[1]) * sx).astype(int), W - 1)
                frame = frame[y_idx[:, None], x_idx]
            t_frame = torch.from_numpy(np.ascontiguousarray(frame)).float()
            if t_frame.max() > 1.0:
                t_frame = t_frame / 255.0
            t_frame = t_frame.permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            t_frame = frame.to(self.device)
            if t_frame.ndim == 3:
                t_frame = t_frame.unsqueeze(0)

        # 4-Channel Preprocessing
        x_4ch = self.preprocessor(t_frame)

        # Model Forward with temporal recurrent state persistence
        out: CortexV2Output = self.model(
            x_4ch,
            prev_state=self.temporal_state,
            timestamp=ts,
            dropped_frame=dropped_frame,
        )
        self.temporal_state = out.temporal_state
        self.last_output = out
        latency = (time.perf_counter() - start_t) * 1000.0

        # Parse Head 1: Temporal Change & Spinner
        change_map = out.change_heatmap.squeeze().cpu().numpy()
        mean_energy = float(np.mean(change_map))
        change_detected = mean_energy > 0.15
        spinner_prob = float(torch.sigmoid(out.spinner_activity_logit).squeeze().cpu().item())
        spinner_active = spinner_prob >= 0.50

        # Parse Head 2: 5-Class UI Detection
        obj_map = torch.sigmoid(out.ui_objectness_logits).squeeze().cpu().numpy()
        cls_logits = out.ui_class_logits.squeeze().cpu().numpy()  # (5, 16, 16)
        box_reg = out.ui_box_regression.squeeze().cpu().numpy()  # (4, 16, 16)

        boxes: List[Tuple[int, float, List[float]]] = []
        detected_classes: List[str] = []

        valid_y, valid_x = np.where(obj_map >= self.presence_threshold)
        for y, x in zip(valid_y, valid_x):
            conf = float(obj_map[y, x])
            pred_cls = int(np.argmax(cls_logits[:, y, x]))
            cx = float(box_reg[0, y, x])
            cy = float(box_reg[1, y, x])
            w = max(0.0, float(box_reg[2, y, x]))
            h = max(0.0, float(box_reg[3, y, x]))
            boxes.append((pred_cls, conf, [cx, cy, w, h]))
            detected_classes.append(CLASS_NAMES.get(pred_cls, "Unknown"))

        # Parse Head 3: Post-Action Response
        post_prob = float(torch.sigmoid(out.post_action_response_logit).squeeze().cpu().item())
        response_verified = post_prob >= 0.50

        obs = StreamingV2Observation(
            spinner_active=spinner_active,
            spinner_probability=spinner_prob,
            detected_ui_count=len(boxes),
            detected_classes=detected_classes,
            candidate_boxes=boxes,
            change_detected=change_detected,
            mean_change_energy=mean_energy,
            visual_response_verified=response_verified,
            visual_response_confidence=post_prob,
            forward_latency_ms=latency,
        )
        self.last_observation = obs
        return obs

    def to_system2_summary(
        self,
        observation: Optional[StreamingV2Observation] = None,
        as_text: bool = False,
    ) -> Union[System2SymbolicSummary, str]:
        """Provides pure symbolic summary adhering to System 2 Privacy Boundary Contract.

        Strictly encapsulates internal coordinates and exposes only high-level status
        booleans and categorical strings for System 2 dialogue and reasoning.
        """
        if as_text:
            if observation is not None:
                return observation.to_system2_summary()
            if self.last_observation is not None:
                return self.last_observation.to_system2_summary()
            return "Static non-interactive canvas."

        if self.last_output is not None:
            return self.last_output.to_symbolic_summary(
                objectness_threshold=self.presence_threshold
            )
        # Cold start fallback if no frame has been processed yet
        return System2SymbolicSummary(
            has_button=False,
            has_input_box=False,
            has_icon_toggle=False,
            has_dialog_popup=False,
            has_text_area=False,
            visual_change_detected=False,
            spinner_active=False,
            post_action_response_detected=False,
            uncertainty_level="HIGH",
            primary_ui_class=None,
            change_coverage_pct=0.0,
        )


def load_streaming_v2_weights(
    model: torch.nn.Module,
    path: Optional[str] = None,
    device: str = "cpu",
    strict: bool = True,
) -> bool:
    """Helper function to load streaming V2 weights into any matching module."""
    target_path = path or (
        DEFAULT_WEIGHTS_PATH if os.path.exists(DEFAULT_WEIGHTS_PATH) else STAGING_WEIGHTS_PATH
    )
    if not os.path.exists(target_path):
        return False
    state = torch.load(target_path, map_location=device, weights_only=False)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    elif "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=strict)
    return True
