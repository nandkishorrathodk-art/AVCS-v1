"""Compatibility exports and enhanced HumanEye intelligence engine for :mod:`avcs.perception.human_eye`."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from avcs.perception.human_eye import *  # noqa: F403
from avcs.perception.human_eye import HumanEye as _BaseHumanEye
from avcs.goal_grounder import GoalConditionedGrounder, GroundedTarget


def _human_eye_locate(
    self: Any,
    query: str,
    frame: Optional[np.ndarray] = None,
) -> GroundedTarget:
    """Grounds natural language target onto screen coordinates (<15ms on CPU)."""
    if not isinstance(self.grounder, GoalConditionedGrounder):
        feat_dim = (
            getattr(self.backbone, "feature_dim", 128)
            if hasattr(self, "backbone")
            else 128
        )
        in_sz = (
            self.input_size[0]
            if hasattr(self, "input_size") and isinstance(self.input_size, (tuple, list))
            else 128
        )
        device_to_use = getattr(self, "device", "cpu")
        self.grounder = GoalConditionedGrounder(
            feature_dim=feat_dim, input_size=in_sz
        ).to(device_to_use)
        from avcs.weights import load_avcs_weights
        load_avcs_weights(self, device=str(device_to_use), strict=False)

    if frame is None:
        if hasattr(self, "source") and hasattr(self.source, "grab"):
            frame = self.source.grab()
        elif getattr(self, "_latest_raw_frame", None) is not None:
            frame = self._latest_raw_frame
        else:
            in_sz = (
                self.input_size[0]
                if hasattr(self, "input_size") and isinstance(self.input_size, (tuple, list))
                else 128
            )
            frame = np.zeros((in_sz, in_sz, 3), dtype=np.uint8)

    return self.grounder.locate(query, frame=frame)


def _human_eye_predict_point(
    self: Any,
    image_tensor: torch.Tensor,
    query: str,
) -> Dict[str, Any]:
    """Infers candidate point, confidence, and bounding box via GoalConditionedGrounder."""
    if not isinstance(self.grounder, GoalConditionedGrounder):
        feat_dim = (
            getattr(self.backbone, "feature_dim", 128)
            if hasattr(self, "backbone")
            else 128
        )
        in_sz = (
            self.input_size[0]
            if hasattr(self, "input_size") and isinstance(self.input_size, (tuple, list))
            else 128
        )
        device_to_use = getattr(self, "device", "cpu")
        self.grounder = GoalConditionedGrounder(
            feature_dim=feat_dim, input_size=in_sz
        ).to(device_to_use)
        from avcs.weights import load_avcs_weights
        load_avcs_weights(self, device=str(device_to_use), strict=False)

    return self.grounder.predict_point(image_tensor, query)


# Dynamically bind helpers onto _BaseHumanEye
_BaseHumanEye.locate = _human_eye_locate
_BaseHumanEye.predict_point = _human_eye_predict_point

# Hook __init__ to ensure grounder is GoalConditionedGrounder and eval() is called
if not getattr(_BaseHumanEye, "_grounder_hooked", False):
    _orig_init = _BaseHumanEye.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not isinstance(self.grounder, GoalConditionedGrounder):
            in_sz = (
                self.input_size[0]
                if hasattr(self, "input_size") and isinstance(self.input_size, (tuple, list))
                else 128
            )
            self.grounder = GoalConditionedGrounder(
                feature_dim=getattr(self.backbone, "feature_dim", 128),
                input_size=in_sz,
            ).to(self.device)
            if kwargs.get("load_weights", True):
                from avcs.weights import load_avcs_weights
                load_avcs_weights(
                    self,
                    path=kwargs.get("weights_path"),
                    device=str(self.device),
                    strict=False,
                )
        self.eval()

    _BaseHumanEye.__init__ = _patched_init
    _BaseHumanEye._grounder_hooked = True

HumanEye = _BaseHumanEye
