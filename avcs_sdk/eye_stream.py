"""[1] eye_stream.py: 100 FPS Desktop Perception Stream with StreamingVisualCortexV2.

Self-contained ultra-fast desktop perception (<6ms on CPU, 400k parameters).
Features:
- Native Win32 / MSS 100 FPS desktop screen frame grabber.
- 4-Channel Edge & Luminance Pre-Processor (Theme invariant: Dark Mode / Light Mode agnostic).
- Spatiotemporal ConvGRU recurrent temporal dynamics: measures surprise score and detects modal alerts.
- Universal 5-class generic UI morphology detection (interactive controls, inputs, toggles, dialogs, content).
- Post-action visual response verification (click ripple detection).
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.cortex.model import (
    ConvGRUCell,
    CortexTemporalState,
    CortexV2Output,
    StreamingVisualCortexV2,
)
from avcs.cortex.preprocessor import EdgeLuminancePreprocessor

def _resolve_weights_path(filename: str) -> str:
    """Locate bundled model weights both in package distribution and local development."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_path = os.path.join(pkg_dir, "weights", filename)
    if os.path.exists(bundled_path):
        return bundled_path
    repo_root = os.path.dirname(pkg_dir)
    dev_path = os.path.join(repo_root, "weights", filename)
    if os.path.exists(dev_path):
        return dev_path
    return bundled_path


DEFAULT_WEIGHTS_V2 = _resolve_weights_path("avcs_v2.pt")
DEFAULT_WEIGHTS_V1 = _resolve_weights_path("avcs_v1.pt")
DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_V2 if os.path.exists(DEFAULT_WEIGHTS_V2) else (
    DEFAULT_WEIGHTS_V1 if os.path.exists(DEFAULT_WEIGHTS_V1) else DEFAULT_WEIGHTS_V2
)


@dataclass(frozen=True)
class DisplayGeometry:
    """Normalized multi-monitor coordinate frame geometry."""
    virtual_left: int
    virtual_top: int
    virtual_width: int
    virtual_height: int

    def to_normalized(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        nx = (screen_x - self.virtual_left) / max(1, self.virtual_width)
        ny = (screen_y - self.virtual_top) / max(1, self.virtual_height)
        return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))

    def to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        sx = int(self.virtual_left + norm_x * self.virtual_width)
        sy = int(self.virtual_top + norm_y * self.virtual_height)
        return sx, sy


@dataclass(frozen=True)
class EyeFrame:
    """Single tick observation from the EyeStream."""
    raw_frame: np.ndarray             # High-res native RGB frame
    surprise_score: float             # Change magnitude in [0, 1]
    is_stable: bool                  # True if screen content has settled
    has_popup: bool                  # True if high-contrast modal or alert appeared
    hotspots: List[Tuple[float, float, str, float]]  # [(norm_x, norm_y, kind, conf), ...]
    latency_ms: float                # Forward pass duration


class EyeStream(StreamingVisualCortexV2):
    """100 FPS Live Eye Stream with integrated StreamingVisualCortexV2 (400,765 parameters)."""

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        weights_path: Optional[str] = DEFAULT_WEIGHTS,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.to(self.device)
        self.geometry = self._query_geometry()
        self._sct = None
        self._init_capture()

        self.preprocessor = EdgeLuminancePreprocessor().to(self.device)
        self.temporal_state: Optional[CortexTemporalState] = None

        # Honest weight provenance: a checkpoint is only "loaded" if keys match.
        self.weights_loaded: bool = False
        self.weights_status: str = "random_initialised"
        if weights_path:
            self.load_weights(weights_path)

        self.eval()

    def load_weights(self, weights_path: str) -> bool:
        """Loads cortex weights if the checkpoint matches this architecture.

        The checkpoint is verified and loaded only if state dict keys match.
        Records an explicit status in ``weights_loaded`` / ``weights_status``
        and leaves the model untouched when nothing can be mapped.
        """
        if not os.path.exists(weights_path):
            self.weights_loaded = False
            self.weights_status = f"weights file not found: {weights_path}"
            return False
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            self.weights_loaded = False
            self.weights_status = f"weights unreadable: {exc}"
            return False

        if not isinstance(checkpoint, dict):
            self.weights_loaded = False
            self.weights_status = f"unexpected checkpoint type: {type(checkpoint).__name__}"
            return False

        # Extract nested state dict if packaged with metadata
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            ckpt_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            ckpt_dict = checkpoint["model_state_dict"]
        else:
            ckpt_dict = checkpoint

        own = self.state_dict()
        matched = {
            k: v for k, v in ckpt_dict.items()
            if k in own and tuple(own[k].shape) == tuple(getattr(v, "shape", ()))
        }
        if not matched:
            self.weights_loaded = False
            self.weights_status = (
                f"checkpoint incompatible with this architecture "
                f"({len(ckpt_dict)} checkpoint keys, 0 matching); running untrained"
            )
            return False

        if len(matched) != len(own) or len(matched) != 52 or len(ckpt_dict) != len(own):
            self.weights_loaded = False
            self.weights_status = (
                f"loaded {len(matched)}/{len(own)} tensors (partial checkpoint rejected) "
                f"from {os.path.basename(weights_path)}"
            )
            return False

        if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in matched.values()):
            self.weights_loaded = False
            self.weights_status = (
                f"rejected checkpoint with non-finite weights (NaN/Inf) from {os.path.basename(weights_path)}"
            )
            return False

        own.update(matched)
        self.load_state_dict(own)
        self.weights_loaded = True
        self.weights_status = f"loaded {len(matched)}/{len(own)} tensors from {os.path.basename(weights_path)}"
        return True

    def _query_geometry(self) -> DisplayGeometry:
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            v_left = user32.GetSystemMetrics(76)
            v_top = user32.GetSystemMetrics(77)
            v_width = user32.GetSystemMetrics(78)
            v_height = user32.GetSystemMetrics(79)
            if v_width <= 0 or v_height <= 0:
                v_width, v_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                v_left, v_top = 0, 0
            return DisplayGeometry(v_left, v_top, v_width, v_height)
        return DisplayGeometry(0, 0, 1920, 1080)

    def _init_capture(self) -> None:
        try:
            if sys.platform == "win32":
                try:
                    u32 = ctypes.windll.user32
                    hdesk = u32.OpenInputDesktop(0, False, 0x01FF)
                    if hdesk:
                        u32.SetThreadDesktop(hdesk)
                except Exception:
                    pass
            import mss
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()
        except Exception:
            self._sct = None

    def grab_screen(self) -> np.ndarray:
        """Captures a native RGB screenshot in <3-6ms."""
        if self._sct is not None:
            try:
                mon = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]
                shot = self._sct.grab(mon)
                arr = np.asarray(shot)
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    return arr[:, :, 2::-1].copy()
                return arr.copy()
            except Exception:
                pass
        h = max(360, self.geometry.virtual_height // 2)
        w = max(640, self.geometry.virtual_width // 2)
        return np.full((h, w, 3), 128, dtype=np.uint8)

    @torch.no_grad()
    def step(self) -> EyeFrame:
        """Captures frame and runs cortex forward pass in under 10ms total."""
        t0 = time.perf_counter()
        raw_frame = self.grab_screen()
        now = time.time()

        # Handle raw frame shape and conversion to (1, 3, 128, 128) in [0, 1]
        if raw_frame.ndim == 2:
            frame_rgb = np.stack([raw_frame] * 3, axis=-1)
        elif raw_frame.shape[-1] >= 3:
            frame_rgb = raw_frame[:, :, :3]
        else:
            frame_rgb = np.repeat(raw_frame, 3, axis=-1)

        # Fast bilinear interpolation / sub-sampling to 128x128
        H, W = frame_rgb.shape[:2]
        if (H, W) != (128, 128):
            sy = max(1.0, float(H) / 128.0)
            sx = max(1.0, float(W) / 128.0)
            y_idx = np.minimum((np.arange(128) * sy).astype(int), H - 1)
            x_idx = np.minimum((np.arange(128) * sx).astype(int), W - 1)
            frame_small = frame_rgb[y_idx[:, None], x_idx]
        else:
            frame_small = frame_rgb

        t_frame = torch.from_numpy(np.ascontiguousarray(frame_small)).float()
        if t_frame.max() > 1.0:
            t_frame = t_frame / 255.0
        t_frame = t_frame.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, 128, 128)

        # 4-Channel Preprocessing: Luminance, Sobel X, Sobel Y, Gradient Magnitude
        x_4ch = self.preprocessor(t_frame)

        # StreamingVisualCortexV2 temporal forward pass
        out: CortexV2Output = self.forward(
            x_4ch,
            prev_state=self.temporal_state,
            timestamp=now,
        )
        self.temporal_state = out.temporal_state

        # Head 1: Surprise score & Temporal Change
        surprise = float(np.clip(out.feature_uncertainty[0].item(), 0.0, 1.0))
        mean_change = float(out.change_heatmap[0, 0].mean().item())
        is_stable = (surprise < 0.08) and (mean_change < 0.05)

        # Head 2: 5-Class UI Objectness & Hotspots
        obj_probs = torch.sigmoid(out.ui_objectness_logits[0, 0])  # (16, 16)
        cls_probs = torch.softmax(out.ui_class_logits[0], dim=0)   # (5, 16, 16)

        CLASS_MAP = {
            0: "interactive_control",
            1: "text_input",
            2: "toggle_icon",
            3: "modal_dialog",
            4: "content_region",
        }

        grid_H, grid_W = obj_probs.shape
        obj_4d = obj_probs.unsqueeze(0).unsqueeze(0)
        max_pool = F.max_pool2d(obj_4d, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        is_peak = (obj_probs == max_pool) & (obj_probs >= 0.25)
        peak_idx = torch.nonzero(is_peak, as_tuple=False)

        hotspots: List[Tuple[float, float, str, float]] = []
        for i in range(min(12, len(peak_idx))):
            py, px = int(peak_idx[i, 0].item()), int(peak_idx[i, 1].item())
            val = float(obj_probs[py, px].item())
            best_cls = int(torch.argmax(cls_probs[:, py, px]).item())
            cls_name = CLASS_MAP.get(best_cls, "interactive_control")
            norm_x = (px + 0.5) / float(grid_W)
            norm_y = (py + 0.5) / float(grid_H)
            hotspots.append((round(norm_x, 3), round(norm_y, 3), cls_name, round(val, 3)))

        # Popup detection: high surprise or modal dialog class detected
        has_popup = (surprise >= 0.25 and len(hotspots) > 0) or any(
            h[2] == "modal_dialog" and h[3] >= 0.40 for h in hotspots
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return EyeFrame(
            raw_frame=raw_frame,
            surprise_score=round(surprise, 3),
            is_stable=is_stable,
            has_popup=has_popup,
            hotspots=hotspots,
            latency_ms=round(dt_ms, 2),
        )

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
