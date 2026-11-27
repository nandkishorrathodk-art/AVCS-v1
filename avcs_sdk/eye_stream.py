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
from typing import Dict, Generator, List, Optional, Tuple

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
    repo_root = os.path.dirname(pkg_dir)

    search_dirs = [
        os.path.join(pkg_dir, "weights"),
        os.path.join(repo_root, "weights"),
        os.path.join(repo_root, "runs", "weights"),
        os.path.join(repo_root, "..", "splendid-babbage", "runs", "weights"),
        os.path.join(repo_root, "..", "avcs-sdk", "avcs", "weights"),
    ]

    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # Check known v2 weights filenames in search directories (including runs/weights/)
    v2_candidates = ["avcs_v2_weights.pt", "avcs_v2.pt", "avcs_v2.avcs"]
    if filename in v2_candidates or "avcs_v2" in filename:
        for candidate_name in v2_candidates:
            for d in search_dirs:
                candidate = os.path.join(d, candidate_name)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)

    return os.path.join(pkg_dir, "weights", filename)


DEFAULT_VAULT_V2 = _resolve_weights_path("avcs_v2.avcs")
DEFAULT_WEIGHTS_V2 = _resolve_weights_path("avcs_v2_weights.pt")
DEFAULT_WEIGHTS_V1 = _resolve_weights_path("avcs_v1.pt")
if os.path.exists(DEFAULT_WEIGHTS_V2):
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_V2
elif os.path.exists(DEFAULT_VAULT_V2):
    DEFAULT_WEIGHTS = DEFAULT_VAULT_V2
elif os.path.exists(DEFAULT_WEIGHTS_V1):
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_V1
else:
    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_V2


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
    hotspots: List[Tuple[float, float, str, float, float]]  # [(norm_x, norm_y, kind, obj_conf, cls_conf), ...]
    latency_ms: float                # Forward pass duration


# Backwards-compatible alias: the observation dataclass is named EyeFrame.
EyeObservation = EyeFrame

# The 5-class UI morphology vocabulary emitted in EyeFrame.hotspots. Consumers
# must branch on these exact tokens; testing for a token no producer emits
# silently disables the branch.
UI_CLASS_MAP: Dict[int, str] = {
    0: "interactive_control",
    1: "text_input",
    2: "toggle_icon",
    3: "modal_dialog",
    4: "content_region",
}
UI_CATEGORIES = frozenset(UI_CLASS_MAP.values())



class EyeStream(StreamingVisualCortexV2):
    """100 FPS Live Eye Stream with integrated StreamingVisualCortexV2 (400,765 parameters)."""

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        weights_path: Optional[str] = DEFAULT_WEIGHTS,
        device: str = "cpu",
        source: str = "screen",
        load_weights: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.to(self.device)
        self.geometry = self._query_geometry()
        self._sct = None
        self._source = source
        # `source="synthetic"` forces the deterministic 128-gray fallback buffer
        # instead of a real screen capture, so tests and headless runs never
        # depend on a display server.
        self._use_synthetic = str(source).lower() in ("synthetic", "virtual", "test")
        if not self._use_synthetic:
            self._init_capture()

        self.preprocessor = EdgeLuminancePreprocessor().to(self.device)
        self.temporal_state: Optional[CortexTemporalState] = None

        # Honest weight provenance: a checkpoint is only "loaded" if keys match.
        self.weights_loaded: bool = False
        self.weights_status: str = "random_initialised"
        if load_weights and weights_path:
            self.load_weights(weights_path)

        self.eval()

    def load_weights(self, weights_path: str, api_key: Optional[str] = None) -> bool:
        """Loads cortex weights if the checkpoint matches this architecture.

        Supports both unencrypted (.pt) checkpoints and encrypted (.avcs) vaults.
        The checkpoint is verified and loaded only if state dict keys match.
        Records an explicit status in ``weights_loaded`` / ``weights_status``
        and leaves the model untouched when nothing can be mapped.
        """
        if not os.path.exists(weights_path):
            self.weights_loaded = False
            self.weights_status = f"weights file not found: {weights_path}"
            return False

        is_vault = False
        try:
            with open(weights_path, "rb") as f:
                header = f.read(5)
            is_vault = (header == b"AVCS\x01")
        except Exception:
            header = b""
            is_vault = False

        if not is_vault and str(weights_path).endswith(".avcs"):
            # Only treat .avcs as encrypted vault unless it is clearly a plain PyTorch archive
            if header[:2] not in (b"PK", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
                is_vault = True

        if is_vault:
            try:
                from avcs.vault import InvalidKeyError, ModelVault, VaultSecurityError
            except Exception:
                try:
                    from .vault import InvalidKeyError, ModelVault, VaultSecurityError
                except Exception:
                    from avc.vault import InvalidKeyError, ModelVault, VaultSecurityError

            key = (
                api_key
                or os.environ.get("AVC_API_KEY")
                or os.environ.get("AVCS_API_KEY")
                or os.environ.get("AVCS_KEY")
            )
            if not key:
                self.weights_loaded = False
                self.weights_status = "Access Denied: Missing AVC_API_KEY for encrypted model vault (.avcs)"
                return False
            try:
                checkpoint = ModelVault.decrypt_to_ram(weights_path, api_key=key, map_location=self.device)
            except InvalidKeyError:
                self.weights_loaded = False
                self.weights_status = "Access Denied: Invalid AVC_API_KEY for model vault"
                return False
            except Exception as exc:
                self.weights_loaded = False
                self.weights_status = f"vault decryption failed: {exc}"
                return False
        else:
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            except Exception:
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
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            ckpt_dict = checkpoint["model"]
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
        if is_vault:
            try:
                from avcs.vault import apply_anti_dump_guard
                apply_anti_dump_guard(self)
            except Exception:
                try:
                    from .vault import apply_anti_dump_guard
                    apply_anti_dump_guard(self)
                except Exception:
                    pass
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
        if self._sct is not None and not getattr(self, "_use_synthetic", False):
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

    @property
    def capture_is_live(self) -> bool:
        """True only when frames come from a real screen, not the grey fallback.

        ``grab_screen`` returns a flat 128-grey buffer whenever ``mss`` is
        missing or the capture device fails. That buffer is indistinguishable
        from a real grey screen, so a caller that reported "live screen" would be
        claiming a capture it never made. This exposes the distinction so the
        provenance can be shown instead of assumed.
        """
        return self._sct is not None and not getattr(self, "_use_synthetic", False)

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

        # Head 1: Surprise score & Temporal Change calibrated to inter-frame optical change
        if hasattr(self, "_prev_features") and self._prev_features is not None:
            spatial_delta = float(torch.abs(out.temporal_state.previous_features - self._prev_features).mean().item())
        else:
            spatial_delta = 0.0
        self._prev_features = out.temporal_state.previous_features.clone()

        surprise = float(np.clip(spatial_delta * 10.0, 0.0, 1.0))
        is_stable = spatial_delta < 0.015

        # Head 2: 5-Class UI Objectness & Hotspots
        obj_probs = torch.sigmoid(out.ui_objectness_logits[0, 0])  # (16, 16)
        cls_probs = torch.softmax(out.ui_class_logits[0], dim=0)   # (5, 16, 16)

        grid_H, grid_W = obj_probs.shape
        obj_4d = obj_probs.unsqueeze(0).unsqueeze(0)
        max_pool = F.max_pool2d(obj_4d, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        is_peak = (obj_probs == max_pool) & (obj_probs >= 0.25)
        peak_idx = torch.nonzero(is_peak, as_tuple=False)

        hotspots: List[Tuple[float, float, str, float, float]] = []
        for i in range(min(12, len(peak_idx))):
            py, px = int(peak_idx[i, 0].item()), int(peak_idx[i, 1].item())
            obj_conf = float(obj_probs[py, px].item())
            best_cls = int(torch.argmax(cls_probs[:, py, px]).item())
            cls_conf = float(cls_probs[best_cls, py, px].item())
            cls_name = UI_CLASS_MAP.get(best_cls, "interactive_control")
            norm_x = (px + 0.5) / float(grid_W)
            norm_y = (py + 0.5) / float(grid_H)
            hotspots.append((round(norm_x, 3), round(norm_y, 3), cls_name, round(obj_conf, 3), round(cls_conf, 3)))

        # Popup detection: require physical optical delta (spatial_delta >= 0.04) AND cls_name == "modal_dialog" AND cls_conf >= 0.70
        has_popup = (spatial_delta >= 0.04) and any(
            h[2] == "modal_dialog" and (h[4] if len(h) > 4 else h[3]) >= 0.70 for h in hotspots
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
