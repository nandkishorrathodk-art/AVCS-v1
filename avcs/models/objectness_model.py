"""Class-Agnostic UI Objectness Model Architecture & Loss Formulation.

Provides pure-pixel, class-agnostic UI objectness detection for AVCS System 1:
- Input: [B, 3, 128, 128] RGB raw pixel buffers (or [B, 4, 128, 128] preprocessed tensors)
- Preprocessor: EdgeLuminancePreprocessor (ITU-R BT.601 + SobelX + SobelY + GradientMagnitude)
- Backbone: RealTimeCorticalBackbone4Ch (~10k parameters, 4 channels in, downsampling to 16x16)
- Global Presence Head: Linear head on pooled features -> [B, 1] logit (blank/wallpaper rejection)
- Dense Objectness Heatmap Head: ConvBNAct + 1x1 Conv -> [B, 1, 16, 16] interactive centroid logits
- Bounding Box Regressor Head: ConvBNAct + 1x1 Conv -> [B, 4, 16, 16] (cx_offset, cy_offset, w, h in [0, 1])
- Output contract: UIObjectnessPrediction dataclass matching PROJECT.md §6.3.

Conforms strictly to the Pure-Pixel Security Contract: zero OCR, zero UIA, zero DOM imports.
Forward pass latency p50 <= 10.0ms on standard 4-core CPU at 128x128 resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.perception.edge_preprocessor import EdgeLuminancePreprocessor


@dataclass(frozen=True)
class UIObjectnessPrediction:
    """Canonical UI Objectness Prediction dataclass conforming to PROJECT.md §6.3."""

    global_presence_prob: float                               # [0.0, 1.0]
    candidate_boxes: List[Tuple[float, float, float, float]]   # Normalized [x1, y1, x2, y2]
    candidate_centroids: List[Tuple[float, float]]             # Normalized (cx, cy)
    objectness_confidences: List[float]                       # [0.0, 1.0] per candidate
    is_blank_rejected: bool                                   # True if global_presence_prob < 0.35
    heatmap: torch.Tensor                                     # [16, 16] spatial objectness map

    @property
    def num_candidates(self) -> int:
        """Number of detected candidate bounding boxes."""
        return len(self.candidate_boxes)


class ConvBNAct(nn.Module):
    """Standard Conv2d + BatchNorm2d + GELU block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """Channel attention mechanism to recalibrate spatial features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class InvertedResidual(nn.Module):
    """Inverted residual block with depthwise separation and SE attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: float = 1.5,
    ) -> None:
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        hidden = int(round(in_ch * expand_ratio))

        layers: List[nn.Module] = []
        if expand_ratio != 1.0:
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, padding=0))

        # Depthwise Conv
        layers.append(
            ConvBNAct(
                hidden,
                hidden,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden,
            )
        )
        # Squeeze-and-Excitation
        layers.append(SqueezeExcitation(hidden))
        # Pointwise Linear
        layers.append(ConvBNAct(hidden, out_ch, kernel_size=1, padding=0, act=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.block(x)
        return self.block(x)


class RealTimeCorticalBackbone4Ch(nn.Module):
    """Real-Time 4-Channel Cortical Feature Backbone (~10k parameters).

    Takes [B, 4, 128, 128] preprocessed edge/luminance frames and downsamples to
    [B, 128, 16, 16] conscious feature maps in sub-3ms on CPU.

    Downsampling Stages:
      - Stem: 128x128 -> 64x64 (ConvBNAct, in_ch=4, out_ch=16, stride=2)
      - Stage 1: 64x64 -> 32x32 (InvertedResidual, in_ch=16, out_ch=24, stride=2)
      - Stage 2: 32x32 -> 16x16 (InvertedResidual, in_ch=24, out_ch=32, stride=2)
      - Proj: 16x16 -> 16x16 (ConvBNAct, in_ch=32, out_ch=feature_dim=128, kernel=1)
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # 128x128 -> 64x64
        self.stem = ConvBNAct(in_channels, 16, stride=2)
        # 64x64 -> 32x32
        self.stage1 = InvertedResidual(16, 24, stride=2, expand_ratio=1.5)
        # 32x32 -> 16x16
        self.stage2 = InvertedResidual(24, 32, stride=2, expand_ratio=1.5)
        # 16x16 -> feature_dim (128)
        self.proj = ConvBNAct(32, feature_dim, kernel_size=1, padding=0)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, 4, 128, 128] -> [B, feature_dim, 16, 16]."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.proj(x)


class GlobalPresenceHead(nn.Module):
    """Linear head on pooled feature map for blank/wallpaper rejection.

    Takes [B, feature_dim, 16, 16] features, applies adaptive average pooling to
    [B, feature_dim], and projects via linear layer to [B, 1] presence logit.
    """

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, feature_dim, 16, 16] -> [B, 1]."""
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.linear(pooled)


class DenseObjectnessHead(nn.Module):
    """Dense 16x16 objectness heatmap head for interactive UI centroids.

    Takes [B, feature_dim, 16, 16], applies ConvBNAct(feature_dim, 32, kernel_size=1) followed by
    1x1 Conv2d(32, 1) to produce [B, 1, 16, 16] interactive centroid logits in <1ms.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        mid_channels: int = 32,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = ConvBNAct(feature_dim, mid_channels, kernel_size=kernel_size, padding=padding)
        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # Prior initialization: initialize bias to -2.0 so sigmoid initially predicts ~0.12
        with torch.no_grad():
            if self.head.bias is not None:
                self.head.bias.fill_(-2.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, feature_dim, 16, 16] -> [B, 1, 16, 16]."""
        return self.head(self.conv(feat))


class BoxRegressorHead(nn.Module):
    """Bounding box regressor head predicting [cx_offset, cy_offset, w, h] in [0, 1].

    Takes [B, feature_dim, 16, 16], applies ConvBNAct(feature_dim, 32, kernel_size=1) followed by
    1x1 Conv2d(32, 4) and sigmoid activation to predict:
      - Channel 0: cx_offset in [0, 1] (local x offset within grid cell)
      - Channel 1: cy_offset in [0, 1] (local y offset within grid cell)
      - Channel 2: w in [0, 1] (normalized width across image)
      - Channel 3: h in [0, 1] (normalized height across image)
    """

    def __init__(
        self,
        feature_dim: int = 128,
        mid_channels: int = 32,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = ConvBNAct(feature_dim, mid_channels, kernel_size=kernel_size, padding=padding)
        self.head = nn.Conv2d(mid_channels, 4, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, feature_dim, 16, 16] -> [B, 4, 16, 16] in [0.0, 1.0]."""
        logits = self.head(self.conv(feat))
        return torch.sigmoid(logits)


class UIObjectnessModel(nn.Module):
    """End-to-End Class-Agnostic UI Objectness Perception Model.

    Combines:
      1. EdgeLuminancePreprocessor: converts raw RGB [B, 3, 128, 128] into 4-channel representation.
      2. RealTimeCorticalBackbone4Ch: ~10k param backbone downsampling to [B, 128, 16, 16].
      3. GlobalPresenceHead: linear head for blank/wallpaper rejection.
      4. DenseObjectnessHead: 16x16 centroid heatmap head.
      5. BoxRegressorHead: 16x16 bounding box offsets and dimensions.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 128,
        presence_threshold: float = 0.70,
        heatmap_threshold: float = 0.35,
        top_k: int = 20,
        preprocessor: Optional[EdgeLuminancePreprocessor] = None,
        edge_rejection: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.presence_threshold = float(presence_threshold)
        self.heatmap_threshold = float(heatmap_threshold)
        self.top_k = int(top_k)
        self.edge_rejection = bool(edge_rejection)

        # 1. 4-Channel Preprocessor (zero learned weights)
        self.preprocessor = (
            preprocessor if preprocessor is not None else EdgeLuminancePreprocessor()
        )

        # 2. Cortical Feature Backbone
        self.backbone = RealTimeCorticalBackbone4Ch(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )

        # 3. Task Heads
        self.presence_head = GlobalPresenceHead(feature_dim=feature_dim)
        self.objectness_head = DenseObjectnessHead(feature_dim=feature_dim)
        self.box_head = BoxRegressorHead(feature_dim=feature_dim)

    @property
    def total_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        return_prediction: bool = False,
    ) -> Union[Dict[str, torch.Tensor], UIObjectnessPrediction, List[UIObjectnessPrediction]]:
        """Forward pass of UIObjectnessModel.

        Args:
            x: Input image tensor of shape [B, 3, 128, 128] RGB, [B, 4, 128, 128],
               or unbatched 3D tensor [3, 128, 128] / [4, 128, 128].
            return_prediction: If True, executes candidate extraction and returns
               `UIObjectnessPrediction` dataclass(es). If False, returns raw output dict.

        Returns:
            Dict containing:
              - 'presence_logit': [B, 1] float32 logit
              - 'presence_prob': [B, 1] float32 in [0, 1]
              - 'heatmap_logits': [B, 1, 16, 16] float32 logits
              - 'heatmap': [B, 1, 16, 16] float32 in [0, 1]
              - 'box_preds': [B, 4, 16, 16] float32 in [0, 1]
              - 'features': [B, feature_dim, 16, 16] float32
              - 'four_ch': [B, 4, 128, 128] preprocessed tensor
        """
        was_3d = False
        if x.ndim == 3:
            was_3d = True
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        # If 3-channel RGB, run through 4-channel preprocessor
        if x.shape[1] == 3:
            four_ch = self.preprocessor(x)
        elif x.shape[1] == 4:
            four_ch = x
        else:
            raise ValueError(
                f"Expected 3 or 4 channels at dimension 1, got {x.shape[1]} channels"
            )

        feat = self.backbone(four_ch)
        presence_logit = self.presence_head(feat)
        presence_prob = torch.sigmoid(presence_logit)
        heatmap_logits = self.objectness_head(feat)
        heatmap = torch.sigmoid(heatmap_logits)
        box_preds = self.box_head(feat)

        if return_prediction:
            return self._extract_predictions(
                presence_prob=presence_prob,
                heatmap=heatmap,
                box_preds=box_preds,
                was_single=was_3d or (x.shape[0] == 1),
                four_ch=four_ch,
            )

        return {
            "presence_logit": presence_logit,
            "presence_prob": presence_prob,
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "box_preds": box_preds,
            "features": feat,
            "four_ch": four_ch,
        }

    def predict(
        self,
        x: torch.Tensor,
        presence_threshold: Optional[float] = None,
        heatmap_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        edge_rejection: Optional[bool] = None,
    ) -> Union[UIObjectnessPrediction, List[UIObjectnessPrediction]]:
        """Inference method returning UIObjectnessPrediction dataclass matching PROJECT.md §6.3.

        Args:
            x: RGB image tensor [3, 128, 128] or [B, 3, 128, 128] (or 4-channel equivalent).
            presence_threshold: Threshold for blank-screen rejection (default: self.presence_threshold).
            heatmap_threshold: Minimum heatmap confidence for peak detection (default: self.heatmap_threshold).
            top_k: Maximum candidate boxes to return (default: self.top_k).
            edge_rejection: Whether to reject unstructured blank/noise screens based on gradient energy.

        Returns:
            UIObjectnessPrediction for a single image, or List[UIObjectnessPrediction] for batches.
        """
        was_3d = False
        if x.ndim == 3:
            was_3d = True
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        with torch.no_grad():
            outputs = self.forward(x, return_prediction=False)
            presence_prob = outputs["presence_prob"]
            heatmap = outputs["heatmap"]
            box_preds = outputs["box_preds"]
            four_ch = outputs.get("four_ch")

            return self._extract_predictions(
                presence_prob=presence_prob,
                heatmap=heatmap,
                box_preds=box_preds,
                was_single=was_3d or (x.shape[0] == 1),
                presence_threshold=presence_threshold,
                heatmap_threshold=heatmap_threshold,
                top_k=top_k,
                four_ch=four_ch,
                edge_rejection=edge_rejection,
            )

    def _extract_predictions(
        self,
        presence_prob: torch.Tensor,
        heatmap: torch.Tensor,
        box_preds: torch.Tensor,
        was_single: bool,
        presence_threshold: Optional[float] = None,
        heatmap_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        four_ch: Optional[torch.Tensor] = None,
        edge_rejection: Optional[bool] = None,
    ) -> Union[UIObjectnessPrediction, List[UIObjectnessPrediction]]:
        """Internal helper to decode heatmap and box predictions into UIObjectnessPrediction."""
        p_thresh = (
            float(presence_threshold)
            if presence_threshold is not None
            else self.presence_threshold
        )
        h_thresh = (
            float(heatmap_threshold)
            if heatmap_threshold is not None
            else self.heatmap_threshold
        )
        k_max = int(top_k) if top_k is not None else self.top_k
        do_edge_reject = (
            bool(edge_rejection)
            if edge_rejection is not None
            else getattr(self, "edge_rejection", True)
        )

        b_size = heatmap.shape[0]
        predictions: List[UIObjectnessPrediction] = []

        # 3x3 MaxPool for 2D spatial Non-Maximum Suppression
        pooled_heat = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)

        for b in range(b_size):
            p_prob = float(presence_prob[b, 0].item())
            if p_thresh <= 0.0:
                is_blank = False
            else:
                is_blank = bool(p_prob < p_thresh)
                # Enhance blank rejection: check gradient energy / edge activity from 4th channel
                if not is_blank and do_edge_reject and four_ch is not None and four_ch.shape[1] >= 4:
                    mag_b = four_ch[b, 3]
                    mean_mag = float(mag_b.mean().item())
                    max_mag = float(mag_b.max().item())
                    contrast_ratio = max_mag / (mean_mag + 1e-6)
                    # Flat screens, gradients, or random unstructured noise lack sharp UI edges
                    if max_mag < 0.18 or contrast_ratio < 6.0:
                        is_blank = True

            hmap_b = heatmap[b, 0]  # [16, 16]
            box_b = box_preds[b]    # [4, 16, 16]

            if is_blank:
                # Blank screen / static wallpaper: suppress all candidates
                pred = UIObjectnessPrediction(
                    global_presence_prob=p_prob,
                    candidate_boxes=[],
                    candidate_centroids=[],
                    objectness_confidences=[],
                    is_blank_rejected=True,
                    heatmap=hmap_b.detach().cpu(),
                )
            else:
                # Local peak detection: cell must equal 3x3 max and exceed heatmap_threshold
                is_peak = (hmap_b == pooled_heat[b, 0]) & (hmap_b >= h_thresh)
                peak_coords = torch.nonzero(is_peak, as_tuple=False)  # [N, 2] -> (row, col)

                candidate_boxes: List[Tuple[float, float, float, float]] = []
                candidate_centroids: List[Tuple[float, float]] = []
                objectness_confidences: List[float] = []

                if peak_coords.shape[0] > 0:
                    peak_confidences = [
                        float(hmap_b[r, c].item()) for r, c in peak_coords
                    ]
                    # Sort peaks descending by confidence
                    sorted_order = sorted(
                        range(len(peak_confidences)),
                        key=lambda idx: peak_confidences[idx],
                        reverse=True,
                    )[:k_max]

                    for idx in sorted_order:
                        r = int(peak_coords[idx, 0].item())
                        c = int(peak_coords[idx, 1].item())
                        conf = peak_confidences[idx]

                        cx_off = float(box_b[0, r, c].item())
                        cy_off = float(box_b[1, r, c].item())
                        w = float(box_b[2, r, c].item())
                        h = float(box_b[3, r, c].item())

                        # Centroid in normalized [0, 1] coordinate space
                        cx = (float(c) + cx_off) / 16.0
                        cy = (float(r) + cy_off) / 16.0
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))

                        # Bounding box [x1, y1, x2, y2] in normalized [0, 1]
                        x1 = max(0.0, min(1.0, cx - w / 2.0))
                        y1 = max(0.0, min(1.0, cy - h / 2.0))
                        x2 = max(0.0, min(1.0, cx + w / 2.0))
                        y2 = max(0.0, min(1.0, cy + h / 2.0))

                        candidate_boxes.append((x1, y1, x2, y2))
                        candidate_centroids.append((cx, cy))
                        objectness_confidences.append(conf)

                pred = UIObjectnessPrediction(
                    global_presence_prob=p_prob,
                    candidate_boxes=candidate_boxes,
                    candidate_centroids=candidate_centroids,
                    objectness_confidences=objectness_confidences,
                    is_blank_rejected=False,
                    heatmap=hmap_b.detach().cpu(),
                )

            predictions.append(pred)

        if was_single and len(predictions) == 1:
            return predictions[0]
        return predictions


def build_objectness_targets(
    target_boxes: torch.Tensor,
    target_centroids: torch.Tensor,
    box_mask: torch.Tensor,
    has_target: torch.Tensor,
    grid_size: int = 16,
    sigma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Build ground-truth heatmap and box targets from UnifiedBatchDict.

    Args:
        target_boxes: [B, max_objs, 4] normalized [x1, y1, x2, y2].
        target_centroids: [B, max_objs, 2] normalized (cx, cy).
        box_mask: [B, max_objs] bool (True for valid objects).
        has_target: [B] float32 (1.0 for UI screens, 0.0 for blanks).
        grid_size: Spatial grid dimension (default: 16).
        sigma: Standard deviation of Gaussian heatmap splat (default: 1.0).

    Returns:
        Dict containing:
          - 'target_heatmap': [B, 1, 16, 16] Gaussian centroid heatmap in [0.0, 1.0]
          - 'target_boxes_map': [B, 4, 16, 16] normalized [cx_offset, cy_offset, w, h]
          - 'pos_mask': [B, 16, 16] boolean mask for positive cells
    """
    b_size = target_boxes.shape[0]
    device = target_boxes.device

    target_heatmap = torch.zeros((b_size, 1, grid_size, grid_size), dtype=torch.float32, device=device)
    target_boxes_map = torch.zeros((b_size, 4, grid_size, grid_size), dtype=torch.float32, device=device)
    pos_mask = torch.zeros((b_size, grid_size, grid_size), dtype=torch.bool, device=device)

    radius = max(1, int(round(2.0 * sigma)))

    for b in range(b_size):
        if float(has_target[b].item()) < 0.5:
            # Blank screen: heatmap remains zeros, pos_mask remains False
            continue

        num_objs = target_boxes.shape[1]
        for k in range(num_objs):
            if not bool(box_mask[b, k].item()):
                continue

            cx = float(target_centroids[b, k, 0].item())
            cy = float(target_centroids[b, k, 1].item())
            x1 = float(target_boxes[b, k, 0].item())
            y1 = float(target_boxes[b, k, 1].item())
            x2 = float(target_boxes[b, k, 2].item())
            y2 = float(target_boxes[b, k, 3].item())

            w = max(1e-4, min(1.0, x2 - x1))
            h = max(1e-4, min(1.0, y2 - y1))

            # Grid cell assignment
            gx = min(grid_size - 1, max(0, int(cx * grid_size)))
            gy = min(grid_size - 1, max(0, int(cy * grid_size)))

            cx_off = max(0.0, min(1.0, cx * grid_size - float(gx)))
            cy_off = max(0.0, min(1.0, cy * grid_size - float(gy)))

            pos_mask[b, gy, gx] = True
            target_boxes_map[b, 0, gy, gx] = cx_off
            target_boxes_map[b, 1, gy, gx] = cy_off
            target_boxes_map[b, 2, gy, gx] = w
            target_boxes_map[b, 3, gy, gx] = h

            # Gaussian centroid splat
            cgx = cx * grid_size
            cgy = cy * grid_size
            r_min = max(0, gy - radius)
            r_max = min(grid_size - 1, gy + radius)
            c_min = max(0, gx - radius)
            c_max = min(grid_size - 1, gx + radius)

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    d2 = (float(c) + 0.5 - cgx) ** 2 + (float(r) + 0.5 - cgy) ** 2
                    val = math.exp(-d2 / (2.0 * sigma * sigma))
                    if val > target_heatmap[b, 0, r, c].item():
                        target_heatmap[b, 0, r, c] = val

    target_heatmap = torch.clamp(target_heatmap, 0.0, 1.0)

    return {
        "target_heatmap": target_heatmap,
        "target_boxes_map": target_boxes_map,
        "pos_mask": pos_mask,
    }


def compute_ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Vectorized Complete IoU (CIoU) loss for normalized bounding boxes.

    Args:
        pred_boxes: [N, 4] float tensor of (cx, cy, w, h) in [0, 1].
        target_boxes: [N, 4] float tensor of (cx, cy, w, h) in [0, 1].
        eps: Small epsilon for numerical stability.

    Returns:
        [N] float tensor of CIoU loss (1.0 - CIoU) in [0.0, 2.0].
    """
    if pred_boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.float32, device=pred_boxes.device)

    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2.0
    p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2.0
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2.0
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2.0

    t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2.0
    t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2.0
    t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2.0
    t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2.0

    # Intersection
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    # Union
    p_area = torch.clamp(pred_boxes[:, 2], min=eps) * torch.clamp(pred_boxes[:, 3], min=eps)
    t_area = torch.clamp(target_boxes[:, 2], min=eps) * torch.clamp(target_boxes[:, 3], min=eps)
    union_area = p_area + t_area - inter_area + eps
    iou = inter_area / union_area

    # Smallest enclosing box
    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Centroid distance
    rho2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # Aspect ratio consistency term
    v = (4.0 / (math.pi ** 2)) * torch.pow(
        torch.atan(target_boxes[:, 2] / torch.clamp(target_boxes[:, 3], min=eps))
        - torch.atan(pred_boxes[:, 2] / torch.clamp(pred_boxes[:, 3], min=eps)),
        2,
    )
    with torch.no_grad():
        alpha = v / torch.clamp(1.0 - iou + v, min=eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return 1.0 - torch.clamp(ciou, min=-1.0, max=1.0)


class UIObjectnessLoss(nn.Module):
    """Multi-task loss formulation for Class-Agnostic UI Objectness.

    Total Loss:
      L_total = lambda_pres * L_presence + lambda_heat * L_heatmap + lambda_box * L_box

    Terms:
      - L_presence: BCE with focal scaling on presence logit (0.0 for blanks, 1.0 for UI screens)
      - L_heatmap: Weighted Gaussian centroid heatmap MSE loss
      - L_box: Complete IoU (CIoU) + L1 loss over positive centroid cells
    """

    def __init__(
        self,
        presence_weight: float = 1.0,
        heatmap_weight: float = 2.0,
        box_weight: float = 1.0,
        presence_focal_gamma: float = 2.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.presence_weight = float(presence_weight)
        self.heatmap_weight = float(heatmap_weight)
        self.box_weight = float(box_weight)
        self.presence_focal_gamma = float(presence_focal_gamma)
        self.sigma = float(sigma)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            model_outputs: Dict from UIObjectnessModel.forward containing:
               'presence_logit', 'heatmap', 'box_preds'.
            batch_dict: Dict from UnifiedBatchDict containing:
               'target_boxes', 'target_centroids', 'box_mask', 'has_target'.

        Returns:
            Dict containing:
              - 'loss': total scalar loss
              - 'loss_presence': scalar presence focal BCE loss
              - 'loss_heatmap': scalar heatmap MSE loss
              - 'loss_box': scalar box CIoU + L1 loss
        """
        presence_logit = model_outputs["presence_logit"]
        heatmap = model_outputs["heatmap"]
        box_preds = model_outputs["box_preds"]

        target_boxes = batch_dict["target_boxes"]
        target_centroids = batch_dict["target_centroids"]
        box_mask = batch_dict["box_mask"]
        has_target = batch_dict["has_target"].view(-1, 1).float()

        # 1. Global Presence Loss: Binary Cross-Entropy with focal scaling
        p = torch.sigmoid(presence_logit)
        p_t = p * has_target + (1.0 - p) * (1.0 - has_target)
        focal_weight = torch.pow(1.0 - p_t, self.presence_focal_gamma).detach()
        bce = F.binary_cross_entropy_with_logits(presence_logit, has_target, reduction="none")
        loss_presence = (focal_weight * bce).mean()

        # 2. Build 16x16 Heatmap & Box Ground Truth
        targets = build_objectness_targets(
            target_boxes=target_boxes,
            target_centroids=target_centroids,
            box_mask=box_mask,
            has_target=batch_dict["has_target"],
            grid_size=16,
            sigma=self.sigma,
        )
        target_heatmap = targets["target_heatmap"]
        target_boxes_map = targets["target_boxes_map"]
        pos_mask = targets["pos_mask"]

        # 2. Heatmap MSE Loss (positive centroids weighted 5x)
        heat_weights = 1.0 + 4.0 * (target_heatmap > 0.1).float()
        loss_heatmap = (heat_weights * torch.pow(heatmap - target_heatmap, 2)).mean()

        # 3. Box Regressor Loss (CIoU + L1 on positive cells)
        if pos_mask.sum() > 0:
            # Positive predicted offsets and dimensions
            pred_pos = box_preds.permute(0, 2, 3, 1)[pos_mask]       # [N_pos, 4]
            tgt_pos = target_boxes_map.permute(0, 2, 3, 1)[pos_mask]  # [N_pos, 4]

            b_idx, y_idx, x_idx = torch.where(pos_mask)
            pred_cx = (x_idx.float() + pred_pos[:, 0]) / 16.0
            pred_cy = (y_idx.float() + pred_pos[:, 1]) / 16.0
            tgt_cx = (x_idx.float() + tgt_pos[:, 0]) / 16.0
            tgt_cy = (y_idx.float() + tgt_pos[:, 1]) / 16.0

            pred_boxes_norm = torch.stack(
                [pred_cx, pred_cy, pred_pos[:, 2], pred_pos[:, 3]], dim=1
            )
            tgt_boxes_norm = torch.stack(
                [tgt_cx, tgt_cy, tgt_pos[:, 2], tgt_pos[:, 3]], dim=1
            )

            ciou_loss = compute_ciou_loss(pred_boxes_norm, tgt_boxes_norm).mean()
            l1_loss = F.l1_loss(pred_pos, tgt_pos)
            loss_box = ciou_loss + l1_loss
        else:
            # Maintain backward gradient computation graph when batch has zero positive cells
            loss_box = 0.0 * box_preds.sum()

        loss_total = (
            self.presence_weight * loss_presence
            + self.heatmap_weight * loss_heatmap
            + self.box_weight * loss_box
        )

        return {
            "loss": loss_total,
            "loss_presence": loss_presence,
            "loss_heatmap": loss_heatmap,
            "loss_box": loss_box,
        }
