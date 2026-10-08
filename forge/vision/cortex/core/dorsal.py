"""Dorsal Stream ("WHERE / HOW") - Spatial Affordances & Action Fields.

In primate neurobiology, the dorsal stream (parietal pathway) specializes in
spatial location, motion, and motor affordance ("How can I interact with this?").

For a computer screen, this module transforms visual features into a dense
affordance field, predicting where actions are physically possible:
- Clickable (Buttons, Links, Icons, Tabs)
- Typeable (Input fields, Textareas, Search boxes)
- Scrollable (Feeds, Long panels, Code editors)
- Draggable (Sliders, Splitters, Window bars)
- Static (Non-interactive text, backgrounds)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

AFFORDANCE_CLASSES = [
    "static",      # 0: non-interactive content
    "clickable",   # 1: buttons, links, icons, tabs
    "typeable",    # 2: input fields, search bars
    "scrollable",  # 3: scrollable containers, lists
    "draggable",   # 4: sliders, window frames, splitters
]


class DorsalAffordanceStream(nn.Module):
    """Predicts a dense spatial affordance probability tensor."""

    def __init__(self, in_dim: int = 128, num_classes: int = len(AFFORDANCE_CLASSES)) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Spatial refinement head: (B, in_dim, 16, 16) -> (B, num_classes, 32, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # Upsample 2x
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, num_classes, H_affordance, W_affordance)."""
        return self.head(visual_features)

    @torch.no_grad()
    def get_affordance_map(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities of shape (B, num_classes, H, W)."""
        logits = self.forward(visual_features)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def find_action_hotspots(
        self,
        visual_features: torch.Tensor,
        action_type: str = "clickable",
        top_k: int = 3,
        threshold: float = 0.4,
    ) -> List[Tuple[float, float, float]]:
        """Finds normalized (x, y, confidence) for top interactive points."""
        probs = self.get_affordance_map(visual_features)[0] # (num_classes, H, W)
        if action_type not in AFFORDANCE_CLASSES:
            action_type = "clickable"
        cls_idx = AFFORDANCE_CLASSES.index(action_type)
        heat = probs[cls_idx] # (H, W)

        H, W = heat.shape
        flat = heat.flatten()
        vals, indices = torch.topk(flat, k=min(top_k, flat.numel()))

        hotspots = []
        for val, idx in zip(vals.tolist(), indices.tolist()):
            if val < threshold:
                continue
            y_idx = idx // W
            x_idx = idx % W
            norm_x = (x_idx + 0.5) / W
            norm_y = (y_idx + 0.5) / H
            hotspots.append((float(norm_x), float(norm_y), float(val)))
        return hotspots

