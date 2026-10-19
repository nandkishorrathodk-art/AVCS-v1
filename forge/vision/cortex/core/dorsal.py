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

from typing import List

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

# Classes that represent a point the hand can act on. Derived from the class
# list so adding an affordance class cannot leave this set stale.
INTERACTIVE_CLASSES = frozenset(
    c for c in AFFORDANCE_CLASSES if c != "static"
)


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
