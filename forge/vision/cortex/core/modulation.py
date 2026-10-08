"""Top-Down Language Modulation for Non-Generative Visual Grounding.

In cognitive neuroscience, the prefrontal cortex modulates early visual cortex
via top-down attention (e.g. searching for a red apple primes red-sensitive neurons).

In AVCS, language guides visual attention without generating any text tokens:
1. An incoming language query (or precomputed embedding) is projected into
   the visual feature space.
2. A cross-attention correlation computes a dense 2D spatial heatmap:
   Relevance(x, y) = Softmax(Visual(x, y) . Text_Query / sqrt(D)).
3. The model pinpoints where the query resides on the screen in a single forward pass.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModulation(nn.Module):
    """Zero-generation language conditioning over visual feature maps."""

    def __init__(self, visual_dim: int = 128, text_dim: int = 128) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # Project text embedding to match visual channel dimensions
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )

        # Spatial temperature scaling
        self.scale = visual_dim ** -0.5

        # Coordinate regression head from language-visual cross attention
        self.coord_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(visual_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2), # (norm_x, norm_y) in [0, 1]
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute language-guided relevance map and direct target coordinate.

        Parameters
        ----------
        visual_features: (B, visual_dim, H, W)
        text_embedding:  (B, text_dim)

        Returns
        -------
        relevance_map: (B, 1, H, W) - normalized spatial heatmap
        target_coords: (B, 2)      - predicted (x, y) normalized coordinates
        """
        B, C, H, W = visual_features.shape

        # 1. Project text embedding: (B, visual_dim)
        t_proj = self.text_proj(text_embedding) # (B, C)

        # 2. Compute spatial dot-product alignment
        # (B, C, H, W) * (B, C, 1, 1) -> sum along C -> (B, 1, H, W)
        t_expanded = t_proj.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        sim = (visual_features * t_expanded).sum(dim=1, keepdim=True) * self.scale # (B, 1, H, W)

        # 3. Spatial Softmax Heatmap
        flat_sim = sim.view(B, 1, -1)
        relevance_map = F.softmax(flat_sim, dim=-1).view(B, 1, H, W)

        # 4. Modulated visual features for direct coordinate regression
        modulated_features = visual_features * (1.0 + relevance_map)
        target_coords = self.coord_regressor(modulated_features)

        return relevance_map, target_coords

