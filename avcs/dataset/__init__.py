"""AVCS Dataset Module: synthetic UI desktop generation and PyTorch data loading."""
from avcs.dataset.loader import UIGroundingDataset, ui_grounding_collate_fn

__all__ = [
    "UIGroundingDataset",
    "ui_grounding_collate_fn",
]
