"""AVCS Train Department.

Behavioral cloning (BC) and DAgger training orchestration,
Kaggle GPU automation, and loss optimization loops.
Strictly Linux/Kaggle import safe (never imports Windows OS or win32 modules).
"""
from __future__ import annotations

from avcs.train.train_vla import (
    VLAPolicyTrainingWrapper,
    run_phase4_scaled_bc_training,
)

__all__ = [
    "VLAPolicyTrainingWrapper",
    "run_phase4_scaled_bc_training",
]
