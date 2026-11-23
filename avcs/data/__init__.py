"""AVCS Data Department.

Pure-RGB dataset loaders, LeRobot v3.0 schemas, trajectory cleaner,
QA doctor, and data mixture generators.
Strictly Linux/Kaggle import safe (never imports Windows OS or win32 modules).
"""
from __future__ import annotations

from avcs.data.lerobot_v3 import (
    LeRobotV3Dataset,
    LeRobotV3TrajectoryEpisode,
    Phase4DataMixtureDataset,
    QADoctor,
    TrajectoryCleaner,
    generate_golden_diagnostic_dataset,
    make_lerobot_v3_dataloader,
    normalize_open_source_action,
    normalize_open_source_point,
)
from avcs.data.loader import (
    FastUIGroundingDataset,
    GroundingSample,
    UIGroundingDataset,
    fast_ui_grounding_collate_fn,
    make_fast_ui_grounding_dataloader,
    make_ui_grounding_dataloader,
    ui_grounding_collate_fn,
)

__all__ = [
    "FastUIGroundingDataset",
    "GroundingSample",
    "LeRobotV3Dataset",
    "LeRobotV3TrajectoryEpisode",
    "Phase4DataMixtureDataset",
    "QADoctor",
    "TrajectoryCleaner",
    "UIGroundingDataset",
    "fast_ui_grounding_collate_fn",
    "generate_golden_diagnostic_dataset",
    "make_fast_ui_grounding_dataloader",
    "make_lerobot_v3_dataloader",
    "make_ui_grounding_dataloader",
    "normalize_open_source_action",
    "normalize_open_source_point",
    "ui_grounding_collate_fn",
]
