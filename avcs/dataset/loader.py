"""Legacy shim for avcs.dataset.loader.

Re-exports UI grounding and LeRobot v3 datasets from canonical avcs.data package.
"""
from __future__ import annotations

from avcs.data.loader import *  # noqa: F401, F403
from avcs.data.lerobot_v3 import *  # noqa: F401, F403
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
    encode_queries,
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
    "encode_queries",
    "fast_ui_grounding_collate_fn",
    "generate_golden_diagnostic_dataset",
    "make_fast_ui_grounding_dataloader",
    "make_lerobot_v3_dataloader",
    "make_ui_grounding_dataloader",
    "normalize_open_source_action",
    "normalize_open_source_point",
    "ui_grounding_collate_fn",
]

import avcs.data.lerobot_v3 as _lerobot
import avcs.data.loader as _loader


def __getattr__(name: str):
    if hasattr(_loader, name):
        return getattr(_loader, name)
    return getattr(_lerobot, name)
