"""avcs.vlm: Local Vision-Language Foundation Model Engines for Pure-Pixel Perception."""
from __future__ import annotations

from .florence_engine import (
    TASK_CAPTION,
    TASK_DETAILED_CAPTION,
    TASK_MORE_DETAILED_CAPTION,
    TASK_OD,
    TASK_OCR,
    TASK_OCR_WITH_REGION,
    TASK_DENSE_REGION_CAPTION,
    FlorenceConfig,
    FlorenceDetection,
    FlorenceEngine,
    FlorenceOCRRegion,
    FlorenceResult,
)

__all__ = [
    "TASK_CAPTION",
    "TASK_DETAILED_CAPTION",
    "TASK_MORE_DETAILED_CAPTION",
    "TASK_OD",
    "TASK_OCR",
    "TASK_OCR_WITH_REGION",
    "TASK_DENSE_REGION_CAPTION",
    "FlorenceConfig",
    "FlorenceDetection",
    "FlorenceEngine",
    "FlorenceOCRRegion",
    "FlorenceResult",
]
