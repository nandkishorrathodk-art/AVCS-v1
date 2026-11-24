"""Legacy shim for avcs.ace.planner.

Re-exports CognitivePlanner from canonical avcs.runtime.planner package.
"""
from __future__ import annotations

import avcs.runtime.planner as _planner
from avcs.runtime.planner import *  # noqa: F401, F403
from avcs.runtime.planner import (
    CognitivePlanner,
    ReflectionDiagnostic,
    SubGoal,
    _validate_no_coordinates,
)

__all__ = [
    "CognitivePlanner",
    "ReflectionDiagnostic",
    "SubGoal",
    "_validate_no_coordinates",
]


def __getattr__(name: str):
    return getattr(_planner, name)
