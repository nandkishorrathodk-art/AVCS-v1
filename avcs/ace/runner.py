"""Legacy shim for avcs.ace.runner.

Re-exports AceRunner and verifiers from canonical avcs.runtime package.
"""
from __future__ import annotations

import avcs.runtime.runner as _runner
from avcs.runtime.runner import *  # noqa: F401, F403
from avcs.runtime.runner import (
    AceRunner,
    DualStageVerificationResult,
    DualStageVerifier,
    SafetyBreakoutError,
    Stage1Result,
    Stage2Result,
    TrajectoryLogger,
    logger,
)

__all__ = [
    "AceRunner",
    "DualStageVerificationResult",
    "DualStageVerifier",
    "SafetyBreakoutError",
    "Stage1Result",
    "Stage2Result",
    "TrajectoryLogger",
    "logger",
]


def __getattr__(name: str):
    return getattr(_runner, name)
