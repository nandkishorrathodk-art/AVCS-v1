"""Legacy shim for avcs.ace.computer.

Re-exports Windows computer actuation from canonical avcs.os_io package.
"""
from __future__ import annotations

import avcs.os_io.computer as _computer
from avcs.os_io.computer import *  # noqa: F401, F403
from avcs.os_io.computer import (
    INPUT,
    ExecutionSafetyFirewall,
    MonitorInfo,
    Observation,
    SafetyCheckResult,
    SafetyRiskLevel,
    WindowsComputer,
)

WindowsComputerAdapter = WindowsComputer

__all__ = [
    "ExecutionSafetyFirewall",
    "INPUT",
    "MonitorInfo",
    "Observation",
    "SafetyCheckResult",
    "SafetyRiskLevel",
    "WindowsComputer",
    "WindowsComputerAdapter",
]


def __getattr__(name: str):
    return getattr(_computer, name)

