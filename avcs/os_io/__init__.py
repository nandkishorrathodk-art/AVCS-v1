"""AVCS OS I/O Department.

Provides Windows hardware, display, DPI awareness, input actuation,
and Dynamic Island HUD overlays.
"""
from __future__ import annotations

from avcs.os_io.computer import (
    ExecutionSafetyFirewall,
    MonitorInfo,
    Observation,
    SafetyCheckResult,
    SafetyRiskLevel,
    WindowsComputer,
)
from avcs.os_io.hud import AVCSVisualHUD

WindowsComputerAdapter = WindowsComputer

__all__ = [
    "AVCSVisualHUD",
    "ExecutionSafetyFirewall",
    "MonitorInfo",
    "Observation",
    "SafetyCheckResult",
    "SafetyRiskLevel",
    "WindowsComputer",
    "WindowsComputerAdapter",
]
