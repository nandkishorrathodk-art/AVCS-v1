"""Ace-Control-Medium: Production-Grade Autonomous Pure-Vision GUI Computer-Use Autopilot."""

from avcs.ace.actions import Action, ActionKind
from avcs.ace.agent import AceAgent
from avcs.ace.client import AceClientError, CustomLLMClient
from avcs.ace.computer import Observation, WindowsComputer
from avcs.ace.planner import CognitivePlanner, ReflectionDiagnostic, SubGoal
from avcs.ace.session import AceSession, StepRecord

__all__ = [
    "Action",
    "ActionKind",
    "WindowsComputer",
    "Observation",
    "CustomLLMClient",
    "AceClientError",
    "AceAgent",
    "AceSession",
    "StepRecord",
    "AceRunner",
    "SafetyBreakoutError",
    "CognitivePlanner",
    "SubGoal",
    "ReflectionDiagnostic",
]


def __getattr__(name: str):
    if name == "AceRunner":
        from avcs.ace.runner import AceRunner
        return AceRunner
    if name == "SafetyBreakoutError":
        from avcs.ace.runner import SafetyBreakoutError
        return SafetyBreakoutError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

