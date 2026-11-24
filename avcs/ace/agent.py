"""Legacy shim for avcs.ace.agent.

Re-exports AceAgent from canonical avcs.runtime package.
"""
from __future__ import annotations

import avcs.runtime.agent as _agent
from avcs.runtime.agent import *  # noqa: F401, F403
from avcs.runtime.agent import AceAgent, AceAgentSession

__all__ = [
    "AceAgent",
    "AceAgentSession",
]


def __getattr__(name: str):
    return getattr(_agent, name)

