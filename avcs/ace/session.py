"""Legacy shim for avcs.ace.session.

Re-exports AceSession and StepRecord from canonical avcs.runtime package.
"""
from __future__ import annotations

import avcs.runtime.session as _session
from avcs.runtime.session import *  # noqa: F401, F403
from avcs.runtime.session import AceSession, StepRecord

__all__ = [
    "AceSession",
    "StepRecord",
]


def __getattr__(name: str):
    return getattr(_session, name)
