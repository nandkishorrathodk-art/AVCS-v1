"""Legacy shim for avcs.ace.local_model.

Re-exports LocalVAMClient and EmbeddedVAMRuntime from canonical avcs.models package.
"""
from __future__ import annotations

import avcs.models.local_model as _local_model
from avcs.models.local_model import *  # noqa: F401, F403
from avcs.models.local_model import AceClientError, EmbeddedVAMRuntime, LocalVAMClient

LocalVAMEngine = LocalVAMClient

__all__ = [
    "AceClientError",
    "EmbeddedVAMRuntime",
    "LocalVAMClient",
    "LocalVAMEngine",
]


def __getattr__(name: str):
    return getattr(_local_model, name)

