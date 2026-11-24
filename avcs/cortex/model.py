"""Legacy shim for avcs.cortex.model.

Re-exports VLA Cortex architectures from avcs.models and streaming cortex from avcs.streaming_v2.
"""
from __future__ import annotations

import avcs.models.vla_cortex as _vla
import avcs.streaming_v2.model as _stream_v2
from avcs.streaming_v2.model import *  # noqa: F401, F403
from avcs.models.vla_cortex import *  # noqa: F401, F403


def __getattr__(name: str):
    if hasattr(_vla, name):
        return getattr(_vla, name)
    return getattr(_stream_v2, name)

