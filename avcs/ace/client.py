"""Legacy shim for avcs.ace.client.

Re-exports CustomLLMClient and AceClientError from canonical avcs.runtime package.
"""
from __future__ import annotations

import avcs.runtime.client as _client
from avcs.runtime.client import *  # noqa: F401, F403
from avcs.runtime.client import (
    AceClientError,
    CustomLLMClient,
    repair_truncated_json,
)

__all__ = [
    "AceClientError",
    "CustomLLMClient",
    "repair_truncated_json",
]


def __getattr__(name: str):
    return getattr(_client, name)

