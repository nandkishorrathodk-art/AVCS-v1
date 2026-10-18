"""AVCS remote client and server subsystem.

Provides:
- A cloud inference server that keeps model weights on the server.
- A legacy visual-inference client that sends captured frames to the server.
- A separate text-only System 2 client for local System 1 observations.
"""
from __future__ import annotations

from avcs.server.client import AVCSCloudClient
from avcs.server.system2_client import TextOnlySystem2Client

__all__ = ["AVCSCloudClient", "TextOnlySystem2Client"]
