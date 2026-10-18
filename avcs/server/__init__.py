"""AVCS Cloud Model Serving & Remote Client Subsystem.

Provides:
- Zero-leakage cloud inference server (FastAPI/Uvicorn) protecting proprietary neural weights.
- Lightweight remote client for external projects (no PyTorch, no local weights required).
"""
from __future__ import annotations

from avcs.server.client import AVCSCloudClient

__all__ = ["AVCSCloudClient"]

