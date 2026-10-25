"""Compatibility re-export of EyeStream and its observations from avcs_sdk.eye_stream."""
from __future__ import annotations

from avcs_sdk.eye_stream import EyeFrame, EyeObservation, EyeStream

__all__ = ["EyeStream", "EyeObservation", "EyeFrame"]

