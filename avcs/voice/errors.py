"""Typed errors for the AVCS voice layer.

Every failure that a caller might want to recover from (no engine installed, no
audio device, no model file) has its own type, so a headless or partially
provisioned machine degrades to "silent but working" instead of raising an
opaque subprocess error deep in a render loop.
"""
from __future__ import annotations


class VoiceError(RuntimeError):
    """Base class for every voice-layer failure."""


class EngineNotAvailable(VoiceError):
    """The requested engine binary or Python package is not installed."""


class AudioDeviceError(VoiceError):
    """No output/input audio device is reachable (common on headless CI)."""


class ModelNotFound(VoiceError):
    """A speech-recognition model directory or file is missing."""


class SynthesisError(VoiceError):
    """The engine ran but produced no usable audio."""
