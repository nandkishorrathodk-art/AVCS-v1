"""Regression tests for the avcs_sdk live-agent entrypoint.

The entrypoint was not runnable: it constructed ``EyeStream(api_key=...)``
although ``EyeStream`` (pure-pixel local perception) accepts no such argument,
so ``run_agent``, ``run_monitor`` and ``run_check`` all raised ``TypeError``
before turn 1. These tests pin the constructor contract and the honesty gate.
"""
from __future__ import annotations

import inspect

import pytest

from avcs_sdk.eye_stream import EyeStream
from avcs_sdk import run_live_agent


def test_eye_stream_does_not_accept_api_key():
    # Perception is local and pure-pixel; the LLM key belongs to the bridge only.
    params = inspect.signature(EyeStream.__init__).parameters
    assert "api_key" not in params


def test_entrypoint_never_passes_api_key_to_the_eye():
    src = inspect.getsource(run_live_agent)
    assert "EyeStream(api_key" not in src


def test_check_reports_incomplete_and_exits_two_without_a_checkpoint():
    # No trained checkpoint ships, so a fresh checkout must exit 2, not 0.
    assert run_live_agent.run_check(api_key=None) == 2
