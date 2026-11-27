"""Regression tests pinning 100% self-contained single-brain local VLA invariants."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from avcs.os_io.computer import Observation
from avcs.protocol.actions import Action, ActionKind
from avcs.runtime.agent import AceAgent
from avcs.runtime.client import AceClientError, CustomLLMClient, is_local_endpoint
from avcs.voice.tts import NeuralVoiceModelEngine, Speaker
from avcs_sdk.llm_bridge import LLMBridge
from avcs_sdk.narrator import LiveCommentary


def test_is_local_endpoint_only_permits_loopback():
    assert is_local_endpoint("http://127.0.0.1:8000/v1/chat/completions")
    assert is_local_endpoint("http://localhost:8000/v1")
    assert is_local_endpoint("ws://127.0.0.1:8000/ws/act")
    assert not is_local_endpoint("https://api.openai.com/v1/chat/completions")
    assert not is_local_endpoint("https://api.anthropic.com/v1/messages")
    assert not is_local_endpoint("https://api.parthlabs.in/v1")
    assert not is_local_endpoint("")


@pytest.mark.parametrize(
    "cloud_url",
    [
        "https://api.openai.com/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
        "https://generativelanguage.googleapis.com/v1beta/models",
        "https://api.parthlabs.in/v1/chat/completions",
    ],
)
def test_custom_llm_client_quarantines_external_cloud_endpoints(cloud_url: str):
    client = CustomLLMClient(endpoint=cloud_url)
    with pytest.raises(AceClientError):
        client.predict_action(b"fake-bytes", goal="Click Save")
    assert client.complete("system", "user") == ""


def test_ace_agent_defaults_to_no_cloud_client_and_fails_closed():
    agent = AceAgent()
    assert agent.client is None
    obs = Observation(image_bytes=b"fake", width=1280, height=800, format="jpeg")
    act, _lat = agent.step(obs, goal="Click Save")
    assert act.kind == ActionKind.WAIT
    assert (act.metadata or {}).get("fail_closed") is True


def test_ace_agent_vla_failure_returns_fail_closed_wait_without_cloud_fallback():
    mock_vla = MagicMock()
    mock_vla.act.side_effect = RuntimeError("local vla disconnected")
    mock_client = MagicMock()

    agent = AceAgent(client=mock_client, vla_client=mock_vla)
    obs = Observation(
        image_bytes=b"fake",
        width=1280,
        height=800,
        format="jpeg",
        raw_numpy=np.zeros((800, 1280, 3), dtype=np.uint8),
    )
    act, _lat = agent.step(obs, goal="Click Search")

    mock_client.predict_action.assert_not_called()
    assert act.kind == ActionKind.WAIT
    assert (act.metadata or {}).get("fail_closed") is True


def test_sdk_llm_bridge_rejects_external_urls_and_runs_locally():
    with pytest.raises(ValueError):
        LLMBridge(base_url="https://api.parthlabs.in/v1")

    bridge = LLMBridge()
    assert "127.0.0.1" in bridge.base_url or "localhost" in bridge.base_url

    commentary = LiveCommentary(
        narrative="Desktop idle",
        active_window="",
        is_stable=True,
        has_popup=False,
        needs_llm_attention=True,
        surprise_score=0.0,
        visible_elements=[],
    )
    cmd = bridge.plan_next_step("Click Save button", commentary, fovea_patches=None)
    assert cmd.source in ("local_vla", "heuristic_fallback")


def test_tts_disables_cloud_edge_tts_even_when_requested():
    neural = NeuralVoiceModelEngine()
    assert neural.available() is False
    speaker = Speaker(allow_cloud_tts=True)
    assert not isinstance(speaker.engine, NeuralVoiceModelEngine)
    assert speaker.engine_name != "neural-voice"
