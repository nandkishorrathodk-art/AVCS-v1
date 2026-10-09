"""Unit tests for AVCS Unified LLM Drivers (Anthropic, OpenAI, Ollama)."""
from __future__ import annotations

import json
import pytest

from avcs.drivers import (
    AnthropicAVCSDriver,
    OpenAIAVCSDriver,
    OllamaAVCSDriver,
    BaseAVCSDriver,
)
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


@pytest.fixture
def eye():
    screen = ScreenEye(source=SyntheticScreen(width=320, height=240), device="cpu")
    yield screen
    screen.close()


def test_anthropic_driver_step(eye):
    driver = AnthropicAVCSDriver(eye=eye, force_virtual_motor=True)

    tool_call = {
        "id": "toolu_01",
        "name": "locate",
        "input": {"query": "Submit button"},
    }
    result = driver.step(tool_call)
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_01"
    parsed = json.loads(result["content"])
    assert parsed["target"] == "Submit button"
    assert "x" in parsed and "y" in parsed

    # Verify step history
    assert len(driver.step_history) == 1
    s = driver.step_history[0]
    assert s.tool_name == "locate"
    assert s.success is True
    assert s.screen_category is not None

    driver.close()


def test_openai_driver_step(eye):
    driver = OpenAIAVCSDriver(eye=eye, force_virtual_motor=True)

    tool_call = {
        "id": "call_99",
        "function": {
            "name": "type_into",
            "arguments": json.dumps({"query": "Search input", "text": "AVCS docs"}),
        },
    }
    result = driver.step(tool_call)
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_99"
    parsed = json.loads(result["content"])
    assert parsed["action"] == "type_into"
    assert parsed["target"] == "Search input"

    driver.close()


def test_ollama_driver_step(eye):
    driver = OllamaAVCSDriver(eye=eye, force_virtual_motor=True)

    tool_call = {
        "function": {
            "name": "get_affordances",
            "arguments": {"action_type": "clickable"},
        },
    }
    result = driver.step(tool_call)
    assert result["role"] == "tool"
    parsed = json.loads(result["content"])
    assert parsed["action_type"] == "clickable"

    driver.close()

