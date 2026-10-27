"""End-to-end tests for the real OpenAI-compatible (cloud) LLM path.

The voice tests drive ``ConversationBrain`` with *fake* providers, so the real
HTTP transport in ``LLMReasoningProvider`` (payload build, SSE stream, JSON
parse) was never exercised. These tests stand up a real local HTTP server that
speaks the OpenAI ``/chat/completions`` protocol and drive the *real* provider
against it, so a bug in the transport surfaces here rather than in production.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

import pytest

from avcs.reasoning.contracts import ActionOp
from avcs.reasoning.provider import LLMReasoningProvider
from avcs.voice.brain import (
    STREAM_SENTINEL,
    ConversationBrain,
    TurnKind,
)


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /chat/completions server (stream + non-stream)."""

    reply_text: str = ""
    stream_chunks: List[str] = []
    fail_status: int = 0
    fail_times: int = 0          # fail this many times, then succeed
    last_body: Dict[str, Any] = {}
    request_count: int = 0
    auth_header: str = ""

    def log_message(self, *a):
        pass

    def _send(self, code: int, payload: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        cls = type(self)
        cls.request_count += 1
        cls.auth_header = self.headers.get("Authorization", "")
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        cls.last_body = body
        if cls.fail_times > 0:
            cls.fail_times -= 1
            self._send(cls.fail_status, b'{"error": {"message": "transient"}}')
            return
        if body.get("stream"):
            self._stream()
            return
        payload = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": cls.reply_text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }).encode("utf-8")
        self._send(200, payload)


    def _stream(self):
        cls = type(self)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in cls.stream_chunks:
            event = json.dumps({"choices": [{"delta": {"content": piece}}]})
            self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
def _reset_handler() -> None:
    _MockOpenAIHandler.reply_text = ""
    _MockOpenAIHandler.stream_chunks = []
    _MockOpenAIHandler.fail_status = 0
    _MockOpenAIHandler.fail_times = 0
    _MockOpenAIHandler.last_body = {}
    _MockOpenAIHandler.request_count = 0
    _MockOpenAIHandler.auth_header = ""


@pytest.fixture
def openai_server():
    """A real HTTP server on a free port that speaks the OpenAI protocol."""
    _reset_handler()
    httpd = HTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _make_provider(endpoint: str, **kwargs) -> LLMReasoningProvider:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("retry_delay_seconds", 0.001)
    kwargs.setdefault("max_retries", 3)
    return LLMReasoningProvider(endpoint=endpoint, **kwargs)


_DECISION = {
    "thought": "The Save button is grounded; clicking it.",
    "task_status": "in_progress",
    "next_intent": {
        "op": "click",
        "target_description": "Save button",
        "rationale": "user asked to save the file",
        "expected_outcome": "the file is saved",
    },
}


# --------------------------------------------------------------------------- #
# Non-streaming transport
# --------------------------------------------------------------------------- #
def test_plan_step_non_stream_parses_decision(openai_server):
    _MockOpenAIHandler.reply_text = json.dumps(_DECISION)
    provider = _make_provider(openai_server)

    decision = provider.plan_step(
        "save the file", {"frame_sequence": 4}, [], step_index=2
    )

    assert decision.next_intent.op == ActionOp.CLICK
    assert decision.next_intent.target_description == "Save button"
    assert decision.next_intent.frame_sequence == 4
    assert decision.thought.startswith("The Save button")
    assert _MockOpenAIHandler.request_count == 1
    # The body really is OpenAI chat-completions shaped.
    assert _MockOpenAIHandler.last_body["model"] == provider.model
    assert _MockOpenAIHandler.last_body["messages"][0]["role"] == "system"


# --------------------------------------------------------------------------- #
# SSE streaming transport
# --------------------------------------------------------------------------- #
def test_stream_step_yields_sse_fragments(openai_server):
    _MockOpenAIHandler.stream_chunks = ["Hello ", "sir. ", "Doing ", "it now."]
    provider = _make_provider(openai_server)

    chunks = list(provider.stream_step("do it", {"frame_sequence": 1}, []))

    assert chunks == ["Hello ", "sir. ", "Doing ", "it now."]
    assert "".join(chunks) == "Hello sir. Doing it now."
    assert _MockOpenAIHandler.last_body["stream"] is True


def test_brain_streams_hinglish_reply_and_recovers_action(openai_server):
    _MockOpenAIHandler.stream_chunks = [
        "Ji sir, ",
        "main Save button dabata hoon. ",
        "\n" + STREAM_SENTINEL + "\n",
        json.dumps({
            "kind": "act",
            "thought": "click the save button",
            "action": {"op": "click", "target_description": "Save button"},
        }),
    ]
    provider = _make_provider(openai_server)
    brain = ConversationBrain(provider=provider)

    spoken: List[str] = []
    turn = brain.respond_stream(
        "file save karo",
        observation=ConversationBrain.build_observation(
            screen_description="a Save button", target_present=True
        ),
        on_speech=spoken.append,
    )

    assert turn.kind == TurnKind.ACT
    assert turn.source == "llm"
    assert turn.intent is not None
    assert turn.intent.op == ActionOp.CLICK
    assert turn.intent.target_description == "Save button"
    assert spoken and "Ji sir," in spoken[0]
    assert STREAM_SENTINEL not in turn.speech


# --------------------------------------------------------------------------- #
# Retry / backoff
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_post_retries_transient_status_then_succeeds(openai_server, status):
    _MockOpenAIHandler.reply_text = json.dumps(_DECISION)
    _MockOpenAIHandler.fail_status = status
    _MockOpenAIHandler.fail_times = 2
    provider = _make_provider(openai_server, max_retries=3, retry_delay_seconds=0.001)

    decision = provider.plan_step("save", {}, [], step_index=1)

    assert decision.next_intent.op == ActionOp.CLICK
    assert provider.request_count == 3
    assert provider.retry_count == 2
    assert provider.error_count == 2


def test_post_raises_after_exhausting_retries(openai_server):
    _MockOpenAIHandler.fail_status = 503
    _MockOpenAIHandler.fail_times = 10
    provider = _make_provider(openai_server, max_retries=2, retry_delay_seconds=0.001)

    with pytest.raises(RuntimeError) as excinfo:
        provider.plan_step("save", {}, [], step_index=1)

    assert "503" in str(excinfo.value)
    assert provider.request_count == 2
    assert provider.retry_count == 1


# --------------------------------------------------------------------------- #
# Auth + telemetry
# --------------------------------------------------------------------------- #
def test_authorization_bearer_header_is_sent(openai_server):
    _MockOpenAIHandler.reply_text = json.dumps(_DECISION)
    provider = _make_provider(openai_server, api_key="sk-secret-abc123")

    provider.plan_step("save", {}, [], step_index=1)

    assert _MockOpenAIHandler.auth_header == "Bearer sk-secret-abc123"


def test_token_usage_telemetry_accumulates(openai_server):
    _MockOpenAIHandler.reply_text = json.dumps(_DECISION)
    provider = _make_provider(openai_server)

    provider.plan_step("save", {}, [], step_index=1)
    assert provider.last_usage == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }

    provider.plan_step("save", {}, [], step_index=2)
    assert provider.total_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 14,
        "total_tokens": 24,
    }
    assert provider.request_count == 2
