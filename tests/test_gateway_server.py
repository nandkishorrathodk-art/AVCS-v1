"""Unit and Integration Tests for Parth Labs Unified Gateway Server."""
from __future__ import annotations

import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Generator

import pytest

from avcs.auth.server import ParthLabsGatewayHandler


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gateway_server() -> Generator[str, None, None]:
    port = find_free_port()
    server_address = ("127.0.0.1", port)
    httpd = http.server.HTTPServer(server_address, ParthLabsGatewayHandler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    time.sleep(0.1)

    yield base_url

    httpd.shutdown()
    httpd.server_close()


def test_gateway_health(gateway_server: str) -> None:
    req = urllib.request.Request(f"{gateway_server}/health")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "healthy"
        assert "Parth Labs AVC Unified Gateway" in data["service"]


def test_gateway_models(gateway_server: str) -> None:
    req = urllib.request.Request(f"{gateway_server}/v1/models")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "avc-cortex-v1" in model_ids


def test_gateway_auth_verify_success(gateway_server: str) -> None:
    payload = json.dumps({"api_key": "avcs-community-free-2026"}).encode("utf-8")
    req = urllib.request.Request(
        f"{gateway_server}/v1/auth/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["valid"] is True
        assert data["status"] == "authorized"
        assert data["tier"] == "developer"


def test_gateway_auth_verify_developer_key(gateway_server: str) -> None:
    payload = json.dumps({"api_key": "avc_live_9921_premium"}).encode("utf-8")
    req = urllib.request.Request(
        f"{gateway_server}/v1/auth/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["valid"] is True
        assert data["status"] == "authorized"


def test_gateway_auth_verify_rejected(gateway_server: str) -> None:
    payload = json.dumps({"api_key": "hacker_invalid_key"}).encode("utf-8")
    req = urllib.request.Request(
        f"{gateway_server}/v1/auth/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5.0)
    assert exc_info.value.code == 403


def test_gateway_chat_completions(gateway_server: str) -> None:
    payload = json.dumps({
        "model": "avc-reasoner-v1",
        "messages": [{"role": "user", "content": "What is on screen?"}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{gateway_server}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer avc_live_client_token",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"

