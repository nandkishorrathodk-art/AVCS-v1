from __future__ import annotations

from types import SimpleNamespace

import pytest

from avcs.server.system2_client import TextOnlySystem2Client


def test_system2_client_sends_allowlisted_text_only_observation(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"goal": "Inspect the dialog before continuing."}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("avcs.server.system2_client.requests.post", fake_post)
    client = TextOnlySystem2Client(
        "https://brain.example",
        api_key="test-key",
    )
    observation = {
        "timestamp": 12.5,
        "frame_sequence": 42,
        "primary_category": "dialog",
        "stability": "stable",
        "description": "A modal dialog is visible.",
        "visible_entities": ["2 buttons"],
        "has_active_modal": True,
        "is_stable": True,
        "confidence": 0.8,
        "uncertainty": 0.2,
        "image_b64": "must never be forwarded",
        "embedding": [1.0, 2.0],
        "raw_frame": object(),
    }

    goal = client.request_goal("Review the dialog", observation, consent_confirmed=True)

    assert goal == "Inspect the dialog before continuing."
    url, request = calls[0]
    assert url == "https://brain.example/v1/system2/goal"
    assert request["json"]["task"] == "Review the dialog"
    assert request["json"]["observation"] == {
        key: value
        for key, value in observation.items()
        if key in {
            "timestamp",
            "frame_sequence",
            "primary_category",
            "stability",
            "description",
            "visible_entities",
            "has_active_modal",
            "is_stable",
            "confidence",
            "uncertainty",
        }
    }
    assert "image_b64" not in request["json"]["observation"]
    assert "embedding" not in request["json"]["observation"]
    assert "raw_frame" not in request["json"]["observation"]


def test_system2_client_accepts_scene_observation_object(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"goal": "Wait for the screen to stabilize."}

    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr("avcs.server.system2_client.requests.post", fake_post)
    client = TextOnlySystem2Client("https://brain.example")
    observation = SimpleNamespace(
        to_dict=lambda: {
            "timestamp": 1.0,
            "frame_sequence": 7,
            "primary_category": "desktop",
            "stability": "busy",
            "description": "The display is changing.",
        }
    )

    assert client.request_goal(
        "Wait", observation, consent_confirmed=True
    ) == "Wait for the screen to stabilize."
    assert captured["observation"]["frame_sequence"] == 7


def test_system2_client_rejects_invalid_observation_numbers(monkeypatch):
    client = TextOnlySystem2Client("https://brain.example")
    with pytest.raises(ValueError, match="finite number"):
        client.request_goal("Inspect", {"timestamp": float("nan")}, consent_confirmed=True)


def test_system2_client_requires_https_for_remote_endpoint():
    with pytest.raises(ValueError, match="must use HTTPS"):
        TextOnlySystem2Client("http://brain.example")


def test_system2_client_rejects_empty_or_malformed_goal(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"goal": "  "}

    monkeypatch.setattr("avcs.server.system2_client.requests.post", lambda *a, **k: Response())
    client = TextOnlySystem2Client("https://brain.example")
    with pytest.raises(ValueError, match="non-empty text 'goal'"):
        client.request_goal("Continue", {}, consent_confirmed=True)


def test_system2_client_requires_consent_before_network_send(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "avcs.server.system2_client.requests.post",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )
    client = TextOnlySystem2Client("https://brain.example")

    with pytest.raises(PermissionError, match="explicit user consent"):
        client.request_goal("Continue", {})
    with pytest.raises(PermissionError, match="explicit user consent"):
        client.request_goal("Continue", {}, consent_confirmed=False)
    assert sent == []


def test_system2_client_redacts_common_sensitive_text_patterns(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"goal": "Continue carefully."}

    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr("avcs.server.system2_client.requests.post", fake_post)
    client = TextOnlySystem2Client("https://brain.example")

    client.request_goal(
        "Review OTP: 123456 for user@example.com",
        {"description": "OTP: 654321; user@example.com"},
        consent_confirmed=True,
    )

    assert "123456" not in captured["task"]
    assert "user@example.com" not in captured["task"]
    assert "654321" not in captured["observation"]["description"]
