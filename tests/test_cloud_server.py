"""Tests for AVCS Cloud Inference Server and Remote Client."""
from __future__ import annotations

import base64
import io
import os
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

import avcs.server.api as server_api
from avcs.server.api import app, decode_image
from avcs.server.client import AVCSCloudClient
from forge.vision.capture import SyntheticScreen
from forge.vision.cortex.motor.driver import VirtualCursorDriver

TEST_INFERENCE_KEY = "avcs-inference-test-key-0123456789abcdef"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AVCS_INFERENCE_API_KEY", TEST_INFERENCE_KEY)
    return TestClient(app, headers={"X-API-Key": TEST_INFERENCE_KEY})


def test_decode_image():
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[10:20, 10:20] = 255
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    out = decode_image(b64)
    assert out.shape == (32, 32, 3)
    assert np.all(out[10:20, 10:20] == 255)


def test_health_endpoint(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["weights_loaded"] is True
    assert data["device"] == "cpu"
    assert data["perceive_ready"] is True
    assert data["ground_ready"] is True


def test_health_endpoint_does_not_treat_untrained_grounder_as_ready(client, monkeypatch):
    from types import SimpleNamespace

    eye = SimpleNamespace(weights_loaded=False)
    fovea = SimpleNamespace(weights_loaded=False)
    untrained_grounder = SimpleNamespace(is_trained=False)
    monkeypatch.setattr(server_api, "get_engine", lambda: (eye, fovea, None, None))
    monkeypatch.setattr(server_api, "get_grounder", lambda: untrained_grounder)
    monkeypatch.setattr(server_api, "_grounder_loaded", True)

    response = client.get("/v1/health")
    data = response.json()

    assert data["status"] == "degraded"
    assert data["weights_loaded"] is False
    assert data["grounder_weights_loaded"] is False
    assert data["ground_ready"] is False


def test_perceive_endpoint(client):
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Draw a simulated button
    arr[40:60, 40:80] = 220
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.post(
        "/v1/perceive",
        json={"image_b64": b64, "action_type": "all", "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "category" in data
    assert "category_confidence" in data
    assert "stability" in data
    assert "affordances" in data
    assert isinstance(data["affordances"], list)
    assert data["latency_ms"] > 0
    assert data["weights_loaded"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"image_b64": "unused", "action_type": "execute", "top_k": 5},
        {"image_b64": "unused", "action_type": "all", "top_k": -1},
        {"image_b64": "unused", "action_type": "all", "top_k": 51},
    ],
)
def test_perceive_endpoint_rejects_invalid_filters(client, payload):
    response = client.post("/v1/perceive", json=payload)

    assert response.status_code == 422


def test_ground_endpoint_uses_loaded_demonstration_grounder(client, monkeypatch):
    from types import SimpleNamespace

    class StubGrounder:
        is_trained = True

        def predict(self, frame, query):
            assert frame.shape == (64, 96, 3)
            assert query == "Find the submit button"
            return SimpleNamespace(
                candidate_point_norm=(0.63, 0.42),
                peak_probability=0.81,
                target_present=True,
                target_present_probability=0.92,
            )

    monkeypatch.setattr(server_api, "_grounder", StubGrounder())
    monkeypatch.setattr(server_api, "_grounder_loaded", True)
    monkeypatch.setattr(server_api, "_grounder_load_attempted", True)

    arr = np.zeros((64, 96, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    response = client.post(
        "/v1/ground",
        json={"image_b64": encoded, "query": "Find the submit button"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_coord"] == [0.63, 0.42]
    assert data["confidence"] == 0.81
    assert data["target_present"] is True
    assert data["target_presence_probability"] == 0.92
    assert data["frame_width"] == 96
    assert data["frame_height"] == 64
    assert data["checkpoint"] == "demonstration_trained"


def test_ground_endpoint_fails_closed_without_trained_checkpoint(client, monkeypatch):
    monkeypatch.setattr(server_api, "_grounder", None)
    monkeypatch.setattr(server_api, "_grounder_loaded", False)
    monkeypatch.setattr(server_api, "_grounder_load_attempted", True)

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    response = client.post(
        "/v1/ground",
        json={"image_b64": encoded, "query": "Find the submit button"},
    )
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"]


@pytest.mark.parametrize(
    "query, detail",
    [
        ("   ", "cannot be empty"),
        ("x" * 97, "exceeds 96 UTF-8 bytes"),
    ],
)
def test_ground_endpoint_rejects_invalid_query_before_decoding(client, query, detail):
    response = client.post(
        "/v1/ground",
        json={"image_b64": "not-an-image", "query": query},
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_foveate_endpoint(client):
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[50:70, 50:70] = 200
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.post(
        "/v1/foveate",
        json={"image_b64": b64, "target": [0.5, 0.5]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "element_kind" in data
    assert "confidence" in data
    assert "contrast_score" in data
    assert "bbox" in data


def test_verify_focus_endpoint(client):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.post(
        "/v1/verify_focus",
        json={"current_image_b64": b64, "target": [0.5, 0.5]},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Blank frame has no focus
    assert data["is_focused"] is False


def test_plan_trajectory_endpoint(client):
    resp = client.post(
        "/v1/plan_trajectory",
        json={"start": [0.1, 0.1], "target": [0.8, 0.8], "duration_s": 0.4},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["duration_s"] == 0.4
    assert data["waypoints_count"] > 10
    assert len(data["waypoints"]) == data["waypoints_count"]
    # Check boundary coordinates
    first_pt = data["waypoints"][0]
    last_pt = data["waypoints"][-1]
    assert np.isclose(first_pt["x"], 0.1, atol=1e-2)
    assert np.isclose(first_pt["y"], 0.1, atol=1e-2)
    assert np.isclose(last_pt["x"], 0.8, atol=1e-2)
    assert np.isclose(last_pt["y"], 0.8, atol=1e-2)


def test_api_key_auth(client, monkeypatch):
    monkeypatch.setenv("AVCS_INFERENCE_API_KEY", TEST_INFERENCE_KEY)
    unauthenticated_client = TestClient(app)
    # Request without header should fail 401
    resp = unauthenticated_client.post(
        "/v1/plan_trajectory",
        json={"start": [0.1, 0.1], "target": [0.2, 0.2]},
    )
    assert resp.status_code == 401

    # Request with valid key should succeed
    resp_authed = unauthenticated_client.post(
        "/v1/plan_trajectory",
        headers={"X-API-Key": TEST_INFERENCE_KEY},
        json={"start": [0.1, 0.1], "target": [0.2, 0.2]},
    )
    assert resp_authed.status_code == 200
    resp_wrong_key = unauthenticated_client.post(
        "/v1/plan_trajectory",
        headers={"X-API-Key": "x" * len(TEST_INFERENCE_KEY)},
        json={"start": [0.1, 0.1], "target": [0.2, 0.2]},
    )
    assert resp_wrong_key.status_code == 401


def test_service_refuses_requests_when_secret_is_unconfigured(monkeypatch):
    monkeypatch.delenv("AVCS_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("AVCS_API_KEY", raising=False)
    response = TestClient(app).get("/v1/health")
    assert response.status_code == 503


def test_service_rejects_short_configured_secret(monkeypatch):
    monkeypatch.setenv("AVCS_INFERENCE_API_KEY", "too-short")
    response = TestClient(app).get(
        "/v1/health",
        headers={"X-API-Key": "too-short"},
    )
    assert response.status_code == 503


def test_cloud_client_local_workflow(monkeypatch):
    """Verifies that AVCSCloudClient operates cleanly without local neural weights."""
    monkeypatch.setenv("AVCS_INFERENCE_API_KEY", TEST_INFERENCE_KEY)
    test_client = TestClient(app)

    # Monkeypatch requests calls to route through TestClient
    def mock_get(url, headers=None, timeout=None):
        path = url.replace("http://avcs.test", "")
        return test_client.get(path, headers=headers)

    def mock_post(url, json=None, headers=None, timeout=None):
        path = url.replace("http://avcs.test", "")
        return test_client.post(path, headers=headers, json=json)

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("requests.post", mock_post)

    source = SyntheticScreen()
    c = AVCSCloudClient(
        endpoint_url="http://avcs.test",
        api_key=TEST_INFERENCE_KEY,
        source=source,
        force_virtual_motor=True,
        allow_insecure_http=True,
    )

    # 1. ping
    health = c.ping()
    assert health["status"] == "healthy"

    # 2. see
    state = c.see()
    assert "category" in state

    # 3. get_affordances
    affs = c.get_affordances("clickable")
    assert isinstance(affs, list)

    # 4. foveate
    fov = c.foveate((0.5, 0.5))
    assert "element_kind" in fov

    # 5. verify_focus on blank screen -> False
    focused = c.verify_focus((0.5, 0.5))
    assert focused is False

    # 6. move_to
    moved = c.move_to((0.7, 0.7), realtime=False)
    assert moved is True

    # 7. click
    clicked = c.click((0.7, 0.7))
    assert clicked is True

    # 8. type_into with focus check -> should raise RuntimeError (fail-closed)
    with pytest.raises(RuntimeError, match="visual focus verification failed closed"):
        c.type_into((0.5, 0.5), "hello world", enforce_focus=True)

    c.close()


def test_cloud_client_rejects_remote_http_by_default():
    with pytest.raises(ValueError, match="use HTTPS for remote servers"):
        AVCSCloudClient(
            endpoint_url="http://avcs.example",
            source=SyntheticScreen(),
            force_virtual_motor=True,
        )


def test_cloud_client_ground_calls_authenticated_route(monkeypatch):
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"target_coord": [0.25, 0.75], "confidence": 0.6}

    def mock_post(url, json=None, headers=None, timeout=None):
        requests.append((url, json, headers, timeout))
        return Response()

    monkeypatch.setattr("avcs.server.client.requests.post", mock_post)
    c = AVCSCloudClient(
        endpoint_url="https://avcs.example",
        api_key=TEST_INFERENCE_KEY,
        source=SyntheticScreen(),
        force_virtual_motor=True,
    )
    try:
        result = c.ground("  Save button  ", frame=np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        c.close()

    assert result["target_coord"] == [0.25, 0.75]
    assert requests[0][0] == "https://avcs.example/v1/ground"
    assert requests[0][1]["query"] == "Save button"
    assert requests[0][2]["X-API-Key"] == TEST_INFERENCE_KEY


def test_cloud_client_rejects_malformed_focus_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"is_focused": "false"}

    monkeypatch.setattr("avcs.server.client.requests.post", lambda *args, **kwargs: Response())
    c = AVCSCloudClient(
        endpoint_url="https://avcs.example",
        source=SyntheticScreen(),
        force_virtual_motor=True,
    )
    c._last_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    try:
        with pytest.raises(ValueError, match="must contain a boolean"):
            c.verify_focus((0.5, 0.5))
    finally:
        c.close()


def test_cloud_client_does_not_type_after_malformed_focus_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"is_focused": "false"}

    monkeypatch.setattr("avcs.server.client.requests.post", lambda *args, **kwargs: Response())
    c = AVCSCloudClient(
        endpoint_url="https://avcs.example",
        source=SyntheticScreen(),
        force_virtual_motor=True,
    )
    c._last_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(c, "click", lambda target: True)
    monkeypatch.setattr(c, "see", lambda: {})
    try:
        with pytest.raises(ValueError, match="must contain a boolean"):
            c.type_into((0.5, 0.5), "must not be typed")
        assert c.driver.typed_history == []
        assert c.driver.key_history == []
    finally:
        c.close()
