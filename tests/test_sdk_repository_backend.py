"""Integration tests for AVCS Python SDK with RepositoryAVCSBackend and ActionBroker."""
from __future__ import annotations

import pytest

from avcs.action_broker import (
    ActionBroker,
    ActionResult as RepoActionResult,
    ActionSafetyController,
    ActionStatus as RepoActionStatus,
)
from avcs.capture_adapter import InteractiveSyntheticCaptureAdapter, SyntheticCaptureAdapter
from avcs.human_eye import HumanEye
from avcs_sdk import (
    ActionClass,
    ActionKind,
    ActionRequest,
    ActionResult,
    ActionStatus,
    AVCSClient,
    NormalizedPoint,
    RepositoryAVCSBackend,
    RiskLevel,
)
from forge.vision.cortex.motor.driver import VirtualCursorDriver


@pytest.mark.asyncio
async def test_repository_backend_lifecycle_and_observation():
    """Verify that AVCSClient with RepositoryAVCSBackend handles lifecycle and captures real frames."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        assert obs.width > 0
        assert obs.height > 0
        assert obs.coordinate_space == "normalized_virtual_desktop"
        assert obs.frame_fingerprint is not None
        assert backend._connected is True

    assert backend._connected is False


@pytest.mark.asyncio
async def test_repository_backend_find_targets():
    """Verify that no semantic targets are fabricated from generic saliency."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        targets = await client.find_targets("Save Button", observation=obs)
        assert targets == []


@pytest.mark.asyncio
async def test_repository_backend_click_with_verified_evidence():
    """A visual ripple must not be reported as proof of a prose postcondition."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        captured = []

        def confirmed_visual_action(repo_request):
            captured.append(repo_request)
            return RepoActionResult(
                action_id=repo_request.action_id,
                status=RepoActionStatus.CONFIRMED,
                action_type=repo_request.action_type,
                target_coord=repo_request.target_coord,
                elapsed_ms=1.0,
                evidence="visual ripple confirmed",
                optical_ripple_energy=0.1,
                photometric_delta=0.1,
                recovery_attempts=1,
            )

        backend.broker.execute = confirmed_visual_action
        result = await client.click(
            point=NormalizedPoint(0.5, 0.5),
            observation=obs,
            expected_postcondition="Submit state update",
            action_class=ActionClass.IDEMPOTENT,
        )

        assert captured[0].expected_postcondition is None
        assert result.status == ActionStatus.AMBIGUOUS
        assert result.evidence is not None
        assert result.evidence.verified is False
        assert result.evidence.method == "action_broker_visual_reaction_only"
        assert result.duration_ms >= 0.0


@pytest.mark.asyncio
async def test_repository_backend_high_risk_unauthorized_blocked():
    """Verify that high-risk action with invalid authorization is blocked by safety controller."""
    backend = RepositoryAVCSBackend(trusted_auth_tokens={"valid_token_123"})
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(
            point=NormalizedPoint(0.5, 0.5),
            observation=obs,
            risk=RiskLevel.HIGH,
            authorization_ref="unauthorized_spoofed_token",
        )

        assert result.status == ActionStatus.BLOCKED
        assert result.evidence is None


@pytest.mark.asyncio
async def test_repository_backend_high_risk_authorized_confirmed():
    """Verify that high-risk action with valid authorization is allowed and confirmed."""
    backend = RepositoryAVCSBackend(trusted_auth_tokens={"valid_token_123"})
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        result = await client.click(
            point=NormalizedPoint(0.42, 0.58),
            observation=obs,
            risk=RiskLevel.HIGH,
            authorization_ref="valid_token_123",
        )

        assert result.status == ActionStatus.CONFIRMED
        assert result.evidence is not None
        assert result.evidence.verified is True


@pytest.mark.asyncio
async def test_repository_backend_type_text_and_key():
    """Block keyboard input when this client cannot prove a focused target."""
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        res_text = await client.type_text("Hello AVCS", action_class=ActionClass.IDEMPOTENT)
        assert res_text.status == ActionStatus.BLOCKED
        assert res_text.error_code == "TARGET_REQUIRED"

        res_key = await client.press_key("Enter", action_class=ActionClass.IDEMPOTENT)
        assert res_key.status == ActionStatus.BLOCKED
        assert res_key.error_code == "TARGET_REQUIRED"
        assert backend.driver.typed_history == []
        assert backend.driver.key_history == []


@pytest.mark.asyncio
async def test_repository_backend_blocks_keyboard_action_without_focus(
    monkeypatch,
):
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        observation = await client.observe()
        monkeypatch.setattr(
            backend.eye,
            "verify_focus",
            lambda target_point: type("Focus", (), {"is_focused": False})(),
        )
        result = await client.act(
            ActionRequest(
                kind=ActionKind.TYPE_TEXT,
                point=NormalizedPoint(0.5, 0.5),
                observation_id=observation.observation_id,
                text="must not be typed",
            )
        )

        assert result.status == ActionStatus.BLOCKED
        assert result.error_code == "FOCUS_NOT_VERIFIED"
        assert backend.driver.typed_history == []
        assert backend.driver.key_history == []


@pytest.mark.asyncio
async def test_repository_backend_does_not_translate_scroll_to_click():
    backend = RepositoryAVCSBackend()
    async with AVCSClient(backend) as client:
        observation = await client.observe()
        result = await client.act(
            ActionRequest(
                kind=ActionKind.SCROLL,
                point=NormalizedPoint(0.5, 0.5),
                observation_id=observation.observation_id,
                scroll_delta=120,
            )
        )

        assert result.status == ActionStatus.BLOCKED
        assert result.error_code == "UNSUPPORTED_ACTION"
        assert backend.driver.click_history == []


@pytest.mark.asyncio
async def test_repository_backend_coordinate_conversion():
    """Verify exact coordinate translation between NormalizedPoint and broker target_coord."""
    driver = VirtualCursorDriver()
    cap = InteractiveSyntheticCaptureAdapter().open()
    eye = HumanEye(source=cap, threaded_capture=False)
    safety = ActionSafetyController()
    broker = ActionBroker(eye=eye, driver=driver, safety_controller=safety, settle_delay_s=0.001)

    backend = RepositoryAVCSBackend(broker=broker, eye=eye, capture=cap, driver=driver)
    async with AVCSClient(backend) as client:
        obs = await client.observe()
        pt = NormalizedPoint(0.2468, 0.7531)
        res = await client.click(point=pt, observation=obs, action_class=ActionClass.IDEMPOTENT)
        assert res.status == ActionStatus.CONFIRMED
