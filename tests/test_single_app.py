from __future__ import annotations

from types import SimpleNamespace

from avcs.server.system2_client import TextOnlySystem2Client
from avcs.single_app import run_single_app_loop


class StubController:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.capture_backend = "Win32FastCapture"
        self.closed = False
        self.observations = 0
        self.source = StubSequenceCapture()

    def describe_screen(self, *, frame, frame_sequence):
        self.observations += 1
        return SimpleNamespace(
            frame_sequence=frame_sequence,
            confidence=0.7,
            uncertainty=0.3,
            description="OTP: 123456 on a changing desktop.",
            to_dict=lambda: {
                "frame_sequence": frame_sequence,
                "primary_category": "desktop",
                "description": "OTP: 123456 on a changing desktop.",
                "confidence": 0.7,
                "uncertainty": 0.3,
                "raw_frame": "must never be included",
            },
        )

    def close(self):
        self.closed = True


class StubSequenceCapture:
    def __init__(self):
        self.sequence = 0
        self.sequence_values = None

    def grab_with_sequence(self):
        if self.sequence_values is not None:
            return object(), next(self.sequence_values)
        self.sequence += 1
        return object(), self.sequence


class StubSystem2Client:
    def __init__(self):
        self.requests = []

    def prepare_payload(self, task, observation):
        return TextOnlySystem2Client.prepare_payload(task, observation)

    def request_goal(self, task, observation, *, consent_confirmed):
        self.requests.append((task, observation, consent_confirmed))
        return "Review the screen before taking the next step."


def test_single_app_loop_uses_real_capture_and_only_sends_after_consent():
    controllers = []
    client = StubSystem2Client()

    def controller_factory(**kwargs):
        controller = StubController(**kwargs)
        controllers.append(controller)
        return controller

    answers = iter(("yes", "q"))
    output = []
    result = run_single_app_loop(
        "Notepad",
        "Prepare a harmless live check",
        client,
        controller_factory=controller_factory,
        input_fn=lambda _prompt: next(answers),
        output_fn=output.append,
    )

    assert result == 0
    assert controllers[0].kwargs == {
        "source": "desktop",
        "device": "cpu",
        "force_virtual_motor": True,
        "enable_streaming": False,
        "require_real_capture": True,
    }
    assert controllers[0].closed
    assert len(client.requests) == 1
    sent_task, sent_observation, consent = client.requests[0]
    assert "Operator-provided app context: Notepad" in sent_task
    assert "User task: Prepare a harmless live check" in sent_task
    assert sent_observation["frame_sequence"] == 1
    assert consent is True
    assert any("Advisory only" in line for line in output)
    assert any("No synthetic fallback" in line for line in output)
    assert any("Exact redacted, allowlisted payload" in line for line in output)
    joined_output = "\n".join(output)
    assert "[REDACTED]" in joined_output
    assert "123456" not in joined_output
    assert "raw_frame" not in joined_output


def test_single_app_loop_does_not_send_when_operator_declines():
    controllers = []
    client = StubSystem2Client()

    def controller_factory(**kwargs):
        controller = StubController(**kwargs)
        controllers.append(controller)
        return controller

    output = []
    result = run_single_app_loop(
        "Notepad",
        "Observe only",
        client,
        controller_factory=controller_factory,
        input_fn=lambda _prompt: "no",
        output_fn=output.append,
    )

    assert result == 0
    assert client.requests == []
    assert controllers[0].closed
    assert any("Not sent" in line for line in output)


def test_single_app_loop_reobserves_after_operator_step():
    controllers = []
    client = StubSystem2Client()

    def controller_factory(**kwargs):
        controller = StubController(**kwargs)
        controllers.append(controller)
        return controller

    answers = iter(("y", "", "y", "q"))
    run_single_app_loop(
        "Notepad",
        "Observe this app",
        client,
        controller_factory=controller_factory,
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _line: None,
    )

    assert controllers[0].observations == 2
    assert [entry[1]["frame_sequence"] for entry in client.requests] == [1, 2]


def test_single_app_loop_waits_for_a_new_capture_sequence():
    controllers = []
    client = StubSystem2Client()

    def controller_factory(**kwargs):
        controller = StubController(**kwargs)
        controller.source.sequence_values = iter((5, 5, 6))
        controllers.append(controller)
        return controller

    answers = iter(("y", "", "y", "q"))
    run_single_app_loop(
        "Notepad",
        "Observe this app",
        client,
        controller_factory=controller_factory,
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _line: None,
    )

    assert [entry[1]["frame_sequence"] for entry in client.requests] == [5, 6]


def test_single_app_loop_closes_controller_when_system2_fails():
    controllers = []

    def controller_factory(**kwargs):
        controller = StubController(**kwargs)
        controllers.append(controller)
        return controller

    class FailingClient:
        def prepare_payload(self, task, observation):
            return StubSystem2Client().prepare_payload(task, observation)

        def request_goal(self, *_args, **_kwargs):
            raise RuntimeError("bridge unavailable")

    try:
        run_single_app_loop(
            "Notepad",
            "Observe",
            FailingClient(),
            controller_factory=controller_factory,
            input_fn=lambda _prompt: "yes",
            output_fn=lambda _line: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "bridge unavailable"
    else:
        raise AssertionError("System 2 failure should remain visible to the caller")

    assert controllers[0].closed
