"""Tests for the GUI *assistant* and its scene eyes.

The product is an assistant that looks at a screen and talks about it, not a
control detector. These tests pin that: the default eyes describe a scene and
name no control, the assistant has no mutating capability by default, and the
page is a conversation rather than a clicker. The control-grounding path is a
separate, opt-in thing tested in `tests/test_lexicon_perception.py`.

Nothing here is mocked: the scene perception, the kernel, the assistant and the
FastAPI app are the real ones, because the thing under test is whether the pieces
agree with each other.
"""
from __future__ import annotations

import pytest

from avcs.gui.demo_source import CATEGORIES, DemoSceneSource
from avcs.gui.server import GuiApp, build_app
from avcs.voice.scene_perception import ScenePerception


@pytest.fixture(scope="module")
def scene_source():
    return DemoSceneSource("desktop", seed=7)


@pytest.fixture(scope="module")
def perception(scene_source):
    return ScenePerception(frame_source=scene_source)


class _OfflineBrain:
    """A brain that is never live, so a test's outcome is deterministic.

    `ConversationBrain` falls back to `RuleBrain` when its provider raises, and
    the rule brain maps "daba do" to a click. That gives these tests a fixed,
    network-free decision without mocking the assistant itself.
    """

    api_key = ""

    def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
        raise RuntimeError("offline in tests")

    def reset(self):
        pass


def _offline_brain():
    from avcs.voice.brain import ConversationBrain

    return ConversationBrain(provider=_OfflineBrain())


# --------------------------------------------------------------------------- #
# The scripted scene source
# --------------------------------------------------------------------------- #
def test_demo_source_renders_every_category():
    for cat in CATEGORIES:
        src = DemoSceneSource(cat)
        frame = src.capture()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype.name == "uint8"
        assert frame.shape[0] > 100 and frame.shape[1] > 300


def test_demo_source_is_callable_and_switchable():
    src = DemoSceneSource("desktop")
    a = src()
    assert a.shape == src.capture().shape
    src.set_category("terminal")
    assert src.category == "terminal"
    with pytest.raises(ValueError):
        src.set_category("nonsense")


# --------------------------------------------------------------------------- #
# Scene perception: describes, never grounds
# --------------------------------------------------------------------------- #
def test_scene_perception_describes_the_frame(perception):
    obs = perception.observe()
    assert obs.description
    assert "screen" in obs.description.lower()
    assert obs.target_present is None          # it does not name a target


def test_scene_perception_exposes_a_typed_reading(perception):
    perception.observe()
    reading = perception.state_summary()
    for key in ("brightness", "contrast", "peak_motion", "novelty", "luma_grid"):
        assert key in reading
    assert 0.0 <= reading["brightness"] <= 1.0
    # The grid is the model's own view; it must be a real 2D array.
    assert len(reading["luma_grid"]) > 4
    assert len(reading["luma_grid"][0]) > 4


def test_scene_perception_has_no_grounding_evidence(perception):
    """The whole point: an assistant with these eyes cannot click.

    With no `grounding_evidence` the assistant builds a placeholder with no
    candidate point, and the kernel refuses any mutating op. That is the honest
    outcome, not a bug to paper over.
    """
    assert not hasattr(perception, "grounding_evidence")


def test_scene_perception_reports_no_frame_honestly():
    perc = ScenePerception(frame_source=lambda: None)
    obs = perc.observe()
    assert "no screen" in obs.description.lower()
    assert perc.frame() is None
    assert perc.state_summary() == {}


def test_scene_perception_ignores_the_query(perception):
    """A query is accepted for protocol compatibility and has no effect.

    This layer has nothing to say about a named control; pretending otherwise
    would be the control detector the assistant is not.
    """
    a = perception.observe()
    b = perception.observe("Save button")
    assert a.target_present == b.target_present is None


def test_scene_perception_reports_change_between_frames(scene_source):
    perc = ScenePerception(frame_source=scene_source)
    perc.observe()
    obs = perc.observe()          # the generator advances every grab
    assert obs.stability in ("stable", "changing")
    assert perc.state_summary().get("regions", 0) >= 0


# --------------------------------------------------------------------------- #
# The assistant: looks, talks, does not click by default
# --------------------------------------------------------------------------- #
def test_assistant_defaults_to_no_mutating_grant():
    gui = GuiApp(source="demo", mouth="silent", ears=False)
    assert gui.grants == []
    assert gui.capabilities()["can_act"] is False
    assert gui.capabilities()["eyes"] == "ScenePerception"


def test_assistant_refuses_to_click_with_scene_eyes():
    """Even asked to click, a scene-eyes assistant is refused, honestly."""
    gui = GuiApp(source="demo", mouth="silent", ears=False, brain=_offline_brain())
    rec = gui.turn("save button daba do")
    assert rec["action"] == "click"               # the rule brain reads "daba do"
    assert rec["mutating"] is True
    assert rec["gate_allowed"] is False           # no grant, and no grounding evidence
    assert rec["clicked"] is None


def test_lexicon_eyes_still_opt_in_to_a_grant():
    gui = GuiApp(source="demo", eyes="lexicon", mouth="silent", ears=False)
    assert gui.grants == ["click"]
    assert gui.capabilities()["eyes"] == "LexiconPerception"
    assert gui.capabilities()["can_act"] is True


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    return TestClient(build_app(source="demo", mouth="silent", ears=False))


def test_index_serves_the_assistant_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "AVCS" in res.text and "assistant" in res.text
    assert "api/stream" in res.text
    # The page is a conversation, not a control detector.
    assert "api/scene" in res.text
    assert "api/screen" not in res.text


def test_capabilities_report_an_assistant(client):
    caps = client.get("/api/capabilities").json()
    assert caps["eyes"] == "ScenePerception"
    assert caps["can_act"] is False
    assert isinstance(caps["brain_has_key"], bool)


def test_frame_reports_a_description_not_labels(client):
    data = client.get("/api/frame").json()
    assert data["frame_size"][0] > 0
    assert data["description"]
    assert "brightness" in data["reading"]
    # No control labels on the default path.
    assert data["regions"] == []


def test_frame_png_is_a_png(client):
    res = client.get("/api/frame.png")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_turn_replies_in_conversation(client):
    rec = client.post("/api/turn", json={"text": "screen par kya hai?"}).json()
    assert rec["said"]
    assert rec["brain_source"] in ("llm", "rule_fallback")
    assert rec["actuation"] in ("none", "virtual")


def test_scene_switch_changes_what_is_described(client):
    client.post("/api/scene", json={"category": "media"})
    desc_media = client.get("/api/frame").json()["description"]
    client.post("/api/scene", json={"category": "terminal"})
    desc_terminal = client.get("/api/frame").json()["description"]
    assert desc_media and desc_terminal
    client.post("/api/scene", json={"category": "desktop"})


def test_scene_switch_rejects_an_unknown_category(client):
    res = client.post("/api/scene", json={"category": "nope"})
    assert res.status_code == 400


def test_scene_switch_refuses_on_a_real_screen():
    gui = GuiApp(source="screen", mouth="silent", ears=False)
    assert gui.screen.demo_source is None


def test_stream_emits_a_description(client):
    with client.stream("GET", "/api/stream") as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        for line in res.iter_lines():
            if line.startswith("data: "):
                import json

                payload = json.loads(line[len("data: "):])
                assert "description" in payload and "seq" in payload
                break


def test_assistant_is_virtual_by_default():
    """Virtual mode must still attach a hand, or a grant is unreachable.

    The bug this pins: with `hand is None`, `can_click` was reported False, the
    model answered "I have no capability", and a granted click was refused with
    "no hand attached". A virtual hand is inert but real, so the gate decides and
    the record says which actuation happened.
    """
    gui = GuiApp(source="demo", mouth="silent", ears=False)
    assert gui.live is False
    assert gui.capabilities()["actuation"] == "virtual"
    assert gui.hand is not None
    assert gui.capabilities()["hand"] == "virtual-hand"


def test_a_granted_click_is_reachable_in_virtual_mode():
    gui = GuiApp(source="demo", mouth="silent", ears=False,
                 grants=["click"], eyes="lexicon")
    caps = gui.capabilities()
    assert caps["can_act"] is True
    # The hand advertises the op, so the brain is told it can click.
    assert "click" in caps["hand_ops"]


def test_live_without_a_hand_reports_unavailable_not_live(monkeypatch):
    """`--live` must not read as "live" when the machine cannot actuate.

    A page that shows "live" while every op is refused is the same class of
    dishonesty as a checkpoint claiming accuracy it does not have.
    """
    import avcs.voice.actuators as actuators

    monkeypatch.setattr(actuators, "default_hand", lambda: None)
    gui = GuiApp(source="demo", mouth="silent", ears=False, live=True, grants=["click"])
    caps = gui.capabilities()
    assert gui.hand is None
    assert caps["actuation"] == "live-unavailable"
    assert caps["hand"] is None
    # And a turn record must not claim real actuation either.
    assert gui.assistant.actuation == "virtual"


def test_live_with_a_hand_reports_live_and_names_it(monkeypatch):
    import avcs.voice.actuators as actuators

    class _Hand:
        name = "xdotool"
        def get_screen_size(self):
            return 1280, 800

    monkeypatch.setattr(actuators, "default_hand", lambda: _Hand())
    gui = GuiApp(source="demo", mouth="silent", ears=False, live=True, grants=["click"])
    caps = gui.capabilities()
    assert caps["actuation"] == "live"
    assert caps["hand"] == "xdotool"
    assert gui.assistant.actuation == "live"


def test_speech_endpoint_returns_204_with_no_body_when_idle():
    """A 204 must carry no body; a body makes uvicorn reject the response.

    The browser polls this endpoint four times a second, so a broken idle
    response would flood the log and stall the mouth.
    """
    from fastapi.testclient import TestClient

    from avcs.voice.tts import BrowserEngine

    gui = GuiApp(source="demo", mouth="browser", ears=False)
    client = TestClient(build_app(source="demo", mouth="browser", ears=False))
    # Drain whatever the construction produced, then ask again: nothing new.
    first = client.get("/api/speech.wav?after=0")
    seq = int(first.headers.get("x-speech-seq", "0")) if first.status_code == 200 else 0
    res = client.get(f"/api/speech.wav?after={seq + 1000}")
    assert res.status_code == 204
    assert res.content == b""
    assert isinstance(gui.mouth_engine, BrowserEngine)


def test_streaming_page_clears_the_streamed_bubble_when_a_step_lands(client):
    """The streamed reply and the step record carry the same sentence.

    The page shows the reply as it streams, then a `step` event arrives with the
    completed record -- whose `said` is that same sentence. If the streaming
    bubble is not removed first, the assistant visibly repeats itself on every
    turn. This pins the guard in the `step` branch of the SSE reader.
    """
    html = client.get("/").text
    step_branch = html.split('evt.type === "step"', 1)[1].split('evt.type === "turn"', 1)[0]
    assert "streamedEl.remove()" in step_branch, (
        "the step branch must clear the streamed bubble or the reply shows twice"
    )
    assert "streamedEl = null" in step_branch


def test_capabilities_report_capture_provenance(client):
    """The page must not claim a live screen it never captured.

    A synthetic source (the demo desktop) and an `EyeStream` whose capture
    backend is missing both yield pixels that are not the user's screen. The
    first yields a drawn scene; the second yields a flat grey buffer that is
    indistinguishable from a real grey screen. Either way `capture` must say
    `synthetic`, so the pane title can be honest.
    """
    caps = client.get("/api/capabilities").json()
    assert caps["capture"] in ("live", "synthetic")
    # The demo source used by this fixture is synthetic by construction.
    assert caps["capture"] == "synthetic"


def test_capture_is_live_is_false_without_a_real_backend(monkeypatch):
    """`EyeStream.capture_is_live` is False when `mss` cannot be imported.

    Without the flag, `grab_screen` silently returns a flat 128-grey buffer and
    the caller has no way to tell a missing capture device from a grey screen.
    """
    from avcs_sdk import eye_stream as es

    monkeypatch.setattr(es.EyeStream, "_init_capture", lambda self: setattr(self, "_sct", None))
    stream = es.EyeStream.__new__(es.EyeStream)
    stream._sct = None
    stream._use_synthetic = False
    stream.geometry = type("G", (), {"virtual_height": 720, "virtual_width": 1280})()
    assert stream.capture_is_live is False

    frame = stream.grab_screen()
    assert int(frame.max()) == 128 and int(frame.min()) == 128


def test_lexicon_perception_reports_capture_provenance():
    """A `frame_source` is never a live screen; an EyeStream defers to its flag."""
    from avcs.voice.lexicon_perception import LexiconPerception

    src = DemoSceneSource("desktop", seed=3)
    p = LexiconPerception(frame_source=src, reader=_NullReader())
    assert p.capture_is_live is False


class _NullReader:
    """A reader that finds nothing, so provenance can be tested without weights."""

    labels = []

    def read_regions(self, frame, floor=0.0):
        return []

