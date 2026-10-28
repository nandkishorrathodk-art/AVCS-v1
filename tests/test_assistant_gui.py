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
    gui = GuiApp(source="demo", mouth="silent", ears=False)
    assert gui.live is False
    assert gui.capabilities()["actuation"] == "virtual"
    assert gui.hand is None


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
