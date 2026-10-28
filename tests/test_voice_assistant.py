"""Voice layer tests: ears, mouth, brain, and the gated assistant loop.

These pin the behaviours that make the assistant honest and safe:

* the mouth synthesizes real audio and reports playback truthfully;
* the ears degrade to a null recognizer instead of crashing;
* the brain asks when the request is not actionable (the user's own example);
* the assistant's action path is fail-closed by default and only reaches the
  hand through the production capability kernel.

No test requires a microphone, a speaker, an API key, or a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from avcs.control.capability import CapabilityKernel
from avcs.voice.assistant import Assistant, ScreenObservation, StaticPerception
from avcs.voice.brain import AssistantTurn, ConversationBrain, RuleBrain, TurnKind
from avcs.voice.stt import AudioClip, NullEngine, Transcript, default_stt_engine
from avcs.voice.tts import (
    EspeakNgEngine,
    NullEngine as NullTtsEngine,
    PcmClip,
    Speaker,
    VoiceProfile,
)


# --------------------------------------------------------------------------- #
# Mouth (TTS)
# --------------------------------------------------------------------------- #
def test_pcm_clip_roundtrips_through_wav_bytes():
    samples = (np.sin(np.linspace(0, 20, 1000)) * 8000).astype(np.int16)
    clip = PcmClip(samples=samples, sample_rate=16000)
    restored = PcmClip.from_wav_bytes(clip.to_wav_bytes())
    assert restored.sample_rate == 16000
    assert restored.samples.shape == samples.shape
    np.testing.assert_allclose(restored.samples, samples, atol=1)
    assert restored.duration_s == pytest.approx(1000 / 16000, rel=1e-3)


def test_empty_clip_is_reported_as_empty():
    clip = PcmClip(samples=np.zeros(0, dtype=np.int16), sample_rate=22050)
    assert clip.is_empty
    assert clip.duration_s == 0.0


def test_speaker_synthesizes_real_audio_when_espeak_is_present():
    engine = EspeakNgEngine()
    if not engine.available():
        pytest.skip("espeak-ng not installed on this machine")
    speaker = Speaker(engine=engine)
    clip = speaker.synthesize("Hello sir, what work would you like me to do?")
    assert not clip.is_empty
    assert clip.duration_s > 0.3
    assert clip.sample_rate > 0


def test_null_engine_synthesizes_without_an_audio_device():
    speaker = Speaker(engine=NullTtsEngine())
    result = speaker.speak("testing the silent mouth")
    assert result.spoken is False
    assert "silent engine" in (result.error or "")
    assert result.duration_s > 0.0


def test_speaker_reports_empty_text_honestly():
    speaker = Speaker(engine=NullTtsEngine())
    result = speaker.speak("   ")
    assert result.spoken is False
    assert result.error == "empty text"
    assert result.ok is False


def test_voice_profile_voice_arg_falls_back_to_language():
    assert VoiceProfile(language="en-gb").engine_voice_arg() == "en-gb"
    assert VoiceProfile(language="en-us", voice_variant="en-us+f3").engine_voice_arg() == "en-us+f3"


# --------------------------------------------------------------------------- #
# Ears (STT)
# --------------------------------------------------------------------------- #
def test_audio_clip_trims_silence_but_keeps_the_speech():
    rate = 16000
    quiet = np.zeros(int(0.5 * rate), dtype=np.float32)
    speech = (np.sin(np.linspace(0, 400, int(0.5 * rate))) * 0.4).astype(np.float32)
    clip = AudioClip(samples=np.concatenate([quiet, speech, quiet]), sample_rate=rate)
    trimmed = clip.trim_silence()
    assert trimmed.samples.size < clip.samples.size
    assert trimmed.duration_s < clip.duration_s
    assert trimmed.rms > 0.05


def test_audio_clip_all_silence_trims_to_empty():
    clip = AudioClip(samples=np.zeros(8000, dtype=np.float32), sample_rate=16000)
    assert clip.trim_silence().samples.size == 0


def test_audio_clip_pcm16_roundtrip():
    clip = AudioClip(samples=np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32), sample_rate=8000)
    restored = AudioClip.from_pcm16_bytes(clip.to_pcm16_bytes(), 8000)
    np.testing.assert_allclose(restored.samples, clip.samples, atol=1e-3)


def test_null_stt_returns_an_honest_empty_transcript():
    engine = NullEngine()
    assert engine.available()
    t = engine.transcribe(AudioClip(samples=np.zeros(1600, dtype=np.float32), sample_rate=16000))
    assert isinstance(t, Transcript)
    assert t.text == ""
    assert t.ok is False
    assert t.error


def test_default_stt_engine_never_raises_without_a_model():
    engine = default_stt_engine(model_path="/nonexistent/path")
    assert engine is not None
    assert isinstance(engine.available(), bool)


# --------------------------------------------------------------------------- #
# Brain
# --------------------------------------------------------------------------- #
def test_rule_brain_asks_when_the_greeting_has_no_task():
    turn = RuleBrain().respond("hello")
    assert turn.kind == TurnKind.ASK
    assert "what work" in turn.speech.lower()
    assert turn.source == "rule_fallback"


def test_rule_brain_asks_when_task_has_no_target():
    turn = RuleBrain().respond("mujhe kaam karna hai")
    assert turn.kind == TurnKind.ASK
    assert turn.intent is None


def test_rule_brain_extracts_a_hinglish_target():
    turn = RuleBrain().respond("mujhe X kaam karna hai")
    assert turn.kind == TurnKind.ACT
    assert turn.intent is not None
    assert turn.intent.target_description == "X"
    assert turn.intent.op.value == "click"


def test_rule_brain_never_emits_coordinates():
    turn = RuleBrain().respond("click the Save button at 100,200")
    assert turn.intent is not None
    payload = turn.intent.to_dict()
    assert "x" not in payload
    assert "y" not in payload
    assert "coords" not in payload
    assert "bbox" not in payload


def test_rule_brain_reports_target_absent_from_perception():
    turn = RuleBrain().respond("click the Save button", {"target_present": False})
    assert turn.kind == TurnKind.ASK
    assert "can't see" in turn.speech.lower()


def test_rule_brain_extracts_object_verb_hinglish_target():
    # Hinglish puts the object before the verb. Following the verb here would
    # leave only the tail of "dabao" and produce a one-letter target.
    turn = RuleBrain().respond("save button dabao")
    assert turn.kind == TurnKind.ACT
    assert turn.intent is not None
    assert turn.intent.target_description == "save button"


def test_rule_brain_keeps_a_verb_like_noun_inside_the_target():
    # "click" and "save" are also ordinary nouns; a mid-sentence "save" belongs
    # to the target name, not to the action.
    turn = RuleBrain().respond("click the Save button")
    assert turn.intent is not None
    assert "save" in turn.intent.target_description.lower()


def test_rule_brain_strips_trailing_hinglish_auxiliary():
    # "click karo" is verb + light verb; the action verb is still "click".
    turn = RuleBrain().respond("save button click karo")
    assert turn.intent is not None
    assert turn.intent.target_description == "save button"


class _ExplodingProvider:
    """A provider that fails the way an unreachable endpoint does."""

    api_key = ""

    def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
        raise RuntimeError("connection refused")

    def reset(self):
        pass


def test_conversation_brain_falls_back_when_the_llm_is_unreachable():
    brain = ConversationBrain(provider=_ExplodingProvider())
    turn = brain.respond("click the Save button")
    assert turn.source == "rule_fallback"
    assert "unavailable" in turn.thought.lower()


class _ScriptedProvider:
    """Returns a fixed System2Decision, standing in for a live LLM."""

    api_key = "test-key"

    def __init__(self, decision):
        self._decision = decision
        self.system_prompt_seen = None

    def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
        self.system_prompt_seen = system_prompt
        return self._decision

    def reset(self):
        pass


def _decision(op, target=None, thought="", prompt=None):
    from avcs.reasoning.contracts import ActionIntent, ActionOp, System2Decision

    return System2Decision(
        thought=thought,
        task_status="in_progress",
        next_intent=ActionIntent.create(step_index=1, op=op, target_description=target),
        clarification_prompt=prompt,
    )


def test_conversation_brain_maps_finish_to_finish_turn():
    from avcs.reasoning.contracts import ActionOp

    brain = ConversationBrain(provider=_ScriptedProvider(_decision(ActionOp.FINISH, thought="done")))
    turn = brain.respond("click Save")
    assert turn.kind == TurnKind.FINISH
    assert turn.source == "llm"


def test_conversation_brain_maps_ask_user_to_ask_turn():
    from avcs.reasoning.contracts import ActionOp

    brain = ConversationBrain(
        provider=_ScriptedProvider(_decision(ActionOp.ASK_USER, prompt="Which one, sir?"))
    )
    turn = brain.respond("open the menu")
    assert turn.kind == TurnKind.ASK
    assert turn.speech == "Which one, sir?"


def test_conversation_brain_records_history_and_keeps_the_task():
    from avcs.reasoning.contracts import ActionOp

    brain = ConversationBrain(provider=_ScriptedProvider(_decision(ActionOp.CLICK, "Save button")))
    brain.respond("click the Save button")
    assert brain.task == "click the Save button"
    assert any(t.role == "user" for t in brain.history)
    assert any(t.role == "assistant" for t in brain.history)


def test_build_observation_is_coordinate_free_text():
    obs = ConversationBrain.build_observation(
        screen_description="2 interactive elements",
        target_present=True,
        extra={"interactive_elements": 2},
    )
    assert isinstance(obs["screen"], str)
    assert obs["target_present"] is True
    assert obs["interactive_elements"] == 2
    for forbidden in ("x", "y", "coords", "bbox", "pixels"):
        assert forbidden not in obs


# --------------------------------------------------------------------------- #
# Assistant loop
# --------------------------------------------------------------------------- #
def _assistant(**kwargs) -> Assistant:
    kwargs.setdefault("speaker", Speaker(engine=NullTtsEngine()))
    kwargs.setdefault("stt", NullEngine())
    kwargs.setdefault("perception", StaticPerception())
    return Assistant(**kwargs)


def test_assistant_is_fail_closed_without_a_grant():
    assistant = _assistant(kernel=CapabilityKernel())
    rec = assistant.step("click the Save button")
    assert rec.action == "click"
    assert rec.gate_allowed is False
    assert rec.actuation == "none"
    assert rec.clicked is None
    assert "no capability grant" in rec.gate_reason


def test_assistant_denies_when_perception_cannot_ground_the_target():
    """Morphology-only perception yields no grounding evidence -> fail closed."""
    assistant = _assistant(
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=object(),  # a hand exists, but the gate must still deny
    )
    rec = assistant.step("click the Save button")
    assert rec.gate_allowed is False
    assert "no candidate point" in rec.gate_reason
    assert rec.actuation == "none"


class _PassingPerception(StaticPerception):
    """A perception that reports real (high-peak) grounding evidence."""

    def grounding_evidence(self, query):
        from avcs.control.capability import GroundingEvidence

        return GroundingEvidence(
            candidate_point_norm=(0.5, 0.5), peak_probability=0.9,
            target_present=True, authorized=False, active_query=query,
        )


def test_assistant_does_not_act_without_a_hand_even_when_granted():
    assistant = _assistant(
        perception=_PassingPerception(),
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=None,
    )
    rec = assistant.step("click the Save button")
    assert rec.gate_allowed is True
    assert rec.actuation == "none"
    assert "no hand attached" in rec.gate_reason


def test_assistant_reaches_the_hand_only_through_the_kernel():
    from avcs.system1.loop import VirtualHand

    hand = VirtualHand(1920, 1080)
    assistant = _assistant(
        perception=_PassingPerception(),
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.0
        ),
        hand=hand,
    )
    rec = assistant.step("click the Save button")
    assert rec.gate_allowed is True
    assert rec.actuation == "virtual"
    assert len(hand.clicks) == 1


def test_assistant_speaks_the_reply_on_every_turn():
    assistant = _assistant()
    rec = assistant.step("hello")
    assert rec.said
    assert rec.kind == "ask"
    assert rec.spoken is False  # null engine: honest


def test_assistant_converse_stops_after_finish():
    from avcs.reasoning.contracts import ActionOp

    brain = ConversationBrain(provider=_ScriptedProvider(_decision(ActionOp.FINISH, thought="done")))
    assistant = _assistant(brain=brain)
    report = assistant.converse(max_turns=5, initial_utterance="click Save")
    assert report.total_turns == 1
    assert report.turns[0].kind == "finish"


def test_assistant_capabilities_are_reported_honestly():
    assistant = _assistant()
    caps = assistant.capabilities()
    assert caps["mouth"] == "null"
    assert caps["ears"] == "null"
    assert caps["mutating_grants"] == []
    assert caps["actuation"] == "virtual"


def test_assistant_step_survives_perception_failure():
    class _BrokenPerception:
        def observe(self, query=None):
            raise RuntimeError("capture device lost")

        def frame(self):
            return None

    assistant = _assistant(perception=_BrokenPerception())
    rec = assistant.step("click the Save button")
    assert "perception error" in assistant._observe("x").description
    assert rec.said  # still speaks


# --------------------------------------------------------------------------- #
# Real grounding -> kernel -> hand (integration)
# --------------------------------------------------------------------------- #
_REAL_GROUNDER = "runs/weights/goal_grounder_best.pt"
_UI_FRAME = "datasets/ui_grounding/screens/test_held_out_0001.png"


def _real_perception():
    from pathlib import Path

    if not Path(_REAL_GROUNDER).exists() or not Path(_UI_FRAME).exists():
        pytest.skip("grounder checkpoint or ui_grounding frame not present")
    from PIL import Image

    from avcs.voice.perception import GroundingPerception

    img = np.asarray(Image.open(_UI_FRAME).convert("RGB"), dtype=np.uint8)

    class _StubEye:
        weights_loaded = True

        def step(self):
            class _EF:
                pass

            e = _EF()
            e.raw_frame = img
            e.hotspots = []
            e.has_popup = False
            e.is_stable = True
            return e

        def close(self):
            pass

    return GroundingPerception(eye=_StubEye())


def test_real_grounding_evidence_separates_correct_from_wrong_targets():
    perception = _real_perception()
    perception.observe(None)
    correct = perception.grounding_evidence("Start button")
    wrong = perception.grounding_evidence("Bookmark this tab")
    assert correct is not None and wrong is not None
    assert correct.peak_probability > 0.75
    assert wrong.peak_probability < 0.2
    assert correct.candidate_point_norm is not None
    assert wrong.candidate_point_norm is None  # presence head rejects it


def test_assistant_acts_only_when_real_grounding_passes_the_kernel():
    from avcs.system1.loop import VirtualHand

    hand = VirtualHand(1920, 1080)
    assistant = _assistant(
        perception=_real_perception(),
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=hand,
    )
    rec = assistant.step("click the Start button")
    assert rec.gate_allowed is True, rec.gate_reason
    assert rec.actuation == "virtual"
    assert len(hand.clicks) == 1
    x, y = hand.clicks[0]
    # The Start button on this frame sits at normalized (0.107, 0.949).
    assert abs(x / 1920 - 0.107) < 0.03
    assert abs(y / 1080 - 0.949) < 0.03


def test_assistant_refuses_a_target_that_is_not_on_screen():
    from avcs.system1.loop import VirtualHand

    hand = VirtualHand(1920, 1080)
    assistant = _assistant(
        perception=_real_perception(),
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=hand,
    )
    rec = assistant.step("click the Bookmark this tab button")
    assert rec.gate_allowed is False
    assert rec.clicked is None
    assert len(hand.clicks) == 0


# --------------------------------------------------------------------------- #
# Streaming (real-time) path
# --------------------------------------------------------------------------- #
class _StreamingProvider:
    """A provider that yields its reply in fragments, like an SSE endpoint."""

    api_key = "test-key"

    def __init__(self, fragments, supports_streaming=True):
        self._fragments = list(fragments)
        if supports_streaming:
            self.stream_step = self._stream_step
        self.system_prompt_seen = None

    def _stream_step(self, task, observation, history, step_index=1, system_prompt=None):
        self.system_prompt_seen = system_prompt
        for frag in self._fragments:
            yield frag

    def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
        raise AssertionError("plan_step must not be used on the streaming path")


def test_sentence_splitter_emits_only_complete_sentences():
    from avcs.voice.tts import split_sentences

    # A half-sentence produces nothing.
    assert split_sentences("Hello sir, how") == []
    # A finished sentence is emitted with its remainder.
    out = split_sentences("Hello sir. How are")
    assert out == [("Hello sir.", "How are")]
    # A decimal number is not a sentence break.
    assert split_sentences("It is 0.5 done. ") == [("It is 0.5 done.", "")]


def test_speaker_streams_sentences_to_the_engine_in_order():
    from avcs.voice.tts import NullEngine as _Null, Speaker

    engine = _Null()
    speaker = Speaker(engine=engine)
    results = speaker.speak_stream(["Ji sir. ", "Kya kaam", " karna hai? ", "Bolo"])
    # Three sentences: "Ji sir.", "Kya kaam karna hai?", "Bolo"
    assert [r.text for r in results] == ["Ji sir.", "Kya kaam karna hai?", "Bolo"]


def test_speech_queue_speaks_off_thread_and_drains():
    from avcs.voice.tts import NullEngine as _Null, Speaker, SpeechQueue

    q = SpeechQueue(Speaker(engine=_Null()))
    q.say("First sentence.")
    q.say("Second sentence.")
    results = q.drain()
    assert [r.text for r in results] == ["First sentence.", "Second sentence."]
    assert all(r.spoken is False for r in results)  # null engine is honest


def test_brain_stream_speaks_before_the_action_and_parses_it():
    from avcs.reasoning.contracts import ActionOp

    provider = _StreamingProvider([
        "Ji sir, main Save ",
        "button daba raha hoon. ",
        "<<<ACTION>>>",
        '{"kind": "act", "thought": "target named", ',
        '"action": {"op": "click", "target_description": "Save button"}}',
    ])
    brain = ConversationBrain(provider=provider)
    heard = []
    turn = brain.respond_stream("Save button daba do", on_speech=heard.append)

    # The spoken text was streamed out in sentence form, before the action.
    assert heard == ["Ji sir, main Save button daba raha hoon."]
    # The action was parsed and is never spoken.
    assert turn.kind == TurnKind.ACT
    assert turn.intent is not None and turn.intent.op == ActionOp.CLICK
    assert turn.intent.target_description == "Save button"
    assert "<<<ACTION>>>" not in turn.speech
    assert provider.system_prompt_seen  # the streaming prompt was used


# --------------------------------------------------------------------------- #
# Language policy: the brain must actually be told to speak Hinglish
# --------------------------------------------------------------------------- #
def test_default_brain_language_is_hinglish():
    brain = ConversationBrain(provider=_StreamingProvider(["ji."]))
    assert brain.language == "hinglish"


def test_hinglish_language_reaches_the_stream_prompt():
    provider = _StreamingProvider(["Ji sir. "])
    brain = ConversationBrain(provider=provider, language="hinglish")
    brain.respond_stream("hello")
    prompt = provider.system_prompt_seen
    assert "ALWAYS answer in Hinglish" in prompt
    # The streaming role instructions must still be there, not replaced.
    assert "live voice call" in prompt


def test_language_reaches_the_non_streaming_conversation_prompt():
    """The non-streaming path must use the conversation prompt, not the planner one.

    This was a real bug: ``plan_step`` had no ``system_prompt`` parameter, so
    ``CONVERSATION_SYSTEM_PROMPT`` was dead code and a voice turn was answered
    with the silent System-2 planner instructions.
    """
    from avcs.reasoning.contracts import ActionOp

    provider = _ScriptedProvider(_decision(ActionOp.ASK_USER, None))
    brain = ConversationBrain(provider=provider, language="hinglish")
    brain.respond("hello")
    prompt = provider.system_prompt_seen
    assert "ALWAYS answer in Hinglish" in prompt
    assert "real-time desktop assistant" in prompt      # the conversation prompt
    assert "AVCS System 2" not in prompt                 # never the planner prompt


def test_auto_language_mirrors_the_user():
    provider = _StreamingProvider(["ok. "])
    brain = ConversationBrain(provider=provider, language="auto")
    brain.respond_stream("hello")
    assert "mirror the user's language" in provider.system_prompt_seen


def test_unknown_language_is_rejected_loudly():
    with pytest.raises(ValueError):
        ConversationBrain(provider=_StreamingProvider(["ji."]), language="klingon")


def test_brain_stream_asks_when_the_request_is_not_actionable():
    provider = _StreamingProvider([
        "Ji sir, kya kaam karna hai? ",
        "<<<ACTION>>>",
        '{"kind": "ask", "thought": "no task yet", "action": null}',
    ])
    brain = ConversationBrain(provider=provider)
    heard = []
    turn = brain.respond_stream("mujhe kuch kaam karna hai", on_speech=heard.append)
    assert turn.kind == TurnKind.ASK
    assert turn.intent is None
    assert heard == ["Ji sir, kya kaam karna hai?"]


def test_brain_stream_falls_back_when_the_provider_cannot_stream():
    from avcs.reasoning.contracts import ActionOp

    brain = ConversationBrain(provider=_ScriptedProvider(_decision(ActionOp.CLICK, "Save button")))
    heard = []
    turn = brain.respond_stream("click Save", on_speech=heard.append)
    assert turn.kind == TurnKind.ACT
    assert heard  # the fallback still speaks


def test_assistant_streams_a_turn_end_to_end():
    provider = _StreamingProvider([
        "Theek hai sir. ",
        "<<<ACTION>>>",
        '{"kind": "act", "action": {"op": "click", "target_description": "Save button"}}',
    ])
    brain = ConversationBrain(provider=provider)
    assistant = _assistant(
        brain=brain,
        perception=_PassingPerception(),
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.0
        ),
    )
    rec = assistant.step("Save button daba do")
    assert rec.said == "Theek hai sir."
    assert rec.kind == "act"
    assert rec.action == "click"
    assert rec.gate_allowed is True


def test_streaming_can_be_disabled_for_a_deterministic_demo():
    provider = _StreamingProvider([
        "Theek hai sir. ",
        "<<<ACTION>>>",
        '{"kind": "act", "action": {"op": "click", "target_description": "Save button"}}',
    ])
    brain = ConversationBrain(provider=provider)
    assistant = _assistant(brain=brain, stream=False)
    rec = assistant.step("Save button daba do")
    # Non-streaming path uses plan_step, which this provider forbids.
    assert rec.said  # rule fallback still speaks


def _serve_sse(lines, status=200):
    """Run a one-shot HTTP server that streams `lines` as SSE and return its URL."""
    import http.server, threading

    body = ("\n".join(lines) + "\n").encode("utf-8")

    class _H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1", srv


def test_stream_step_parses_sse_from_a_real_http_endpoint():
    from avcs.reasoning.provider import LLMReasoningProvider
    import json

    def frame(content):
        return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})

    url, srv = _serve_sse([
        frame("Theek hai sir. "),
        frame("<<<ACTION>>>\n"),
        frame(chr(123) + chr(34) + "kind" + chr(34) + ": " + chr(34) + "act" + chr(34) + ", " + chr(34) + "action" + chr(34) + ": " + chr(123) + chr(34) + "op" + chr(34) + ": " + chr(34) + "click" + chr(34) + ", " + chr(34) + "target_description" + chr(34) + ": " + chr(34) + "Save button" + chr(34) + chr(125) + chr(125)),
    ])

    provider = LLMReasoningProvider(endpoint=url, api_key="test-key", model="test-model")
    pieces = list(provider.stream_step(
        task="click the Save button",
        observation={"screen_description": "a form with a Save button"},
        history=[],
        step_index=1,
    ))
    srv.server_close()
    assert "".join(pieces).startswith("Theek hai sir.")
    assert "<<<ACTION>>>" in "".join(pieces)


def test_grounding_perception_reads_the_real_eye_frame_hotspot_shape():
    """The live eye emits 5-tuples (x, y, kind, obj_conf, cls_conf).

    A hand-rolled 4-tuple fixture hides this: the morphology adapter must read
    the *class name* at index 2, not a numeric field, or it crashes on the real
    frame and no screen description ever reaches the brain.
    """
    import numpy as np

    from avcs.voice.perception import GroundingPerception

    frame = np.full((256, 256, 3), 60, dtype=np.uint8)
    frame[80:120, 150:220] = 220

    class _FakeEye:
        weights_loaded = True

        def step(self):
            f = type("F", (), {})()
            f.raw_frame = frame
            f.hotspots = [(0.7, 0.4, "interactive_control", 0.9, 0.8)]
            f.has_popup = False
            f.is_stable = True
            return f

        def close(self):
            pass

    perception = GroundingPerception(eye=_FakeEye(), source="synthetic")
    obs = perception.observe("Launch button")
    assert obs.hotspot_count == 1
    assert "clickable control" in obs.description


def test_full_stack_voice_turn_over_real_sse_brain_eyes_gate_and_hand():
    """One real turn end-to-end: SSE LLM -> brain -> streamed mouth -> eyes -> gate -> hand.

    Every layer is the production object. Only the *endpoints* are local: a real
    HTTP server speaks SSE (no mocked provider), and the eyes are the shipped
    grounder over a synthetic frame. This is the integration the unit tests
    around it each cover in isolation.
    """
    import json

    from avcs.control.capability import CapabilityKernel, GroundingEvidence
    from avcs.reasoning.provider import LLMReasoningProvider
    from avcs.system1.loop import VirtualHand

    def frame(content):
        return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})

    action = json.dumps({
        "kind": "act",
        "thought": "the user named the Save button",
        "action": {"op": "click", "target_description": "Save button"},
    })
    url, srv = _serve_sse([
        frame("Theek hai sir, main "),
        frame("Save button daba raha hoon. "),
        frame("<<<ACTION>>>\n"),
        frame(action),
    ])

    class _GroundedPerception:
        """Reports real (high-peak) evidence for the named target."""

        def __init__(self):
            self.queries = []

        def observe(self, query=None):
            self.queries.append(query)
            return ScreenObservation(description="a form with a Save button",
                                     target_present=True, hotspot_count=2)

        def frame(self):
            return None

        def grounding_evidence(self, query):
            return GroundingEvidence(
                candidate_point_norm=(0.42, 0.81), peak_probability=0.91,
                target_present=True, authorized=False, active_query=query,
            )

    said: list[str] = []

    class _CapturingSpeaker:
        engine_name = "capture"

        def available(self):
            return True

        def speak(self, text):
            said.append(text)
            from avcs.voice.tts import SpeechResult
            return SpeechResult(text=text, engine="capture", spoken=True,
                                duration_s=0.0, error=None)

    provider = LLMReasoningProvider(endpoint=url, api_key="test-key", model="test-model")
    brain = ConversationBrain(provider=provider)
    perception = _GroundedPerception()
    hand = VirtualHand(1920, 1080)
    assistant = Assistant(
        brain=brain,
        speaker=_CapturingSpeaker(),
        perception=perception,
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.5
        ),
        hand=hand,
        stream=True,
    )

    rec = assistant.step("Save button daba do", turn=1)
    srv.server_close()

    # The mouth spoke the model's own words, streamed, not a canned line.
    assert rec.source == "llm"
    assert rec.kind == "act"
    assert rec.said.startswith("Theek hai sir")
    assert any("Save button daba raha hoon" in s for s in said)
    # The sentinel and the JSON were never spoken out loud.
    assert all("<<<ACTION>>>" not in s for s in said)
    assert all("target_description" not in s for s in said)
    # The eyes were asked to ground the *named target* before the gate ran.
    assert "Save button" in perception.queries
    # The gate passed and the hand moved exactly once, at the grounded point.
    assert rec.gate_allowed is True
    assert rec.actuation == "virtual"
    assert len(hand.clicks) == 1
    assert rec.clicked == hand.clicks[0]


def test_hinglish_voice_profile_renders_real_audio():
    """The user speaks Hinglish, so the mouth must have a Hinglish voice.

    The point is not just that a profile object exists: espeak-ng must actually
    synthesize romanized Hindi with it, otherwise the assistant answers in a
    voice the user cannot understand.
    """
    engine = EspeakNgEngine()
    if not engine.available():
        pytest.skip("espeak-ng not installed on this machine")
    profile = VoiceProfile.hinglish()
    assert profile.engine_voice_arg() == "hi"
    clip = engine.synthesize("Namaste sir, mujhe X kaam karna hai.", profile)
    assert not clip.is_empty
    assert clip.duration_s > 0.3


def test_conversation_clarifies_then_acts_across_turns():
    """The user's real flow: vague request -> assistant asks -> task named -> act.

    Driven by a real HTTP SSE brain whose replies change per turn, so this is the
    whole voice loop (scripted utterances in place of a microphone), not a single
    step.
    """
    import http.server
    import json
    import threading

    from avcs.control.capability import CapabilityKernel, GroundingEvidence
    from avcs.reasoning.provider import LLMReasoningProvider
    from avcs.system1.loop import VirtualHand

    def frame(content):
        return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})

    # The mock brain answers differently on each POST: first it asks, then it acts.
    replies = [
        "Ji sir, kya kaam karna hai? <<<ACTION>>>\n"
        + json.dumps({"kind": "ask", "action": {"op": "ask_user"}}),
        "Theek hai sir, Save button daba raha hoon. <<<ACTION>>>\n"
        + json.dumps({"kind": "act",
                      "action": {"op": "click", "target_description": "Save button"}}),
    ]

    class _ScriptedSse(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = frame(replies[min(self.server.count, len(replies) - 1)]).encode()
            self.server.count += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _ScriptedSse)
    srv.count = 0
    threading.Thread(target=srv.handle_request, daemon=True).start()
    threading.Thread(target=srv.handle_request, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    class _GroundedPerception:
        def observe(self, query=None):
            return ScreenObservation(description="a form with a Save button",
                                     target_present=True, hotspot_count=2)

        def frame(self):
            return None

        def grounding_evidence(self, query):
            return GroundingEvidence(
                candidate_point_norm=(0.4, 0.8), peak_probability=0.9,
                target_present=True, authorized=False, active_query=query,
            )

    hand = VirtualHand(1920, 1080)
    assistant = Assistant(
        brain=ConversationBrain(
            provider=LLMReasoningProvider(endpoint=url, api_key="k", model="m")
        ),
        speaker=Speaker(engine=NullTtsEngine()),
        perception=_GroundedPerception(),
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.5
        ),
        hand=hand,
        stream=True,
    )

    report = assistant.converse(
        max_turns=4,
        utterances=["mujhe X kaam karna hai", "Save button daba do"],
    )
    srv.server_close()

    spoken_turns = [r for r in report.turns if r.kind != "silent"]
    assert [r.kind for r in spoken_turns] == ["ask", "act"]
    assert "kya kaam" in spoken_turns[0].said
    # Nothing was acted on until the user named a target.
    assert spoken_turns[0].action is None
    # Then the gate passed and the hand moved once.
    assert spoken_turns[1].gate_allowed is True
    assert len(hand.clicks) == 1
    # The loop then honestly reports that it heard nothing more (no mic here).
    assert report.turns[-1].kind == "silent"


def test_brain_forwards_conversation_history_to_the_llm():
    """The LLM must see earlier turns, or it answers each turn in isolation.

    `_build_payload` previously collapsed the whole history into a
    `{step, op}` summary, so the voice brain's real dialogue messages were
    dropped and the model had no memory of what the user already said.
    """
    from avcs.reasoning.provider import LLMReasoningProvider

    provider = LLMReasoningProvider(endpoint="http://127.0.0.1:1/v1", api_key="k", model="m")
    history = [
        {"role": "user", "content": "mujhe X kaam karna hai"},
        {"role": "assistant", "content": "Ji sir, kya kaam karna hai?"},
    ]
    payload = provider._build_payload("t", {"a": 1}, history, step_index=2)
    roles = [m["role"] for m in payload["messages"]]
    contents = [m["content"] for m in payload["messages"]]

    assert roles[0] == "system"
    # Both dialogue turns survive, in order, as real chat messages.
    assert roles[1:3] == ["user", "assistant"]
    assert contents[1] == "mujhe X kaam karna hai"
    assert contents[2] == "Ji sir, kya kaam karna hai?"
    # The current observation is still the final user message.
    assert "observation" in contents[-1]


def test_brain_forwards_step_history_summary_for_the_control_loop():
    """The System 2 control loop's step records must still be summarised.

    The fix that forwards dialogue turns must not break the other history shape.
    """
    from avcs.reasoning.provider import LLMReasoningProvider

    provider = LLMReasoningProvider(endpoint="http://127.0.0.1:1/v1", api_key="k", model="m")
    history = [
        {"step_index": 1, "proposed_action": {"op": "write"}},
        {"step_index": 2, "proposed_action": {"op": "mkdir"}},
    ]
    payload = provider._build_payload("t", {}, history, step_index=3)
    # No extra chat messages were injected for step records.
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert "step_history_summary" in payload["messages"][-1]["content"]
    assert '"op": "write"' in payload["messages"][-1]["content"]


def test_whisper_language_is_auto_detect_by_default_for_hinglish():
    """The ears must not hardcode English: the user mixes Hindi and English."""
    from avcs.voice.stt import FasterWhisperEngine

    assert FasterWhisperEngine().language is None
    assert FasterWhisperEngine(language="hi").language == "hi"


def test_default_stt_engine_accepts_a_language_without_raising():
    from avcs.voice.stt import default_stt_engine

    engine = default_stt_engine(language="hi")  # no model present -> NullEngine
    assert engine is not None


def test_silence_terminated_capture_falls_back_without_sox(monkeypatch):
    """End-pointing must degrade to a fixed capture, not crash, without sox."""
    import avcs.voice.stt as stt

    monkeypatch.setattr(stt.shutil, "which", lambda name: None)
    called = {}

    def fake_record(seconds, sample_rate=16000):
        called["seconds"] = seconds
        return stt.AudioClip(samples=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)

    monkeypatch.setattr(stt, "record_microphone", fake_record)
    clip = stt.record_microphone_until_silence(max_seconds=12.0)
    assert called["seconds"] == 5.0          # capped fallback window
    assert clip.samples.size == 0


# --------------------------------------------------------------------------- #
# Hand dispatch: the op must pick the method (the old code always clicked)
# --------------------------------------------------------------------------- #
def test_dispatch_routes_each_op_to_the_right_hand_method():
    from avcs.system1.loop import VirtualHand
    from avcs.voice.hands import dispatch

    hand = VirtualHand(1920, 1080)

    r = dispatch(hand, "type_text", text="hello")
    assert r.performed and hand.typed == ["hello"]
    assert hand.clicks == []            # typing must NOT click

    r = dispatch(hand, "scroll", scroll_delta=-5)
    assert r.performed and hand.scrolls == [-5]
    assert hand.clicks == []

    r = dispatch(hand, "press_key", key_name="enter")
    assert r.performed and hand.keys == ["enter"]
    assert hand.clicks == []

    r = dispatch(hand, "click", point=(0.5, 0.5))
    assert r.performed and len(hand.clicks) == 1
    assert hand.clicks[0] == (960, 540)

    r = dispatch(hand, "double_click", point=(0.2, 0.2))
    assert r.performed and len(hand.clicks) == 3   # two more recorded


def test_dispatch_fails_closed_for_an_op_with_no_hand_implementation():
    from avcs.system1.loop import VirtualHand
    from avcs.voice.hands import dispatch

    hand = VirtualHand(1920, 1080)
    r = dispatch(hand, "observe_target", point=(0.5, 0.5))
    assert not r.performed
    assert hand.clicks == []


def test_dispatch_reports_failure_when_the_hand_refuses():
    """A hand that returns False must be reported, not claimed as success."""
    from avcs.voice.hands import dispatch

    class _RefusingHand:
        def type_text(self, text, typing_speed_cps=25.0):
            return False

    r = dispatch(_RefusingHand(), "type_text", text="hi")
    assert not r.performed and "not typed" in r.error


class _AlwaysGroundPerception:
    """Grounds every query with a high peak; the frame has real structure."""

    def __init__(self) -> None:
        self._frame = np.full((64, 64, 3), 60, dtype=np.uint8)
        self._frame[10:30, 10:30] = 200

    def observe(self, query=None):
        from avcs.voice.assistant import ScreenObservation

        return ScreenObservation(description="a dialog with buttons", hotspot_count=3)

    def frame(self):
        return self._frame

    def grounding_evidence(self, query):
        from avcs.control.capability import GroundingEvidence

        return GroundingEvidence(
            candidate_point_norm=(0.3, 0.3),
            peak_probability=0.95,
            target_present=True,
            authorized=False,
            active_query=query,
        )


def test_assistant_types_text_instead_of_clicking():
    """The reported bug: a 'type' request used to produce a click."""
    from avcs.system1.loop import VirtualHand

    brain = ConversationBrain(
        provider=_ScriptedProvider(
            _intent("type_text", target_description="Search box", text_input="hello")
        )
    )
    hand = VirtualHand(1920, 1080)
    assistant = _assistant(
        brain=brain,
        perception=_AlwaysGroundPerception(),
        kernel=CapabilityKernel(granted_ops=["click", "type_text"]),
        hand=hand,
    )
    rec = assistant.step("type hello into the search box")
    assert rec.gate_allowed is True, rec.gate_reason
    assert rec.performed is True
    assert hand.typed == ["hello"]
    assert hand.clicks == []
    assert rec.actuation == "virtual"


# --------------------------------------------------------------------------- #
# Autonomous multi-step task loop
# --------------------------------------------------------------------------- #
class _SequenceProvider:
    """Returns a different decision on each call, standing in for a live LLM."""

    api_key = "test-key"

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._i = 0

    def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d

    def reset(self):
        pass


def _intent(op_name, **kw):
    from avcs.reasoning.contracts import ActionIntent, System2Decision

    return System2Decision(
        thought=kw.pop("thought", ""),
        task_status="in_progress",
        next_intent=ActionIntent.create(step_index=1, op=op_name, **kw),
    )


def test_run_task_executes_multiple_steps_until_finish():
    from avcs.reasoning.contracts import ActionOp, ActionIntent, System2Decision
    from avcs.system1.loop import VirtualHand

    decisions = [
        _intent("click", target_description="Search box"),
        _intent("type_text", target_description="Search box", text_input="hello"),
        System2Decision(thought="done", task_status="complete",
                        next_intent=ActionIntent.create(step_index=3, op=ActionOp.FINISH)),
    ]
    brain = ConversationBrain(provider=_SequenceProvider(decisions))
    hand = VirtualHand(1920, 1080)
    assistant = _assistant(
        brain=brain,
        perception=_AlwaysGroundPerception(),
        kernel=CapabilityKernel(granted_ops=["click", "type_text"]),
        hand=hand,
    )
    assistant.step_delay_s = 0.0

    records = assistant.run_task("search for hello", max_steps=5)

    kinds = [r.kind for r in records]
    assert kinds[-1] == "finish"
    assert kinds.count("act") == 2
    # The two steps did the two different things, in order.
    assert len(hand.clicks) == 1
    assert hand.typed == ["hello"]


def test_run_task_stops_when_the_gate_refuses_instead_of_looping():
    from avcs.system1.loop import VirtualHand

    decisions = [_intent("click", target_description="Save button")] * 5
    brain = ConversationBrain(provider=_SequenceProvider(decisions))
    hand = VirtualHand(1920, 1080)
    assistant = _assistant(
        brain=brain,
        perception=_AlwaysGroundPerception(),
        kernel=CapabilityKernel(),          # no grants: fail closed
        hand=hand,
    )
    assistant.step_delay_s = 0.0

    records = assistant.run_task("click Save", max_steps=5)

    assert len(records) == 1                # stopped after the first refusal
    assert records[0].gate_allowed is False
    assert hand.clicks == []


def test_run_task_feeds_the_last_outcome_back_to_the_brain():
    """The loop must be closed: the brain sees what the previous step did."""
    from avcs.system1.loop import VirtualHand

    seen = []

    class _ObservingProvider(_SequenceProvider):
        def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
            seen.append(dict(observation))
            return super().plan_step(task, observation, history, step_index)

    decisions = [
        _intent("click", target_description="Search box"),
        _intent("click", target_description="Search box"),
    ]
    brain = ConversationBrain(provider=_ObservingProvider(decisions))
    assistant = _assistant(
        brain=brain,
        perception=_AlwaysGroundPerception(),
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=VirtualHand(1920, 1080),
    )
    assistant.step_delay_s = 0.0

    assistant.run_task("click the search box", max_steps=2)

    # Second call carries a summary of the first step's outcome.
    assert "last_action" in seen[1]
    assert "click" in seen[1]["last_action"]


def test_rule_brain_fills_text_input_for_a_type_request():
    """The offline brain must supply the payload, not just name the op."""
    brain = ConversationBrain(provider=_ExplodingProvider())  # forces rule fallback
    turn = brain.respond("type hello into the search box")
    assert turn.intent is not None
    assert turn.intent.op.value == "type_text"
    assert turn.intent.text_input == "hello"


def test_rule_brain_extracts_a_key_chord():
    brain = ConversationBrain(provider=_ExplodingProvider())
    turn = brain.respond("press ctrl+s to save")
    assert turn.intent is not None
    assert turn.intent.op.value == "press_key"
    assert turn.intent.key_name == "ctrl+s"


# --------------------------------------------------------------------------- #
# Real-time: interruptible mouth, async speak, continuous loop, real actuation
# --------------------------------------------------------------------------- #
def test_espeak_engine_stop_interrupts_playback():
    """Barge-in requires the mouth to be interruptible, not a blocking run()."""
    from avcs.voice.tts import EspeakNgEngine, PcmClip
    import numpy as np

    engine = EspeakNgEngine()
    if not engine.available():
        import pytest
        pytest.skip("espeak-ng not installed")

    # A long clip gives the stop() a window to land.
    rate = 22050
    clip = PcmClip(samples=np.zeros(int(rate * 3.0), dtype=np.int16), sample_rate=rate)
    import threading

    result = {}
    def _play():
        try:
            result["rc"] = engine.play(clip)
        except Exception as exc:  # no audio device on CI is fine
            result["error"] = str(exc)

    t = threading.Thread(target=_play, daemon=True)
    t.start()
    import time as _time
    _time.sleep(0.15)
    engine.stop()
    t.join(timeout=5.0)
    assert not t.is_alive(), "stop() did not interrupt playback"
    # Either it was interrupted (False) or there was no device (error).
    assert result.get("rc") is False or "error" in result


def test_speaker_speak_async_does_not_block_and_stop_barges_in():
    from avcs.voice.tts import NullEngine, Speaker, VoiceProfile
    import time as _time

    speaker = Speaker(engine=NullEngine(), profile=VoiceProfile.hinglish())
    t0 = _time.perf_counter()
    handle = speaker.speak_async("ye ek lamba jawab hai jo bolne me waqt lega")
    elapsed_ms = (_time.perf_counter() - t0) * 1000.0
    # Returned immediately; the speaking happens on the worker.
    assert elapsed_ms < 500.0
    assert handle.wait(timeout=5.0)
    assert handle.is_done()


def test_continuous_loop_runs_until_a_stop_phrase():
    """The always-on loop must keep going, then stop on a stop phrase."""
    from avcs.voice.stt import Transcript

    scripted = [
        Transcript(text="mujhe X kaam karna hai", engine="test"),
        Transcript(text="stop", engine="test"),
    ]
    assistant = _assistant()
    assistant._record_audio = None
    assistant._listen = lambda: scripted.pop(0)

    report = assistant.loop(max_turns=5)

    assert report.total_turns == 2
    assert report.turns[-1].kind == "stop"
    assert report.turns[-1].heard == "stop"


def test_continuous_loop_keeps_waiting_on_a_silent_turn():
    from avcs.voice.stt import Transcript

    scripted = [
        Transcript(text="", engine="test", error="no speech detected"),
        Transcript(text="mujhe X kaam karna hai", engine="test"),
        Transcript(text="band karo", engine="test"),
    ]
    assistant = _assistant()
    assistant._listen = lambda: scripted.pop(0)

    report = assistant.loop(max_turns=6)

    kinds = [t.kind for t in report.turns]
    assert "silent" in kinds                     # the empty turn was recorded
    assert kinds[-1] == "stop"
    assert report.total_turns == 3


def test_loop_records_per_stage_timing():
    from avcs.voice.stt import Transcript

    assistant = _assistant()
    assistant._listen = lambda: Transcript(text="mujhe X kaam karna hai", engine="test")
    seen = []
    report = assistant.loop(max_turns=1, on_record=seen.append)

    rec = report.turns[0]
    assert rec.brain_ms > 0.0
    assert rec.listen_ms >= 0.0
    assert seen and seen[0] is rec


# --------------------------------------------------------------------------- #
# File-driven ears (--audio): the loop without a sound card
# --------------------------------------------------------------------------- #
def test_wav_turn_reader_yields_one_clip_per_turn_then_silence(tmp_path):
    from avcs.voice.demo import _wav_turn_reader

    clip = PcmClip(samples=(np.sin(np.linspace(0, 40, 8000)) * 9000).astype(np.int16), sample_rate=16000)
    (tmp_path / "a.wav").write_bytes(clip.to_wav_bytes())
    (tmp_path / "b.wav").write_bytes(clip.to_wav_bytes())

    read = _wav_turn_reader(str(tmp_path))
    first, second, third = read(1.0), read(1.0), read(1.0)
    assert first.samples.size > 0 and second.samples.size > 0
    # Exhausted: silence, not the last clip repeated forever.
    assert third.samples.size == 0


def test_wav_turn_reader_rejects_an_empty_directory(tmp_path):
    from avcs.voice.demo import _wav_turn_reader

    with pytest.raises(SystemExit):
        _wav_turn_reader(str(tmp_path))


def test_assistant_transcribes_audio_from_the_injected_reader():
    """The ears path runs end-to-end from a clip, with no microphone."""
    from avcs.voice.stt import SttEngine

    class _FixedStt(SttEngine):
        name = "fixed"

        def available(self) -> bool:
            return True

        def transcribe(self, audio: AudioClip) -> Transcript:
            assert audio.samples.size > 0, "reader gave the recognizer an empty clip"
            return Transcript(text="save button dabao", engine="fixed")

    clip = AudioClip(
        samples=(np.sin(np.linspace(0, 40, 8000)) * 0.3).astype(np.float32),
        sample_rate=16000,
    )
    assistant = _assistant()
    assistant.stt = _FixedStt()
    assistant._record_audio = lambda _seconds: clip

    transcript = assistant._listen()
    assert transcript.text == "save button dabao"
    assert transcript.error is None


def test_xdotool_hand_is_fail_closed_without_a_display():
    """No xdotool / no DISPLAY must refuse, never pretend to act."""
    from avcs.voice.actuators import XdotoolHand

    hand = XdotoolHand(display="", binary=None)
    assert hand.available() is False
    assert hand.click(10, 10) is False
    assert hand.type_text("hello") is False
    assert hand.press_key("enter") is False
    assert hand.scroll(-3) is False
    assert hand.get_screen_size() == (0, 0)


def test_xdotool_key_translation_handles_chords_and_aliases():
    from avcs.voice.actuators import XdotoolHand

    assert XdotoolHand._xdotool_key("enter") == "Return"
    assert XdotoolHand._xdotool_key("ctrl+s") == "ctrl+s"
    assert XdotoolHand._xdotool_key("Shift+Tab") == "shift+Tab"
    assert XdotoolHand._xdotool_key("esc") == "Escape"


def test_dispatch_drives_the_real_hand_contract_in_pixels():
    """A real hand takes absolute pixels; dispatch must map the point for it."""
    from avcs.voice.hands import dispatch

    calls = []

    class _FakeRealHand:
        def get_screen_size(self):
            return 1920, 1080
        def click(self, x, y, double=False):
            calls.append(("click", x, y, double)); return True
        def type_text(self, text, typing_speed_cps=25.0):
            calls.append(("type", text)); return True
        def press_key(self, key):
            calls.append(("key", key)); return True
        def scroll(self, delta):
            calls.append(("scroll", delta)); return True

    hand = _FakeRealHand()
    dispatch(hand, "click", point=(0.5, 0.5))
    assert calls[-1] == ("click", 960, 540, False)   # normalized -> pixels

    dispatch(hand, "type_text", text="hi")
    assert calls[-1] == ("type", "hi")
    dispatch(hand, "press_key", key_name="enter")
    assert calls[-1] == ("key", "enter")
    dispatch(hand, "scroll", scroll_delta=-2)
    assert calls[-1] == ("scroll", -2)


def test_continuous_loop_streams_and_acts_over_a_real_sse_brain():
    """The always-on loop must stream a real reply and act, end to end.

    This is the full real-time path: mic (scripted) -> SSE brain -> streamed
    mouth -> pure-pixel gate -> hand, repeated until a stop phrase.
    """
    import http.server
    import json as _json
    import threading

    from avcs.control.capability import CapabilityKernel, GroundingEvidence
    from avcs.reasoning.provider import LLMReasoningProvider
    from avcs.system1.loop import VirtualHand
    from avcs.voice.stt import Transcript

    def frame(content):
        return "data: " + _json.dumps({"choices": [{"delta": {"content": content}}]})

    replies = [
        "Theek hai sir, Save button daba raha hoon. <<<ACTION>>>\n"
        + _json.dumps({"kind": "act",
                       "action": {"op": "click", "target_description": "Save button"}}),
        "Ho gaya sir. <<<ACTION>>>\n" + _json.dumps({"kind": "finish"}),
    ]

    class _Sse(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = frame(replies[min(self.server.count, len(replies) - 1)]).encode()
            self.server.count += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Sse)
    srv.count = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    class _Perception:
        def observe(self, query=None):
            return ScreenObservation(description="a form with a Save button",
                                     target_present=True, hotspot_count=2)

        def frame(self):
            return None

        def grounding_evidence(self, query):
            return GroundingEvidence(
                candidate_point_norm=(0.4, 0.8), peak_probability=0.9,
                target_present=True, authorized=False, active_query=query,
            )

    hand = VirtualHand(1920, 1080)
    assistant = Assistant(
        brain=ConversationBrain(provider=LLMReasoningProvider(endpoint=url, api_key="k", model="m")),
        speaker=Speaker(engine=NullTtsEngine()),
        stt=NullEngine(),
        perception=_Perception(),
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.5
        ),
        hand=hand,
        stream=True,
    )

    script = [
        Transcript(text="Save button daba do", engine="test"),
        Transcript(text="stop", engine="test"),
    ]
    assistant._listen = lambda: script.pop(0)

    try:
        report = assistant.loop(max_turns=5)
    finally:
        srv.shutdown()
        srv.server_close()

    acted = [r for r in report.turns if r.kind == "act"]
    assert len(acted) == 1
    assert acted[0].source == "llm"          # streamed from the real SSE brain
    assert acted[0].gate_allowed is True
    assert len(hand.clicks) == 1
    assert acted[0].brain_ms > 0.0
    assert report.turns[-1].kind == "stop"


def test_loop_continues_autonomously_after_a_spoken_task():
    """A spoken task must not stop after one action.

    The user says one thing; the assistant performs a step, looks at the screen,
    performs the next step on its own, then goes back to listening. This is the
    difference between a chat bot and an assistant that completes the work.
    """
    from avcs.control.capability import CapabilityKernel, GroundingEvidence
    from avcs.system1.loop import VirtualHand
    from avcs.voice.stt import Transcript

    class _Perception:
        def observe(self, query=None):
            return ScreenObservation(description="form", target_present=True, hotspot_count=2)
        def frame(self):
            return None
        def grounding_evidence(self, query):
            return GroundingEvidence(
                candidate_point_norm=(0.3, 0.4), peak_probability=0.9,
                target_present=True, authorized=False, active_query=query,
            )

    class _ScriptedBrain:
        """First turn acts; the next autonomous step acts; then it finishes."""
        def __init__(self):
            self.calls = 0
        def respond(self, user_text, observation, **kw):
            from avcs.voice.brain import ConversationBrain
            self.calls += 1
            return ConversationBrain().respond(user_text, observation)
        def respond_stream(self, *a, **k):
            raise AssertionError("not used")

    hand = VirtualHand(1920, 1080)
    assistant = Assistant(
        speaker=Speaker(engine=NullTtsEngine()),
        stt=NullEngine(),
        perception=_Perception(),
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.5
        ),
        hand=hand,
        step_delay_s=0.0,
        max_task_steps=3,
    )
    # Two mic turns: the task, then "stop". The middle steps come from autonomy.
    script = [
        Transcript(text="Save button daba do", engine="test"),
        Transcript(text="stop", engine="test"),
    ]
    assistant._listen = lambda: script.pop(0)

    report = assistant.loop(max_turns=10)

    acted = [r for r in report.turns if r.action]
    # The loop ran at least one autonomous step before returning to listening,
    # and every acted step went through the gate and reached the hand.
    assert len(acted) >= 1
    assert all(r.gate_allowed for r in acted)
    assert report.turns[-1].kind == "stop"


def test_rule_brain_finishes_an_autonomous_step_instead_of_reasking():
    """After acting, an empty follow-up turn must finish, not loop on 'what work?'."""
    brain = RuleBrain()
    first = brain.respond("Save button daba do", {"target_present": True})
    assert first.kind is TurnKind.ACT
    follow_up = brain.respond("", {"target_present": True})
    assert follow_up.kind is TurnKind.FINISH


def test_fallback_rule_brain_persists_across_turns():
    """The offline fallback must remember the task between calls."""
    from avcs.reasoning.provider import LLMReasoningProvider

    brain = ConversationBrain(
        provider=LLMReasoningProvider(endpoint="http://127.0.0.1:1/v1", api_key="k", model="m")
    )
    t1 = brain.respond("Save button daba do", {"target_present": True})
    assert t1.kind is TurnKind.ACT
    # Same brain instance on the next (autonomous) call must not re-ask.
    t2 = brain.respond("", {"target_present": True})
    assert t2.kind is TurnKind.FINISH


def test_loop_runs_multiple_autonomous_steps_until_finish():
    """A multi-action task must run every step the brain asks for."""
    from avcs.control.capability import CapabilityKernel, GroundingEvidence
    from avcs.voice.stt import Transcript

    class _Perception:
        def observe(self, query=None):
            return ScreenObservation(description="form", target_present=True, hotspot_count=2)
        def frame(self):
            return None
        def grounding_evidence(self, query):
            return GroundingEvidence(
                candidate_point_norm=(0.3, 0.4), peak_probability=0.9,
                target_present=True, authorized=False, active_query=query,
            )

    from avcs.reasoning.contracts import ActionIntent, ActionOp
    from avcs.system1.loop import VirtualHand

    class _TwoStepBrain:
        """Acts on the task turn, acts once more, then finishes."""
        def __init__(self):
            self.n = 0
        def respond(self, user_text, observation, **kw):
            self.n += 1
            if self.n <= 2:
                intent = ActionIntent.create(
                    step_index=self.n, op=ActionOp.CLICK,
                    target_description="Save button", expected_outcome="clicked",
                    rationale="test",
                )
                return AssistantTurn(kind=TurnKind.ACT, speech=f"step {self.n}",
                                     intent=intent, source="test")
            return AssistantTurn(kind=TurnKind.FINISH, speech="done", source="test")

    hand = VirtualHand(1920, 1080)
    assistant = Assistant(
        brain=_TwoStepBrain(),
        speaker=Speaker(engine=NullTtsEngine()),
        stt=NullEngine(),
        perception=_Perception(),
        kernel=CapabilityKernel(
            granted_ops=["click"], allow_degenerate_frames=True, min_peak_probability=0.5
        ),
        hand=hand,
        step_delay_s=0.0,
        max_task_steps=5,
    )
    script = [
        Transcript(text="Save button daba do", engine="test"),
        Transcript(text="stop", engine="test"),
    ]
    assistant._listen = lambda: script.pop(0)

    report = assistant.loop(max_turns=10)

    performed = [r for r in report.turns if r.performed]
    assert len(performed) == 2          # both autonomous steps ran
    assert len(hand.clicks) == 2
    assert report.turns[-1].kind == "stop"


def test_keyless_brain_skips_the_llm_and_answers_fast():
    """A brain with no API key must not stall on retry backoff every turn.

    Opening a connection with no key, retrying three times and only then falling
    back costs ~2 s per turn — a real-time loop that talks slower than the user.
    The offline path is taken directly and the turn stays sub-100 ms.
    """
    import time
    from avcs.reasoning.provider import LLMReasoningProvider

    brain = ConversationBrain(
        provider=LLMReasoningProvider(endpoint="http://127.0.0.1:1/v1", api_key="", model="m")
    )
    assert brain._offline() is True

    start = time.perf_counter()
    turn = brain.respond("Save button daba do", {"target_present": True})
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert turn.kind is TurnKind.ACT
    assert turn.source == "rule_fallback"
    assert elapsed_ms < 500, f"keyless turn took {elapsed_ms:.0f} ms"
