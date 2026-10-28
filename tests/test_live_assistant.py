"""Tests for the live-LLM assistant wiring.

These cover the pieces that turn the assistant from "plumbing that degrades to a
rule brain" into a working AI assistant, and each one pins a bug that was real:

- credential resolution (user env -> platform-managed key -> offline),
- the conversation schema vs the System 2 planner schema,
- reasoning-model replies where ``content`` is null,
- the task updating on a new spoken instruction but not on an autonomous step,
- a natural-language target ("Save button") reaching a closed-vocabulary label.

No network is used: the managed-key fetch is monkeypatched, and the provider is
driven through a fake transport.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from avcs.reasoning import credentials
from avcs.reasoning.credentials import (
    extract_message_text,
    resolve_llm_credentials,
)


# --------------------------------------------------------------------------- #
# extract_message_text: the reasoning-model trap
# --------------------------------------------------------------------------- #
def test_extract_message_text_reads_content():
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert extract_message_text(body) == "hello"


def test_extract_message_text_falls_through_null_content():
    """A reasoning model can spend the budget before emitting visible content.

    Indexing ``["content"]`` returned None and the caller crashed with
    ``'NoneType' object has no attribute 'strip'``; the probe then reported a
    healthy endpoint as failed.
    """
    body = {"choices": [{"message": {"content": None, "reasoning_content": "thought"}}]}
    assert extract_message_text(body) == "thought"


def test_extract_message_text_handles_nested_reasoning_dict():
    body = {"choices": [{"message": {"content": "", "reasoning": {"text": "deep"}}}]}
    assert extract_message_text(body) == "deep"


def test_extract_message_text_empty_body_is_empty_string():
    assert extract_message_text({}) == ""
    assert extract_message_text({"choices": []}) == ""


# --------------------------------------------------------------------------- #
# Credential resolution
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    """Isolate every test from the developer's own environment."""
    for var in (
        "AVCS_LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY",
        "AVCS_LLM_ENDPOINT", "AVCS_LLM_MODEL", "SESSION_API_KEY",
        "OH_LLM_API_KEY_REFRESH_URL", "OH_LLM_API_KEY_REFRESH_BASE_URLS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(credentials, "_cached_key", None, raising=False)
    monkeypatch.setattr(credentials, "_cached_at", 0.0, raising=False)
    monkeypatch.setattr(credentials, "_cached_models", None, raising=False)


def test_resolve_prefers_the_users_own_key(monkeypatch):
    monkeypatch.setenv("AVCS_LLM_API_KEY", "sk-user")
    monkeypatch.setenv("AVCS_LLM_ENDPOINT", "http://localhost:11434/v1")
    creds = resolve_llm_credentials()
    assert creds.api_key == "sk-user"
    assert creds.source == "user_env"
    assert creds.live is True


def test_resolve_uses_the_managed_key_when_no_user_key(monkeypatch):
    monkeypatch.setenv("OH_LLM_API_KEY_REFRESH_URL", "http://platform/refresh")
    monkeypatch.setenv("OH_LLM_API_KEY_REFRESH_BASE_URLS", "https://proxy.example/v1,https://b/v1")
    monkeypatch.setattr(credentials, "fetch_managed_key", lambda **_: "sk-managed")
    monkeypatch.setattr(credentials, "list_models", lambda *a, **k: ["deepseek-v4.1-flash"])
    creds = resolve_llm_credentials()
    assert creds.api_key == "sk-managed"
    assert creds.source == "managed"
    # The first advertised base URL is the proxy to use, not the vendor default.
    assert creds.endpoint == "https://proxy.example/v1"
    assert creds.model == "deepseek-v4.1-flash"


def test_resolve_explicit_empty_key_forces_offline(monkeypatch):
    """``api_key=""`` is a deliberate opt-out and must not reach the platform."""
    monkeypatch.setenv("OH_LLM_API_KEY_REFRESH_URL", "http://platform/refresh")
    monkeypatch.setattr(credentials, "fetch_managed_key", lambda **_: "sk-managed")
    creds = resolve_llm_credentials(api_key="")
    assert creds.api_key == ""
    assert creds.source == "none"
    assert creds.live is False


def test_resolve_offline_when_nothing_available(monkeypatch):
    monkeypatch.setattr(credentials, "fetch_managed_key", lambda **_: None)
    creds = resolve_llm_credentials()
    assert creds.live is False
    assert creds.source == "none"


def test_fetch_managed_key_returns_none_without_refresh_url():
    assert credentials.fetch_managed_key() is None


def test_mask_never_exposes_the_middle():
    masked = credentials.mask("sk-abcdefghijklmnop")
    assert "abcdefghij" not in masked
    assert masked.startswith("sk-abc")
    assert masked.endswith("mnop")


# --------------------------------------------------------------------------- #
# ConversationBrain: schema and task
# --------------------------------------------------------------------------- #
class _FakeProvider:
    """A provider whose raw reply is fixed, so the parse is what is tested."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_observation: Optional[Dict[str, Any]] = None

    def raw_step(self, task, observation, history, step_index=1, system_prompt=None):
        self.last_observation = observation
        return self.reply

    def plan_step(self, task, observation, history, step_index=1, system_prompt=None):
        raise AssertionError("the conversation path must use raw_step, not the planner")


def _brain_with(reply: str):
    from avcs.voice.brain import ConversationBrain

    return ConversationBrain(provider=_FakeProvider(reply))


def test_conversation_action_survives_a_non_streamed_reply():
    """The non-streaming path used the planner schema and dropped ``action``.

    The model speaks the conversation contract (``speech``/``kind``/``action``);
    parsing it as ``next_intent`` turned every turn into a bare ``observe``.
    """
    reply = json.dumps({
        "speech": "Theek hai sir.",
        "kind": "act",
        "action": {"type": "click", "target": "Save button"},
    })
    turn = _brain_with(reply).respond("save karo", {"screen": "toolbar"})
    assert turn.kind.value == "act"
    assert turn.speech == "Theek hai sir."
    assert turn.intent is not None
    assert turn.intent.op.value == "click"
    assert "save" in (turn.intent.target_description or "").lower()


def test_plain_reply_is_still_a_conversational_answer():
    turn = _brain_with("Ji sir, kya kaam karna hai?").respond("hello", {})
    assert turn.kind.value == "reply"
    assert "kya kaam" in turn.speech.lower()


def test_task_updates_on_a_new_instruction_and_holds_on_autonomous_steps():
    brain = _brain_with(json.dumps({"speech": "ok", "kind": "act",
                                    "action": {"type": "click", "target": "save"}}))
    brain.respond("mujhe save karna hai", {})
    assert brain.task == "mujhe save karna hai"

    # A new spoken instruction replaces the task.
    brain.respond("ab close karo", {})
    assert brain.task == "ab close karo"

    # An autonomous step (empty text) keeps it.
    brain.respond("", {})
    assert brain.task == "ab close karo"


# --------------------------------------------------------------------------- #
# LexiconPerception: a natural-language target reaches a vocabulary label
# --------------------------------------------------------------------------- #
def test_candidate_words_strip_generic_suffixes():
    from avcs.voice.lexicon_perception import LexiconPerception

    cands = LexiconPerception._candidate_words("the save button")
    assert "save" in cands
    assert "the save button" in cands
    # Best (most specific) first.
    assert cands.index("the save button") < cands.index("save")


def test_candidate_words_reach_underscored_vocabulary_entries():
    from avcs.voice.lexicon_perception import LexiconPerception

    assert "edit_menu" in LexiconPerception._candidate_words("edit menu")


def test_candidate_words_keep_unknown_words_as_is():
    from avcs.voice.lexicon_perception import LexiconPerception

    cands = LexiconPerception._candidate_words("print")
    assert cands[0] == "print"
    assert "save" not in cands


# --------------------------------------------------------------------------- #
# Browser mouth: the clip is delivered to the page, not the local sound card
# --------------------------------------------------------------------------- #
def test_browser_engine_parks_the_newest_clip_for_the_page():
    import numpy as np

    from avcs.voice.tts import BrowserEngine, PcmClip

    engine = BrowserEngine()
    # Avoid needing espeak-ng in the test: feed clips straight to play().
    seq0, wav0 = engine.take_latest(after_seq=0)
    assert wav0 is None

    engine.play(PcmClip(samples=np.array([0, 1000, -1000, 0], dtype=np.int16), sample_rate=16000))
    seq1, wav1 = engine.take_latest(after_seq=seq0)
    assert wav1 is not None and wav1[:4] == b"RIFF"
    assert seq1 > seq0

    # The same clip is not handed out twice.
    seq2, wav2 = engine.take_latest(after_seq=seq1)
    assert wav2 is None


def test_browser_engine_reports_the_browser_as_the_mouth():
    from avcs.voice.tts import BrowserEngine, Speaker, VoiceProfile

    speaker = Speaker(engine=BrowserEngine(), profile=VoiceProfile.hinglish())
    assert speaker.engine_name == "browser"
