"""The conversational brain: an LLM that talks with the user and drives AVCS.

This is the piece the user asked for. AVCS System 1 is the eyes and hands
(pure pixels, no OCR, real-time). System 2 is an LLM that:

1. **Talks like a person.** "Mujhe X kaam karna hai" -> the assistant answers
   out loud, and asks a question back when the request is not actionable yet
   (the user's own example: it asks "kya kaam hai, sir?").
2. **Acts on the screen.** When the user has said enough, the LLM emits a
   machine-readable action for the AVCS loop to carry out, using the same
   coordinate-free contract the rest of the repo uses: it names a target in
   natural language, it never emits (x, y).

The brain is deliberately provider-agnostic. It speaks to any OpenAI-compatible
``/chat/completions`` endpoint (a local llama, NVIDIA NIM, Gemini's compat
layer, ...) through the transport already in ``avcs.reasoning.provider``, so a
new model is a config change.

When no model is reachable the brain falls back to ``RuleBrain``, which keeps
the exact same interface. That fallback is intentionally simple and is labelled
as such in every reply -- the point is that the voice loop stays honest and
runnable on a laptop with no API key, not that the fallback is clever.
"""
from __future__ import annotations

import dataclasses
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from avcs.reasoning.contracts import ActionIntent, ActionOp, System2Decision
from avcs.voice.tts import split_sentences as _split_sentences
from avcs.reasoning.provider import ReasoningProvider as _ReasoningProvider

_BASE_STREAM_STEP = _ReasoningProvider.stream_step


class TurnKind(str, Enum):
    """What the assistant's turn is doing."""

    REPLY = "reply"            # Answering the user, no screen action needed
    ASK = "ask"                # Needs clarification before it can act
    ACT = "act"                # Has a concrete action for the AVCS loop
    FINISH = "finish"          # Task is done
    STOP = "stop"              # Aborting safely


@dataclasses.dataclass
class AssistantTurn:
    """One thing the assistant says, plus the action (if any) it wants taken."""

    kind: TurnKind
    speech: str                                   # what the assistant says out loud
    intent: Optional[ActionIntent] = None         # machine-readable next action
    thought: str = ""                             # private reasoning, not spoken
    source: str = "llm"                           # "llm" | "rule_fallback"
    confidence: float = 1.0

    @property
    def wants_action(self) -> bool:
        return self.intent is not None and self.kind == TurnKind.ACT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "speech": self.speech,
            "thought": self.thought,
            "source": self.source,
            "confidence": self.confidence,
            "intent": self.intent.to_dict() if self.intent else None,
        }


@dataclasses.dataclass
class ConversationTurn:
    """A single exchange entry kept in the rolling conversation history."""

    role: str          # "user" | "assistant"
    text: str

    def to_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.text}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
CONVERSATION_SYSTEM_PROMPT = """You are AVCS, a real-time desktop assistant.
You SEE the screen through a local visual cortex (it streams raw pixels; you are
given short factual descriptions of what is on screen, never images). You HEAR
the user through speech-to-text and you SPEAK your replies out loud through
text-to-speech.

You are holding a live conversation, like one person talking to another:
- Speak naturally and briefly. One or two short sentences. This text is read
  aloud, so no markdown, no bullet lists, no code blocks, no emoji.
- If the user's request is not yet actionable (no target, no clear task), ASK a
  short clarifying question instead of guessing. For example, if the user says
  "I have some work to do" with no task named, ask "What work would you like me
  to do, sir?"
- If the user has told you what to do and the screen shows the target, act.
- If you already did the thing and the screen confirms it, say so and finish.

Hard rules:
1. NEVER output screen coordinates (x, y, bbox, pixels). The local cortex owns
   all geometry. You name targets in plain words: "the Save button".
2. Base what you say on the observation you are given. If the observation says a
   target is absent, do not claim to click it; say you cannot see it and ask or
   suggest a next step.
3. Keep the user's language where you can. If the user speaks Hindi/Hinglish,
   you may reply in Hinglish; otherwise reply in English.

Reply with ONE JSON object and nothing else:
{
  "speech": "what you say out loud",
  "thought": "one line of private reasoning",
  "kind": "reply" | "ask" | "act" | "finish" | "stop",
  "action": {
    "op": "observe_target" | "click" | "double_click" | "type_text" | "press_key" | "scroll" | "wait" | "ask_user" | "stop" | "finish",
    "target_description": "natural-language target, e.g. 'Save button', or null",
    "text_input": "text to type, or null",
    "key_name": "key name, or null"
  }
}
`action` is required when kind is "act". For "reply"/"ask" you may omit it or set op to "wait".
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model reply, tolerating fences."""
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if m:
            cleaned = m.group(1).strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        m = re.search(r"(\{[\s\S]*\})", cleaned)
        if m:
            cleaned = m.group(1)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None



# --------------------------------------------------------------------------- #
# Streaming protocol
# --------------------------------------------------------------------------- #
# A voice agent cannot wait for a whole JSON reply before it starts talking. The
# streaming prompt therefore asks the model for *plain spoken text first*, then a
# sentinel line, then the machine-readable action. That way the first sentence is
# audible while the rest (including the action) is still being generated, and the
# action is never spoken out loud.
STREAM_SENTINEL = "<<<ACTION>>>"

STREAM_SYSTEM_PROMPT = """You are AVCS, a real-time desktop assistant in a live voice call.
You SEE the screen through a local visual cortex (short factual descriptions, never images).
You HEAR the user through speech-to-text. Everything you write before the sentinel is READ
ALOUD by text-to-speech, so it must be natural spoken language.

Answer like one person talking to another:
- First, say your reply in one or two short spoken sentences. No markdown, no lists,
  no emoji, no coordinates, no JSON here.
- If the request is not actionable yet (no target named), ASK a short question.
  Example: user says "mujhe kuch kaam karna hai" -> you say "Ji sir, kya kaam karna hai?"
- If the user named a task and the screen shows the target, say briefly that you are doing it.
- Keep the user's language: reply in Hinglish if the user speaks Hinglish.

After your spoken sentences, output the sentinel on its own line, then ONE JSON object:
<<<ACTION>>>
{{"kind": "reply"|"ask"|"act"|"finish"|"stop",
  "thought": "one line of private reasoning",
  "action": {{"op": "observe_target"|"click"|"double_click"|"type_text"|"press_key"|"scroll"|"wait"|"ask_user"|"stop"|"finish",
             "target_description": "natural-language target or null",
             "text_input": "text to type or null",
             "key_name": "key name or null"}}}}

Rules: never output coordinates. `action` is required when kind is "act". Use kind "ask"
when you need clarification, "reply" for plain conversation, "finish" when the task is done.
"""


def _split_stream_reply(text: str) -> tuple[str, Optional[Dict[str, Any]]]:
    """Split a streamed reply into (spoken_text, action_object).

    Tolerates the sentinel being absent (then the whole reply is spoken and there
    is no action) or the JSON being malformed (then the spoken text is still
    returned and the action is dropped).
    """
    if STREAM_SENTINEL in text:
        spoken, _, tail = text.partition(STREAM_SENTINEL)
        return spoken.strip(), _extract_json_object(tail)
    # No sentinel: try to treat a trailing JSON object as the action anyway.
    obj = _extract_json_object(text)
    if obj is not None and ("kind" in obj or "action" in obj):
        brace = text.rfind("{")
        return text[:brace].strip(), obj
    return text.strip(), None


class ConversationBrain:
    """LLM-backed brain for the real-time voice assistant.

    Parameters
    ----------
    provider:
        Any object exposing ``plan_step(task, observation, history, step_index)``
        and returning a ``System2Decision``. Defaults to ``LLMReasoningProvider``
        which reads its endpoint/key from the environment.
    """

    def __init__(
        self,
        provider: Any = None,
        task: str = "",
        history_limit: int = 12,
    ) -> None:
        if provider is None:
            from avcs.reasoning.provider import LLMReasoningProvider

            provider = LLMReasoningProvider()
        self.provider = provider
        self.task = task
        self.history_limit = int(history_limit)
        self.history: List[ConversationTurn] = []
        self.step_index = 0

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #
    def record_user(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.history.append(ConversationTurn(role="user", text=text))
            self._trim()

    def record_assistant(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.history.append(ConversationTurn(role="assistant", text=text))
            self._trim()

    def _trim(self) -> None:
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

    def set_task(self, task: str) -> None:
        self.task = (task or "").strip()

    # ------------------------------------------------------------------ #
    # Observation payload
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_observation(
        screen_description: str = "",
        target_present: Optional[bool] = None,
        stability: str = "stable",
        has_modal: bool = False,
        active_window: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble the coordinate-free observation the LLM is allowed to see.

        This is the System 1 -> System 2 boundary: short factual text about the
        screen. No images, no embeddings, no coordinates.
        """
        obs: Dict[str, Any] = {
            "screen": screen_description or "no visual detail available",
            "stability": stability,
            "has_modal_dialog": bool(has_modal),
            "active_window": active_window or "",
        }
        if target_present is not None:
            obs["target_present"] = bool(target_present)
        if extra:
            obs.update(extra)
        return obs

    # ------------------------------------------------------------------ #
    # The turn
    # ------------------------------------------------------------------ #
    def respond(self, user_text: str, observation: Optional[Dict[str, Any]] = None) -> AssistantTurn:
        """Produce the assistant's next turn from the user's utterance.

        Never raises for an unreachable model: it falls back to ``RuleBrain`` and
        labels the result ``rule_fallback``.
        """
        self.record_user(user_text)
        if user_text and not self.task:
            # The first actionable utterance becomes the task.
            self.task = user_text.strip()

        self.step_index += 1
        observation = observation or self.build_observation()
        observation = dict(observation)
        observation.setdefault("task", self.task)

        try:
            decision = self.provider.plan_step(
                task=self.task,
                observation=observation,
                history=[t.to_message() for t in self.history[:-1]],
                step_index=self.step_index,
            )
            turn = self._turn_from_decision(decision)
        except Exception as exc:  # unreachable endpoint, bad key, timeout, ...
            turn = self._fallback(user_text, observation, reason=str(exc))

        self.record_assistant(turn.speech)
        return turn

    _DEFAULT_SPEECH = {
        TurnKind.REPLY: "Okay sir.",
        TurnKind.ASK: "Could you tell me a bit more, sir?",
        TurnKind.ACT: "Okay sir, doing that now.",
        TurnKind.FINISH: "Done, sir.",
        TurnKind.STOP: "Stopping, sir.",
    }


    def respond_stream(
        self,
        user_text: str,
        observation: Optional[Dict[str, Any]] = None,
        on_speech=None,
    ) -> AssistantTurn:
        """Stream one turn: speak sentences as they arrive, then return the turn.

        This is the real-time path. The provider yields text fragments; the brain
        buffers them, forwards each *complete sentence* to ``on_speech`` the moment
        it is finished (so the mouth can start talking immediately), and keeps
        scanning for the action sentinel. Everything after the sentinel is parsed
        as the machine-readable action and is never spoken.

        If the provider has no streaming support, or the endpoint refuses
        ``stream=true``, this degrades to a single non-streamed fragment and still
        returns a valid turn.
        """
        self.record_user(user_text)
        if user_text and not self.task:
            self.task = user_text.strip()
        self.step_index += 1
        observation = observation or self.build_observation()
        observation = dict(observation)
        observation.setdefault("task", self.task)

        speak = on_speech if on_speech is not None else (lambda _s: None)

        # If the provider has no real streaming implementation (it still uses the
        # base-class hook), the reply arrives whole and its action cannot be
        # recovered from a text fragment. Take the ordinary path instead of
        # guessing, so a non-streaming model still acts correctly.
        _impl = getattr(self.provider, "stream_step", None)
        if _impl is None or getattr(_impl, "__func__", _impl) is _BASE_STREAM_STEP:
            turn = self.respond(user_text, observation)
            if turn.speech:
                speak(turn.speech)
            return turn

        spoken_parts: List[str] = []
        buffer = ""
        action_seen = False

        def _flush_sentences() -> None:
            nonlocal buffer
            while True:
                pieces = _split_sentences(buffer)
                if not pieces:
                    break
                sentence, remainder = pieces[0]
                sentence = sentence.strip()
                if sentence:
                    spoken_parts.append(sentence)
                    speak(sentence)
                buffer = remainder

        try:
            for chunk in self.provider.stream_step(
                task=self.task,
                observation=observation,
                history=[t.to_message() for t in self.history[:-1]],
                step_index=self.step_index,
                system_prompt=STREAM_SYSTEM_PROMPT,
            ):
                if not chunk:
                    continue
                if action_seen:
                    buffer += chunk
                    continue
                buffer += chunk
                if STREAM_SENTINEL in buffer:
                    spoken, _, tail = buffer.partition(STREAM_SENTINEL)
                    tail = tail.strip()
                    if spoken.strip():
                        spoken_parts.append(spoken.strip())
                        speak(spoken.strip())
                    buffer = tail
                    action_seen = True
                    continue
                _flush_sentences()

            if not action_seen:
                # No sentinel: the tail is more speech, not an action.
                tail = buffer.strip()
                if tail:
                    spoken_parts.append(tail)
                    speak(tail)
                action_obj = None
            else:
                action_obj = _extract_json_object(buffer)

            spoken_text = " ".join(spoken_parts).strip()
            turn = self._turn_from_stream(spoken_text, action_obj)
        except Exception as exc:
            turn = self._fallback(user_text, observation, reason=str(exc))
            if turn.speech:
                speak(turn.speech)

        self.record_assistant(turn.speech)
        return turn

    def _turn_from_stream(self, spoken_text: str, action_obj: Optional[Dict[str, Any]]) -> AssistantTurn:
        """Build an AssistantTurn from streamed speech plus an optional action."""
        kind = TurnKind.REPLY
        intent: Optional[ActionIntent] = None
        thought = ""
        if isinstance(action_obj, dict):
            thought = str(action_obj.get("thought") or "")
            raw_kind = str(action_obj.get("kind") or "").strip().lower()
            kind = {
                "reply": TurnKind.REPLY, "ask": TurnKind.ASK, "act": TurnKind.ACT,
                "finish": TurnKind.FINISH, "stop": TurnKind.STOP,
            }.get(raw_kind, TurnKind.REPLY)
            action = action_obj.get("action") or {}
            op_name = str(action.get("op") or "").strip().lower()
            if kind == TurnKind.ACT and op_name:
                try:
                    intent = ActionIntent.create(
                        step_index=self.step_index,
                        op=op_name,
                        target_description=action.get("target_description"),
                        text_input=action.get("text_input"),
                        key_name=action.get("key_name"),
                        rationale=thought,
                    )
                except Exception:
                    intent = None
            if kind == TurnKind.ACT and intent is None:
                kind = TurnKind.REPLY

        speech = spoken_text.strip() or self._DEFAULT_SPEECH[kind]
        return AssistantTurn(
            kind=kind, speech=speech, intent=intent, thought=thought, source="llm"
        )

    def _turn_from_decision(self, decision: System2Decision) -> AssistantTurn:
        intent = decision.next_intent
        op = intent.op
        if op == ActionOp.FINISH:
            kind = TurnKind.FINISH
        elif op == ActionOp.STOP:
            kind = TurnKind.STOP
        elif op == ActionOp.ASK_USER:
            kind = TurnKind.ASK
        elif op == ActionOp.WAIT:
            # A bare "wait" is the model stalling; treat it as a conversational
            # reply unless it also carries a target to look at.
            kind = TurnKind.REPLY
        else:
            kind = TurnKind.ACT

        # The assistant must always say *something* -- this text is read aloud,
        # and a model that returns an empty string would leave the user in
        # silence. Fall back to the private thought, then to a short default.
        speech = (decision.clarification_prompt or decision.thought or "").strip()
        if not speech:
            speech = self._DEFAULT_SPEECH[kind]
        return AssistantTurn(
            kind=kind,
            speech=speech,
            intent=intent,
            thought=decision.thought,
            source="llm",
            confidence=float(getattr(decision, "confidence", 1.0) or 1.0),
        )

    def _fallback(self, user_text: str, observation: Dict[str, Any], reason: str) -> AssistantTurn:
        rule = RuleBrain()
        turn = rule.respond(user_text, observation)
        turn.source = "rule_fallback"
        turn.thought = f"LLM unavailable ({reason[:120]}); used the offline rule brain."
        return turn


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #
_ACTION_WORDS = {
    "click": ("click", "press", "tap", "dabao", "daba", "kholo", "open"),
    "type": ("type", "write", "enter", "likho", "likh", "bharo"),
    "save": ("save",),
    "search": ("search", "dhundo", "find"),
    "scroll": ("scroll", "neeche", "upar"),
}


class RuleBrain:
    """Deterministic offline brain with the same interface as ``ConversationBrain``.

    It is a keyword matcher, not reasoning. It exists so the voice loop runs and
    is testable without an API key, and every turn it produces is marked
    ``source="rule_fallback"`` so no caller can mistake it for the LLM.
    """

    def __init__(self) -> None:
        self.task = ""
        self.step_index = 0

    def respond(self, user_text: str, observation: Optional[Dict[str, Any]] = None) -> AssistantTurn:
        self.step_index += 1
        observation = observation or {}
        text = (user_text or "").strip()
        low = text.lower()
        if text:
            self.task = text

        # No usable task yet -> ask, which is the behavior the user described.
        if not low or low in {"hi", "hello", "hey", "namaste"}:
            return AssistantTurn(
                kind=TurnKind.ASK,
                speech="Yes sir, what work would you like me to do?",
                thought="Greeting with no task named.",
                source="rule_fallback",
            )

        target = self._extract_target(text)
        if target is None:
            return AssistantTurn(
                kind=TurnKind.ASK,
                speech=f"Understood sir. Which item should I act on for '{text}'?",
                thought="Task named but no target could be extracted.",
                source="rule_fallback",
            )

        if observation.get("target_present") is False:
            return AssistantTurn(
                kind=TurnKind.ASK,
                speech=f"I can't see {target} on screen right now, sir. Should I wait or look elsewhere?",
                thought="Target reported absent by perception.",
                source="rule_fallback",
            )

        op = ActionOp.CLICK
        if any(w in low for w in _ACTION_WORDS["type"]):
            op = ActionOp.TYPE_TEXT
        elif any(w in low for w in _ACTION_WORDS["scroll"]):
            op = ActionOp.SCROLL

        intent = ActionIntent.create(
            step_index=self.step_index,
            op=op,
            target_description=target,
            expected_outcome=f"{target} responds",
            rationale="Offline keyword match.",
        )
        return AssistantTurn(
            kind=TurnKind.ACT,
            speech=f"Okay sir, {op.value.replace('_', ' ')} on {target}.",
            intent=intent,
            thought="Keyword match produced an action.",
            source="rule_fallback",
        )

    # Words that describe the *task*, not the target. Kept bilingual because the
    # user speaks Hinglish; a target named by the user survives these.
    _STOPWORDS = frozenset({
        "the", "a", "an", "this", "that", "please", "and", "on", "to", "for",
        "mujhe", "mere", "mera", "ko", "karo", "kar", "karna", "karna", "hai",
        "do", "dena", "chahiye", "kaam", "kuch", "bhi", "ye", "yeh", "wo",
    })

    @classmethod
    def _extract_target(cls, text: str) -> Optional[str]:
        """Find a likely target phrase, preferring a quoted string or a known noun."""
        quoted = re.search(r"['\"]([^'\"]{1,40})['\"]", text)
        if quoted:
            return quoted.group(1).strip()

        low = text.lower()
        # Prefer text that follows an action verb ("click the Save button").
        for words in _ACTION_WORDS.values():
            for w in words:
                idx = low.find(w)
                if idx != -1:
                    tail = text[idx + len(w):].strip(" .!?,")
                    tail_words = [t for t in re.findall(r"[A-Za-z0-9_-]+", tail)
                                  if t.lower() not in cls._STOPWORDS]
                    if tail_words:
                        return " ".join(tail_words[:4])

        # Otherwise take the remaining content tokens (drops "mujhe"/"kaam"/"karna").
        tokens = [t for t in re.findall(r"[A-Za-z0-9_-]+", text)
                  if t.lower() not in cls._STOPWORDS]
        if tokens:
            return " ".join(tokens[:3])
        return None
