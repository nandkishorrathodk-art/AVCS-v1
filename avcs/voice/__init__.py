"""AVCS voice assistant: the LLM brain with ears and a mouth, on top of the eyes.

System 1 (`avcs/`) is the visual cortex: pure-pixel perception, a fail-closed
capability gate, and motor hands. This package adds the conversation:

    ears (stt)  ->  brain (llm)  ->  mouth (tts)
                        ^
                        +-- AVCS eyes (perception) and gated hands (assistant loop)

Quickstart::

    from avcs.voice import Assistant, Speaker, default_stt_engine

    assistant = Assistant()               # best available engines, fail-closed gate
    print(assistant.capabilities())       # what this machine can actually do
    print(assistant.speaker.speak("Hello sir, what work would you like me to do?"))

    turn = assistant.step("mujhe X kaam karna hai")
    print(turn.said)                      # what the assistant said back
"""
from __future__ import annotations

from avcs.voice.assistant import (
    Assistant,
    AssistantTurnRecord,
    ConversationReport,
    ScreenObservation,
    StaticPerception,
)
from avcs.voice.brain import (
    AssistantTurn,
    ConversationBrain,
    RuleBrain,
    TurnKind,
    CONVERSATION_SYSTEM_PROMPT,
)
from avcs.voice.errors import (
    AudioDeviceError,
    EngineNotAvailable,
    ModelNotFound,
    SynthesisError,
    VoiceError,
)
from avcs.voice.stt import (
    AudioClip,
    FasterWhisperEngine,
    NullEngine as NullSttEngine,
    SttEngine,
    Transcript,
    VoskEngine,
    default_stt_engine,
    record_microphone,
    record_microphone_until_silence,
)
from avcs.voice.perception import EyePerception, GroundingPerception
from avcs.voice.tts import (
    EspeakNgEngine,
    NullEngine as NullTtsEngine,
    PcmClip,
    Speaker,
    SpeechQueue,
    SpeechResult,
    TtsEngine,
    VoiceProfile,
)

__all__ = [
    # assistant loop
    "Assistant", "AssistantTurnRecord", "ConversationReport",
    "ScreenObservation", "StaticPerception",
    # brain
    "ConversationBrain", "RuleBrain", "AssistantTurn", "TurnKind",
    "CONVERSATION_SYSTEM_PROMPT",
    # eyes (pure pixels)
    "EyePerception", "GroundingPerception",
    # mouth
    "Speaker", "TtsEngine", "EspeakNgEngine", "NullTtsEngine",
    "VoiceProfile", "PcmClip", "SpeechResult", "SpeechQueue",
    # ears
    "SttEngine", "VoskEngine", "FasterWhisperEngine", "NullSttEngine",
    "AudioClip", "Transcript", "default_stt_engine", "record_microphone",
    "record_microphone_until_silence",
    # errors
    "VoiceError", "EngineNotAvailable", "AudioDeviceError", "ModelNotFound",
    "SynthesisError",
]
