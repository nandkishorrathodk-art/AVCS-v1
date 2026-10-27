"""Text-to-speech: the mouth of the AVCS assistant.

The assistant speaks its answers out loud while its eyes stream the screen and
an LLM does the reasoning. This module owns only the speaking.

Design rules:

1. **No hard dependency.** The default engine is ``espeak-ng`` (a system binary),
   but a machine without it must still run the rest of AVCS. ``available()``
   answers honestly and ``speak()`` returns a status instead of crashing.
2. **Never block the perception loop for longer than the sentence.** Speech is
   rendered to a PCM buffer first (fast, deterministic, testable) and then
   played. ``synthesize()`` is pure enough to unit-test without an audio device.
3. **One voice, one place.** Voice/pitch/rate live in ``VoiceProfile`` so a
   product decision about how the assistant sounds is a data change, not a code
   change scattered across demos.

The engine boundary is ``synthesize(text) -> PcmClip``. Adding a new engine
(piper, a cloud TTS) means implementing that one method.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import tempfile
import wave
from typing import Iterable, List, Optional, Tuple

import numpy as np

from avcs.voice.errors import AudioDeviceError, EngineNotAvailable, SynthesisError

# Sentence terminators, kept explicit so the streaming splitter never breaks on
# an abbreviation inside a decimal number ("0.5") or a version string ("v1.2").
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TERMINATOR = re.compile(r"[.!?]+(?=\s|$)")


def _split_sentences(text: str) -> List[Tuple[str, str]]:
    """Split complete sentences from a partial buffer.

    Returns ``[(sentence_with_terminator, remainder), ...]``. Only sentences that
    are actually terminated are returned, so a caller can stream deltas in and
    speak each sentence the moment it is complete.
    """
    out: List[Tuple[str, str]] = []
    remainder = text
    while True:
        m = _TERMINATOR.search(remainder)
        if not m:
            break
        end = m.end()
        sentence = remainder[:end].strip()
        remainder = remainder[end:].lstrip()
        if sentence:
            out.append((sentence, remainder))
        else:
            remainder = remainder.lstrip()
    return out


# Public name: the brain streams sentences to the mouth through this.
split_sentences = _split_sentences


@dataclasses.dataclass(frozen=True)
class VoiceProfile:
    """How the assistant sounds. Values are engine-neutral where possible."""

    language: str = "en-us"
    speed_wpm: int = 165          # espeak-ng words-per-minute; 175 is its default
    pitch: int = 45               # 0-99, 50 is neutral
    amplitude: int = 175          # 0-200, 100 is neutral
    word_gap_ms: int = 2          # tiny pause between words, sounds less robotic
    voice_variant: Optional[str] = None  # e.g. "en-us+f3" for a female variant

    def engine_voice_arg(self) -> str:
        return self.voice_variant or self.language

    @classmethod
    def hinglish(cls) -> "VoiceProfile":
        """A profile that reads Hinglish (romanized Hindi) intelligibly.

        espeak-ng's ``hi`` voice reads romanized Hindi the way the user writes it
        ("mujhe X kaam karna hai"), which an ``en`` voice mangles into English
        phonetics. The default Hinglish profile uses the Hindi voice at a
        slightly slower pace, since the user mixes English words in.
        """
        return cls(
            language="hi",
            speed_wpm=150,
            pitch=45,
            amplitude=175,
            word_gap_ms=4,
        )


@dataclasses.dataclass
class PcmClip:
    """A rendered utterance held in RAM.

    Kept as raw samples plus a sample rate so a caller can either play it,
    measure its duration, or write it to a WAV for inspection -- all without
    re-invoking the engine.
    """

    samples: np.ndarray          # int16 mono
    sample_rate: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)

    @property
    def is_empty(self) -> bool:
        return self.samples.size == 0

    def to_wav_bytes(self) -> bytes:
        import io

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.samples.astype("<i2").tobytes())
        return buf.getvalue()

    @classmethod
    def from_wav_bytes(cls, data: bytes) -> "PcmClip":
        import io

        with wave.open(io.BytesIO(data), "rb") as w:
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
            channels = w.getnchannels()
        samples = np.frombuffer(raw, dtype="<i2")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return cls(samples=samples.astype(np.int16), sample_rate=rate)


@dataclasses.dataclass
class SpeechResult:
    """Outcome of one `speak()` call, honest about whether audio actually played."""

    text: str
    engine: str
    spoken: bool                 # True only if audio was dispatched to a device
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.spoken and self.error is None


def _which_any(*names: str) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


class TtsEngine:
    """Base class: an engine renders text to a `PcmClip` and plays a clip."""

    name = "base"
    silent = False   # True when this engine deliberately never opens an audio device

    def available(self) -> bool:
        raise NotImplementedError

    def synthesize(self, text: str, profile: VoiceProfile) -> PcmClip:
        raise NotImplementedError

    def play(self, clip: PcmClip) -> bool:
        """Dispatch a clip to the audio device. Returns False when there is none."""
        raise NotImplementedError

    def stop(self) -> None:
        """Interrupt playback if the engine supports it."""


class EspeakNgEngine(TtsEngine):
    """Offline TTS via the ``espeak-ng`` system binary.

    espeak-ng is small, has no Python dependency, and writes a WAV we can hold
    in RAM, which makes it a good default for a CPU-only, privacy-preserving
    assistant: nothing is sent to a cloud service.
    """

    name = "espeak-ng"

    def __init__(self, binary: Optional[str] = None) -> None:
        self.binary = binary or _which_any("espeak-ng", "espeak")

    def available(self) -> bool:
        return bool(self.binary and os.path.exists(self.binary))

    def _engine_args(self, profile: VoiceProfile) -> list[str]:
        return [
            "-v", profile.engine_voice_arg(),
            "-s", str(int(profile.speed_wpm)),
            "-p", str(int(profile.pitch)),
            "-a", str(int(profile.amplitude)),
            "-g", str(int(profile.word_gap_ms)),
        ]

    def synthesize(self, text: str, profile: VoiceProfile) -> PcmClip:
        if not self.available():
            raise EngineNotAvailable("espeak-ng binary not found on PATH")
        text = (text or "").strip()
        if not text:
            return PcmClip(samples=np.zeros(0, dtype=np.int16), sample_rate=22050)

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="avcs_tts_")
        os.close(fd)
        try:
            cmd = [self.binary, "-w", path, *self._engine_args(profile), text]
            proc = subprocess.run(
                cmd, capture_output=True, timeout=30
            )
            if proc.returncode != 0 or not os.path.exists(path):
                raise SynthesisError(
                    f"espeak-ng failed (rc={proc.returncode}): "
                    f"{proc.stderr.decode('utf-8', 'replace').strip()[:200]}"
                )
            with open(path, "rb") as f:
                data = f.read()
            if not data:
                raise SynthesisError("espeak-ng produced an empty file")
            return PcmClip.from_wav_bytes(data)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def play(self, clip: PcmClip) -> bool:
        if clip.is_empty:
            return False
        player = _which_any("aplay", "paplay", "play")
        if not player:
            raise AudioDeviceError(
                "no audio player found (install alsa-utils for aplay or pulseaudio for paplay)"
            )
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="avcs_play_")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(clip.to_wav_bytes())
            args = [player, path]
            if os.path.basename(player) == "aplay":
                args = [player, "-q", path]
            proc = subprocess.run(args, capture_output=True, timeout=120)
            return proc.returncode == 0
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class NullEngine(TtsEngine):
    """A silent engine for headless machines and tests.

    It still *synthesizes* (via espeak-ng when present) so durations and
    buffers are real, but it never touches an audio device. This is what makes
    the voice layer testable on CI without sound hardware.
    """

    name = "null"
    silent = True

    def __init__(self, synthesize_with: Optional[TtsEngine] = None) -> None:
        self._synth = synthesize_with or EspeakNgEngine()

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, profile: VoiceProfile) -> PcmClip:
        if self._synth.available():
            try:
                return self._synth.synthesize(text, profile)
            except SynthesisError:
                pass
        # Deterministic fallback: ~90 ms of silence per word at 22.05 kHz, so
        # timing-sensitive callers still get a plausible duration.
        words = max(1, len((text or "").split()))
        rate = 22050
        n = int(0.09 * words * rate)
        return PcmClip(samples=np.zeros(n, dtype=np.int16), sample_rate=rate)

    def play(self, clip: PcmClip) -> bool:
        return False


class Speaker:
    """The assistant's mouth: synthesize once, play once, report honestly.

    ``speak()`` never raises for the ordinary "this machine has no speaker"
    case; it returns a ``SpeechResult`` with ``spoken=False`` and a reason. Only
    genuine programming errors propagate.

    ``speak_stream()`` is the real-time path: a reply is spoken sentence by
    sentence as it is produced, instead of waiting for the model to finish the
    whole answer. That is what makes the conversation feel live -- the first
    sentence is already audible while the rest is still being generated.
    """

    def __init__(
        self,
        engine: Optional[TtsEngine] = None,
        profile: Optional[VoiceProfile] = None,
    ) -> None:
        self.profile = profile or VoiceProfile()
        if engine is not None:
            self.engine = engine
        else:
            espeak = EspeakNgEngine()
            self.engine = espeak if espeak.available() else NullEngine()

    @property
    def engine_name(self) -> str:
        return self.engine.name

    def available(self) -> bool:
        return self.engine.available()

    def synthesize(self, text: str) -> PcmClip:
        return self.engine.synthesize(text, self.profile)

    def speak(self, text: str) -> SpeechResult:
        """Render and play `text`. Returns a status rather than raising."""
        text = (text or "").strip()
        if not text:
            return SpeechResult(text="", engine=self.engine.name, spoken=False, error="empty text")
        try:
            clip = self.synthesize(text)
        except EngineNotAvailable as exc:
            return SpeechResult(text=text, engine=self.engine.name, spoken=False, error=str(exc))
        except SynthesisError as exc:
            return SpeechResult(text=text, engine=self.engine.name, spoken=False, error=str(exc))
        try:
            played = self.engine.play(clip)
        except AudioDeviceError as exc:
            return SpeechResult(
                text=text, engine=self.engine.name, spoken=False,
                duration_s=clip.duration_s, error=str(exc),
            )
        if self.engine.silent:
            return SpeechResult(
                text=text, engine=self.engine.name, spoken=False,
                duration_s=clip.duration_s, error="silent engine: audio not played by design",
            )
        return SpeechResult(
            text=text, engine=self.engine.name, spoken=played, duration_s=clip.duration_s,
            error=None if played else "playback did not complete",
        )

    def speak_stream(self, chunks) -> List[SpeechResult]:
        """Speak a token/chunk stream sentence by sentence, as it arrives.

        Parameters
        ----------
        chunks:
            An iterable of text fragments (LLM deltas, or any producer). Each
            complete sentence is spoken as soon as its terminator arrives; the
            final fragment is flushed at the end.

        Returns
        -------
        One ``SpeechResult`` per spoken sentence, in order. Sentences that render
        to nothing are skipped. Never raises for missing audio hardware.
        """
        results: List[SpeechResult] = []
        buffer = ""
        for chunk in chunks:
            if not chunk:
                continue
            buffer += chunk
            for sentence, remainder in _split_sentences(buffer):
                sentence = sentence.strip()
                if sentence:
                    results.append(self.speak(sentence))
                buffer = remainder
        tail = buffer.strip()
        if tail:
            results.append(self.speak(tail))
        return results


class SpeechQueue:
    """Speaks sentences on a background thread so the loop never waits on audio.

    Playing a sentence takes seconds; perceiving a frame takes milliseconds. If
    the assistant spoke synchronously, the eyes would stall for the length of
    every reply. This queue lets a caller push sentences as they are generated
    and keep working, then ``drain()`` before ending the session.

    Ordering is preserved. ``errors`` collects anything that went wrong, so a
    caller can report it after draining rather than losing it on the worker.
    """

    def __init__(self, speaker: Optional[Speaker] = None) -> None:
        import queue
        import threading

        self.speaker = speaker or Speaker()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.results: List[SpeechResult] = []
        self._thread = threading.Thread(target=self._run, name="avcs-speech", daemon=True)
        self._started = False

    def start(self) -> "SpeechQueue":
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def say(self, text: str) -> None:
        """Queue one sentence. Starts the worker on first use."""
        if not self._started:
            self.start()
        self._queue.put(text)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if item.strip():
                    self.results.append(self.speaker.speak(item))
            finally:
                self._queue.task_done()

    def drain(self, timeout: float = 120.0) -> List[SpeechResult]:
        """Wait for everything queued to finish, then stop the worker."""
        if self._started:
            self._queue.join()
            self._queue.put(None)
            self._thread.join(timeout=timeout)
            self._started = False
        return list(self.results)

    @property
    def errors(self) -> List[str]:
        return [r.error for r in self.results if r.error]
