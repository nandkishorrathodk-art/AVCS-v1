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
import sys
import tempfile
import threading
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

    @classmethod
    def natural(cls) -> "VoiceProfile":
        """Clean, natural American English voice profile without robotic accents or native distortions."""
        return cls(
            language="en-us",
            speed_wpm=160,
            pitch=50,
            amplitude=180,
            word_gap_ms=2,
            voice_variant="natural-en",
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
        # The player subprocess is held so `stop()` can interrupt it. Without
        # this, `play()` blocks for the whole clip and the assistant cannot be
        # talked over -- which is the difference between a live conversation and
        # a monologue.
        self._player_proc: Optional[subprocess.Popen] = None
        self._stop_flag = False

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
            self._stop_flag = False
            # Popen (not run) so `stop()` can terminate playback mid-sentence.
            self._player_proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            rc = self._player_proc.wait()
            self._player_proc = None
            if self._stop_flag:
                # Interrupted on purpose: not a playback failure.
                return False
            return rc == 0
        finally:
            self._player_proc = None
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop(self) -> None:
        """Interrupt the current clip so the user can take the floor back."""
        self._stop_flag = True
        proc = self._player_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


class NeuralVoiceModelEngine(TtsEngine):
    """Deep learning neural voice model engine.

    Synthesizes rich, human-quality neural speech using transformer-based neural vocoder models
    (Microsoft Neural Voice models: en-US-AriaNeural, en-US-GuyNeural, en-US-JennyNeural).
    Produces natural prosody, breathing, intonation, and zero robotic native-ness.
    """

    name = "neural-voice"

    def __init__(self, voice_model: str = "en-US-AriaNeural") -> None:
        self.voice_model = voice_model
        self._current_alias: Optional[str] = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, profile: VoiceProfile) -> PcmClip:
        if not self.available():
            raise EngineNotAvailable("edge_tts neural speech package is not available")
        text = (text or "").strip()
        if not text:
            return PcmClip(samples=np.zeros(0, dtype=np.int16), sample_rate=24000)

        import asyncio
        import edge_tts

        voice = profile.voice_variant if profile.voice_variant and "Neural" in profile.voice_variant else self.voice_model

        rate_delta = int(round((profile.speed_wpm - 165) / 1.65))
        rate_str = f"{rate_delta:+d}%" if rate_delta != 0 else "+0%"

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            audio_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.extend(chunk["data"])
            return bytes(audio_bytes)

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        mp3_bytes = ex.submit(lambda: asyncio.run(_synthesize())).result(timeout=20)
                else:
                    mp3_bytes = loop.run_until_complete(_synthesize())
            except RuntimeError:
                mp3_bytes = asyncio.run(_synthesize())

            clip = PcmClip(samples=np.zeros(len(mp3_bytes), dtype=np.int16), sample_rate=24000)
            clip._mp3_data = mp3_bytes
            return clip
        except Exception as exc:
            raise SynthesisError(f"Neural voice model synthesis failed: {exc}") from exc

    def play(self, clip: PcmClip) -> bool:
        if clip.is_empty:
            return False
        mp3_data = getattr(clip, "_mp3_data", None)
        if not mp3_data:
            wav_bytes = clip.to_wav_bytes()
            if sys.platform == "win32":
                try:
                    import winsound
                    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
                    return True
                except Exception:
                    return False
            return False

        if sys.platform != "win32":
            return False

        import ctypes
        import tempfile
        import time

        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="avcs_neural_play_")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(mp3_data)

        winmm = ctypes.windll.winmm
        alias = f"avcs_voice_{int(time.time() * 1000) % 100000}"
        with self._lock:
            self._current_alias = alias

        try:
            winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
            winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
            winmm.mciSendStringW(f'close {alias}', None, 0, None)
            return True
        except Exception as exc:
            raise AudioDeviceError(f"Neural voice playback error: {exc}") from exc
        finally:
            with self._lock:
                if self._current_alias == alias:
                    self._current_alias = None
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop(self) -> None:
        if sys.platform == "win32":
            import ctypes
            winmm = ctypes.windll.winmm
            with self._lock:
                if self._current_alias:
                    try:
                        winmm.mciSendStringW(f"stop {self._current_alias}", None, 0, None)
                        winmm.mciSendStringW(f"close {self._current_alias}", None, 0, None)
                    except Exception:
                        pass
                    self._current_alias = None


class WindowsNaturalSpeechEngine(TtsEngine):
    """Natural, high-clarity speech synthesis for Windows using SAPI5 / Windows Media.

    Provides clean, natural English pronunciation with zero robotic native-ness or awkward phonetics.
    Outputs high-quality 22.05kHz mono PCM and plays via hardware audio with stop/barge-in support.
    """

    name = "windows-natural"

    def __init__(self, voice_name: Optional[str] = None) -> None:
        self.voice_name = voice_name
        self._available = False
        if sys.platform == "win32":
            co_init = False
            try:
                import pythoncom
                pythoncom.CoInitialize()
                co_init = True
            except Exception:
                pass
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                self._available = True
            except Exception:
                self._available = False
            finally:
                if co_init:
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

    def available(self) -> bool:
        return self._available and sys.platform == "win32"

    def synthesize(self, text: str, profile: VoiceProfile) -> PcmClip:
        if not self.available():
            raise EngineNotAvailable("Windows SAPI5 voice engine is not available on this platform")
        text = (text or "").strip()
        if not text:
            return PcmClip(samples=np.zeros(0, dtype=np.int16), sample_rate=22050)

        import win32com.client
        import tempfile
        import os
        import wave

        co_init = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            co_init = True
        except Exception:
            pass

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="avcs_natural_tts_")
        os.close(fd)
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            if self.voice_name:
                for v in speaker.GetVoices():
                    if self.voice_name.lower() in v.GetDescription().lower():
                        speaker.Voice = v
                        break
            else:
                for v in speaker.GetVoices():
                    desc = v.GetDescription().lower()
                    if "zira" in desc or "natural" in desc or "jenny" in desc or "aria" in desc:
                        speaker.Voice = v
                        break

            rate_val = int(round((profile.speed_wpm - 160) / 15.0))
            speaker.Rate = max(-10, min(10, rate_val))
            speaker.Volume = max(0, min(100, int(profile.amplitude / 2.0)))

            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(path, 3)  # 3 = SSFMCreateForWrite
            speaker.AudioOutputStream = stream
            speaker.Speak(text)
            stream.Close()

            with open(path, "rb") as f:
                data = f.read()
            if not data:
                raise SynthesisError("SAPI produced empty audio")
            return PcmClip.from_wav_bytes(data)
        except Exception as exc:
            raise SynthesisError(f"Windows natural speech synthesis failed: {exc}") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
            if co_init:
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def play(self, clip: PcmClip) -> bool:
        if clip.is_empty:
            return False
        if sys.platform != "win32":
            return False
        try:
            import winsound
            wav_bytes = clip.to_wav_bytes()
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
            return True
        except Exception as exc:
            raise AudioDeviceError(f"Audio playback error: {exc}") from exc

    def stop(self) -> None:
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass


class BrowserEngine(TtsEngine):
    """A mouth for a browser page instead of the local sound card.

    The GUI serves the assistant to a browser, which may be on a different
    machine than the process doing the reasoning (a container with no sound card
    is the normal case here). espeak-ng still *synthesizes* the audio -- so the
    voice is identical to the local path -- but instead of handing the clip to
    ``aplay`` this engine parks the most recent WAV for the page to fetch and
    play through the Web Audio API.

    ``play()`` therefore returns ``True`` (the clip was delivered to the page,
    which is the client's mouth) and the honest "did the speaker actually move
    air" signal is the browser's, not ours.
    """

    name = "browser"
    silent = False

    def __init__(self, synthesize_with: Optional[TtsEngine] = None) -> None:
        self._synth = synthesize_with or EspeakNgEngine()
        self._lock = threading.Lock()
        self._latest: Optional[PcmClip] = None
        self._seq = 0

    def available(self) -> bool:
        return self._synth.available()

    def synthesize(self, text: str, profile: VoiceProfile) -> PcmClip:
        return self._synth.synthesize(text, profile)

    def play(self, clip: PcmClip) -> bool:
        if clip.is_empty:
            return False
        with self._lock:
            self._latest = clip
            self._seq += 1
        return True

    def take_latest(self, after_seq: int = 0) -> tuple[int, Optional[bytes]]:
        """Return ``(seq, wav_bytes)`` for the newest clip, or ``(seq, None)``.

        The page polls with the last sequence it played, so a clip is fetched
        exactly once even if the poll is slower than speech.
        """
        with self._lock:
            if self._latest is None or self._seq <= after_seq:
                return self._seq, None
            return self._seq, self._latest.to_wav_bytes()

    def stop(self) -> None:
        self._synth.stop()


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


@dataclasses.dataclass
class SpeechHandle:
    """A spoken utterance running on a background thread.

    `done` is set when the thread finishes. `stop()` barges in: it interrupts the
    audio engine and marks the result so a caller can tell an interrupted clip
    from a completed one.
    """

    text: str
    engine: str
    result: Optional["SpeechResult"] = None
    error: Optional[str] = None
    interrupted: bool = False
    done: "threading.Event" = dataclasses.field(default_factory=threading.Event)
    _thread: Optional["threading.Thread"] = None
    _speaker: Optional["Speaker"] = None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the utterance finishes. Returns True if it did."""
        return self.done.wait(timeout)

    def is_done(self) -> bool:
        return self.done.is_set()

    def stop(self) -> None:
        """Interrupt this utterance mid-sentence."""
        self.interrupted = True
        if self._speaker is not None:
            self._speaker.stop()


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
            neural = NeuralVoiceModelEngine()
            if neural.available():
                self.engine = neural
            else:
                win_nat = WindowsNaturalSpeechEngine()
                if win_nat.available():
                    self.engine = win_nat
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

    def stop(self) -> None:
        """Interrupt whatever is playing, so the user can take the floor back."""
        self.engine.stop()

    def speak_async(self, text: str) -> "SpeechHandle":
        """Speak `text` on a background thread and return immediately.

        The real-time loop must not block on audio: playing a sentence takes
        seconds, and the eyes/brain should keep running while the mouth talks.
        `SpeechHandle.wait()` blocks until done; `SpeechHandle.stop()` barges in.
        """
        import threading

        handle = SpeechHandle(text=text, engine=self.engine.name)
        handle._speaker = self
        def _run() -> None:
            try:
                handle.result = self.speak(text)
            except Exception as exc:  # never let the mouth kill the loop
                handle.error = str(exc)
            finally:
                handle.done.set()

        handle._thread = threading.Thread(target=_run, name="avcs-mouth", daemon=True)
        handle._thread.start()
        return handle

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
