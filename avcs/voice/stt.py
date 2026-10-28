"""Speech-to-text: the ears of the AVCS assistant.

The user talks to the assistant ("mujhe X kaam karna hai"); this module turns
that audio into text the reasoning brain can act on.

Engines are pluggable and every one is optional, because the assistant must
still run (silently listening to nothing) on a machine with no microphone, no
model file, or no recognizer installed:

* ``VoskEngine``      -- offline, small models, no network. Preferred default.
* ``FasterWhisperEngine`` -- offline, higher accuracy, needs a model download.
* ``NullEngine``      -- never returns a transcript; keeps the pipeline testable.

The important contract is ``transcribe(audio) -> Transcript``: a transcript
carries the text *and* the confidence/source, so a caller can decide whether to
trust it. An empty transcript is a first-class, honest result -- not an error.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import wave
from typing import Optional

import numpy as np

from avcs.voice.errors import EngineNotAvailable, ModelNotFound


@dataclasses.dataclass
class AudioClip:
    """Mono float32 PCM in [-1, 1] at a known sample rate."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim > 1:
            self.samples = self.samples.mean(axis=-1)
        self.samples = self.samples.astype(np.float32)

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples))))

    def to_pcm16_bytes(self) -> bytes:
        clipped = np.clip(self.samples, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes()

    @classmethod
    def from_pcm16_bytes(cls, data: bytes, sample_rate: int) -> "AudioClip":
        arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
        return cls(samples=arr, sample_rate=sample_rate)

    @classmethod
    def read_wav(cls, path: str) -> "AudioClip":
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        elif width == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width} bytes")
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        return cls(samples=arr, sample_rate=rate)

    @classmethod
    def read_wav_bytes(cls, data: bytes) -> "AudioClip":
        """Parse an in-memory WAV. Used for audio posted from a browser."""
        import io

        with wave.open(io.BytesIO(data), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        elif width == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width} bytes")
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        return cls(samples=arr, sample_rate=rate)

    def write_wav(self, path: str) -> None:
        pcm = self.to_pcm16_bytes()
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.sample_rate))
            w.writeframes(pcm)

    def trim_silence(self, threshold: float = 0.008, pad_s: float = 0.05) -> "AudioClip":
        """Drop leading/trailing near-silence so a recognizer sees speech only.

        A fixed-length capture usually has a long quiet tail (the user stopped
        talking before the window closed). Feeding that to a recognizer wastes
        time and can add hallucinated words, so trim it, keeping a small pad.
        """
        if self.samples.size == 0:
            return self
        mag = np.abs(self.samples)
        voiced = np.where(mag > threshold)[0]
        if voiced.size == 0:
            return AudioClip(samples=np.zeros(0, dtype=np.float32), sample_rate=self.sample_rate)
        pad = int(pad_s * self.sample_rate)
        lo = max(0, int(voiced[0]) - pad)
        hi = min(len(self.samples), int(voiced[-1]) + pad)
        return AudioClip(samples=self.samples[lo:hi], sample_rate=self.sample_rate)


@dataclasses.dataclass
class Transcript:
    """One recognition result, honest about where the text came from."""

    text: str
    confidence: float = 0.0
    engine: str = "null"
    is_final: bool = True
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    def normalized(self) -> str:
        return " ".join(self.text.strip().split())


class SttEngine:
    """Base class for a recognizer."""

    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def transcribe(self, audio: AudioClip) -> Transcript:
        raise NotImplementedError


class NullEngine(SttEngine):
    """Always available, never recognizes. Keeps the pipeline runnable."""

    name = "null"

    def available(self) -> bool:
        return True

    def transcribe(self, audio: AudioClip) -> Transcript:
        return Transcript(text="", engine=self.name, confidence=0.0, error="no recognizer configured")


class VoskEngine(SttEngine):
    """Offline recognition with a Vosk model directory.

    Vosk needs 16 kHz mono PCM. We resample linearly when the caller's capture
    rate differs, which is accurate enough for speech at these rates and avoids
    pulling in a resampling dependency.
    """

    name = "vosk"

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000) -> None:
        self.model_path = model_path or os.environ.get("AVCS_VOSK_MODEL")
        self.sample_rate = sample_rate
        self._model = None

    def available(self) -> bool:
        if not self.model_path or not os.path.isdir(self.model_path):
            return False
        try:
            import vosk  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._model is None:
            if not self.model_path or not os.path.isdir(self.model_path):
                raise ModelNotFound(f"Vosk model directory not found: {self.model_path}")
            try:
                import vosk
            except ImportError as exc:
                raise EngineNotAvailable("vosk package is not installed") from exc
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(self.model_path)
        return self._model

    def _resample(self, audio: AudioClip) -> np.ndarray:
        if audio.sample_rate == self.sample_rate:
            return audio.samples
        if audio.samples.size == 0:
            return audio.samples
        n_out = int(round(audio.samples.size * self.sample_rate / audio.sample_rate))
        src_idx = np.linspace(0.0, audio.samples.size - 1, num=max(1, n_out))
        return np.interp(src_idx, np.arange(audio.samples.size), audio.samples).astype(np.float32)

    def transcribe(self, audio: AudioClip) -> Transcript:
        if not self.available():
            return Transcript(
                text="", engine=self.name,
                error=f"vosk unavailable (model={self.model_path}, package missing or path absent)",
            )
        if audio.samples.size == 0:
            return Transcript(text="", engine=self.name, error="empty audio")
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover - guarded by available()
            return Transcript(text="", engine=self.name, error=str(exc))

        model = self._load()
        samples = self._resample(audio)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2").tobytes()

        rec = vosk.KaldiRecognizer(model, self.sample_rate)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm16)
        result = json.loads(rec.FinalResult())
        text = str(result.get("text", "")).strip()
        return Transcript(text=text, engine=self.name, confidence=1.0 if text else 0.0)


class FasterWhisperEngine(SttEngine):
    """Offline recognition via ``faster-whisper`` (CTranslate2).

    Higher accuracy than the small Vosk models, at a larger download and CPU
    cost. Kept as an opt-in alternative so the default path stays light.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        # None lets Whisper auto-detect, which is the right default for a
        # Hinglish speaker (mixed Hindi/English in one utterance). Pass "en" or
        # "hi" to force a language.
        self.language = language
        self._model = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise EngineNotAvailable("faster-whisper is not installed") from exc
            self._model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio: AudioClip) -> Transcript:
        if not self.available():
            return Transcript(text="", engine=self.name, error="faster-whisper not installed")
        if audio.samples.size == 0:
            return Transcript(text="", engine=self.name, error="empty audio")
        model = self._load()
        segments, info = model.transcribe(
            audio.samples, language=self.language, beam_size=1
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        conf = float(getattr(info, "language_probability", 0.0) or 0.0)
        return Transcript(text=text, engine=self.name, confidence=conf)


def find_vosk_model() -> Optional[str]:
    """Locate a Vosk model directory on this machine, or ``None``.

    A recognizer that only works when the caller passes an explicit path is not
    usable out of the box. This checks the environment, then the conventional
    places a downloaded model lands (``models/`` in the repo, ``~/.cache/vosk``,
    ``/opt/vosk``), so installing a model is enough to give the assistant ears.
    """
    env = os.environ.get("AVCS_VOSK_MODEL")
    if env and os.path.isdir(env):
        return env

    roots = [
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"),
        os.path.expanduser("~/.cache/vosk"),
        os.path.expanduser("~/.local/share/vosk"),
        "/opt/vosk",
        "/usr/local/share/vosk",
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        # Prefer a Hindi/Indian-English model when several are present: the
        # assistant speaks Hinglish, so that is the one it should hear with.
        candidates = sorted(
            (d for d in os.listdir(root) if d.startswith("vosk-model")),
            key=lambda d: ("hi" not in d.lower() and "in" not in d.lower(), d),
        )
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
    return None


def download_vosk_model(dest_dir: str = "models", model: Optional[str] = None) -> Optional[str]:
    """Download and extract a small Vosk model so the assistant has ears.

    Returns the model directory, or ``None`` on failure. Kept dependency-free
    (urllib + zipfile) and CPU-only, matching the rest of this repo. A model
    already present in ``dest_dir`` is reused rather than re-downloaded.
    """
    import urllib.request
    import zipfile

    url = model or os.environ.get("AVCS_VOSK_MODEL_URL") or (
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    )
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".zip"):
        name = name[:-4]
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, name)
    if os.path.isdir(target):
        return target

    zip_path = os.path.join(dest_dir, name + ".zip")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    except Exception:
        return None
    finally:
        try:
            os.unlink(zip_path)
        except OSError:
            pass
    return target if os.path.isdir(target) else None


def default_stt_engine(model_path: Optional[str] = None, language: Optional[str] = None) -> SttEngine:
    """Best available offline recognizer, or a silent one. Never raises.

    `language` is passed to Whisper when it is the chosen engine; ``None`` lets
    Whisper auto-detect, which is what a Hinglish speaker needs (Hindi and
    English mixed in one utterance). Vosk ignores it: a Vosk model is bound to
    the language it was trained on, so choose a Hindi/Indian-English Vosk model
    by path instead.
    """
    vosk = VoskEngine(model_path=model_path or find_vosk_model())
    if vosk.available():
        return vosk
    fw = FasterWhisperEngine(language=language)
    if fw.available():
        return fw
    return NullEngine()


def _capture_error(tool: str, stderr: bytes) -> str:
    """One readable line from a capture tool's stderr, without ALSA internals.

    A machine with no capture device prints pages of ALSA `confmisc` noise; the
    useful fact is only that there is no usable input device.
    """
    raw = stderr.decode("utf-8", "replace").strip()
    if "cannot find card" in raw or "No such file or directory" in raw or "no default" in raw.lower():
        return f"{tool} found no input device (no microphone on this machine)"
    first = raw.splitlines()[0] if raw else "unknown error"
    return f"{tool} capture failed: {first[:160]}"


def record_microphone(seconds: float = 4.0, sample_rate: int = 16000) -> AudioClip:
    """Capture `seconds` of mono audio from the default input device.

    Uses the ``arecord`` ALSA utility when present (no Python audio dependency),
    falling back to ``sox``. Raises ``EngineNotAvailable`` when neither exists so
    a caller can fall back to reading a WAV file instead.
    """
    if seconds <= 0:
        return AudioClip(samples=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)

    import tempfile
    import subprocess

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="avcs_mic_")
    os.close(fd)
    try:
        arecord = shutil.which("arecord")
        if arecord:
            cmd = [arecord, "-q", "-f", "S16_LE", "-r", str(sample_rate),
                   "-c", "1", "-d", str(int(round(seconds))), path]
        else:
            sox = shutil.which("sox")
            if not sox:
                raise EngineNotAvailable(
                    "no microphone capture tool found (install alsa-utils for arecord or sox)"
                )
            cmd = [sox, "-q", "-d", "-r", str(sample_rate), "-c", "1", path, "trim", "0", str(seconds)]
        proc = subprocess.run(cmd, capture_output=True, timeout=int(seconds) + 30)
        if proc.returncode != 0 or not os.path.exists(path):
            raise EngineNotAvailable(_capture_error(os.path.basename(cmd[0]), proc.stderr))
        return AudioClip.read_wav(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def record_microphone_until_silence(
    max_seconds: float = 15.0,
    silence_seconds: float = 0.8,
    sample_rate: int = 16000,
    threshold: str = "3%",
) -> AudioClip:
    """Record until the speaker stops, instead of a fixed window.

    A fixed capture makes the conversation feel laggy: the user finishes a
    sentence and then waits out the rest of the window before the assistant
    replies. This stops on trailing silence, so the reply starts as soon as the
    user does. Falls back to a fixed capture when ``sox`` is unavailable (it is
    the only tool here that does end-pointing without a Python VAD dependency).
    """
    import subprocess
    import tempfile

    sox = shutil.which("sox")
    if not sox:
        return record_microphone(seconds=min(max_seconds, 5.0), sample_rate=sample_rate)

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="avcs_mic_")
    os.close(fd)
    try:
        # silence 1 <dur> <thr>: start once sound rises above thr; 1 <dur> <thr>:
        # stop once it stays below thr for <dur>. trim caps the total length.
        cmd = [
            sox, "-q", "-d", "-r", str(sample_rate), "-c", "1", path,
            "silence", "1", "0.2", threshold, "1", str(silence_seconds), threshold,
            "trim", "0", str(max_seconds),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=int(max_seconds) + 30)
        if proc.returncode != 0 or not os.path.exists(path):
            raise EngineNotAvailable(_capture_error(os.path.basename(cmd[0]), proc.stderr))
        return AudioClip.read_wav(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
