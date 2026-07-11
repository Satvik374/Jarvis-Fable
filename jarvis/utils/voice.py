"""Voice I/O: speak replies (TTS) and hear commands (STT).

TTS  - Gemini TTS on Vertex AI, using a model separate from Jarvis's brain.
STT  - record the microphone with sounddevice (simple energy-based
       start/stop), then transcribe with the existing Gemini backend, which
       accepts audio natively - no local Whisper/torch on this machine.
"""

from __future__ import annotations

import io
import math
import os
import struct
import tempfile
import wave

from ..config import VoiceConfig
from . import logging as log

_RATE = 16000          # 16 kHz mono int16 - plenty for speech
_CHUNK = 1600          # 0.1 s per energy reading


# --------------------------------------------------------------------------- #
# Gemini TTS
# --------------------------------------------------------------------------- #

_brain = None
_tts_config = VoiceConfig()
_async_wav_path: str | None = None


def configure(brain, config: VoiceConfig) -> None:
    """Connect voice output to Jarvis's authenticated Gemini brain."""
    global _brain, _tts_config
    _brain = brain
    _tts_config = config


def reset() -> None:
    """Clear configured TTS state and stop any asynchronous playback."""
    global _brain, _tts_config
    _stop_async_playback()
    _brain = None
    _tts_config = VoiceConfig()


def _synthesize(text: str) -> bytes:
    if _brain is None:
        raise RuntimeError("Gemini TTS is not configured")
    synthesize = getattr(_brain, "synthesize_speech", None)
    if synthesize is None:
        raise RuntimeError(
            "the active brain cannot authenticate Gemini TTS; "
            "use the gemini/Vertex backend"
        )
    return synthesize(
        text,
        model=_tts_config.model,
        voice_name=_tts_config.voice,
        language_code=_tts_config.language_code,
    )


def _wav_bytes(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _stop_async_playback() -> None:
    global _async_wav_path
    try:
        import winsound
        winsound.PlaySound(None, 0)
    except Exception:
        pass
    if _async_wav_path:
        try:
            os.unlink(_async_wav_path)
        except OSError:
            pass
        _async_wav_path = None


def _play_wav(data: bytes, wait: bool) -> None:
    """Play WAV bytes on Windows while preserving sync/async voice behavior."""
    import winsound

    global _async_wav_path
    _stop_async_playback()
    if wait:
        winsound.PlaySound(
            data,
            winsound.SND_MEMORY | winsound.SND_NODEFAULT,
        )
        return

    # Windows cannot reliably combine SND_MEMORY and SND_ASYNC. Keep a
    # temporary WAV alive until the next utterance replaces it.
    handle, path = tempfile.mkstemp(prefix="jarvis-tts-", suffix=".wav")
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
        _async_wav_path = path
        winsound.PlaySound(
            path,
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except Exception:
        try:
            os.close(handle)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        _async_wav_path = None
        raise


def speak(text: str, wait: bool = False) -> None:
    """Say text out loud. Async + purge by default (a new utterance interrupts
    the old, the loop never blocks); ``wait=True`` speaks synchronously - use
    it before the process exits so speech isn't cut off."""
    text = (text or "").strip()
    if not text:
        return
    try:
        _play_wav(_wav_bytes(_synthesize(text)), wait)
    except Exception as exc:
        log.warn(f"TTS unavailable: {exc}")


def speak_to_wav(text: str, path: str) -> bool:
    """Render speech to a WAV file (used by the self-test; no speakers needed)."""
    try:
        with open(path, "wb") as output:
            output.write(_wav_bytes(_synthesize(text)))
        return True
    except Exception as exc:
        log.warn(f"TTS-to-file failed: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Wake word
# --------------------------------------------------------------------------- #

def wait_for_wake(phrase: str = "hey jarvis", timeout: float | None = None,
                  _wav_file: str | None = None) -> bool:
    """Block until the wake phrase is spoken. Offline, near-zero CPU.

    Uses Windows' built-in SAPI recognizer with a one-phrase grammar - no
    audio ever leaves the machine while waiting. Returns True on wake,
    False on timeout or if the recognizer is unavailable. Ctrl+C propagates
    so the caller can exit hands-free mode.

    ``_wav_file`` feeds a file instead of the microphone (self-tests).
    """
    import time as _t
    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore
    except Exception as exc:
        log.warn(f"wake word needs pywin32: {exc}")
        return False

    pythoncom.CoInitialize()
    grammar = None
    try:
        rec = win32com.client.Dispatch("SAPI.SpInprocRecognizer")
        if _wav_file:
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(_wav_file)
            rec.AudioInputStream = stream
        else:
            rec.AudioInput = win32com.client.Dispatch("SAPI.SpMMAudioIn")
        ctx = rec.CreateRecoContext()
        grammar = ctx.CreateGrammar()
        rule = grammar.Rules.Add("wake", 1 | 2)      # SRATopLevel|SRADynamic
        rule.InitialState.AddWordTransition(None, phrase)
        grammar.Rules.Commit()
        grammar.CmdSetRuleState("wake", 1)           # SGDSActive

        heard: list[int] = []

        class _Sink:
            def OnRecognition(self, *args):
                heard.append(1)

        win32com.client.WithEvents(ctx, _Sink)
        t0 = _t.time()
        while not heard:
            pythoncom.PumpWaitingMessages()
            _t.sleep(0.05)
            if timeout is not None and _t.time() - t0 > timeout:
                return False
        return True
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        log.warn(f"wake-word listener failed: {exc}")
        return False
    finally:
        try:
            if grammar is not None:
                grammar.CmdSetRuleState("wake", 0)   # release before mic reuse
        except Exception:
            pass
        pythoncom.CoUninitialize()


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #

class _EnergyVAD:
    """Streaming energy VAD tuned to ignore steady background (e.g. fans).

    Feed it one RMS level per audio chunk (``_CHUNK`` = 0.1 s).

    START: the fan/room level is tracked continuously as the noise floor (an
    EWMA that only learns from below-threshold chunks, so speech never inflates
    it). Recording begins only when a chunk is ``START_FACTOR`` * above that
    floor AND stays there for ``START_HOLD`` chunks - a fan surge or a thud is
    brief and never sustains, so it can't trip recording; a spoken word does.

    STOP: end-of-speech is judged relative to the speech's own loudness
    (``speech_ref``), not the floor, so it can't get wedged when the background
    sits higher than it did at warmup - the failure that made recording run to
    ``max_seconds`` every time.
    """
    START_FACTOR = 3.0     # speech must exceed the noise floor by this ratio
    START_HOLD = 3         # ... for this many chunks (~0.3 s) - rejects surges
    STOP_FRAC = 0.35       # end when level falls below this * speech loudness
    SPEECH_DECAY = 0.97    # speech_ref bleeds down so one loud blip can't stick
    MIN_FLOOR = 120.0      # a near-zero floor (muted mic) can't drive the gate
    WARMUP = 5             # ~0.5 s to let the floor settle on the fan noise

    def __init__(self, silence_after: float, chunk: float = 0.1):
        self.silence_after = silence_after
        self.chunk = chunk
        self.floor: float | None = None
        self.speech_ref = 0.0
        self.started = False
        self.quiet = 0.0
        self.warmed = 0
        self.hold = 0

    def _track_floor(self, level: float) -> None:
        self.floor = level if self.floor is None else 0.9 * self.floor + 0.1 * level

    def feed(self, level: float) -> str:
        """Return 'listening' (pre-speech), 'recording', or 'stop'."""
        if not self.started:
            floor = max(self.MIN_FLOOR,
                        self.floor if self.floor is not None else level)
            if self.warmed < self.WARMUP:         # settle the floor on the fans
                self.warmed += 1                  # before the start gate arms
                self._track_floor(level)
                return "listening"
            if level >= self.START_FACTOR * floor:
                self.hold += 1                    # loud - but is it sustained?
                if self.hold >= self.START_HOLD:
                    self.started = True
                    self.speech_ref = level
                    return "recording"
                return "listening"
            self.hold = 0                         # dropped back to the fans
            self._track_floor(level)              # learn the floor from them only
            return "listening"
        # Recording: track the speech's loudness (jump up fast, bleed down slow),
        # and call it silence once the level drops well below that.
        self.speech_ref = max(level, self.SPEECH_DECAY * self.speech_ref)
        if level < self.STOP_FRAC * self.speech_ref:
            self.quiet += self.chunk
        else:
            self.quiet = 0.0
        return "stop" if self.quiet >= self.silence_after else "recording"


def listen(start_timeout: float = 6.0, max_seconds: float = 12.0,
           silence_after: float | None = None) -> bytes | None:
    """Record one spoken phrase from the default microphone.

    Waits up to ``start_timeout`` for speech to begin, then records until
    ``silence_after`` seconds of quiet (or ``max_seconds`` total). Returns
    WAV bytes, or None if nothing was heard / no microphone.
    """
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        log.warn(f"microphone unavailable: {exc}")
        return None
    if silence_after is None:
        silence_after = _tts_config.silence_after

    def rms(chunk: bytes) -> float:
        n = len(chunk) // 2
        if not n:
            return 0.0
        samples = struct.unpack(f"<{n}h", chunk)
        return math.sqrt(sum(s * s for s in samples) / n)

    from collections import deque
    frames: list[bytes] = []
    waited = 0.0
    det = _EnergyVAD(silence_after)
    # Keep the last few pre-speech chunks so the start-hold delay doesn't clip
    # the onset of the first word.
    preroll: deque = deque(maxlen=_EnergyVAD.START_HOLD)
    try:
        with sd.RawInputStream(samplerate=_RATE, channels=1, dtype="int16",
                               blocksize=_CHUNK) as stream:
            while True:
                chunk, _ = stream.read(_CHUNK)
                chunk = bytes(chunk)
                state = det.feed(rms(chunk))
                if state == "listening":
                    preroll.append(chunk)
                    waited += 0.1
                    if waited >= start_timeout:
                        return None               # heard nothing
                    continue
                if not frames:                    # just started - recover onset
                    frames.extend(preroll)
                frames.append(chunk)
                if state == "stop" or len(frames) * 0.1 >= max_seconds:
                    hit_cap = state != "stop"
                    break
    except Exception as exc:
        log.warn(f"recording failed: {exc}")
        return None

    secs = len(frames) * 0.1
    log.info(f"recorded {secs:.1f}s"
             + (" (hit max_seconds - silence never detected)" if hit_cap else ""))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def transcribe(wav_bytes: bytes, brain) -> str:
    """Turn recorded speech into text using the configured brain backend."""
    fn = getattr(brain, "transcribe_audio", None)
    if fn is None:
        log.warn("this brain backend cannot transcribe audio "
                 "(voice input needs the gemini backend).")
        return ""
    try:
        return (fn(
            wav_bytes,
            model=_tts_config.transcription_model,
        ) or "").strip()
    except Exception as exc:
        log.warn(f"transcription failed: {exc}")
        return ""
