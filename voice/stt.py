"""
voice/stt.py — Speech-to-Text module

Improvements:
  - Whisper model is loaded ONCE at module level (singleton) — no cold-start
    delay on the second question you ask.
  - Recording uses Voice Activity Detection (VAD): it stops automatically
    the moment you go silent for SILENCE_DURATION seconds, up to a hard cap
    of MAX_RECORD_SECONDS.  No more sitting in silence waiting for a timer!
"""

import sys
import os
import numpy as np
import sounddevice as sd
import warnings

# ── Allow running this file directly (python voice/stt.py) ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings

# ── Whisper Singleton ────────────────────────────────────────────────────────
# Loaded ONCE when this module is first imported.  Every subsequent call to
# speech_to_text() reuses the already-loaded model — no repeated disk reads.
_whisper_model = None

def _get_whisper_model():
    """Returns the cached Whisper model, loading it on the very first call."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        warnings.filterwarnings(
            "ignore", message="FP16 is not supported on CPU; using FP32 instead"
        )
        print(f"[Loading Whisper '{settings.WHISPER_MODEL_SIZE}' model — one-time cost...]")
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE)
        print("[Whisper model ready!]")
    return _whisper_model


# ── VAD Recording ────────────────────────────────────────────────────────────

def record_until_silence(
    sample_rate: int = None,
    silence_threshold: float = None,
    silence_duration: float = None,
    max_seconds: int = None,
) -> tuple[np.ndarray, int]:
    """
    Records from the microphone and stops automatically once the user has been
    silent for `silence_duration` seconds (or `max_seconds` has elapsed).

    Algorithm:
      - Audio is captured in small blocks (0.1 s each).
      - For each block the RMS amplitude is measured.
      - If RMS < silence_threshold the block is counted as "silent".
      - Once the cumulative silence exceeds `silence_duration` AND at least
        0.5 s of speech was heard, recording stops.

    Returns:
        (audio_array, sample_rate) — 1-D float32 array ready for Whisper.
    """
    sr               = sample_rate       or settings.SAMPLE_RATE
    threshold        = silence_threshold or settings.SILENCE_THRESHOLD
    silence_secs     = silence_duration  or settings.SILENCE_DURATION
    max_secs         = max_seconds       or settings.MAX_RECORD_SECONDS

    block_duration   = 0.1                        # seconds per chunk
    block_size       = int(sr * block_duration)   # samples per chunk
    silence_blocks   = int(silence_secs / block_duration)
    max_blocks       = int(max_secs    / block_duration)

    print(
        f"\n[Mic] Listening — speak now! "
        f"(stops after {silence_secs}s of silence, max {max_secs}s)]"
    )

    collected_chunks   : list[np.ndarray] = []
    consecutive_silent : int              = 0
    total_speech_secs  : float            = 0.0

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(block_size)    # shape: (block_size, 1)
            chunk    = block.flatten()
            collected_chunks.append(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms < threshold:
                consecutive_silent += 1
            else:
                consecutive_silent  = 0
                total_speech_secs  += block_duration

            # Only stop on silence if we already captured some real speech
            if consecutive_silent >= silence_blocks and total_speech_secs >= 0.5:
                print(f"[Silence detected — stopping after {total_speech_secs:.1f}s of speech]")
                break
        else:
            print(f"[Max recording time ({max_secs}s) reached]")

    audio = np.concatenate(collected_chunks, axis=0)
    print("[Recording complete!]")
    return audio, sr


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_audio(audio_array: np.ndarray) -> str:
    """
    Transcribes a 1-D float32 NumPy audio array using the cached Whisper model.
    No FFmpeg required — Whisper accepts raw arrays directly.
    """
    model = _get_whisper_model()
    print("[Transcribing your speech...]")
    result = model.transcribe(audio_array)
    text   = result["text"].strip()
    print(f'[You said: "{text}"]')
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def speech_to_text(
    silence_threshold: float = None,
    silence_duration: float  = None,
    max_seconds: int         = None,
) -> str:
    """
    Full STT pipeline:
      1. Record with VAD (stops on silence automatically)
      2. Transcribe with cached Whisper model
      3. Return the transcribed text
    """
    audio, _ = record_until_silence(
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        max_seconds=max_seconds,
    )
    return transcribe_audio(audio)


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Speech-to-Text Test (VAD + Singleton Whisper) ===\n")
    result = speech_to_text()
    print(f"\n--- Transcribed: ---\n{result}")
