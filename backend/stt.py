"""
SPEECH-TO-TEXT (STT) ENGINE
=============================
This file turns spoken audio (what the participant said) into written text,
using "faster-whisper" — a faster reimplementation of OpenAI's Whisper
speech recognition model (same underlying model and accuracy, using the
CTranslate2 engine internally to run significantly faster, especially on
GPU). Fully local — no internet connection needed at runtime, no audio ever
leaves this computer.

WHY faster-whisper INSTEAD OF openai-whisper (Step 5 of the project plan,
"reduce latency"): this is a pure speed upgrade, not a quality trade-off —
it's the same Whisper models, just run through a faster engine. Automatic
device/precision selection (see WhisperModel below) means it also picks up
a GPU automatically if the study computer has one, without any code changes.

It is only used for the "voice" condition of the study. In the "text"
condition, this file is never called, so no speech model needs to be loaded.

You should not need to change anything in this file for normal study
configuration changes (wording, question order, language, etc.) — those live
in the study_content/ folder instead.

Used by: backend/speech.py (which connects recording, transcription, and
speaking together for the voice condition).
"""
import numpy as np
from faster_whisper import WhisperModel

# The model is loaded once, when this file is first imported, and then
# reused for every transcription (loading it fresh every time would be slow).
# The first time this runs on a machine, the model is downloaded automatically
# (needs an internet connection just that first time; fully offline after that).
#
# "base" is the model size — a good balance of accuracy and speed for a
# single local machine. Larger sizes (e.g. "small", "medium") are more
# accurate but slower and need more resources.
#
# device="auto" uses a GPU automatically if one is available, otherwise CPU.
# compute_type="default" picks a sensible numeric precision for whichever
# device gets used (e.g. faster int8 math on CPU, float16 on GPU) — no
# manual GPU/CPU flag needed, unlike the previous openai-whisper setup.
#
# compute_type = "default" — CTranslate2 wählt automatisch die für dein Gerät passende, meist schnellste Option (auf CPU meist int8)
# compute_type = "float32" — volle Genauigkeit, wie es vorher mit openai-whisper war, dafür langsamer
# compute_type = "int8" — explizit die schnelle, leicht ungenauere Variante erzwingen (statt sich auf "default" zu verlassen)
stt_model = WhisperModel("base", device="auto", compute_type="default")


def transcribe(audio_np: np.ndarray, language: str = "de") -> str:
    """
    Convert recorded audio into text.

    Args:
        audio_np: The recorded audio as a numpy array (16kHz, mono, float32,
            values between -1 and 1). This is the format produced by
            backend/speech.py after recording or decoding an uploaded file.
        language: Language code for transcription, e.g. "de" for German,
            "en" for English. Telling Whisper the language in advance makes
            transcription faster and more accurate than auto-detecting it.

    Returns:
        The transcribed text, with leading/trailing whitespace removed.
    """
    segments, _info = stt_model.transcribe(audio_np, language=language)
    # faster-whisper returns a generator of segments (one per detected
    # phrase/pause) rather than one combined string — join them into the
    # full transcript, same as the single "text" field the old engine gave us.
    text = "".join(segment.text for segment in segments).strip()
    return text
