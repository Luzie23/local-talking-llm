"""
SPEECH-TO-TEXT (STT) ENGINE
=============================
This file turns spoken audio (what the participant said) into written text,
using the "Whisper" speech recognition model from OpenAI (it runs fully
locally — no internet connection needed, no audio ever leaves this computer).

It is only used for the "voice" condition of the study. In the "text"
condition, this file is never called, so no speech model needs to be loaded.

You should not need to change anything in this file for normal study
configuration changes (wording, question order, language, etc.) — those live
in the study_content/ folder instead.

Used by: backend/speech.py (which connects recording, transcription, and
speaking together for the voice condition).
"""
import whisper
import numpy as np

# The Whisper model is loaded once, when this file is first imported, and then
# reused for every transcription (loading it fresh every time would be slow).
# "base" is the model size — a good balance of accuracy and speed for a single
# local machine. Larger sizes (e.g. "small", "medium") are more accurate but
# slower and need more VRAM.
stt_model = whisper.load_model("base")


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
    # fp16=False keeps this compatible with CPU-only machines. Set fp16=True
    # if this is running on an NVIDIA GPU, for faster transcription.
    result = stt_model.transcribe(audio_np, fp16=False, language=language)
    text = result["text"].strip()
    return text
