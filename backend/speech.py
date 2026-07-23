"""
SPEECH — VOICE-CONDITION GLUE LAYER
=====================================
This file only exists for the "voice" (Sprache) condition of the study. It
connects the two audio engines (backend/stt.py for listening, backend/tts.py
for speaking) with the actual hardware and network plumbing:

  - recording from the participant's microphone (terminal mode)
  - playing the assistant's reply out loud (terminal mode)
  - converting uploaded browser audio into the format the STT engine expects
  - converting the assistant's spoken reply into the format the browser expects
  - a simple heuristic that makes the voice sound a bit more expressive when
    the reply text seems more emotional

In the "text" condition, none of this file is used: the participant types,
the model replies in writing, done. That is what keeps the two conditions
cleanly separated — Modalität (voice vs. text) only ever touches this one
file, never backend/llm_provider.py.

Used by: backend/app.py.
"""
import base64
import io
import time
import wave
from queue import Queue

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


def record_audio(stop_event, data_queue: Queue) -> None:
    """
    Record audio from the microphone until told to stop (terminal mode only).

    Args:
        stop_event: A threading.Event. Recording stops as soon as this is set
            (this happens when the user presses Enter again in the terminal).
        data_queue: A queue that raw audio chunks are pushed into as they are
            captured, so the main thread can collect them afterwards.
    """
    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def play_audio(sample_rate: int, audio_array: np.ndarray) -> None:
    """Play a generated voice reply out loud through the computer's speakers (terminal mode only)."""
    sd.play(audio_array, sample_rate)
    sd.wait()


def decode_uploaded_wav(audio_bytes: bytes) -> np.ndarray:
    """
    Convert an uploaded WAV audio file (from the browser) into the plain
    numpy array format that backend/stt.py's transcribe() expects: 16kHz,
    mono, float32 values between -1 and 1.

    Args:
        audio_bytes: The raw bytes of the uploaded .wav file.

    Returns:
        The audio as a numpy array, ready to pass to transcribe().
    """
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            # Mix stereo/multi-channel audio down to a single (mono) channel.
            audio_np = audio_np.reshape(-1, channels).mean(axis=1)

    if sample_rate != 16000:
        # Whisper expects 16kHz audio, so resample if the browser recorded at
        # a different rate (this does not change the pitch or speed of the
        # recording, only how many samples per second represent it).
        audio_np = resample_poly(audio_np, 16000, sample_rate)

    return audio_np


def encode_wav_base64(sample_rate: int, audio_array: np.ndarray) -> str:
    """
    Convert a generated voice reply (numpy array) into a base64-encoded WAV
    string, so it can be sent to the browser as part of a JSON response and
    played there.

    Args:
        sample_rate: The audio's sample rate, as returned by the TTS engine.
        audio_array: The generated audio, as floating-point values.

    Returns:
        A base64-encoded string containing the audio as a .wav file.
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        clipped = np.clip(audio_array, -1.0, 1.0)
        int16_audio = (clipped * 32767).astype(np.int16)
        wf.writeframes(int16_audio.tobytes())
    return base64.b64encode(wav_buffer.getvalue()).decode("ascii")


def analyze_emotion(text: str) -> float:
    """
    A simple, rough heuristic that estimates how "emotional" a reply sounds,
    based on a few keywords and punctuation marks. The result is used to
    make the synthesized voice sound a little more or less expressive
    (see the "exaggeration" setting in backend/tts.py).

    This is intentionally simple and NOT a clinical or diagnostic measure —
    it only adjusts voice delivery, nothing else.

    Returns:
        A value between 0.3 (calm) and 0.9 (expressive).
    """
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'wonderful', 'awful', '!', '?!', '...']

    emotion_score = 0.5  # Neutral starting point.
    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    return min(0.9, max(0.3, emotion_score))
