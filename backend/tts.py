"""
TEXT-TO-SPEECH (TTS) ENGINE
============================
This file turns written text into spoken audio (a voice reading the text out
loud), using Piper — a fast, fully local, open-source neural TTS engine.

It is only used for the "voice" condition of the study. In the "text"
condition, this file is never called, so no voice model needs to be loaded.

WHY PIPER (as of Step 4 of the project plan): the previous engine
(ChatterBox) speaks German with a noticeable English accent, and offers
voice cloning plus "emotion" controls (exaggeration/cfg_weight) that this
study doesn't need. Piper ships genuinely natural-sounding, purpose-built
German voices, runs fast enough for real-time use even on a plain CPU, and
is actively maintained. The trade-off: one fixed voice per model, no
cloning, no per-line emotional expressiveness — for a calm, consistent
study interview, that's a reasonable trade, not a loss.

SETUP — downloading a voice (one-time, per machine):
    python -m piper.download_voices de_DE-thorsten-high --download-dir tts_models
This downloads two files into tts_models/ (a .onnx model and a .onnx.json
config) — see study_content/study_settings.yaml's "tts_voice" setting for
which voice name is currently active, and the project README for other
available German voices.

Used by: backend/app.py.
"""
import os

import numpy as np
from piper import PiperVoice

# Where downloaded voice model files (.onnx + .onnx.json) live. Not inside
# study_content/, since these are large binary downloads you fetch once per
# machine, not something you hand-edit — see the setup instructions above.
TTS_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts_models"
)


class TextToSpeechService:
    """Loads one Piper voice and turns text into audio for it."""

    def __init__(self, voice_name: str = "de_DE-thorsten-high", use_cuda: bool = False):
        """
        Args:
            voice_name: Which downloaded Piper voice to use, e.g.
                "de_DE-thorsten-high" (see study_content/study_settings.yaml's
                "tts_voice" setting — this is normally passed in from there,
                not hardcoded by callers).
            use_cuda: Whether to run the voice model on an NVIDIA GPU via
                onnxruntime-gpu. Piper is fast enough on a plain CPU that
                this is usually unnecessary — only turn it on if you've
                installed onnxruntime-gpu and want to double-check it helps.

        Raises:
            FileNotFoundError: if the voice hasn't been downloaded yet (see
                the setup instructions in this file's module docstring).
        """
        model_path = os.path.join(TTS_MODELS_DIR, f"{voice_name}.onnx")
        config_path = os.path.join(TTS_MODELS_DIR, f"{voice_name}.onnx.json")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Piper voice '{voice_name}' not found in {TTS_MODELS_DIR}.\n"
                f"Download it once with:\n"
                f"  python -m piper.download_voices {voice_name} --download-dir {TTS_MODELS_DIR}"
            )

        self.voice_name = voice_name
        self.voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """
        Convert text into spoken audio.

        Piper internally splits the text into sentences and generates one
        audio chunk per sentence; this method stitches them back together
        into a single array, matching how backend/app.py expects to receive
        audio (as one clip, not several).

        Args:
            text: The text to speak aloud. Can be more than one sentence.

        Returns:
            (sample_rate, audio_array) — audio_array is a 1-D float32 numpy
            array (mono), with values between -1 and 1. If text is empty,
            audio_array will be empty too.
        """
        if not text.strip():
            return self.voice.config.sample_rate, np.zeros(0, dtype=np.float32)

        chunks = list(self.voice.synthesize(text))
        if not chunks:
            return self.voice.config.sample_rate, np.zeros(0, dtype=np.float32)

        sample_rate = chunks[0].sample_rate
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return sample_rate, audio
