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

    def synthesize(self, text: str, sentence_silence: float = 0.3) -> tuple[int, np.ndarray]:
        """
        Convert text into spoken audio.

        Piper internally splits the text into sentences and generates one
        audio chunk per sentence, with no gap between them. This method
        stitches the chunks back together with a short silence inserted
        between each one — without it, sentences run directly into each
        other with no natural pause, which sounds abrupt/"choppy" rather
        than like normal speech.

        Args:
            text: The text to speak aloud. Can be more than one sentence.
            sentence_silence: Seconds of silence to insert between
                sentences. 0.3 is a natural-sounding default; raise it a
                bit (e.g. 0.4-0.5) for an even calmer, slower-paced
                delivery, or lower it for snappier pacing.

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
        silence = np.zeros(int(sample_rate * sentence_silence), dtype=np.float32)

        pieces = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                pieces.append(silence)
            pieces.append(chunk.audio_float_array)

        audio = np.concatenate(pieces)
        return sample_rate, audio

    def synthesize_streaming(self, text: str, sentence_silence: float = 0.3):
        """
        Like synthesize(), but yields one (sample_rate, audio_array) pair
        per sentence as soon as each one is ready, instead of waiting for
        the entire text to finish and returning it all as a single array.

        This is what makes Step 5 ("reduce latency") possible for spoken
        replies: a caller can start playing the first sentence while later
        sentences are still being synthesized, instead of waiting for the
        whole reply before any sound plays at all — see backend/app.py's
        terminal-mode speak() for how this is used.

        Args:
            text: The text to speak aloud. Can be more than one sentence.
            sentence_silence: Seconds of silence appended after each
                sentence except the last one, so consecutive playback
                calls still sound like natural speech with a pause in
                between (see synthesize()'s docstring for why this matters).

        Yields:
            (sample_rate, audio_array) tuples, one per sentence, in the
            order they should be played.
        """
        if not text.strip():
            return

        silence = None
        pending_chunk = None
        pending_sample_rate = None

        # One chunk is held back at a time so we know, once the loop ends,
        # which chunk was the LAST one — that one should have no trailing
        # silence appended (nothing comes after it).
        for chunk in self.voice.synthesize(text):
            if pending_chunk is not None:
                yield pending_sample_rate, np.concatenate([pending_chunk, silence])

            pending_sample_rate = chunk.sample_rate
            pending_chunk = chunk.audio_float_array
            if silence is None:
                silence = np.zeros(int(chunk.sample_rate * sentence_silence), dtype=np.float32)

        if pending_chunk is not None:
            yield pending_sample_rate, pending_chunk
