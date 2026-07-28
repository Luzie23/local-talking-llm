"""Unit tests for backend/tts.py. These use a mocked PiperVoice instead of a
real downloaded voice model, so they run without needing any model files."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import tts  # noqa: E402


def _make_service_with_fake_voice(chunks, sample_rate=22050):
    """Build a TextToSpeechService instance without loading a real model,
    wired up with a fake PiperVoice that returns the given chunks."""
    service = tts.TextToSpeechService.__new__(tts.TextToSpeechService)
    service.voice_name = "fake-voice"
    service.voice = MagicMock()
    service.voice.synthesize.return_value = chunks
    service.voice.config.sample_rate = sample_rate
    return service


def _fake_chunk(sample_rate, values):
    chunk = MagicMock()
    chunk.sample_rate = sample_rate
    chunk.audio_float_array = np.array(values, dtype=np.float32)
    return chunk


class TestTextToSpeechService(unittest.TestCase):
    def test_missing_voice_raises_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            tts.TextToSpeechService(voice_name="de_DE-does-not-exist")
        message = str(ctx.exception)
        self.assertIn("de_DE-does-not-exist", message)
        self.assertIn("python -m piper.download_voices", message)

    def test_concatenates_multiple_sentence_chunks(self):
        chunks = [
            _fake_chunk(22050, [0.1, 0.2, 0.3]),
            _fake_chunk(22050, [0.4, 0.5]),
        ]
        service = _make_service_with_fake_voice(chunks)
        sample_rate, audio = service.synthesize("Erster Satz. Zweiter Satz.")
        self.assertEqual(sample_rate, 22050)
        np.testing.assert_allclose(audio, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_empty_text_returns_empty_audio(self):
        service = _make_service_with_fake_voice([])
        sample_rate, audio = service.synthesize("   ")
        self.assertEqual(sample_rate, 22050)
        self.assertEqual(len(audio), 0)
        # Should not have even called the voice for whitespace-only text.
        service.voice.synthesize.assert_not_called()

    def test_no_chunks_returned_gives_empty_audio(self):
        service = _make_service_with_fake_voice([])
        sample_rate, audio = service.synthesize("Text das keine Chunks liefert.")
        self.assertEqual(sample_rate, 22050)
        self.assertEqual(len(audio), 0)


if __name__ == "__main__":
    unittest.main()
