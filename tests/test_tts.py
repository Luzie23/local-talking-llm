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
        sample_rate, audio = service.synthesize("Erster Satz. Zweiter Satz.", sentence_silence=0.0)
        self.assertEqual(sample_rate, 22050)
        np.testing.assert_allclose(audio, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_inserts_silence_between_sentences(self):
        chunks = [
            _fake_chunk(1000, [1.0, 1.0]),
            _fake_chunk(1000, [1.0, 1.0]),
        ]
        service = _make_service_with_fake_voice(chunks, sample_rate=1000)
        sample_rate, audio = service.synthesize("Satz eins. Satz zwei.", sentence_silence=0.1)
        # 0.1s at 1000Hz = 100 silent samples inserted between the two chunks.
        self.assertEqual(len(audio), 2 + 100 + 2)
        silent_region = audio[2:2 + 100]
        self.assertTrue(np.all(silent_region == 0.0))

    def test_no_silence_added_before_or_after_a_single_chunk(self):
        chunks = [_fake_chunk(1000, [1.0, 1.0, 1.0])]
        service = _make_service_with_fake_voice(chunks, sample_rate=1000)
        sample_rate, audio = service.synthesize("Nur ein Satz.", sentence_silence=0.5)
        np.testing.assert_allclose(audio, [1.0, 1.0, 1.0])

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


class TestSynthesizeStreaming(unittest.TestCase):
    def test_yields_one_chunk_per_sentence_in_order(self):
        chunks = [
            _fake_chunk(1000, [1.0, 1.0]),
            _fake_chunk(1000, [2.0, 2.0]),
            _fake_chunk(1000, [3.0, 3.0]),
        ]
        service = _make_service_with_fake_voice(chunks, sample_rate=1000)
        results = list(service.synthesize_streaming("Eins. Zwei. Drei.", sentence_silence=0.1))
        self.assertEqual(len(results), 3)

    def test_trailing_silence_between_but_not_after_last_sentence(self):
        chunks = [
            _fake_chunk(1000, [1.0, 1.0]),
            _fake_chunk(1000, [2.0, 2.0]),
        ]
        service = _make_service_with_fake_voice(chunks, sample_rate=1000)
        results = list(service.synthesize_streaming("Eins. Zwei.", sentence_silence=0.1))

        # First sentence: audio + 100 silent samples (0.1s at 1000Hz).
        first_sr, first_audio = results[0]
        self.assertEqual(len(first_audio), 2 + 100)
        np.testing.assert_allclose(first_audio[:2], [1.0, 1.0])
        self.assertTrue(np.all(first_audio[2:] == 0.0))

        # Last sentence: no trailing silence appended.
        _second_sr, second_audio = results[1]
        np.testing.assert_allclose(second_audio, [2.0, 2.0])

    def test_empty_text_yields_nothing(self):
        service = _make_service_with_fake_voice([])
        results = list(service.synthesize_streaming("   "))
        self.assertEqual(results, [])
        service.voice.synthesize.assert_not_called()

    def test_single_sentence_no_silence_appended(self):
        chunks = [_fake_chunk(1000, [1.0, 1.0, 1.0])]
        service = _make_service_with_fake_voice(chunks, sample_rate=1000)
        results = list(service.synthesize_streaming("Nur ein Satz.", sentence_silence=0.5))
        self.assertEqual(len(results), 1)
        np.testing.assert_allclose(results[0][1], [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
