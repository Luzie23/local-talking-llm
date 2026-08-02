"""Unit tests for backend/stt.py. Uses a mocked faster_whisper.WhisperModel
instead of a real downloaded model, so these run without needing any model
files or network access."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def _import_stt_with_mocked_model():
    """Import backend/stt.py with WhisperModel mocked out, so module import
    doesn't try to load/download a real model."""
    with patch("faster_whisper.WhisperModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()
        if "stt" in sys.modules:
            del sys.modules["stt"]
        import stt
        return stt


class TestTranscribe(unittest.TestCase):
    def test_joins_multiple_segments_into_one_transcript(self):
        stt = _import_stt_with_mocked_model()
        seg1 = MagicMock(text=" Hallo,")
        seg2 = MagicMock(text=" wie geht es Ihnen?")
        stt.stt_model.transcribe.return_value = ([seg1, seg2], MagicMock())

        result = stt.transcribe(np.zeros(1000, dtype=np.float32), language="de")
        self.assertEqual(result, "Hallo, wie geht es Ihnen?")

    def test_single_segment(self):
        stt = _import_stt_with_mocked_model()
        seg = MagicMock(text=" Ganz okay.")
        stt.stt_model.transcribe.return_value = ([seg], MagicMock())

        result = stt.transcribe(np.zeros(1000, dtype=np.float32), language="de")
        self.assertEqual(result, "Ganz okay.")

    def test_no_segments_returns_empty_string(self):
        stt = _import_stt_with_mocked_model()
        stt.stt_model.transcribe.return_value = ([], MagicMock())

        result = stt.transcribe(np.zeros(1000, dtype=np.float32), language="de")
        self.assertEqual(result, "")

    def test_language_is_passed_through(self):
        stt = _import_stt_with_mocked_model()
        stt.stt_model.transcribe.return_value = ([], MagicMock())

        audio = np.zeros(1000, dtype=np.float32)
        stt.transcribe(audio, language="en")

        _args, kwargs = stt.stt_model.transcribe.call_args
        self.assertEqual(kwargs["language"], "en")


if __name__ == "__main__":
    unittest.main()
