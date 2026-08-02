"""Unit tests for backend/speech.py (audio format conversion helpers only —
recording/playback need real audio hardware and aren't covered here)."""

import base64
import io
import os
import sys
import tempfile
import unittest
import wave
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# speech.py needs a microphone/speaker library (sounddevice) which may not be
# installed in a plain test environment, so a stand-in is registered here,
# once, before speech.py is imported (numpy's C extension does not support
# being re-imported repeatedly within the same process).
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()
import speech  # noqa: E402


def _make_wav_bytes(sample_rate: int, samples: np.ndarray, channels: int = 1) -> bytes:
    """Build raw .wav bytes from int16 samples, for feeding into decode_uploaded_wav()."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


class TestDecodeUploadedWav(unittest.TestCase):
    def test_decodes_mono_16khz_to_float_array(self):
        samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        wav_bytes = _make_wav_bytes(16000, samples)
        audio_np = speech.decode_uploaded_wav(wav_bytes)
        self.assertEqual(len(audio_np), 4)
        # Values should be roughly in [-1, 1], matching the int16 proportions.
        self.assertAlmostEqual(audio_np[0], 0.0, places=3)
        self.assertGreater(audio_np[1], 0)
        self.assertLess(audio_np[2], 0)

    def test_mixes_stereo_down_to_mono(self):
        # Left channel loud, right channel silent -> average should be non-zero but reduced.
        left = np.full(100, 32767, dtype=np.int16)
        right = np.zeros(100, dtype=np.int16)
        interleaved = np.empty(200, dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right
        wav_bytes = _make_wav_bytes(16000, interleaved, channels=2)
        audio_np = speech.decode_uploaded_wav(wav_bytes)
        self.assertEqual(len(audio_np), 100)
        self.assertTrue(np.all(audio_np > 0))
        self.assertTrue(np.all(audio_np < 1.0))

    def test_resamples_to_16khz(self):
        # Recorded at 48kHz -> should come back resampled to a shorter array.
        samples = np.zeros(4800, dtype=np.int16)  # 0.1s at 48kHz
        wav_bytes = _make_wav_bytes(48000, samples)
        audio_np = speech.decode_uploaded_wav(wav_bytes)
        # 0.1s at 16kHz should be roughly 1600 samples.
        self.assertLess(len(audio_np), 4800)
        self.assertAlmostEqual(len(audio_np), 1600, delta=50)


class TestEncodeWavBase64(unittest.TestCase):
    def test_round_trips_through_wav_format(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        encoded = speech.encode_wav_base64(22050, audio)
        raw = base64.b64decode(encoded)

        with wave.open(io.BytesIO(raw), "rb") as wf:
            self.assertEqual(wf.getframerate(), 22050)
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)

    def test_clips_out_of_range_values(self):
        # Values beyond [-1, 1] shouldn't wrap around / overflow int16.
        audio = np.array([2.0, -2.0], dtype=np.float32)
        encoded = speech.encode_wav_base64(16000, audio)
        raw = base64.b64decode(encoded)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
        self.assertEqual(samples[0], 32767)
        self.assertEqual(samples[1], -32767)


class TestWriteWavFile(unittest.TestCase):
    def test_writes_a_valid_wav_file(self):
        audio = np.array([0.0, 0.25, -0.25], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "out.wav")
            speech.write_wav_file(path, 22050, audio)
            self.assertTrue(os.path.exists(path))
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getframerate(), 22050)
                self.assertEqual(wf.getnframes(), 3)


if __name__ == "__main__":
    unittest.main()
