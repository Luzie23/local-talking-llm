"""Exercise the real Flask text routes without audio libraries or an LLM call."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


class TestTextPilotApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["app.py", "--web", "--text-only", "--pilot"]):
            cls.module = importlib.import_module("app")
        cls.client = cls.module.app.test_client()

    def setUp(self):
        self.client.get("/api/interview/start")

    def test_text_config_and_no_audio_imports(self):
        config = self.client.get("/api/runtime-config.js")
        self.assertIn(b'"textOnly": true', config.data)
        self.assertIn(b'"pilot": true', config.data)
        for module in ("stt", "tts", "speech"):
            self.assertNotIn(module, sys.modules)

    def test_text_end_to_end_and_reset(self):
        def judge(model, criterion, turns):
            turn = turns[-1]
            return {"status": "present", "evidence": [
                {"turn_id": turn["turn_id"], "quote": turn["answer"]}
            ]}, {"seconds": 0}
        with patch.object(self.module.pilot, "judge", judge):
            for i in range(3):
                response = self.client.post("/api/chat", json={"text": f"Fictional answer {i}"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["finished"], i == 2)
        report = self.client.get("/api/pilot/report").json
        self.assertTrue(report["finished"])
        self.assertEqual(len(report["findings"]), 3)
        self.assertEqual(len(report["events"]), 6)
        self.client.get("/api/interview/start")
        self.assertFalse(self.client.get("/api/pilot/report").json["finished"])

    def test_audio_routes_are_explicitly_disabled(self):
        self.assertEqual(self.client.get("/api/interview/start/audio").status_code, 503)
        self.assertEqual(self.client.post("/api/chat/audio").status_code, 503)


if __name__ == "__main__":
    unittest.main()
