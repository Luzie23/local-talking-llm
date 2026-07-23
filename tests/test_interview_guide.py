"""Unit tests for backend/interview_guide.py."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from interview_guide import InterviewGuide

TEST_GUIDE_YAML = """
guide_id: test_guide
title: "Test Guide"
behavior_profile: strict_clinical
framing_message: "Welcome message."
closing_message: "Closing message."
sections:
  - section_title: "Section A"
    questions:
      - id: q1
        text: "Question one?"
      - id: q2
        text: "Question two?"
  - section_title: "Section B"
    questions:
      - id: q3
        text: "Question three?"
"""

EMPTY_GUIDE_YAML = """
guide_id: empty_guide
title: "Empty Guide"
framing_message: "Hello."
closing_message: "Bye."
sections: []
"""


class TestInterviewGuide(unittest.TestCase):
    """Tests for the InterviewGuide class."""

    def _write_guide(self, yaml_text: str) -> str:
        """Write a guide YAML string into a temp file inside interview_guide's
        expected directory and return just the filename (as InterviewGuide expects)."""
        target_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "study_content", "interviews",
        )
        fd, path = tempfile.mkstemp(suffix=".yaml", dir=target_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        self.addCleanup(os.remove, path)
        return os.path.basename(path)

    def test_loads_metadata(self):
        filename = self._write_guide(TEST_GUIDE_YAML)
        guide = InterviewGuide(filename)
        self.assertEqual(guide.guide_id, "test_guide")
        self.assertEqual(guide.behavior_profile, "strict_clinical")
        self.assertEqual(guide.framing_message, "Welcome message.")
        self.assertEqual(guide.closing_message, "Closing message.")

    def test_flattens_questions_across_sections_in_order(self):
        filename = self._write_guide(TEST_GUIDE_YAML)
        guide = InterviewGuide(filename)
        self.assertEqual([q["id"] for q in guide.questions], ["q1", "q2", "q3"])
        self.assertEqual(guide.questions[0]["section_title"], "Section A")
        self.assertEqual(guide.questions[2]["section_title"], "Section B")

    def test_walks_through_questions_in_sequence(self):
        filename = self._write_guide(TEST_GUIDE_YAML)
        guide = InterviewGuide(filename)

        self.assertEqual(guide.current_question()["id"], "q1")
        self.assertFalse(guide.is_finished())

        self.assertEqual(guide.advance()["id"], "q2")
        self.assertEqual(guide.advance()["id"], "q3")
        self.assertFalse(guide.is_finished())

        self.assertIsNone(guide.advance())
        self.assertTrue(guide.is_finished())
        self.assertIsNone(guide.current_question())

    def test_reset_starts_over(self):
        filename = self._write_guide(TEST_GUIDE_YAML)
        guide = InterviewGuide(filename)
        guide.advance()
        guide.advance()
        guide.reset()
        self.assertEqual(guide.current_question()["id"], "q1")
        self.assertFalse(guide.is_finished())

    def test_empty_guide_has_no_questions(self):
        filename = self._write_guide(EMPTY_GUIDE_YAML)
        guide = InterviewGuide(filename)
        self.assertEqual(guide.questions, [])
        self.assertIsNone(guide.current_question())
        self.assertTrue(guide.is_finished())


if __name__ == "__main__":
    unittest.main()
