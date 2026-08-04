"""Unit tests for backend/safety_check.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import safety_check


class TestSafetyCheck(unittest.TestCase):
    """
    Tests against the real study_content/safety_keywords.yaml shipped with
    the project (not a mock file) — if a researcher edits that file and
    removes all keywords, these tests will correctly start failing, which
    is the point: it should stay obvious if the safety net goes empty.
    """

    def test_german_keyword_matches(self):
        self.assertTrue(safety_check.check_for_safety_concern("Ich habe an Suizid gedacht.", "de"))

    def test_german_no_match_on_unrelated_text(self):
        self.assertFalse(safety_check.check_for_safety_concern("Mir geht es eigentlich ganz gut.", "de"))

    def test_english_keyword_matches(self):
        self.assertTrue(safety_check.check_for_safety_concern("Sometimes I want to kill myself.", "en"))

    def test_english_no_match_on_unrelated_text(self):
        self.assertFalse(safety_check.check_for_safety_concern("I feel pretty good these days.", "en"))

    def test_case_insensitive(self):
        self.assertTrue(safety_check.check_for_safety_concern("SUIZID", "de"))
        self.assertTrue(safety_check.check_for_safety_concern("SuIzId", "de"))

    def test_unknown_language_falls_back_to_all_languages(self):
        # No "fr" section exists in safety_keywords.yaml -> falls back to
        # searching across every configured language rather than silently
        # matching nothing.
        self.assertTrue(safety_check.check_for_safety_concern("I think about suicide sometimes.", "fr"))

    def test_empty_text_never_matches(self):
        self.assertFalse(safety_check.check_for_safety_concern("", "de"))
        self.assertFalse(safety_check.check_for_safety_concern(None, "de"))

    def test_only_checks_given_text_not_global_state(self):
        # Sanity check: calling with a clean text after a flagged one
        # doesn't leave any hidden state around.
        self.assertTrue(safety_check.check_for_safety_concern("suizid", "de"))
        self.assertFalse(safety_check.check_for_safety_concern("ein ganz normaler Tag", "de"))


if __name__ == "__main__":
    unittest.main()
