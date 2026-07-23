"""Unit tests for backend/behavior_profiles.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from behavior_profiles import load_behavior_profile, DEFAULT_PROFILE_NAME


class TestBehaviorProfiles(unittest.TestCase):
    """Tests against the real profiles.yaml / .txt files shipped with the project."""

    def test_strict_clinical_is_verbatim(self):
        profile = load_behavior_profile("strict_clinical")
        self.assertEqual(profile.question_delivery, "verbatim")
        self.assertIn("FEEDBACK:", profile.instructions)
        self.assertIn("FOLLOWUP:", profile.instructions)

    def test_flexible_conversational_is_natural_transition(self):
        profile = load_behavior_profile("flexible_conversational")
        self.assertEqual(profile.question_delivery, "natural_transition")

    def test_comment_lines_are_stripped(self):
        profile = load_behavior_profile("strict_clinical")
        for line in profile.instructions.splitlines():
            self.assertFalse(line.strip().startswith("#"))

    def test_none_falls_back_to_default(self):
        profile = load_behavior_profile(None)
        self.assertEqual(profile.name, DEFAULT_PROFILE_NAME)

    def test_unknown_profile_raises_key_error(self):
        with self.assertRaises(KeyError):
            load_behavior_profile("does_not_exist")


if __name__ == "__main__":
    unittest.main()
