"""Check evidence provenance, uncertain navigation, reset, and failed-call recovery."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from interview_guide import InterviewGuide
from pilot import PilotEngine, validate_evidence


class TestEvidencePilot(unittest.TestCase):
    def test_invented_quote_does_not_support_a_positive_finding(self):
        finding, errors = validate_evidence(
            {"status": "present", "evidence": [{"turn_id": 1, "quote": "not in the answer"}]},
            [{"turn_id": 1, "answer": "I do not know."}]
        )
        self.assertEqual(finding["status"], "unknown")
        self.assertIn("unverifiable_quote", errors)

    def test_wrong_turn_cannot_borrow_another_turns_quote(self):
        finding, errors = validate_evidence(
            {"status": "absent", "evidence": [{"turn_id": 2, "quote": "I slept well."}]},
            [{"turn_id": 1, "answer": "I slept well."}, {"turn_id": 2, "answer": "Not sure."}]
        )
        self.assertEqual(finding["status"], "unknown")
        self.assertTrue(errors)

    def test_missing_evidence_does_not_become_absence(self):
        finding, errors = validate_evidence({"status": "absent", "evidence": []}, [])
        self.assertEqual(finding["status"], "unknown")
        self.assertIn("insufficient_cited_evidence", errors)

    def test_conflicting_requires_distinct_quotes(self):
        quote = {"turn_id": 1, "quote": "I slept well."}
        finding, errors = validate_evidence(
            {"status": "conflicting", "evidence": [quote, quote]},
            [{"turn_id": 1, "answer": "I slept well."}]
        )
        self.assertEqual(finding["status"], "unknown")

    def test_ambiguity_is_capped_and_retained_in_report(self):
        def judge(*args):
            return {"status": "unknown", "evidence": []}, {"seconds": 0}
        guide = InterviewGuide("depression_scid_example.yaml")
        engine = PilotEngine(guide, judge=judge)
        engine.answer("I don't know.")
        self.assertEqual(guide.current_question()["id"], "dep_01")
        engine.answer("I still don't know.")
        self.assertEqual(guide.current_question()["id"], "dep_02")
        self.assertEqual(engine.report()["findings"]["dep_01"]["status"], "unknown")
        self.assertEqual(engine.report()["unassessed_nodes"], ["dep_02", "sleep_01"])
        self.assertEqual([e["action"] for e in engine.events if "action" in e], ["clarify", "advance"])
        engine.reset()
        self.assertEqual(guide.current_question()["id"], "dep_01")
        self.assertEqual(engine.report()["findings"], {})
        self.assertEqual(engine.report()["turns"], [])

    def test_failed_model_call_does_not_advance_or_duplicate_answer(self):
        def judge(*args):
            raise TimeoutError("Model timeout")
        guide = InterviewGuide("depression_scid_example.yaml")
        engine = PilotEngine(guide, judge=judge)
        with self.assertRaises(TimeoutError):
            engine.answer("Example answer")
        self.assertEqual(guide.current_question()["id"], "dep_01")
        self.assertEqual(engine.turns, [])
        self.assertEqual(engine.attempts["dep_01"], 0)


if __name__ == "__main__":
    unittest.main()
