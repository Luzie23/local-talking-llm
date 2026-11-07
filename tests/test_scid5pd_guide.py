"""Tests for the SCID-5-PD guide (study_content/interviews/scid5pd_de.yaml) and
the source-defined flow features: items the chatbot does not ask, derived
ratings, the Antisocial PD Criterion C gate and "IF CAN'T ANSWER, MOVE ON".

A scripted fake model is used (software-behaviour tests, not clinical
validation): the participant answer "JA2" is judged a sufficient answer with
provisional rating 2, "NEIN" a sufficient answer with rating 0 and
"WEISSNICHT" an inability to answer.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from langchain_core.runnables import RunnableLambda

from agents.interview_orchestrator import handle_turn
from interview_guide import InterviewGuide
from session_record import SessionRecorder

CONDUCT_CODES = [f"PD{n}" for n in range(86, 101)]


def fake_model(prompt):
    """Scripted model: 'JA2' -> sufficient answer rated 2, 'NEIN' -> rated 0, 'WEISSNICHT' -> cannot answer."""
    text = prompt.to_string()
    if "You are the Judgment Agent" in text:
        answer = re.findall(r"<<<(.*?)>>>", text, re.S)[-1]
        if answer == "WEISSNICHT":
            out = {"intent": "cannot_answer", "sufficiency": "insufficient", "reasoning": "x"}
        else:
            rated = "?" if "Rating anchor" not in text and "SCID-5-PD coding" not in text else None
            out = {"intent": "answer", "sufficiency": "sufficient", "reasoning": "x",
                   "evidence_quote": answer,
                   "provisional_rating": rated or ("2" if answer == "JA2" else "0")}
        return json.dumps(out)
    if "You are the Question Agent" in text:
        return "Können Sie das etwas genauer sagen?"
    return "FEEDBACK: Danke.\nFOLLOWUP: NONE"


class ScidGuideTest(unittest.TestCase):
    """Content checks and source-defined flow rules of the SKID-5-PD guide."""
    def setUp(self):
        """Prepare a fresh interview, scripted model and temporary record location for each test."""
        self.guide = InterviewGuide("scid5pd_de.yaml")
        self.recorder = SessionRecorder(self.guide, enabled=False)
        self.llm = RunnableLambda(fake_model)

    def run_interview(self, answers: dict, default="NEIN", stop_at=None):
        """Answer every asked item (answers by item id, else default) until the end or stop_at."""
        turns = 0
        while not self.guide.is_finished():
            item = self.guide.current_question()
            if item["id"] == stop_at:
                return
            handle_turn(self.llm, self.guide, self.recorder, answers.get(item["id"], default))
            turns += 1
            self.assertLess(turns, 400, "interview did not terminate")

    def record(self, item_id):
        """The record of one item by id."""
        return next(r for r in self.guide.records if r["item_id"] == item_id)

    # --- content ------------------------------------------------------------
    def test_all_106_numbered_scid_questions_present_in_order(self):
        """All 106 numbered scid questions present in order."""
        ids = [q["id"] for q in self.guide.questions if re.fullmatch(r"Q\d+", q["id"])]
        self.assertEqual(ids, [f"Q{n}" for n in range(1, 107)])

    def test_marked_as_unofficial_translation(self):
        """Marked as unofficial translation."""
        self.assertIn("UNOFFICIAL", self.guide.content_source)

    def test_interview_starts_with_overview_and_source_intro(self):
        """Interview starts with overview and source intro."""
        self.assertEqual(self.guide.current_question()["id"], "OV1")
        self.assertTrue(self.guide.presentation_text().startswith("Ich möchte Ihnen jetzt"))

    def test_overview_items_are_not_rated(self):
        """Overview items are not rated."""
        self.assertEqual(self.guide.questions[0]["rating_scheme"], "none")
        self.assertEqual(self.guide.questions[9]["rating_scheme"], "scid5pd")  # Q1

    # --- flow -----------------------------------------------------------------
    def test_cant_answer_moves_on_for_overview_question(self):
        """Cant answer moves on for overview question."""
        handle_turn(self.llm, self.guide, self.recorder, "WEISSNICHT")
        rec = self.record("OV1")
        self.assertEqual((rec["status"], rec["skip_reason"], rec["clarifications_used"]),
                         ("skipped", "cannot_answer", 0))
        self.assertEqual(self.guide.current_question()["id"], "OV2")

    def test_cant_answer_on_rated_item_gets_one_followup(self):
        """Cant answer on rated item gets one followup."""
        self.run_interview({}, stop_at="Q1")
        handle_turn(self.llm, self.guide, self.recorder, "WEISSNICHT")
        self.assertEqual(self.guide.current_question()["id"], "Q1")
        self.assertEqual(self.guide.current_clarification_count(), 1)

    def test_not_asked_items_are_recorded_for_the_clinician(self):
        """Not asked items are recorded for the clinician."""
        self.run_interview({}, stop_at="Q44")
        par_b = self.record("PAR_B")
        self.assertEqual((par_b["status"], par_b["not_asked_reason"]), ("not_asked", "clinician"))
        self.assertEqual(par_b["provisional_rating"]["value"], "?")
        self.assertEqual(self.record("SZT_A4")["not_asked_reason"], "observation")

    def test_derived_schizotypal_a5_from_paranoid_criteria(self):
        """Derived schizotypal a5 from paranoid criteria."""
        self.run_interview({"Q26": "JA2"}, stop_at="Q44")  # Paranoid A2 rated 2
        rec = self.record("SZT_A5")
        self.assertEqual(rec["status"], "derived")
        self.assertEqual(rec["provisional_rating"]["value"], "2")
        self.assertEqual(rec["provisional_rating"]["rated_2"], ["PD28"])

    def test_derived_stays_unknown_without_paranoid_rating_2(self):
        """Derived stays unknown without paranoid rating 2."""
        self.run_interview({}, stop_at="Q44")
        self.assertEqual(self.record("SZT_A5")["provisional_rating"]["value"], "?")

    def test_schizoid_a5_reuses_q44_when_already_rated(self):
        """Schizoid A5 reuses the Q44 assessment when already rated."""
        self.run_interview({"Q44": "JA2"}, stop_at="Q50")

        rec = self.record("SZD_A5")

        self.assertEqual(rec["status"], "reused")
        self.assertEqual(rec["same_as"], "Q44")
        self.assertEqual(rec["provisional_rating"]["value"], "2")


    def test_schizoid_a5_is_asked_when_q44_was_skipped(self):
        """Schizoid A5 is asked when Q44 was not assessed."""
        self.run_interview({"Q44": "überspringen"}, stop_at="SZD_A5")

        current = self.guide.current_question()

        self.assertEqual(current["id"], "SZD_A5")
        self.assertTrue(current["ask"])
        self.assertTrue(current["text"])

        # EMSAG: 02.11.2026
        # Verify that Q44's question content is reused while SZD_A5
        # retains its own SCID criterion identity.
        source = next(q for q in self.guide.questions if q["id"] == "Q44")

        self.assertEqual(current["text"], source["text"])
        self.assertEqual(current["probes"], source["probes"])
        self.assertEqual(current["rating_anchor"], source["rating_anchor"])

        self.assertEqual(current["criterion_id"], "SZD_A5")
        self.assertEqual(current["criterion_code"], "PD51")
        self.assertEqual(
            current["criterion"],
            "Lacks close friends or confidants other than first-degree relatives."
        )

    # EMSAG: 02.11.2026
    # Verify that temporary SCID question modifications are reverted
    # when a new participant session starts.
    def test_reset_restores_original_schizoid_a5_configuration(self):
        """Reset restores the original SCID configuration for a new participant."""
        self.run_interview({"Q44": "überspringen"}, stop_at="SZD_A5")

        # SZD_A5 should temporarily become an active question.
        item = next(q for q in self.guide.questions if q["id"] == "SZD_A5")
        self.assertTrue(item["ask"])

        # Start a new participant's interview.
        self.guide.reset()

        # The original SCID configuration must be restored.
        item = next(q for q in self.guide.questions if q["id"] == "SZD_A5")
        self.assertFalse(item["ask"])
        self.assertEqual(item["text"], "")
        self.assertEqual(item["intro"], "")
        self.assertEqual(item["rating_anchor"], "")
        self.assertEqual(item["probes"], [])

        self.assertEqual(self.guide.current_question()["id"], "OV1")

    def test_suicidality_items_flagged(self):
        """Suicidality items flagged."""
        self.run_interview({}, stop_at="Q86")
        self.assertTrue(self.record("Q84").get("safety_relevant"))
        self.assertTrue(self.record("Q85").get("safety_relevant"))

    def test_antisocial_gate_met_continues_with_adult_criteria(self):
        """Antisocial gate met continues with adult criteria."""
        self.run_interview({"Q92": "JA2", "Q93": "JA2"}, stop_at="ASPD_A2a")
        gate = self.record("ASPD_C_GATE")
        self.assertTrue(gate["gate_met"])
        self.assertEqual(gate["criteria_rated_2"], ["PD86", "PD87"])
        self.assertEqual(self.record("ASPD_A1")["status"], "answered")

    def test_antisocial_gate_not_met_goes_to_other_specified_and_ends(self):
        """Antisocial gate not met goes to other specified and ends."""
        self.run_interview({"Q92": "JA2"})
        gate = self.record("ASPD_C_GATE")
        self.assertFalse(gate["gate_met"])
        self.assertEqual(gate["routed_to"], "OSPD")
        ids = [r["item_id"] for r in self.guide.records]
        self.assertNotIn("ASPD_A1", ids)  # adult criteria not asked (source routing)
        self.assertEqual(ids[-1], "OSPD")
        self.assertTrue(self.guide.is_finished())

    def test_skipped_conduct_items_do_not_count_towards_the_gate(self):
        """Skipped conduct items do not count towards the gate."""
        self.run_interview({"Q92": "JA2", "Q93": "überspringen"})
        self.assertFalse(self.record("ASPD_C_GATE")["gate_met"])

    def test_full_interview_with_gate_met_ends_after_other_specified(self):
        """Full interview with gate met ends after other specified."""
        self.run_interview({"Q95": "JA2", "Q106": "JA2"})
        ids = [r["item_id"] for r in self.guide.records]
        self.assertIn("ASPD_A7", ids)
        self.assertEqual(ids[-1], "OSPD")
        # every item of the guide appears exactly once in the record
        self.assertEqual(sorted(ids), sorted(q["id"] for q in self.guide.questions))
        # no diagnoses or totals are produced (no such fields in any record)
        keys = {k for r in self.recorder.to_dict()["items"] for k in r}
        self.assertFalse([k for k in keys if "diagnos" in k or "total" in k or "score" in k])


if __name__ == "__main__":
    unittest.main()
