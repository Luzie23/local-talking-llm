"""Unit tests for the agent workflow: Navigation decision table, Judgment
validation rules, Question Agent output contract, guide state/transitions and
the deterministic skip check. All model calls use scripted fakes — these are
software-behaviour checks, not clinical validation."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from langchain_core.runnables import RunnableLambda

from agents.interview_orchestrator import is_explicit_skip, validate_decision
from agents.judgment_agent import JudgmentError, JudgmentResult, enforce_rules, judge_answer
from agents.navigation_agent import NavigationDecision, decide_next_action
from agents.question_agent import generate_clarification_question, validate_question
from interview_guide import GuideConfigError, INTERVIEWS_DIR, InterviewGuide

ITEM = {"id": "q1", "text": "Wie war Ihr Schlaf in den letzten zwei Wochen?",
        "criterion": "sleep last two weeks", "rating_scheme": "none"}
PD_ITEM = {"id": "PD1", "text": "Synthetic test item?", "criterion": "synthetic",
           "rating_scheme": "scid5pd", "rating_anchor": "2 = two examples"}


def write_guide(test, yaml_text):
    """Write a temporary guide into the interviews folder and return its file name."""
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=INTERVIEWS_DIR)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    test.addCleanup(os.remove, path)
    return os.path.basename(path)


def judgment(**kw):
    """Build a schema-valid JudgmentResult with sensible defaults, overridden by kw."""
    base = {"intent": "answer", "sufficiency": "sufficient", "missing_information": [],
            "provisional_rating": None, "evidence_quote": "", "reasoning": "r"}
    base.update(kw)
    return JudgmentResult.model_validate(base)


class TestNavigation(unittest.TestCase):
    """The decision table in navigation_agent.py, row by row."""

    def nav(self, intent, suff, used=0, budget=1):
        """Call the Navigation Agent for the current item (successor 'q2')."""
        return decide_next_action(intent, suff, used, budget, next_step="q2")

    def test_sufficient_answer_advances(self):
        """Sufficient answer advances."""
        d = self.nav("answer", "sufficient")
        self.assertEqual((d.action, d.next_step), ("advance", "q2"))

    def test_sufficient_answer_advances_even_without_budget(self):
        """Sufficient answer advances even without budget."""
        self.assertEqual(self.nav("answer", "sufficient", used=1).action, "advance")

    def test_ambiguous_gets_one_clarification_then_skips(self):
        """First ambiguous answer gets clarification; second skips."""
        first = self.nav("answer", "ambiguous", used=0)
        self.assertEqual((first.action, first.question_mode), ("clarify", "clarify"))

        second = self.nav("answer", "ambiguous", used=1)
        self.assertEqual((second.action, second.skip_reason), ("skip", "unresolved"))

    def test_refusal_skips_immediately_regardless_of_budget(self):
        """Refusal skips immediately regardless of budget."""
        for used in (0, 1):
            d = self.nav("refusal", "insufficient", used=used)
            self.assertEqual((d.action, d.skip_reason), ("skip", "refused"))

    def test_clarification_request_rephrases(self):
        """Clarification request rephrases."""
        d = self.nav("clarification_request", "insufficient")
        self.assertEqual((d.action, d.question_mode), ("rephrase", "rephrase"))

    def test_cannot_answer_gets_one_gentle_followup_then_skips(self):
        """First cannot-answer response gets one gentle follow-up; second skips."""
        first = self.nav("cannot_answer", "insufficient", used=0)
        self.assertEqual(
            (first.action, first.question_mode),
            ("clarify", "gentle"),
        )

        second = self.nav("cannot_answer", "insufficient", used=1)
        self.assertEqual(
            (second.action, second.skip_reason),
            ("skip", "cannot_answer"),
        )

    def test_cannot_answer_moves_on_when_source_requires_it(self):
        """Source instruction can require immediate skipping."""
        decision = decide_next_action(
            "cannot_answer",
            "insufficient",
            clarification_count=0,
            max_clarifications=1,
            next_step="q2",
            cannot_answer_moves_on=True,
        )
        self.assertEqual(
            (decision.action, decision.skip_reason),
            ("skip", "cannot_answer"),
        )

    def test_off_topic_and_unclear_redirect(self):
        """First off-topic/unclear response redirects; second skips."""
        for intent in ("off_topic", "unclear"):
            first = self.nav(intent, "insufficient", used=0)
            self.assertEqual(
                (first.action, first.question_mode),
                ("clarify", "redirect"),
            )

            second = self.nav(intent, "insufficient", used=1)
            self.assertEqual(
                (second.action, second.skip_reason),
                ("skip", "unresolved"),
            )

    # EMSAG: RT01: Numeric zero budget does not disable clarification.
    def test_zero_budget_still_allows_clarification(self):
        """Clarification is allowed even when the legacy numeric budget is zero."""
        self.assertEqual(
            self.nav("answer", "ambiguous", budget=0).action,
            "clarify",
        )

    def test_invalid_labels_raise(self):
        """Invalid labels raise."""
        with self.assertRaises(ValueError):
            self.nav("yes", "sufficient")
        with self.assertRaises(ValueError):
            self.nav("answer", "present")


class TestJudgmentRules(unittest.TestCase):
    """Deterministic post-validation of the model output."""

    def test_non_answer_cannot_be_sufficient(self):
        """Non answer cannot be sufficient."""
        r = enforce_rules(judgment(intent="refusal"), "none", ["nein danke"])
        self.assertEqual(r.sufficiency, "insufficient")
        self.assertTrue(r.validation_notes)

    def test_rating_removed_for_items_without_scheme(self):
        """Rating removed for items without scheme."""
        r = enforce_rules(judgment(provisional_rating="2"), "none", ["x"])
        self.assertIsNone(r.provisional_rating)

    def test_rating_kept_with_verbatim_evidence(self):
        """Rating kept with verbatim evidence."""
        r = enforce_rules(judgment(provisional_rating="2", evidence_quote="ich meide Gruppen"),
                          "scid5pd", ["Ja, ich meide Gruppen fast immer."])
        self.assertEqual(r.provisional_rating, "2")

    def test_rating_forced_to_unknown_when_not_sufficient(self):
        """Rating forced to unknown when not sufficient."""
        r = enforce_rules(judgment(sufficiency="ambiguous", provisional_rating="0",
                                   evidence_quote="manchmal"), "scid5pd", ["manchmal"])
        self.assertEqual(r.provisional_rating, "?")

    def test_invented_evidence_is_dropped_and_rating_unknown(self):
        """Invented evidence is dropped and rating unknown."""
        r = enforce_rules(judgment(provisional_rating="2", evidence_quote="I always avoid people"),
                          "scid5pd", ["Ich weiß nicht genau."])
        self.assertEqual((r.evidence_quote, r.provisional_rating), ("", "?"))

    def test_clear_negative_can_be_rated_absent(self):
        """Clear negative can be rated absent."""
        r = enforce_rules(judgment(provisional_rating="0", evidence_quote="Nein"), "scid5pd", ["Nein."])
        self.assertEqual(r.provisional_rating, "0")

     # EMSAG: 03.11.2026
    # Ensure current evidence cannot be taken from an earlier clarification turn.
    def test_evidence_quote_must_come_from_latest_turn(self):
        result = enforce_rules(
            judgment(
                provisional_rating="2",
                evidence_quote="Ich meide Gruppen",
            ),
            "scid5pd",
            [
                "Ich meide Gruppen fast immer.",
                "Ich weiß nicht genau.",
            ],
        )

        self.assertEqual(result.evidence_quote, "")
        self.assertEqual(result.provisional_rating, "?")

    # EMSAG: 03.11.2026
    # Ensure historical evidence is retained separately and validated against earlier turns.
    def test_historical_evidence_is_kept_separately(self):
        result = enforce_rules(
            judgment(
                evidence_quote="Ich weiß nicht genau.",
                historical_evidence=[
                    "Ich meide Gruppen fast immer.",
                    "Ich bin immer glücklich.",
                ],
            ),
            "none",
            [
                "Ich meide Gruppen fast immer.",
                "Ich weiß nicht genau.",
            ],
        )

        self.assertEqual(
            result.evidence_quote,
            "Ich weiß nicht genau.",
        )
        self.assertEqual(
            result.historical_evidence,
            ["Ich meide Gruppen fast immer."],
        )


class TestJudgmentAgentCalls(unittest.TestCase):
    """judge_answer() with scripted model replies."""

    def run_with(self, replies, item=ITEM):
        """Call judge_answer() with a fake model that returns the scripted replies in order; returns (result, prompts sent)."""
        calls = []

        def fake(prompt):
            """Scripted fake model used in this test."""
            calls.append(prompt.to_string())
            reply = replies[min(len(calls), len(replies)) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply
        result = judge_answer(RunnableLambda(fake), item, [{"prompt": item["text"], "answer": "Gut."}])
        return result, calls

    def test_parses_fenced_json(self):
        """Parses fenced json."""
        reply = "```json\n" + json.dumps({"intent": "answer", "sufficiency": "sufficient",
                                          "missing_information": [], "reasoning": "ok"}) + "\n```"
        result, calls = self.run_with([reply])
        self.assertEqual(result.intent, "answer")
        self.assertEqual(len(calls), 1)

    def test_retries_once_then_succeeds(self):
        """Retries once then succeeds."""
        good = json.dumps({"intent": "answer", "sufficiency": "sufficient", "reasoning": "ok"})
        result, calls = self.run_with(["not json at all", good])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.sufficiency, "sufficient")

    def test_disallowed_values_rejected(self):
        """Disallowed values rejected."""
        bad = json.dumps({"intent": "yes", "sufficiency": "sufficient", "reasoning": "ok"})
        with self.assertRaises(JudgmentError):
            self.run_with([bad, bad])

    def test_extra_fields_rejected(self):
        """Extra fields rejected."""
        bad = json.dumps({"intent": "answer", "sufficiency": "sufficient", "reasoning": "ok",
                          "diagnosis": "depression"})
        with self.assertRaises(JudgmentError):
            self.run_with([bad, bad])

    def test_provider_exception_becomes_judgment_error(self):
        """Provider exception becomes judgment error."""
        with self.assertRaises(JudgmentError):
            self.run_with([ConnectionError("ollama down")])

    def test_prompt_contains_item_context(self):
        """Prompt contains item context."""
        good = json.dumps({"intent": "answer", "sufficiency": "sufficient", "reasoning": "ok"})
        _, calls = self.run_with([good])
        self.assertIn("sleep last two weeks", calls[0])
        self.assertIn("<<<Gut.>>>", calls[0])


class TestQuestionAgent(unittest.TestCase):
    """Output contract of the Question Agent."""

    def test_rejects_verbatim_repeat(self):
        """Rejects verbatim repeat."""
        self.assertIsNone(validate_question(ITEM["text"], ITEM["text"]))

    def test_rejects_multiple_questions(self):
        """Rejects multiple questions."""
        self.assertIsNone(validate_question("Wie lange? Und wie oft?", ITEM["text"]))

    def test_strips_label_and_quotes(self):
        """Strips label and quotes."""
        self.assertEqual(validate_question('Frage: "Wie haben Sie geschlafen?"', ITEM["text"]),
                         "Wie haben Sie geschlafen?")

    def test_invalid_output_twice_returns_none(self):
        """Invalid output twice returns none."""
        llm = RunnableLambda(lambda p: "Keine Frage hier.")
        self.assertIsNone(generate_clarification_question(llm, ITEM, [], [], "clarify"))

    def test_mode_instruction_reaches_prompt(self):
        """Mode instruction reaches prompt."""
        seen = []

        def fake(p):
            """Scripted fake model used in this test."""
            seen.append(p.to_string())
            return "Wie haben Sie in den letzten zwei Wochen geschlafen?"
        out = generate_clarification_question(RunnableLambda(fake), ITEM, [], ["duration"], "rephrase")
        self.assertTrue(out.endswith("?"))
        self.assertIn("did not understand", seen[0])


class TestGuideState(unittest.TestCase):
    """Validation and transitions in InterviewGuide."""

    GUIDE = """
guide_id: g
language: de
sections:
  - questions:
      - {id: a, text: "A?", next_step: c}
      - {id: b, text: "B?"}
      - {id: c, text: "C?", max_clarifications: 0}
"""

    def test_next_step_jump_and_default_budget(self):
        """Next step jump and default budget."""
        g = InterviewGuide(write_guide(self, self.GUIDE))
        self.assertEqual(g.max_clarifications(), 1)  # project default
        self.assertEqual(g.next_item_id(), "c")
        self.assertEqual(g.finish_item({"x": 1}, "c")["id"], "c")
        self.assertEqual(g.max_clarifications(), 0)
        self.assertEqual(len(g.records), 1)

    # EMSAG: RT01: Guide accepts repeated clarification requests.
    def test_clarifications_are_unlimited(self):
        """Multiple clarification requests are accepted and counted."""
        g = InterviewGuide(write_guide(self, self.GUIDE))

        self.assertTrue(g.start_clarification("x?"))
        self.assertTrue(g.start_clarification("y?"))
        self.assertTrue(g.start_clarification("z?", kind="rephrase"))

        self.assertEqual(g.current_clarification_count(), 3)

    # EMSAG: RT01: Explicit item-level restriction must remain effective.
    def test_explicitly_disabled_clarification_is_rejected(self):
        """Items with allow_clarification false reject clarification."""
        guide_text = self.GUIDE.replace(
            '{id: a, text: "A?", next_step: c}',
            '{id: a, text: "A?", next_step: c, allow_clarification: false}',
        )
        g = InterviewGuide(write_guide(self, guide_text))

        self.assertFalse(g.start_clarification("Could you explain?"))
        self.assertEqual(g.current_clarification_count(), 0)

    def test_unknown_next_step_rejected(self):
        """Unknown next step rejected."""
        bad = self.GUIDE.replace("next_step: c", "next_step: zz")
        with self.assertRaises(GuideConfigError):
            InterviewGuide(write_guide(self, bad))

    def test_empty_text_rejected(self):
        """Empty text rejected."""
        with self.assertRaises(GuideConfigError):
            InterviewGuide(write_guide(self, self.GUIDE.replace('"A?"', '""')))

    def test_ask_false_items_need_no_text(self):
        """Ask false items need no text."""
        guide = self.GUIDE.replace('{id: b, text: "B?"}', '{id: b, ask: false, not_asked_reason: observation}')
        g = InterviewGuide(write_guide(self, guide))
        self.assertEqual(g.finish_item({"x": 1}, "b")["id"], "c")  # b is recorded and passed over
        self.assertEqual(g.records[-1]["status"], "not_asked")

    def test_active_example_guide_loads_with_budget_one(self):
        """Active example guide loads with budget one."""
        g = InterviewGuide("depression_scid_example.yaml")
        self.assertEqual([q["max_clarifications"] for q in g.questions], [1, 1, 1])
        self.assertIn("NOT SCID", g.content_source)

    # EMSAG: RT01: Orchestrator validation does not enforce a numeric limit.
    def test_validate_decision_keeps_clarify_after_repeated_requests(self):
        """Clarify decisions remain valid after repeated clarification requests."""
        g = InterviewGuide(write_guide(self, self.GUIDE))
        g.start_clarification("x?")
        g.start_clarification("y?")

        d = validate_decision(
            NavigationDecision(action="clarify", reason="r"),
            g,
        )

        self.assertEqual(d.action, "clarify")
    # EMSAG: RT01: Orchestrator must enforce explicit item-level restrictions.
    def test_validate_decision_rejects_clarification_when_disabled(self):
        """Orchestrator rejects clarification for explicitly disabled items."""
        guide_text = self.GUIDE.replace(
            '{id: a, text: "A?", next_step: c}',
            '{id: a, text: "A?", next_step: c, allow_clarification: false}',
        )
        g = InterviewGuide(write_guide(self, guide_text))

        d = validate_decision(
            NavigationDecision(action="clarify", reason="r"),
            g,
        )

        self.assertEqual((d.action, d.skip_reason), ("skip", "unresolved"))

class TestExplicitSkip(unittest.TestCase):
    """Deterministic recognition of explicit skip commands (works without the model)."""
    PHRASES = ["überspringen", "frage überspringen", "nächste frage", "skip"]

    def test_matches_explicit_commands(self):
        """Matches explicit commands."""
        for text in ("überspringen", "Bitte überspringen.", "Frage überspringen!", "skip please"):
            self.assertTrue(is_explicit_skip(text, self.PHRASES), text)

    def test_ordinary_answers_do_not_match(self):
        """Ordinary answers do not match."""
        for text in ("Ich schlafe schlecht", "Ich würde manche Tage gern überspringen, weil sie schwer sind"):
            self.assertFalse(is_explicit_skip(text, self.PHRASES), text)


if __name__ == "__main__":
    unittest.main()
