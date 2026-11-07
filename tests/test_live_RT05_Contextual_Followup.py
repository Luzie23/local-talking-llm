"""RT05 — Regression tests: Follow-up and next-question exclusivity.

Objective:
    The chatbot must not ask a follow-up question and advance to the next
    interview item in the same response.

Test approach:
    - Exercise the real Flask /api/chat endpoint.
    - Use the existing ScriptedModel to make tests deterministic.
    - Inspect visible responses, active item IDs, agent prompts and records.
    - Print detailed diagnostics for every scenario.

Important:
    These are deterministic application-flow tests. They do not validate
    the classification quality of the real Gemma3 model or clinical validity.

Run:
    python -m pytest -v -s tests/test_live_RT05_Contextual_Followup.py
"""

import os
import sys
from unittest.mock import patch

# Make backend and test modules importable.
BACKEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend")
)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from test_interview_flow import (
    FlowTestBase,
    SCRIPT,
    QUESTION_AGENT_REPLIES,
    Q1,
    Q2,
    Q3,
)


FOLLOWUP = QUESTION_AGENT_REPLIES["missing detail"]

# These strings indicate that the bot has started transitioning to another item.
NEXT_QUESTION_MARKERS = (
    "wir gehen zur nächsten frage weiter",
    "gab es zeiten, in denen sie sich über nichts mehr freuen konnten",
    "wie war ihr schlaf in den letzten zwei wochen",
    "wie erfolgreich sind sie ihrer meinung nach darin",
)


def normalize(text):
    """Normalize whitespace and case for robust visible-response checks."""
    return " ".join((text or "").casefold().split())


def print_report(
    test_name,
    participant_input,
    response,
    current_before,
    current_after,
    records,
    question_prompt_count,
    judgment_prompt_count,
):
    """Print a detailed diagnostic report for manual review."""
    print("\n" + "=" * 90)
    print(f"RT05 | TEST: {test_name}")
    print(f"RT05 | PARTICIPANT INPUT: {participant_input!r}")
    print(f"RT05 | BOT RESPONSE: {response!r}")
    print(f"RT05 | CURRENT ITEM BEFORE: {current_before}")
    print(f"RT05 | CURRENT ITEM AFTER:  {current_after}")
    print(f"RT05 | QUESTION AGENT CALLS: {question_prompt_count}")
    print(f"RT05 | JUDGMENT AGENT CALLS: {judgment_prompt_count}")
    print(f"RT05 | INTERVIEW RECORDS: {records!r}")
    print("=" * 90)


def assert_followup_only(test_case, response, next_question):
    """Assert that a follow-up response does not also advance the interview."""
    visible = normalize(response)

    test_case.assertTrue(
        visible,
        "RT05 FAIL: the bot returned an empty response.",
    )

    test_case.assertNotIn(
        normalize(next_question),
        visible,
        "RT05 FAIL: the response contains the next interview question "
        "even though a follow-up was expected.\n"
        f"Actual response: {response!r}",
    )

    for marker in NEXT_QUESTION_MARKERS:
        test_case.assertNotIn(
            marker,
            visible,
            "RT05 FAIL: the follow-up response also contains a transition "
            "or another interview question.\n"
            f"Matched marker: {marker!r}\n"
            f"Actual response: {response!r}",
        )


class TestLiveRT05FollowupNextQuestionExclusivity(FlowTestBase):
    """RT05 regression tests through the browser-facing Flask endpoint."""

    def test_01_ambiguous_answer_returns_followup_only(self):
        """
        Ambiguous answer:
        - ask a focused follow-up;
        - remain on dep_01;
        - do not display Q2;
        - do not finalize the item yet.
        """
        answer = "Manchmal komisch."
        current_before = self.current()

        response_json = self.say(answer)
        response = response_json.get("response", "")
        current_after = self.current()

        print_report(
            "RT05-01 | Ambiguous answer -> follow-up only",
            answer,
            response,
            current_before,
            current_after,
            self.records(),
            len(self.model.question_prompts),
            len(self.model.judgment_prompts),
        )

        assert current_before == "dep_01"
        assert current_after == "dep_01", (
            "RT05 FAIL: the active item changed after the bot asked "
            "a follow-up question."
        )

        assert response == FOLLOWUP, (
            "RT05 FAIL: the response should contain only the expected "
            "follow-up question.\n"
            f"Expected: {FOLLOWUP!r}\n"
            f"Actual:   {response!r}"
        )

        assert_followup_only(self, response, Q2)

        assert len(self.model.question_prompts) == 1, (
            "RT05 FAIL: the Question Agent should be called exactly once "
            "for this ambiguous answer."
        )

        assert len(self.records()) == 0, (
            "RT05 FAIL: the current item was finalized before the "
            "participant answered the follow-up."
        )

    def test_02_followup_answer_advances_only_on_the_next_turn(self):
        """
        First turn: ambiguous answer -> follow-up only.
        Second turn: sufficient clarification answer -> advance to Q2.
        Both participant answers must be preserved in the same item record.
        """
        first_answer = "Manchmal komisch."
        second_answer = "Ich meine eher müde und gereizt."

        first_response = self.say(first_answer)["response"]
        current_after_first = self.current()

        print_report(
            "RT05-02A | Initial ambiguous answer",
            first_answer,
            first_response,
            "dep_01",
            current_after_first,
            self.records(),
            len(self.model.question_prompts),
            len(self.model.judgment_prompts),
        )

        assert first_response == FOLLOWUP
        assert current_after_first == "dep_01"
        assert_followup_only(self, first_response, Q2)

        second_response = self.say(second_answer)["response"]
        current_after_second = self.current()

        print_report(
            "RT05-02B | Follow-up answered",
            second_answer,
            second_response,
            "dep_01",
            current_after_second,
            self.records(),
            len(self.model.question_prompts),
            len(self.model.judgment_prompts),
        )

        assert Q2 in second_response, (
            "RT05 FAIL: after the clarification answer, the next question "
            "was not displayed."
        )

        assert current_after_second == "dep_02"

        record = self.records()[0]
        assert record["status"] == "answered"
        assert record["clarifications_used"] == 1
        assert len(record["turns"]) == 2

        recorded_answers = [turn["answer"] for turn in record["turns"]]
        assert recorded_answers == [first_answer, second_answer], (
            "RT05 FAIL: the original answer and clarification answer "
            "were not preserved in the same item record."
        )

    def test_03_sufficient_first_answer_advances_without_followup(self):
        """
        Sufficient answer:
        - advance to Q2;
        - do not call the Question Agent;
        - do not return the clarification prompt.
        """
        answer = "Sehr schlecht, ich bin fast jeden Tag niedergeschlagen."
        current_before = self.current()

        response = self.say(answer)["response"]
        current_after = self.current()

        print_report(
            "RT05-03 | Sufficient answer -> next question",
            answer,
            response,
            current_before,
            current_after,
            self.records(),
            len(self.model.question_prompts),
            len(self.model.judgment_prompts),
        )

        assert Q2 in response
        assert current_after == "dep_02"

        assert len(self.model.question_prompts) == 0, (
            "RT05 FAIL: the Question Agent was called even though the "
            "answer was classified as sufficient."
        )

        assert normalize(FOLLOWUP) not in normalize(response), (
            "RT05 FAIL: an unnecessary follow-up was included in the "
            "response to a sufficient answer."
        )

        assert len(self.records()) == 1
        assert self.records()[0]["status"] == "answered"

    def test_04_unresolved_answer_after_budget_does_not_loop(self):
        """
        First ambiguous answer -> one follow-up.
        Second unresolved answer -> skip the item and move on.
        The chatbot must not generate a second follow-up after the budget
        has been exhausted.
        """
        answer = "Manchmal komisch."

        first_response = self.say(answer)["response"]
        current_after_first = self.current()

        assert first_response == FOLLOWUP
        assert current_after_first == "dep_01"

        second_response = self.say(answer)["response"]
        current_after_second = self.current()

        print_report(
            "RT05-04 | Clarification budget exhausted",
            f"First answer: {answer!r}; second answer: {answer!r}",
            f"First response: {first_response!r}; "
            f"second response: {second_response!r}",
            "dep_01",
            current_after_second,
            self.records(),
            len(self.model.question_prompts),
            len(self.model.judgment_prompts),
        )

        assert current_after_second == "dep_02", (
            "RT05 FAIL: the interview did not move on after the "
            "clarification budget was exhausted."
        )

        assert Q2 in second_response
        assert second_response != FOLLOWUP, (
            "RT05 FAIL: the bot repeated the follow-up after its "
            "clarification budget had been exhausted."
        )

        record = self.records()[0]
        assert record["status"] == "skipped"
        assert record["skip_reason"] == "unresolved"
        assert record["clarifications_used"] == 1
        assert len(record["turns"]) == 2

    def test_05_core_response_contract_is_mutually_exclusive(self):
        """
        Verify both sides of the response contract:
        - ambiguous -> follow-up only, stay on current item;
        - sufficient -> next question, with no follow-up.
        """
        cases = [
            {
                "name": "ambiguous",
                "answer": "Manchmal komisch.",
                "expected_response": FOLLOWUP,
                "expected_item": "dep_01",
            },
            {
                "name": "sufficient",
                "answer": "Nein.",
                "expected_response": Q2,
                "expected_item": "dep_02",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                # Keep each scenario independent by resetting the test setup.
                # setUp() creates a fresh Flask client, model and interview.
                self.tearDown()
                self.setUp()

                current_before = self.current()
                response = self.say(case["answer"])["response"]
                current_after = self.current()

                print_report(
                    f"RT05-05 | Contract table: {case['name']}",
                    case["answer"],
                    response,
                    current_before,
                    current_after,
                    self.records(),
                    len(self.model.question_prompts),
                    len(self.model.judgment_prompts),
                )

                assert current_after == case["expected_item"]

                if case["name"] == "ambiguous":
                    assert response == case["expected_response"]
                    assert_followup_only(self, response, Q2)
                else:
                    assert Q2 in response
                    assert normalize(FOLLOWUP) not in normalize(response)


    def test_06_contextual_followup_does_not_include_next_scid_question(self):
        """
        A contextual follow-up about family conflicts must not be combined
        with the next SCID question in the same response.
        """
        answer = (
            "Ich habe oft Konflikte mit meiner Familie, "
            "und das belastet mich."
        )
        contextual_followup = (
            "Was passiert normalerweise bei diesen Konflikten?"
        )

        current_before = self.current()

        # Make the scripted Question Agent ask a topic-specific question.
        with patch.dict(
            QUESTION_AGENT_REPLIES,
            {"missing detail": contextual_followup},
        ):
            response = self.say(answer)["response"]

        current_after = self.current()

        print_report(
            "RT05-06 | Contextual follow-up without next SCID question",
            answer,
            response,
            current_before,
            current_after,
            self.records(),
            len(self.model.question_prompts),
            len(self.model.judgment_prompts),
        )

        assert current_before == "dep_01"
        assert response == contextual_followup, (
            "RT05 FAIL: the chatbot should return only the contextual "
            "follow-up question.\n"
            f"Expected: {contextual_followup!r}\n"
            f"Actual: {response!r}"
        )

        assert_followup_only(self, response, Q2)

        assert current_after == "dep_01", (
            "RT05 FAIL: the interview advanced before the participant "
            "answered the contextual follow-up."
        )

        assert len(self.model.question_prompts) == 1
        assert len(self.records()) == 0, (
            "RT05 FAIL: the item was finalized before the follow-up "
            "was answered."
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(
            [
                "-v",
                "-s",
                os.path.abspath(__file__),
            ]
        )
    )