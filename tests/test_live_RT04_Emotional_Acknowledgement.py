
import os
import sys
from unittest.mock import patch

import pytest

# Make backend modules importable.
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

# Reuse the deterministic interview test setup.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as app_module
import llm_provider
from test_interview_flow import FlowTestBase, SCRIPT


@pytest.fixture(scope="module")
def real_llm():
    """Use the configured real LLM, defaulting to local Gemma3."""
    return llm_provider.create_llm(
        provider=os.getenv("RT01_LLM_PROVIDER", "ollama"),
        model=os.getenv("RT01_LLM_MODEL", "gemma3"),
        temperature=0.0,
    )


EMOTIONAL_CASES = [
    (
        "Ich fühle mich sehr allein und habe das Gefühl, "
        "dass sich niemand für mich interessiert.",
        "loneliness",
    ),
    (
        "Seit dem Tod eines Menschen, der mir sehr nahestand, "
        "bin ich sehr traurig.",
        "bereavement",
    ),
    (
        "Ich fühle mich wertlos und habe kaum noch Hoffnung.",
        "hopelessness",
    ),
]


def is_generic_thanks_only(text):
    """Detect an acknowledgement consisting only of generic thanks."""
    normalized = " ".join(
        (text or "").casefold().split()
    ).strip(" .!,:;")

    generic_responses = {
        "danke",
        "danke für ihre antwort",
        "vielen dank",
        "vielen dank für ihre antwort",
    }

    return normalized in generic_responses


def is_mechanical_transition(text):
    """Detect known generic-thanks transitions."""
    normalized = " ".join((text or "").casefold().split())

    mechanical_phrases = (
        "danke. wir gehen zur nächsten frage weiter",
        "danke für ihre antwort. wir gehen zur nächsten frage weiter",
        "vielen dank. wir gehen zur nächsten frage weiter",
        "vielen dank für ihre antwort. wir gehen zur nächsten frage weiter",
    )

    return any(
        phrase in normalized
        for phrase in mechanical_phrases
    )


def assert_empathetic_response(test_case, visible_text, scenario):
    """Shared quality checks for emotional acknowledgement."""
    normalized = " ".join((visible_text or "").casefold().split())

    test_case.assertTrue(
        visible_text and visible_text.strip(),
        f"{scenario}: the visible response must not be empty.",
    )

    test_case.assertFalse(
        is_mechanical_transition(normalized),
        f"{scenario}: a mechanical transition was shown: "
        f"{visible_text!r}",
    )

    test_case.assertNotIn(
        "danke für ihre antwort",
        normalized,
        f"{scenario}: a generic acknowledgement was shown: "
        f"{visible_text!r}",
    )


# -------------------------------------------------------------------
# RT04a: Test acknowledgement quality directly with the real LLM.
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    ("answer", "scenario"),
    EMOTIONAL_CASES,
    ids=["loneliness", "bereavement", "hopelessness"],
)
def test_emotional_answer_gets_human_acknowledgement(
    real_llm, monkeypatch, answer, scenario
):
    """RT04a: Emotional answers should not receive thanks alone."""
    monkeypatch.setattr(
        app_module,
        "chain_with_history",
        llm_provider.build_chain(real_llm),
    )

    prompt_text = (
        "Wie würden Sie Ihre Stimmung in den letzten zwei Wochen "
        "beschreiben?"
    )

    response = app_module._acknowledge(prompt_text, answer)
    acknowledgement = (response or "").strip()

    print(f"\nRT04a | Scenario: {scenario}")
    print(f"RT04a | Answer: {answer}")
    print(f"RT04a | Acknowledgement: {acknowledgement!r}")

    assert acknowledgement, (
        f"RT04a ({scenario}): acknowledgement is empty."
    )

    assert not is_generic_thanks_only(acknowledgement), (
        f"RT04a ({scenario}): the model returned only generic thanks: "
        f"{acknowledgement!r}"
    )


# -------------------------------------------------------------------
# RT04b: Test the actual /api/chat flow with the real LLM.
# -------------------------------------------------------------------

class TestEmotionalAnswerFullFlow(FlowTestBase):

    def test_emotional_answer_is_acknowledged_before_next_question(self):
        """RT04b: A sufficient emotional answer gets an acknowledgement."""
        answer = (
            "Ich fühle mich sehr allein und habe das Gefühl, "
            "dass sich niemand für mich interessiert."
        )

        script_patch = pytest.MonkeyPatch()
        script_patch.setitem(
            SCRIPT,
            answer,
            ("answer", "sufficient", [], answer, None),
        )
        self.addCleanup(script_patch.undo)

        # Use the real LLM for acknowledgement generation.
        real_llm = llm_provider.create_llm(
            provider=os.getenv("RT01_LLM_PROVIDER", "ollama"),
            model=os.getenv("RT01_LLM_MODEL", "gemma3"),
            temperature=0.0,
        )
        app_module.chain_with_history = llm_provider.build_chain(real_llm)

        response = self.say(answer)
        visible_text = (response.get("response") or "").strip()
        normalized = " ".join(visible_text.casefold().split())

        print("\nRT04b | Scenario: loneliness")
        print(f"RT04b | Answer: {answer}")
        print(f"RT04b | Full response: {visible_text!r}")

        self.assertTrue(
            visible_text,
            "RT04b: the visible response must not be empty.",
        )

        self.assertEqual(
            self.current(),
            "dep_02",
            "RT04b: the interview should advance to dep_02.",
        )

        self.assertFalse(
            is_mechanical_transition(normalized),
            "RT04b: generic thanks followed by a mechanical transition: "
            f"{visible_text!r}",
        )

        self.assertNotIn(
            "danke für ihre antwort",
            normalized,
            "RT04b: the response contains a generic acknowledgement: "
            f"{visible_text!r}",
        )

    # ---------------------------------------------------------------
    # RT04c: Detect generic thanks in the normal advance flow.
    # ---------------------------------------------------------------

    def test_generic_acknowledgement_is_detected_in_full_flow(self):
        """
        RT04c: Generic 'Danke.' must not be passed through as the
        acknowledgement for an emotional answer.
        """
        answer = (
            "Ich fühle mich einsam, sogar wenn ich mit anderen "
            "Menschen zusammen bin."
        )

        script_patch = pytest.MonkeyPatch()
        script_patch.setitem(
            SCRIPT,
            answer,
            ("answer", "sufficient", [], answer, None),
        )
        self.addCleanup(script_patch.undo)

        with patch.object(
            app_module.llm_provider,
            "get_feedback_and_followup",
            return_value=("Danke.", None),
        ):
            response = self.say(answer)

        visible_text = (response.get("response") or "").strip()
        normalized = " ".join(visible_text.casefold().split())

        print("\nRT04c | Forced acknowledgement: 'Danke.'")
        print(f"RT04c | Full response: {visible_text!r}")

        self.assertTrue(
            visible_text,
            "RT04c: the visible response must not be empty.",
        )

        self.assertFalse(
            is_mechanical_transition(normalized),
            "RT04c: a mechanical transition was shown: "
            f"{visible_text!r}",
        )

        self.assertNotIn(
            "danke.",
            normalized,
            "RT04c: generic thanks was passed through to the user.",
        )

        self.assertEqual(
            self.current(),
            "dep_02",
            "RT04c: the interview should advance to dep_02.",
        )

    # ---------------------------------------------------------------
    # RT04d: Clarification budget exhausted with an ambiguous answer.
    # ---------------------------------------------------------------

    def test_ambiguous_emotional_answer_after_clarification_limit(self):
        """
        RT04d: If an emotional answer remains ambiguous after the
        clarification budget is exhausted, acknowledge the emotion
        empathetically and continue to the next interview question.

        This test intentionally fails until the application implements
        the required behaviour.
        """
        first_answer = (
            "Ich weiß es nicht, ich fühle mich so fern von denen."
        )
        emotional_answer = (
            "Ich fühle mich einsam, sogar wenn ich mit denen bin."
        )
        empathetic_feedback = (
            "Es klingt belastend, sich selbst in Gesellschaft "
            "einsam zu fühlen."
        )

        script_patch = pytest.MonkeyPatch()
        script_patch.setitem(
            SCRIPT,
            first_answer,
            (
                "answer",
                "ambiguous",
                ["Specific feelings about the relationships"],
                first_answer,
                None,
            ),
        )
        script_patch.setitem(
            SCRIPT,
            emotional_answer,
            (
                "answer",
                "ambiguous",
                ["Specific feelings about the relationships"],
                emotional_answer,
                None,
            ),
        )
        self.addCleanup(script_patch.undo)

        # First answer should consume the clarification opportunity.
        first_response = self.say(first_answer)
        first_text = (
            first_response.get("response") or ""
        ).strip()

        self.assertTrue(
            first_text,
            "RT04d: the first response must not be empty.",
        )

        # Force a deterministic empathetic acknowledgement for the
        # second answer. The application should use it even if it
        # decides to skip the unresolved item.
        with patch.object(
            app_module.llm_provider,
            "get_feedback_and_followup",
            return_value=(empathetic_feedback, None),
        ) as acknowledgement_mock:
            response = self.say(emotional_answer)

        visible_text = (response.get("response") or "").strip()
        normalized = " ".join(visible_text.casefold().split())

        print("\nRT04d | Scenario: ambiguous emotional answer after limit")
        print(f"RT04d | First answer response: {first_text!r}")
        print(f"RT04d | Final response: {visible_text!r}")

        self.assertTrue(
            visible_text,
            "RT04d: the final response must not be empty.",
        )

        self.assertIn(
            empathetic_feedback.casefold(),
            normalized,
            "RT04d: the configured empathetic acknowledgement must "
            "reach the user after the clarification limit.",
        )

        self.assertFalse(
            is_mechanical_transition(normalized),
            "RT04d: the mechanical unresolved-skip message must not "
            "be shown.",
        )

        self.assertNotIn(
            "danke.",
            normalized,
            "RT04d: generic thanks must not replace empathy.",
        )

        acknowledgement_mock.assert_called_once()

        self.assertEqual(
            self.current(),
            "dep_02",
            "RT04d: the interview must continue to dep_02.",
        )
