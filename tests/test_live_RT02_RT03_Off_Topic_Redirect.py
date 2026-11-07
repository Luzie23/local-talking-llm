import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "backend"),
)

import llm_provider
from agents.question_agent import (
    generate_clarification_question,
    validate_question,
)


# Synthetic test item based on OV1 from the development guide.
# This item is intentionally defined locally and is not loaded
# automatically from the active interview guide.
ITEM = {
    "id": "OV1",
    "text": "Wie würden Sie sich selbst als Person beschreiben?",
    "criterion": (
        "Open overview question (not rated): how the participant "
        "describes themselves as a person - how they generally "
        "feel or behave. Any self-description is sufficient."
    ),
    "rating_scheme": "none",
    "probes": [],
}

OFF_TOPIC_ANSWERS = [
    "Ich habe heute Hotdogs gegessen.",
    "Das Wetter ist heute wirklich schön.",
    "Am Wochenende habe ich einen Film gesehen.",
]


@pytest.fixture(scope="module")
def real_llm():
    """Use the configured real LLM, defaulting to local Gemma3."""
    return llm_provider.create_llm(
        provider=os.getenv("RT01_LLM_PROVIDER", "ollama"),
        model=os.getenv("RT01_LLM_MODEL", "gemma3"),
        temperature=0.0,
    )


# ---------------------------------------------------------------------
# RT02: An off-topic response must trigger a natural redirect.
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "off_topic_answer",
    OFF_TOPIC_ANSWERS,
    ids=["hotdogs", "weather", "film"],
)
def test_off_topic_redirect_is_a_genuine_reformulation(
    real_llm, off_topic_answer
):
    """
    Verify that the Question Agent naturally returns to the original
    interview topic after an unrelated participant response.

    A brief acknowledgement of the unrelated response is allowed.
    The follow-up must return to self-description without mechanically
    repeating the original question.
    """

    # Arrange: prepare the original item and unrelated answer.
    item_turns = [
        {
            "prompt": ITEM["text"],
            "answer": off_topic_answer,
        }
    ]

    # Act: generate a redirect using the real LLM.
    result = generate_clarification_question(
        llm=real_llm,
        item=ITEM,
        item_turns=item_turns,
        missing_information=[
            "description of the participant as a person"
        ],
        mode="redirect",
        language="German",
    )

    # Display the actual response for manual review.
    print("\n" + "=" * 70)
    print(f"RT02 | Participant answer: {off_topic_answer}")
    print(f"RT02 | Generated redirect: {result!r}")
    print("=" * 70)

    # Assertion 1: a redirect must be generated.
    assert result is not None and result.strip(), (
        "RT02 failed: the Question Agent returned no redirect."
    )

    # Assertion 2: the response must pass the existing validator.
    assert validate_question(result, ITEM["text"]) is not None, (
        "RT02 failed: the generated response is invalid. "
        f"Response: {result!r}"
    )

    normalized_result = " ".join(result.casefold().split())
    normalized_original = " ".join(ITEM["text"].casefold().split())

    # Assertion 3: reject mechanical redirect prefixes.
    forbidden_prefixes = (
        "anders gefragt:",
        "anders formuliert:",
        "ich frage anders:",
        "noch einmal anders:",
    )

    assert not normalized_result.startswith(forbidden_prefixes), (
        "RT02 failed: a mechanical redirect prefix was used. "
        f"Response: {result!r}"
    )

    # Assertion 4: the original question must not be repeated unchanged.
    assert normalized_result != normalized_original, (
        "RT02 failed: the original question was repeated unchanged."
    )

    # Assertion 5: the response must return to the original topic.
    # These are indicative phrases, not a complete semantic test.
    relevant_terms = (
        "person",
        "eigenschaft",
        "beschreib",
        "charakter",
        "persönlichkeit",
        "persoenlichkeit",
        "selbst sehen",
        "selbst wahrnehmen",
        "wie sie sich sehen",
        "als mensch",
        "wie sie sich selbst",
    )

    assert any(
        term in normalized_result for term in relevant_terms
    ), (
        "RT02 failed: the response does not appear to return to "
        "the self-description topic. "
        f"Response: {result!r}"
    )

    # Assertion 6: require one clear question at the end.
    assert (
        result.count("?") == 1
        and result.rstrip().endswith("?")
    ), (
        "RT02 failed: expected exactly one question ending with '?'. "
        f"Response: {result!r}"
    )

    # Assertion 7: reject known examples of off-topic follow-up questions.
    # Mentioning the unrelated subject in a brief acknowledgement
    # is allowed; continuing that subject with a follow-up is not.
    unrelated_follow_up_phrases = (
        "mehr über den film erzählen",
        "über den film erzählen",
        "wie sie sich während dieses wochenendes gefühlt",
        "wie sie sich bei diesem film gefühlt",
        "wie das wetter",
        "was sie gegessen haben",
        "welche hotdogs",
    )

    assert not any(
        phrase in normalized_result
        for phrase in unrelated_follow_up_phrases
    ), (
        "RT02 failed: the follow-up appears to continue the unrelated "
        "topic instead of returning to self-description. "
        f"Response: {result!r}"
    )


# ---------------------------------------------------------------------
# RT03: If redirect generation fails, use a safe fallback.
# ---------------------------------------------------------------------

def test_invalid_redirect_uses_safe_fallback():
    """
    RT03: If the Question Agent fails, the application must return
    a non-empty response, avoid the mechanical 'Anders gefragt:'
    fallback, and remain on the current interview item.
    """

    from tests.test_interview_flow import FlowTestBase

    # Create the existing test harness and initialise a fresh interview.
    test_case = FlowTestBase()
    test_case.setUp()

    try:
        # Patch the reference used by the orchestrator.
        # Returning None simulates a Question Agent failure.
        with patch(
            "agents.interview_orchestrator.generate_clarification_question",
            return_value=None,
        ) as mock_generate:
            response = test_case.say(
                "Ich habe Hotdogs gegessen."
            )

        visible_text = str(response.get("response", ""))

        print("\n" + "=" * 70)
        print("RT03 | Participant answer: Ich habe Hotdogs gegessen.")
        print(f"RT03 | Fallback response: {visible_text!r}")
        print(f"RT03 | Current item: {test_case.current()!r}")
        print(f"RT03 | Question Agent mock calls: {mock_generate.call_count}")
        print("=" * 70)

        # Assertion 1: confirm that the intended failure was simulated.
        assert mock_generate.called, (
            "RT03 setup failed: the Question Agent mock was not called. "
            "The test did not reach the redirect-generation path."
        )

        # Assertion 2: the application must return a non-empty response.
        assert visible_text.strip(), (
            "RT03 failed: the application returned an empty response."
        )

        # Assertion 3: the fallback must not repeat the unrelated topic.
        assert "hotdogs" not in visible_text.casefold(), (
            "RT03 failed: the fallback continued the unrelated topic. "
            f"Response: {visible_text!r}"
        )

        # Assertion 4: the fallback must not use a mechanical prefix.
        assert "anders gefragt:" not in visible_text.casefold(), (
            "RT03 failed: the fallback uses the mechanical prefix "
            "'Anders gefragt:'. "
            f"Response: {visible_text!r}"
        )

        # Assertion 5: a failed redirect must not silently advance the
        # interview to the next item.
        assert test_case.current() == "dep_01", (
            "RT03 failed: the interview advanced despite the failed "
            "redirect. "
            f"Current item: {test_case.current()!r}"
        )

    finally:
        # Run temporary-directory cleanup registered by setUp().
        test_case.doCleanups()