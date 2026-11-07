
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "backend"),
)

import llm_provider
from agents.judgment_agent import judge_answer


ITEM = {
    "id": "RT01_TEST",
    "text": "Wie würden Sie Ihre Stimmung in den letzten zwei Wochen beschreiben?",
    "criterion": "Beschreibung der Stimmung in den letzten zwei Wochen",
    "rating_scheme": "none",
}


CLARIFICATION_CASES = [
    "Ich verstehe die Frage nicht.",
    "Ich verstehe nicht ganz, was Sie meinen.",
    "Könnten Sie die Frage bitte anders formulieren?",
    "Was genau möchten Sie von mir wissen?",
    "Ich weiß nicht, wie ich diese Frage verstehen soll.",
]


OFF_TOPIC_CASES = [
    "Ich habe heute Hotdogs gegessen.",
    "Apropos, gestern war ich einkaufen.",
    "Das Wetter ist heute wirklich schön.",
    "Mein Nachbar hat mich gestern angerufen.",
    "Am Wochenende habe ich einen Film gesehen.",
]


@pytest.fixture(scope="module")
def real_llm():
    """Create the project's configured real LLM for intent evaluation."""
    return llm_provider.create_llm(
        provider=os.getenv("RT01_LLM_PROVIDER", "ollama"),
        model=os.getenv("RT01_LLM_MODEL", "gemma3"),
        temperature=0.0,
    )


@pytest.mark.parametrize("message", CLARIFICATION_CASES)
def test_german_clarification_variations(real_llm, message):
    result = judge_answer(
        llm=real_llm,
        item=ITEM,
        item_turns=[
            {"prompt": ITEM["text"], "answer": message}
        ],
    )

    assert result.intent == "clarification_request", (
        f"Message: {message!r}; expected clarification_request, "
        f"got {result.intent!r}"
    )


@pytest.mark.parametrize("message", OFF_TOPIC_CASES)
def test_german_off_topic_variations(real_llm, message):
    result = judge_answer(
        llm=real_llm,
        item=ITEM,
        item_turns=[
            {"prompt": ITEM["text"], "answer": message}
        ],
    )

    assert result.intent == "off_topic", (
        f"Message: {message!r}; expected off_topic, "
        f"got {result.intent!r}"
    )
