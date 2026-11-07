import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import agents.interview_orchestrator as orchestrator
from agents.judgment_agent import JudgmentResult
from interview_guide import InterviewGuide


def prepare_test(monkeypatch):
    guide = InterviewGuide("depression_scid_example.yaml")

    recorder = MagicMock()
    recorder.session_id = "current-session-123"

    judgment = JudgmentResult(
        intent="answer",
        sufficiency="sufficient",
        reasoning="The participant provided a sufficient answer.",
        evidence_quote="Gut.",
    )

    monkeypatch.setattr(
        orchestrator,
        "judge_answer",
        lambda llm, item, turns: judgment,
    )

    diagnosis_agent = MagicMock()
    diagnosis_agent.collect_evidence.return_value = []

    monkeypatch.setattr(
        orchestrator,
        "DiagnosisAgent",
        lambda: diagnosis_agent,
    )

    diagnosis_draft = SimpleNamespace(
        model_dump=lambda: {
            "recommendation": "continue_assessment",
            "recommended_item_id": None,
            "reasoning": "Test diagnosis result.",
        }
    )

    monkeypatch.setattr(
        orchestrator,
        "analyze_evidence",
        lambda **kwargs: diagnosis_draft,
    )

    return guide, recorder, diagnosis_agent


def test_authorized_historical_session_ids_are_forwarded(monkeypatch):
    guide, recorder, diagnosis_agent = prepare_test(monkeypatch)

    authorized_ids = ["historical-session-1", "historical-session-2"]

    orchestrator.handle_turn(
        llm=MagicMock(),
        guide=guide,
        recorder=recorder,
        answer_text="Gut.",
        authorized_historical_session_ids=authorized_ids,
    )

    diagnosis_agent.collect_evidence.assert_called_once()

    call_kwargs = diagnosis_agent.collect_evidence.call_args.kwargs

    assert call_kwargs["current_session_id"] == "current-session-123"
    assert call_kwargs["authorized_historical_session_ids"] == authorized_ids


def test_missing_historical_session_ids_default_to_empty_list(monkeypatch):
    guide, recorder, diagnosis_agent = prepare_test(monkeypatch)

    orchestrator.handle_turn(
        llm=MagicMock(),
        guide=guide,
        recorder=recorder,
        answer_text="Gut.",
    )

    call_kwargs = diagnosis_agent.collect_evidence.call_args.kwargs

    assert call_kwargs["authorized_historical_session_ids"] == []


def test_diagnosis_failure_does_not_break_interview(monkeypatch):
    guide, recorder, diagnosis_agent = prepare_test(monkeypatch)

    diagnosis_agent.collect_evidence.side_effect = RuntimeError(
        "Simulated database failure"
    )

    original_item_id = guide.current_question()["id"]

    outcome = orchestrator.handle_turn(
        llm=MagicMock(),
        guide=guide,
        recorder=recorder,
        answer_text="Gut.",
    )

    assert outcome.trace["diagnosis"] is None
    assert "Simulated database failure" in outcome.trace["diagnosis_error"]

    assert outcome.trace["executed"]["action"] == "advance"
    assert guide.current_question()["id"] != original_item_id


def test_diagnosis_recommendation_does_not_override_scid_navigation(monkeypatch):
    guide, recorder, diagnosis_agent = prepare_test(monkeypatch)

    diagnosis_draft = SimpleNamespace(
        model_dump=lambda: {
            "recommendation": "assessment_complete",
            "recommended_item_id": None,
            "reasoning": "Diagnosis Agent suggests completing the assessment.",
        }
    )

    monkeypatch.setattr(
        orchestrator,
        "analyze_evidence",
        lambda **kwargs: diagnosis_draft,
    )

    original_item_id = guide.current_question()["id"]

    outcome = orchestrator.handle_turn(
        llm=MagicMock(),
        guide=guide,
        recorder=recorder,
        answer_text="Gut.",
    )

    assert outcome.trace["diagnosis"]["recommendation"] == "assessment_complete"

    # Navigation still follows the Judgment Agent and InterviewGuide rules.
    assert outcome.trace["executed"]["action"] == "advance"
    assert guide.current_question()["id"] != original_item_id
    assert guide.current_question() is not None

def test_diagnosis_recommendation_is_serialized_into_trace(monkeypatch):
    guide, recorder, diagnosis_agent = prepare_test(monkeypatch)

    outcome = orchestrator.handle_turn(
        llm=MagicMock(),
        guide=guide,
        recorder=recorder,
        answer_text="Gut.",
    )

    assert isinstance(outcome.trace["diagnosis"], dict)
    assert outcome.trace["diagnosis"]["recommendation"] == "continue_assessment"