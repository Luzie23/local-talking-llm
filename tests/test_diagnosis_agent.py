
# EMSAG: 03.11.2026
# Test evidence aggregation, provenance validation, and active-item handling.

# EMSAG: 03.11.2026
# Import evidence models required by aggregation and validation tests.

from backend.agents.diagnosis_agent import (
    DiagnosisAgent,
    build_evidence_summaries,
    DiagnosisAnalysisDraft,
    EvidenceConflict,
    CriterionEvidenceSummary,
    EvidenceEntry,
    validate_analysis_draft,
)

import pytest

def make_session(
    session_id,
    source,
    response,
    evidence_quote,
    item_records=None,
    historical_evidence=None,
):
    return {
        "session_id": session_id,
        "source": source,
        "guide_id": "scid5pd",
        "guide_version": "1.0",
        "status": "completed",
        "participant_turns": [
            {
                "item_id": "Q1",
                "turn_number": 1,
                "question": "Question",
                "response": response,
                "recorded_at": None,
            }
        ],
        "judgment_assessments": [
            {
                "item_id": "Q1",
                "intent": "answer",
                "sufficiency": "sufficient",
                "provisional_rating": "2",
                "evidence_quote": evidence_quote,
                "historical_evidence": historical_evidence or [],
                "missing_information": ["frequency"],
                "source_turn_number": 1,
                "reasoning": "Test reasoning",
                "assessed_at": None,
            }
        ],
        "item_records": item_records or [],
    }


def test_active_item_uses_guide_metadata_without_item_record():
    session = make_session(
        session_id="current",
        source="current_session",
        response="Ich vermeide soziale Situationen.",
        evidence_quote="Ich vermeide soziale Situationen.",
    )

    guide_items = [
        {
            "id": "Q1",
            "criterion_code": "PD1",
        }
    ]

    summaries = build_evidence_summaries(
        session_evidence=[session],
        guide_items=guide_items,
    )

    assert len(summaries) == 1
    assert summaries[0].criterion_code == "PD1"
    assert len(summaries[0].supporting_evidence) == 1
    assert summaries[0].supporting_evidence[0].source == "current_session"


def test_historical_and_current_evidence_keep_provenance():
    current = make_session(
        session_id="current",
        source="current_session",
        response="Ich vermeide soziale Situationen.",
        evidence_quote="Ich vermeide soziale Situationen.",
    )

    historical = make_session(
        session_id="historical",
        source="historical_session",
        response="Ich gehe ungern auf Partys.",
        evidence_quote="Ich gehe ungern auf Partys.",
    )

    guide_items = [
        {
            "id": "Q1",
            "criterion_code": "PD1",
        }
    ]

    summaries = build_evidence_summaries(
        session_evidence=[current, historical],
        guide_items=guide_items,
    )

    entries = summaries[0].supporting_evidence

    assert len(entries) == 2
    assert {entry.source for entry in entries} == {
        "current_session",
        "historical_session",
    }
    assert {entry.session_id for entry in entries} == {
        "current",
        "historical",
    }


def test_unverified_evidence_quote_is_not_added():
    session = make_session(
        session_id="current",
        source="current_session",
        response="Ich bin gern mit Freunden zusammen.",
        evidence_quote="Ich vermeide soziale Situationen.",
    )

    summaries = build_evidence_summaries(
        session_evidence=[session],
        guide_items=[{"id": "Q1", "criterion_code": "PD1"}],
    )

    assert len(summaries) == 1
    assert summaries[0].supporting_evidence == []
    assert "frequency" in summaries[0].missing_information


# EMSAG: 03.11.2026
# Test that diagnosis recommendations cannot reference invalid items or fabricated conflicts.

import pytest


class FakeGuide:
    def is_finished(self):
        return False


def make_evidence():
    return [
        CriterionEvidenceSummary(
            criterion_code="PD1",
            supporting_evidence=[
                EvidenceEntry(
                    criterion_code="PD1",
                    item_id="Q1",
                    source="current_session",
                    session_id="S1",
                    evidence_quote="Ich vermeide soziale Situationen.",
                    turn_number=1,
                ),
                EvidenceEntry(
                    criterion_code="PD1",
                    item_id="Q2",
                    source="historical_session",
                    session_id="S2",
                    evidence_quote="Ich gehe gerne auf Partys.",
                    turn_number=2,
                ),
            ],
        )
    ]


def test_rejects_unknown_recommended_item():
    result = DiagnosisAnalysisDraft(
        recommendation="clarify_evidence",
        recommended_item_id="UNKNOWN",
        reasoning="Test",
    )

    with pytest.raises(ValueError, match="askable SCID item"):
        validate_analysis_draft(
            result,
            make_evidence(),
            FakeGuide(),
            [{"id": "Q1", "ask": True}],
        )


def test_rejects_conflict_from_different_criterion():
    result = DiagnosisAnalysisDraft(
        recommendation="review_conflicting_evidence",
        conflicts=[
            EvidenceConflict(
                criterion_code="PD2",
                first_session_id="S1",
                first_item_id="Q1",
                second_session_id="S2",
                second_item_id="Q2",
                explanation="Test conflict",
            )
        ],
        reasoning="Test",
    )

    with pytest.raises(ValueError, match="same criterion"):
        validate_analysis_draft(
            result,
            make_evidence(),
            FakeGuide(),
            [{"id": "Q1", "ask": True}],
        )


def test_rejects_fabricated_session_reference():
    result = DiagnosisAnalysisDraft(
        recommendation="review_conflicting_evidence",
        conflicts=[
            EvidenceConflict(
                criterion_code="PD1",
                first_session_id="S1",
                first_item_id="Q1",
                second_session_id="UNKNOWN",
                second_item_id="Q2",
                explanation="Test conflict",
            )
        ],
        reasoning="Test",
    )

    with pytest.raises(ValueError, match="same criterion"):
        validate_analysis_draft(
            result,
            make_evidence(),
            FakeGuide(),
            [{"id": "Q1", "ask": True}],
        )


def test_rejects_completion_before_guide_finishes():
    result = DiagnosisAnalysisDraft(
        recommendation="assessment_complete",
        reasoning="Test",
    )

    with pytest.raises(ValueError, match="not finished"):
        validate_analysis_draft(
            result,
            make_evidence(),
            FakeGuide(),
            [{"id": "Q1", "ask": True}],
        )


def test_accepts_valid_conflict_references():
    result = DiagnosisAnalysisDraft(
        recommendation="review_conflicting_evidence",
        conflicts=[
            EvidenceConflict(
                criterion_code="PD1",
                first_session_id="S1",
                first_item_id="Q1",
                second_session_id="S2",
                second_item_id="Q2",
                explanation="The statements appear inconsistent.",
            )
        ],
        reasoning="Requires further review.",
    )

    validated = validate_analysis_draft(
        result,
        make_evidence(),
        FakeGuide(),
        [{"id": "Q1", "ask": True}],
    )

    assert validated is result


# EMSAG: 03.11.2026
# Test active, unsaved participant turns during evidence collection.

class FakeDiagnosisRepository:

    def __init__(self, session_evidence):
        self.session_evidence = session_evidence

    def load_session_evidence(
        self,
        current_session_id,
        authorized_historical_session_ids=None,
    ):
        return self.session_evidence


class FakeEvidenceGuide:

    guide_id = "scid5pd"
    guide_version = "1.0"

    questions = [
        {
            "id": "Q1",
            "criterion_code": "PD1",
        }
    ]

def test_collect_evidence_does_not_duplicate_persisted_responses():

    session = make_session(
        session_id="current",
        source="current_session",
        response="Already saved response.",
        evidence_quote="Already saved response.",
    )

    session["participant_turns"][0]["turn_number"] = 3

    repository = FakeDiagnosisRepository([session])
    agent = DiagnosisAgent(repository=repository)

    summaries = agent.collect_evidence(
        current_session_id="current",
        guide=FakeEvidenceGuide(),
        current_item_id="Q1",
        current_item_turns=[
            {
                "prompt": "Question",
                "answer": "Already saved response.",
            },
            {
                "prompt": "Question",
                "answer": "New participant response.",
            },
        ],
    )

    turns = session["participant_turns"]

    responses = [
        turn["response"]
        for turn in turns
    ]

    assert len(turns) == 2

    assert responses.count("Already saved response.") == 1

    assert "New participant response." in responses

    assert turns[-1]["turn_number"] == 4

    assert len(summaries) == 1

def test_collect_evidence_skips_empty_responses_and_numbers_new_turns():

    session = make_session(
        session_id="current",
        source="current_session",
        response="First saved response.",
        evidence_quote="First saved response.",
    )

    session["participant_turns"][0]["turn_number"] = 2

    repository = FakeDiagnosisRepository([session])
    agent = DiagnosisAgent(repository=repository)

    agent.collect_evidence(
        current_session_id="current",
        guide=FakeEvidenceGuide(),
        current_item_id="Q1",
        current_item_turns=[
            {
                "prompt": "Question",
                "answer": "",
            },
            {
                "prompt": "Question",
                "answer": "   ",
            },
            {
                "prompt": "Question",
                "answer": "Second response.",
            },
            {
                "prompt": "Question",
                "answer": "Third response.",
            },
        ],
    )

    turns = session["participant_turns"]

    assert len(turns) == 3

    assert [
        turn["turn_number"]
        for turn in turns
    ] == [2, 3, 4]

    assert [
        turn["response"]
        for turn in turns
    ] == [
        "First saved response.",
        "Second response.",
        "Third response.",
    ]

# EMSAG: 03.11.2026
# Verify that previous clarification turns remain current-session evidence.

def test_build_evidence_summaries_preserves_previous_turns_from_same_session():

    session = make_session(
        session_id="current",
        source="current_session",
        response="Ich vermeide soziale Situationen inzwischen häufiger.",
        evidence_quote="Ich vermeide soziale Situationen inzwischen häufiger.",
        historical_evidence=[
            "Ich vermeide manchmal soziale Situationen."
        ],
    )

    # Add an earlier participant response from the same session.
    session["participant_turns"].insert(
        0,
        {
            "item_id": "Q1",
            "turn_number": 1,
            "question": "Original question",
            "response": "Ich vermeide manchmal soziale Situationen.",
            "recorded_at": None,
        },
    )

    # The latest response belongs to the second turn.
    session["participant_turns"][1]["turn_number"] = 2
    session["judgment_assessments"][0]["source_turn_number"] = 2

    summaries = build_evidence_summaries(
        session_evidence=[session],
        guide_items=[
            {
                "id": "Q1",
                "criterion_code": "PD1",
            }
        ],
    )

    assert len(summaries) == 1

    entries = summaries[0].supporting_evidence

    assert len(entries) == 2

    assert {
        entry.evidence_quote
        for entry in entries
    } == {
        "Ich vermeide manchmal soziale Situationen.",
        "Ich vermeide soziale Situationen inzwischen häufiger.",
    }

    assert all(
        entry.source == "current_session"
        for entry in entries
    )

    assert all(
        entry.session_id == "current"
        for entry in entries
    )

    assert {
        entry.turn_number
        for entry in entries
    } == {1, 2}