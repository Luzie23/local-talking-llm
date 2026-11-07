import json
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import make_url

from backend.interview_guide import InterviewGuide
from backend.session_record import SessionRecorder, build_item_record

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from backend.agents.judgment_agent import JudgmentResult
from backend.agents.navigation_agent import NavigationDecision
from backend.agents.navigation_agent import NavigationDecision
from backend.database.models import (
    Participant,
    InterviewSession,
    InterviewItemRecord,
    ParticipantTurn,
    JudgmentAssessment,
    NavigationDecision as NavigationDecisionModel,
    SystemEvent,
)


@pytest.fixture
def db_factory(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    env = dotenv_values(project_root / ".env")

    original_url = env.get("DATABASE_URL")
    assert original_url, "DATABASE_URL is missing from .env"

    test_url = make_url(original_url).set(
        database="local_talking_llm_test"
    )

    engine = create_engine(test_url)

    with engine.connect() as connection:
        actual_database = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        assert actual_database == "local_talking_llm_test", (
            f"Unsafe database target: {actual_database}"
        )

        connection.rollback()
        outer_transaction = connection.begin()

        TestSessionLocal = sessionmaker(
            bind=connection,
            autoflush=False,
            autocommit=False,
            join_transaction_mode="create_savepoint",
        )

        monkeypatch.setattr(
            "backend.database.recorder_repository.SessionLocal",
            TestSessionLocal,
        )
        monkeypatch.setattr(
            "backend.database.diagnosis_repository.SessionLocal",
            TestSessionLocal,
        )

        try:
            yield TestSessionLocal
        finally:
            outer_transaction.rollback()
            engine.dispose()


def test_session_recorder_persists_closed_and_active_items(
    db_factory, tmp_path
):
    guide = InterviewGuide("depression_scid_example.yaml")

    recorder = SessionRecorder(
        guide=guide,
        directory=str(tmp_path),
        enabled=True,
        database_enabled=True,
    )

    assert recorder.session_id is not None
    assert recorder.participant_id is not None
    assert recorder.pseudonym.startswith("P-")

    # First item: record a participant answer.
    first_item = guide.current_question()
    assert first_item["id"] == "dep_01"

    guide.record_participant_turn(
        "In den letzten zwei Wochen war meine Stimmung oft gedrückt."
    )

    judgment = JudgmentResult(
        intent="answer",
        sufficiency="sufficient",
        evidence_quote="meine Stimmung oft gedrückt",
        reasoning="The participant describes their mood during the requested period.",
        source_turn_number=1,
    )

    proposed = NavigationDecision(
        action="advance",
        reason="The answer contains sufficient information.",
        next_step="dep_02",
    )

    validated = NavigationDecision(
        action="advance",
        reason="The proposed transition is valid.",
        next_step="dep_02",
    )

    recorder.record_judgment("dep_01", judgment)

    recorder.record_navigation(
        item_id="dep_01",
        proposed=proposed,
        validated=validated,
        executed_next_item_id="dep_02",
    )

    item_record = build_item_record(
        item=first_item,
        item_turns=guide.item_turns,
        status="answered",
        judgment=judgment,
        decision=validated,
    )

    guide.finish_item(item_record, "dep_02")

    # Second item: leave it active to test partial-session persistence.
    second_item = guide.current_question()
    assert second_item["id"] == "dep_02"

    guide.record_participant_turn(
        "Manchmal konnte ich mich über meine Hobbys nicht freuen."
    )

    recorder.log_system_event(
        item_id="dep_02",
        kind="integration_test",
        detail="Testing active item persistence.",
    )

    recorder.save()

    # Verify database persistence.
    with db_factory() as db:
        participant = db.get(Participant, recorder.participant_id)
        session = db.get(InterviewSession, recorder.session_id)

        assert participant is not None
        assert participant.pseudonym == recorder.pseudonym

        assert session is not None
        assert session.status == "in_progress"

        item = (
            db.query(InterviewItemRecord)
            .filter_by(session_id=recorder.session_id, item_id="dep_01")
            .one()
        )

        assert item.status == "answered"

        turns = (
            db.query(ParticipantTurn)
            .filter_by(session_id=recorder.session_id)
            .order_by(ParticipantTurn.item_id, ParticipantTurn.turn_number)
            .all()
        )

        assert len(turns) == 2
        assert {turn.item_id for turn in turns} == {"dep_01", "dep_02"}

        judgment_rows = (
            db.query(JudgmentAssessment)
            .filter_by(session_id=recorder.session_id)
            .all()
        )

        assert len(judgment_rows) == 1
        assert judgment_rows[0].item_record_id == item.id

        navigation_rows = (
            db.query(NavigationDecisionModel)
            .filter_by(session_id=recorder.session_id)
            .all()
        )

        assert len(navigation_rows) == 1
        assert navigation_rows[0].item_record_id == item.id
        assert navigation_rows[0].executed_next_item_id == "dep_02"

        events = (
            db.query(SystemEvent)
            .filter_by(session_id=recorder.session_id)
            .all()
        )

        assert len(events) == 1
        assert events[0].kind == "integration_test"

    # Verify JSON backup.
    json_path = Path(recorder.path)
    assert json_path.exists()

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_json["finished"] is False
    assert len(saved_json["items"]) == 1
    assert saved_json["items"][0]["item_id"] == "dep_01"

def test_session_recorder_persists_paused_status(db_factory, tmp_path):
    guide = InterviewGuide("depression_scid_example.yaml")

    recorder = SessionRecorder(
        guide=guide,
        directory=str(tmp_path),
        enabled=False,
        database_enabled=True,
    )

    guide.record_participant_turn("I would like to pause here.")
    guide.paused = True

    recorder.save()

    with db_factory() as db:
        session = db.get(InterviewSession, recorder.session_id)

        assert session is not None
        assert session.status == "paused"
        assert session.ended_at is None


def test_session_recorder_persists_completed_status(db_factory, tmp_path):
    guide = InterviewGuide("depression_scid_example.yaml")

    recorder = SessionRecorder(
        guide=guide,
        directory=str(tmp_path),
        enabled=False,
        database_enabled=True,
    )

    # Complete all items using the guide's configured transitions.
    while not guide.is_finished():
        item = guide.current_question()
        next_item_id = guide.next_item_id()

        record = build_item_record(
            item=item,
            item_turns=[],
            status="skipped",
            skip_reason="integration test",
        )

        guide.finish_item(record, next_item_id)

    assert guide.is_finished()

    recorder.save()

    with db_factory() as db:
        session = db.get(InterviewSession, recorder.session_id)

        assert session is not None
        assert session.status == "completed"
        assert session.ended_at is not None

def test_session_recorder_persists_historical_retrieval_consent(
    db_factory, tmp_path
):
    # EMSAG: 04.11.2026
    # Verify that explicit historical retrieval consent is stored in PostgreSQL.
    guide = InterviewGuide("depression_scid_example.yaml")

    recorder = SessionRecorder(
        guide=guide,
        directory=str(tmp_path),
        enabled=True,
        database_enabled=True,
    )

    recorder.start(historical_retrieval_consent=True)

    with db_factory() as db:
        saved_session = db.get(InterviewSession, recorder.session_id)

        assert saved_session is not None
        assert saved_session.historical_retrieval_consent is True