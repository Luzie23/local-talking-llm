# EMSAG: 03.11.2026
# Test DiagnosisEvidenceRepository against an isolated PostgreSQL database with rollback isolation.

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from backend.database.models import (
    Participant,
    InterviewSession,
    ParticipantTurn,
    JudgmentAssessment,
    InterviewItemRecord,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATABASE_NAME = "local_talking_llm_test"


@pytest.fixture
def db_factory(monkeypatch):
    # EMSAG: 03.11.2026
    # Derive the test URL from .env but never connect to the development database.
    env_values = dotenv_values(PROJECT_ROOT / ".env")
    original_url = env_values.get("DATABASE_URL")

    if not original_url:
        raise RuntimeError("DATABASE_URL is missing from the project .env file.")

    test_url = make_url(original_url).set(database=TEST_DATABASE_NAME)

    if test_url.database != TEST_DATABASE_NAME:
        raise RuntimeError("Refusing to run tests against a non-test database.")

    engine = create_engine(test_url, pool_pre_ping=True)
    connection = engine.connect()
    outer_transaction = connection.begin()

    try:
        # EMSAG: 03.11.2026
        # Verify the actual PostgreSQL database before allowing any test operation.
        actual_database = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        if actual_database != TEST_DATABASE_NAME:
            raise RuntimeError(
                f"Unsafe test database connection: {actual_database}"
            )

        TestSessionLocal = sessionmaker(
            bind=connection,
            autoflush=False,
            autocommit=False,
            join_transaction_mode="create_savepoint",
        )

        # EMSAG: 03.11.2026
        # Patch the SessionLocal reference used directly by the repository.
        monkeypatch.setattr(
            "backend.database.diagnosis_repository.SessionLocal",
            TestSessionLocal,
        )

        yield TestSessionLocal

    finally:
        # EMSAG: 03.11.2026
        # Roll back all test data without touching existing database records.
        if outer_transaction.is_active:
            outer_transaction.rollback()

        connection.close()
        engine.dispose()

def seed_session(
    db_factory,
    participant_id,
    session_id,
    guide_id="scid5pd_de",
    guide_version="1.0",
    status="completed",
    historical_retrieval_consent=False,
):
    # EMSAG: 03.11.2026
    # Create a participant and interview session using the actual database models.
    with db_factory() as db:
        participant = db.get(Participant, participant_id)

        if participant is None:
            participant = Participant(
                id=participant_id,
                pseudonym=f"TEST-{uuid.uuid4().hex[:12]}",
            )
            db.add(participant)

        interview_session = InterviewSession(
            id=session_id,
            participant_id=participant_id,
            guide_id=guide_id,
            guide_version=guide_version,
            status=status,
            historical_retrieval_consent=historical_retrieval_consent,
        )
        db.add(interview_session)
        db.commit()


def seed_evidence(db_factory, session_id):
    # EMSAG: 03.11.2026
    # Insert a participant response, its AI assessment, and its SCID item record.
    with db_factory() as db:
        db.add(
            ParticipantTurn(
                id=str(uuid.uuid4()),
                session_id=session_id,
                item_id="OV2",
                turn_number=1,
                question_text="Can you describe this experience?",
                participant_response="I have experienced this repeatedly.",
            )
        )

        db.add(
            JudgmentAssessment(
                id=str(uuid.uuid4()),
                session_id=session_id,
                current_item_id="OV2",
                intent="answer",
                sufficiency="sufficient",
                provisional_rating="1",
                evidence_quote="I have experienced this repeatedly.",
                source_turn_number=1,
                historical_evidence=None,
                missing_information=None,
                reasoning="Test assessment.",
            )
        )

        db.add(
            InterviewItemRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                item_id="OV2",
                criterion_id="criterion_1",
                criterion_code="A1",
                question="Can you describe this experience?",
                status="answered",
                provisional_rating="1",
                rating_scheme="scid",
            )
        )

        db.commit()


def test_loads_current_session_evidence(db_factory):
    # EMSAG: 03.11.2026
    # Verify that all three evidence categories are loaded for the current session.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())

    seed_session(db_factory, participant_id, current_session_id)
    seed_evidence(db_factory, current_session_id)

    result = DiagnosisEvidenceRepository().load_session_evidence(
        current_session_id
    )

    assert len(result) == 1
    assert result[0]["session_id"] == current_session_id
    assert result[0]["source"] == "current_session"
    assert len(result[0]["participant_turns"]) == 1
    assert len(result[0]["judgment_assessments"]) == 1
    assert len(result[0]["item_records"]) == 1


def test_loads_authorized_historical_session(db_factory):
    # EMSAG: 03.11.2026
    # Verify that an explicitly supplied historical session is included.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    historical_session_id = str(uuid.uuid4())

    seed_session(db_factory, participant_id, current_session_id)
    seed_session(db_factory, participant_id, historical_session_id)

    seed_evidence(db_factory, current_session_id)
    seed_evidence(db_factory, historical_session_id)

    result = DiagnosisEvidenceRepository().load_session_evidence(
        current_session_id,
        authorized_historical_session_ids=[historical_session_id],
    )

    assert len(result) == 2

    sources = {entry["session_id"]: entry["source"] for entry in result}

    assert sources[current_session_id] == "current_session"
    assert sources[historical_session_id] == "historical_session"


def test_rejects_historical_session_from_another_participant(db_factory):
    # EMSAG: 03.11.2026
    # Prevent evidence belonging to another participant from being loaded.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    current_participant_id = str(uuid.uuid4())
    other_participant_id = str(uuid.uuid4())

    current_session_id = str(uuid.uuid4())
    foreign_session_id = str(uuid.uuid4())

    seed_session(db_factory, current_participant_id, current_session_id)
    seed_session(db_factory, other_participant_id, foreign_session_id)

    with pytest.raises(ValueError, match="not authorized"):
        DiagnosisEvidenceRepository().load_session_evidence(
            current_session_id,
            authorized_historical_session_ids=[foreign_session_id],
        )


def test_rejects_historical_session_with_different_guide_id(db_factory):
    # EMSAG: 03.11.2026
    # Prevent evidence from a different interview guide from being mixed.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    historical_session_id = str(uuid.uuid4())

    seed_session(db_factory, participant_id, current_session_id)
    seed_session(
        db_factory,
        participant_id,
        historical_session_id,
        guide_id="another_guide",
    )

    with pytest.raises(ValueError, match="not authorized"):
        DiagnosisEvidenceRepository().load_session_evidence(
            current_session_id,
            authorized_historical_session_ids=[historical_session_id],
        )


def test_rejects_historical_session_with_different_guide_version(db_factory):
    # EMSAG: 03.11.2026
    # Prevent evidence from a different guide version from being mixed.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    historical_session_id = str(uuid.uuid4())

    seed_session(db_factory, participant_id, current_session_id)
    seed_session(
        db_factory,
        participant_id,
        historical_session_id,
        guide_version="2.0",
    )

    with pytest.raises(ValueError, match="not authorized"):
        DiagnosisEvidenceRepository().load_session_evidence(
            current_session_id,
            authorized_historical_session_ids=[historical_session_id],
        )


def test_rejects_nonexistent_historical_session(db_factory):
    # EMSAG: 03.11.2026
    # Reject explicitly requested historical sessions that do not exist.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    nonexistent_session_id = str(uuid.uuid4())

    seed_session(db_factory, participant_id, current_session_id)

    with pytest.raises(ValueError, match="not authorized"):
        DiagnosisEvidenceRepository().load_session_evidence(
            current_session_id,
            authorized_historical_session_ids=[nonexistent_session_id],
        )


def test_does_not_load_unspecified_historical_sessions(db_factory):
    # EMSAG: 03.11.2026
    # Ensure historical sessions are excluded unless explicitly supplied.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    historical_session_id = str(uuid.uuid4())

    seed_session(db_factory, participant_id, current_session_id)
    seed_session(db_factory, participant_id, historical_session_id)

    result = DiagnosisEvidenceRepository().load_session_evidence(
        current_session_id
    )

    assert len(result) == 1
    assert result[0]["session_id"] == current_session_id
    assert result[0]["source"] == "current_session"

def test_does_not_authorize_historical_sessions_without_consent(db_factory):
    # EMSAG: 04.11.2026
    # Historical retrieval must remain disabled without explicit consent.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    historical_session_id = str(uuid.uuid4())

    seed_session(
        db_factory,
        participant_id,
        current_session_id,
        historical_retrieval_consent=False,
    )
    seed_session(db_factory, participant_id, historical_session_id)

    result = DiagnosisEvidenceRepository().get_authorized_historical_session_ids(
        current_session_id
    )

    assert result == []

def test_authorizes_only_completed_historical_sessions_with_consent(db_factory):
    # EMSAG: 04.11.2026
    # Only completed sessions of the same participant and guide are authorized.
    from backend.database.diagnosis_repository import DiagnosisEvidenceRepository

    participant_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4())
    completed_session_id = str(uuid.uuid4())
    active_session_id = str(uuid.uuid4())

    seed_session(
        db_factory,
        participant_id,
        current_session_id,
        historical_retrieval_consent=True,
    )
    seed_session(db_factory, participant_id, completed_session_id)
    seed_session(
        db_factory,
        participant_id,
        active_session_id,
        status="in_progress",
    )

    result = DiagnosisEvidenceRepository().get_authorized_historical_session_ids(
        current_session_id
    )

    assert completed_session_id in result
    assert active_session_id not in result
    assert current_session_id not in result