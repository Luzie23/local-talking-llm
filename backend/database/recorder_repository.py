# EMSAG: 03.11.2026
# Persist interview data, append-only assessments, navigation decisions and system events in PostgreSQL.
import hashlib
import hmac
import os

import json
import uuid
from datetime import datetime, timezone

from backend.database.connection import SessionLocal
from backend.database.models import (
    InterviewItemRecord,
    InterviewSession,
    JudgmentAssessment,
    NavigationDecision,
    Participant,
    ParticipantTurn,
    SystemEvent,
)


def _stable_id(session_id: str, record_key: str) -> str:
    """Generate a repeatable record ID within one interview session."""
    return str(uuid.uuid5(uuid.UUID(session_id), record_key))


def _parse_datetime(value: str | None) -> datetime:
    """Convert an ISO timestamp into a timezone-aware datetime."""
    if not value:
        return datetime.now(timezone.utc)

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def create_interview_session(
    guide,
    participant_code: str | None = None,
    historical_retrieval_consent: bool = False,
) -> tuple[str, str, str, str | None]:
    """
    Reuse an existing participant when a valid code is supplied.
    Generate a new reusable code when no code is supplied.
    Always create a new interview session.
    """

    with SessionLocal() as db:
        try:
            participant = None
            generated_code = None
            code_hash = None

            secret = os.getenv("PARTICIPANT_CODE_SECRET")

            if not secret:
                raise RuntimeError(
                    "PARTICIPANT_CODE_SECRET is not configured."
                )

            if participant_code is not None:
                normalized_code = participant_code.strip().upper()

                if not normalized_code:
                    raise ValueError("Participant code cannot be empty.")

                code_hash = hmac.new(
                    secret.encode("utf-8"),
                    normalized_code.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()

                participant = (
                    db.query(Participant)
                    .filter(
                        Participant.participant_code_hash == code_hash
                    )
                    .first()
                )

                if participant is None:
                    raise ValueError(
                        "Invalid participant code. Please check your code."
                    )

            else:
                generated_code = (
                    f"MAGI-{uuid.uuid4().hex[:4].upper()}-"
                    f"{uuid.uuid4().hex[:4].upper()}"
                )

                code_hash = hmac.new(
                    secret.encode("utf-8"),
                    generated_code.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()

                pseudonym = f"P-{uuid.uuid4().hex[:8].upper()}"

                participant = Participant(
                    pseudonym=pseudonym,
                    participant_code_hash=code_hash,
                )

                db.add(participant)
                db.flush()

            interview_session = InterviewSession(
                participant_id=participant.id,
                guide_id=guide.guide_id,
                guide_version=guide.guide_version,
                status="in_progress",
                historical_retrieval_consent=historical_retrieval_consent,
            )

            db.add(interview_session)
            db.commit()

            return (
                participant.id,
                interview_session.id,
                participant.pseudonym,
                generated_code,
            )

        except Exception:
            db.rollback()
            raise


# EMSAG: 03.11.2026
# Serialize structured assessment fields before storing them in PostgreSQL Text columns.
def _serialize_optional_text(value) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False)


# EMSAG: 03.11.2026
# Store every Judgment Agent assessment as a separate append-only database record.
def save_judgment_assessment(
    session_id: str,
    item_id: str,
    judgment: dict,
) -> None:
    provisional_rating = judgment.get("provisional_rating")

    if isinstance(provisional_rating, dict):
        provisional_rating = provisional_rating.get("value")

    with SessionLocal() as db:
        try:
            assessment = JudgmentAssessment(
                id=str(uuid.uuid4()),
                session_id=session_id,
                item_record_id=None,
                current_item_id=item_id,
                intent=judgment.get("intent", "unknown"),
                sufficiency=judgment.get("sufficiency", "unknown"),
                provisional_rating=provisional_rating,
                missing_information=_serialize_optional_text(
                    judgment.get("missing_information")
                ),
                evidence_quote=judgment.get("evidence_quote"),

                # EMSAG: 03.11.2026
                # Persist the exact participant turn associated with the latest evidence.
                source_turn_number=judgment.get("source_turn_number"),

                # EMSAG: 03.11.2026
                # Persist earlier supporting evidence separately from the latest response.
                historical_evidence=_serialize_optional_text(
                    judgment.get("historical_evidence")
                ),

                reasoning=judgment.get("reasoning"),
                validation_notes=_serialize_optional_text(
                    judgment.get("validation_notes")
                ),
            )

            db.add(assessment)
            db.commit()

        except Exception:
            db.rollback()
            raise


# EMSAG: 03.11.2026
# Preserve proposed, validated and executed navigation outcomes as separate decision records.
def save_navigation_decision(
    session_id: str,
    item_id: str,
    proposed: dict,
    validated: dict,
    executed_next_item_id: str | None,
) -> None:
    with SessionLocal() as db:
        try:
            navigation = NavigationDecision(
                id=str(uuid.uuid4()),
                session_id=session_id,
                item_record_id=None,
                current_item_id=item_id,
                proposed_action=proposed.get("action"),
                proposed_next_item_id=proposed.get("next_step"),
                recommended_next_item_id=validated.get("next_step"),
                executed_next_item_id=executed_next_item_id,
                action=validated.get("action", "unknown"),
                reason=validated.get("reason") or proposed.get("reason"),
                validation_status="executed",
            )

            db.add(navigation)
            db.commit()

        except Exception:
            db.rollback()
            raise


def _save_turns(db, session_id: str, item_id: str, turns: list[dict]) -> None:
    """Save participant turns using stable IDs so repeated saves update existing rows."""
    for turn_number, turn in enumerate(turns, start=1):
        participant_turn = ParticipantTurn(
            id=_stable_id(session_id, f"turn:{item_id}:{turn_number}"),
            session_id=session_id,
            item_id=item_id,
            turn_number=turn_number,
            question_text=turn.get("prompt", ""),
            participant_response=turn.get("answer", ""),
        )
        db.merge(participant_turn)


def save_session_snapshot(
    session_id: str,
    guide,
    system_events: list[dict],
) -> None:
    """Persist the current interview state in one database transaction."""

    with SessionLocal() as db:
        try:
            interview_session = db.get(InterviewSession, session_id)

            if interview_session is None:
                raise ValueError(f"Interview session not found: {session_id}")

            interview_session.status = (
                "completed" if guide.is_finished()
                else "paused" if guide.paused
                else "in_progress"
            )

            if guide.is_finished() and interview_session.ended_at is None:
                interview_session.ended_at = datetime.now(timezone.utc)

            # EMSAG: 03.11.2026
            # Persist completed item records and link their earlier assessments and navigation decisions.
            for record in guide.records:
                item_id = record["item_id"]
                item_record_id = _stable_id(session_id, f"item:{item_id}")

                rating = record.get("provisional_rating") or {}

                item_record = InterviewItemRecord(
                    id=item_record_id,
                    session_id=session_id,
                    item_id=item_id,
                    criterion_id=record.get("criterion_id"),
                    criterion_code=record.get("criterion_code"),
                    question=record.get("question") or record.get("note") or item_id,
                    status=record.get("status", "unknown"),
                    skip_reason=record.get("skip_reason"),
                    clarifications_used=record.get("clarifications_used", 0),
                    safety_relevant=record.get("safety_relevant", False),
                    provisional_rating=rating.get("value"),
                    rating_scheme=rating.get("scheme"),
                    rating_label=rating.get("label"),
                    closed_at=_parse_datetime(record.get("closed_at")),
                )

                db.merge(item_record)

                # EMSAG: 03.11.2026
                # Flush the item record before assigning its ID to related assessment and navigation rows.
                db.flush()

                # EMSAG: 03.11.2026
                # Link previously stored assessments and navigation decisions to the closed interview item.
                db.query(JudgmentAssessment).filter(
                    JudgmentAssessment.session_id == session_id,
                    JudgmentAssessment.current_item_id == item_id,
                    JudgmentAssessment.item_record_id.is_(None),
                ).update(
                    {"item_record_id": item_record_id},
                    synchronize_session=False,
                )

                db.query(NavigationDecision).filter(
                    NavigationDecision.session_id == session_id,
                    NavigationDecision.current_item_id == item_id,
                    NavigationDecision.item_record_id.is_(None),
                ).update(
                    {"item_record_id": item_record_id},
                    synchronize_session=False,
                )

                _save_turns(
                    db,
                    session_id,
                    item_id,
                    record.get("turns", []),
                )

            # EMSAG: 03.11.2026
            # Persist turns for the currently active item before it is closed.
            current_item = guide.current_question()

            if current_item is not None and guide.item_turns:
                _save_turns(
                    db,
                    session_id,
                    current_item["id"],
                    guide.item_turns,
                )

            # EMSAG: 03.11.2026
            # Persist technical events separately from participant responses.
            for event_number, event in enumerate(system_events, start=1):
                db.merge(SystemEvent(
                    id=_stable_id(session_id, f"event:{event_number}"),
                    session_id=session_id,
                    item_id=event.get("item_id"),
                    kind=event.get("kind", "unknown"),
                    detail=event.get("detail", "")[:500],
                    occurred_at=_parse_datetime(event.get("at")),
                ))

            db.commit()

        except Exception:
            db.rollback()
            raise