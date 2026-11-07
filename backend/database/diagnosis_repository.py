
# EMSAG: 03.11.2026
# Retrieve current and explicitly authorized historical SCID evidence without modifying interview records.

from backend.database.connection import SessionLocal
from backend.database.models import (
    InterviewSession,
    InterviewItemRecord,
    ParticipantTurn,
    JudgmentAssessment,
)


class DiagnosisEvidenceRepository:
    
    def get_authorized_historical_session_ids(
        self,
        current_session_id: str,
    ) -> list[str]:
        """Return completed historical sessions only when current-session consent exists."""

        with SessionLocal() as db:
            current_session = db.get(
                InterviewSession,
                current_session_id,
            )

            if current_session is None:
                raise ValueError("Current interview session not found.")

            # EMSAG: 04.11.2026
            # Never retrieve historical sessions without explicit current-session consent.
            if current_session.historical_retrieval_consent is not True:
                return []

            historical_sessions = (
                db.query(InterviewSession.id)
                .filter(
                    InterviewSession.participant_id
                    == current_session.participant_id,
                    InterviewSession.guide_id
                    == current_session.guide_id,
                    InterviewSession.guide_version
                    == current_session.guide_version,
                    InterviewSession.id != current_session_id,
                    InterviewSession.status == "completed",
                )
                .order_by(InterviewSession.started_at.desc())
                .all()
            )

            return [session_id for (session_id,) in historical_sessions]

    def load_session_evidence(
        self,
        current_session_id: str,
        authorized_historical_session_ids: list[str] | None = None,
    ) -> list[dict]:

        authorized_historical_session_ids = (
            authorized_historical_session_ids or []
        )

        with SessionLocal() as db:

            current_session = db.get(
                InterviewSession,
                current_session_id,
            )

            if current_session is None:
                raise ValueError("Current interview session not found.")

            allowed_sessions = [
                current_session_id,
                *authorized_historical_session_ids,
            ]

            sessions = (
                db.query(InterviewSession)
                .filter(
                    InterviewSession.id.in_(allowed_sessions),
                    InterviewSession.participant_id
                    == current_session.participant_id,
                    InterviewSession.guide_id
                    == current_session.guide_id,
                    # EMSAG: 03.11.2026
                    # Prevent evidence from different SCID guide versions from being mixed.
                    InterviewSession.guide_version == current_session.guide_version,
                )
                .all()
            )

            found_ids = {session.id for session in sessions}

            missing_ids = (
                set(authorized_historical_session_ids) - found_ids
            )

            if missing_ids:
                raise ValueError(
                    "Historical sessions are not authorized, "
                    "do not belong to this participant, or use "
                    "a different interview guide."
                )

            evidence = []

            for session in sessions:

                source = (
                    "current_session"
                    if session.id == current_session_id
                    else "historical_session"
                )

                turns = (
                    db.query(ParticipantTurn)
                    .filter(
                        ParticipantTurn.session_id == session.id
                    )
                    .order_by(
                        ParticipantTurn.item_id,
                        ParticipantTurn.turn_number,
                    )
                    .all()
                )

                assessments = (
                    db.query(JudgmentAssessment)
                    .filter(
                        JudgmentAssessment.session_id == session.id
                    )
                    .order_by(
                        JudgmentAssessment.assessed_at
                    )
                    .all()
                )

                item_records = (
                    db.query(InterviewItemRecord)
                    .filter(
                        InterviewItemRecord.session_id == session.id
                    )
                    .all()
                )

                evidence.append({
                    "session_id": session.id,
                    "guide_id": session.guide_id,
                    "guide_version": session.guide_version,
                    "source": source,
                    "status": session.status,

                    "participant_turns": [
                        {
                            "item_id": turn.item_id,
                            "turn_number": turn.turn_number,
                            "question": turn.question_text,
                            "response": turn.participant_response,
                            "recorded_at": turn.recorded_at,
                        }
                        for turn in turns
                    ],

                    "judgment_assessments": [
                        {
                            "item_id": assessment.current_item_id,
                            "intent": assessment.intent,
                            "sufficiency": assessment.sufficiency,
                            "provisional_rating": assessment.provisional_rating,
                            "evidence_quote": assessment.evidence_quote,
                            "historical_evidence": assessment.historical_evidence,
                            "missing_information": assessment.missing_information,
                            "source_turn_number": assessment.source_turn_number,
                            "reasoning": assessment.reasoning,
                            "assessed_at": assessment.assessed_at,
                        }
                        for assessment in assessments
                    ],

                    "item_records": [
                        {
                            "item_id": record.item_id,
                            "criterion_id": record.criterion_id,
                            "criterion_code": record.criterion_code,
                            "status": record.status,
                            "skip_reason": record.skip_reason,
                            "provisional_rating": record.provisional_rating,
                            "rating_scheme": record.rating_scheme,
                        }
                        for record in item_records
                    ],
                })

            return evidence
