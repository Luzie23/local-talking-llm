
"""
SESSION RECORD
================
Builds and saves the researcher-only record of one interview.
JSON remains available as a local backup, while PostgreSQL provides
structured persistence when database_enabled is enabled.
"""

# EMSAG: 03.11.2026
# Extend SessionRecorder with optional PostgreSQL persistence while preserving JSON recording.

import json
import os
from datetime import datetime, timezone

from backend.database.recorder_repository import (
    create_interview_session,
    save_session_snapshot,
    save_judgment_assessment,
    save_navigation_decision,
)


RECORDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "session_records",
)

RATING_LABEL = "provisional – for clinician review"


def _now() -> str:
    """Current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_item_record(
    item: dict,
    item_turns: list[dict],
    status: str,
    judgment=None,
    skip_reason: str | None = None,
    decision=None,
    clarifications_used: int = 0,
) -> dict:
    """Assemble the record for one closed interview item."""

    record = {
        "item_id": item["id"],
        "criterion_id": item.get("criterion_id"),
        "criterion_code": item.get("criterion_code"),
        "question": item["text"],
        "status": status,
        "skip_reason": skip_reason,
        "turns": item_turns,
        "clarifications_used": clarifications_used,
        "closed_at": _now(),
    }

    if item.get("safety_relevant"):
        record["safety_relevant"] = True

    if judgment is not None:
        record["judgment"] = {
            "intent": judgment.intent,
            "sufficiency": judgment.sufficiency,
            "missing_information": judgment.missing_information,
            "evidence_quote": judgment.evidence_quote,
            "reasoning": judgment.reasoning,
            "validation_notes": judgment.validation_notes,
        }

    if item.get("rating_scheme", "none") != "none":
        rating = (
            judgment.provisional_rating
            if judgment is not None and status == "answered"
            else "?"
        )

        record["provisional_rating"] = {
            "value": rating or "?",
            "scheme": item["rating_scheme"],
            "label": RATING_LABEL,
        }

    if decision is not None:
        record["decision"] = {
            "action": decision.action,
            "reason": decision.reason,
        }

    return record


class SessionRecorder:
    """Keeps the JSON and optional PostgreSQL records of one interview."""

    def __init__(
        self,
        guide,
        directory: str = RECORDS_DIR,
        enabled: bool = True,
        database_enabled: bool = False,
    ):
        """
        Args:
            guide: The InterviewGuide whose records are saved.
            directory: Directory for JSON session records.
            enabled: Enable or disable JSON recording.
            database_enabled: Enable or disable PostgreSQL persistence.
        """
        self.guide = guide
        self.directory = directory
        self.enabled = enabled
        self.database_enabled = database_enabled

        self.participant_id: str | None = None
        self.session_id: str | None = None
        self.pseudonym: str | None = None
        self.generated_participant_code: str | None = None

        self.start()

    def start(
            self,
            participant_code: str | None = None,
            historical_retrieval_consent: bool = False,
        ) -> None:
        
        """Begin a new record and optionally create a PostgreSQL session."""

        self.started_at = _now()
        self.system_events: list[dict] = []

        # EMSAG: 03.11.2026
        # Historical retrieval requires explicit consent for every new session.
        self.historical_retrieval_consent = (
            historical_retrieval_consent is True
        )

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path = os.path.join(
            self.directory,
            f"{self.guide.guide_id}_{stamp}.json",
        )

        # EMSAG: 03.11.2026
        # Create a fresh pseudonymized participant and database session for each interview.
        self.participant_id = None
        self.session_id = None
        self.pseudonym = None
        self.generated_participant_code = None

        if self.database_enabled:
            (
                self.participant_id,
                self.session_id,
                self.pseudonym,
                self.generated_participant_code,
            ) = create_interview_session(
                self.guide,
                participant_code=participant_code,
                historical_retrieval_consent=self.historical_retrieval_consent,
            )

    
    def log_system_event(
        self,
        item_id: str | None,
        kind: str,
        detail: str,
    ) -> None:
        """Log a technical event separately from participant responses."""

        self.system_events.append({
            "at": _now(),
            "item_id": item_id,
            "kind": kind,
            "detail": detail[:500],
        })


    # EMSAG: 03.11.2026
    # Persist every Judgment Agent evaluation immediately without overwriting previous assessments.
    def record_judgment(self, item_id: str, judgment) -> None:
        if not self.database_enabled:
            return

        if not self.session_id:
            raise RuntimeError("No active database session.")

        save_judgment_assessment(
            session_id=self.session_id,
            item_id=item_id,
            judgment=judgment.model_dump(),
        )

    # EMSAG: 03.11.2026
    # Persist proposed, validated and actually executed navigation information separately.
    def record_navigation(
        self,
        item_id: str,
        proposed,
        validated,
        executed_next_item_id: str | None,
    ) -> None:
        if not self.database_enabled:
            return

        if not self.session_id:
            raise RuntimeError("No active database session.")

        save_navigation_decision(
            session_id=self.session_id,
            item_id=item_id,
            proposed=proposed.model_dump(),
            validated=validated.model_dump(),
            executed_next_item_id=executed_next_item_id,
        )

    
    def to_dict(self) -> dict:
        """Return the current researcher-only interview record."""

        return {
            "guide_id": self.guide.guide_id,
            "guide_version": self.guide.guide_version,
            "content_source": self.guide.content_source,
            "started_at": self.started_at,
            "updated_at": _now(),
            "finished": self.guide.is_finished(),
            "paused": self.guide.paused,
            "historical_retrieval_consent": self.historical_retrieval_consent,
            "notice": (
                "Research prototype. Not a diagnosis. "
                "'skipped' items are missing information, not negative answers. "
                "Ratings are provisional and require clinician review. "
                "Source-defined SCID routing gates may use provisional ratings "
                "but do not produce clinical diagnoses."
            ),
            "items": self.guide.records,
            "system_events": self.system_events,
        }

    def save(self) -> None:
        """Save the current record to JSON and optionally PostgreSQL."""

        # EMSAG: 03.11.2026
        # Keep JSON backups and persist structured records through the repository when enabled.
        if self.enabled:
            os.makedirs(self.directory, exist_ok=True)

            with open(self.path, "w", encoding="utf-8") as file:
                json.dump(
                    self.to_dict(),
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        if self.database_enabled:
            if not self.session_id:
                raise RuntimeError(
                    "Cannot save PostgreSQL records: no active database session."
                )

            save_session_snapshot(
                session_id=self.session_id,
                guide=self.guide,
                system_events=self.system_events,
            )
