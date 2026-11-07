
# EMSAG: 03.11.2026
# Register all database models so SQLAlchemy and Alembic can discover their tables.

from backend.database.models.participant import Participant
from backend.database.models.interview_session import InterviewSession
from backend.database.models.interview_item_record import InterviewItemRecord
from backend.database.models.participant_turn import ParticipantTurn
from backend.database.models.judgment_assessment import JudgmentAssessment
from backend.database.models.navigation_decision import NavigationDecision
from backend.database.models.system_event import SystemEvent


__all__ = [
    "Participant",
    "InterviewSession",
    "InterviewItemRecord",
    "ParticipantTurn",
    "JudgmentAssessment",
    "NavigationDecision",
    "SystemEvent",
]
