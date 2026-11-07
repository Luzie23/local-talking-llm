
# EMSAG: 03.11.2026
# Preserve each participant response separately from AI-generated assessments.

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class ParticipantTurn(Base):
    __tablename__ = "participant_turns"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    item_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    turn_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    question_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    participant_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
