
# EMSAG: 03.11.2026
# Store pseudonymized participants and link them to their interview sessions.

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    pseudonym: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    
    # EMSAG: 03.11.2026
    # Store only the hash of the participant's anonymous access code.
    participant_code_hash: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # EMSAG: 03.11.2026
    # Allow one participant to have multiple interview sessions.
    interview_sessions = relationship(
        "InterviewSession",
        back_populates="participant",
        cascade="save-update, merge",
    )
