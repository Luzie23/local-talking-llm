
# EMSAG: 03.11.2026
# Link each interview session to a pseudonymized participant and track its lifecycle.

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    participant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("participants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    guide_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    guide_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="in_progress",
        nullable=False,
    )

    # EMSAG: 03.11.2026
    # Explicit consent for historical retrieval, recorded per interview session.
    historical_retrieval_consent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    participant = relationship(
        "Participant",
        back_populates="interview_sessions",
    )
