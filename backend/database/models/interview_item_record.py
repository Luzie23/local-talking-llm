
# EMSAG: 03.11.2026
# Persist SCID item outcomes without treating skipped items as negative answers.

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class InterviewItemRecord(Base):
    __tablename__ = "interview_item_records"

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
    )

    criterion_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    criterion_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    skip_reason: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    clarifications_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    safety_relevant: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    provisional_rating: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    rating_scheme: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    rating_label: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
