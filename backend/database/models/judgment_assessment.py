
# EMSAG: 03.11.2026
# Store every AI-generated judgment independently, including assessments made before an item is closed.

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class JudgmentAssessment(Base):
    __tablename__ = "judgment_assessments"

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

    # EMSAG: 03.11.2026
    # Allow assessments to exist before the corresponding SCID item record is closed.
    item_record_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("interview_item_records.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # EMSAG: 03.11.2026
    # Identify the SCID item directly so intermediate assessments remain traceable.
    current_item_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    intent: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    sufficiency: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    provisional_rating: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    missing_information: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    evidence_quote: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # EMSAG: 03.11.2026
    # Store the participant turn that supplied the latest evidence quote.
    source_turn_number: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    # EMSAG: 03.11.2026
    # Preserve evidence from earlier participant turns as structured JSON.
    historical_evidence: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reasoning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    validation_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    assessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
