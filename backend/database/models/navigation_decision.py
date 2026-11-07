
# EMSAG: 03.11.2026
# Preserve the original navigation proposal, validated decision and actual executed transition.

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class NavigationDecision(Base):
    __tablename__ = "navigation_decisions"

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

    item_record_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("interview_item_records.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    current_item_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # EMSAG: 03.11.2026
    # Preserve the Navigation Agent's original proposal before orchestrator validation.
    proposed_action: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    proposed_next_item_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    recommended_next_item_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    executed_next_item_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # This field represents the validated action.
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    validation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
