
# EMSAG: 03.11.2026
# Store technical system events separately from participant responses and assessments.

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class SystemEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("interview_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    item_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    kind: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
