from __future__ import annotations

from sqlalchemy import select

from app.core.database import SessionLocal
from app.models import ChatMessage
from app.schemas.assistant import ChatHistoryMessage


_MAX_HISTORY = 100  # rows kept per user+temple pair


def save_exchange(
    *,
    user_id: str,
    temple_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Persist a user↔assistant exchange.  Trims to _MAX_HISTORY rows after insert."""
    with SessionLocal() as session:
        session.add(ChatMessage(user_id=user_id, temple_id=temple_id, role="user", content=user_message))
        session.add(ChatMessage(user_id=user_id, temple_id=temple_id, role="assistant", content=assistant_message))
        session.commit()

        # Keep only the latest _MAX_HISTORY rows
        rows = session.execute(
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id, ChatMessage.temple_id == temple_id)
            .order_by(ChatMessage.created_at.asc())
        ).scalars().all()

        if len(rows) > _MAX_HISTORY:
            for old in rows[: len(rows) - _MAX_HISTORY]:
                session.delete(old)
            session.commit()


def load_history(*, user_id: str, temple_id: str, limit: int = 40) -> list[ChatHistoryMessage]:
    """Return the last `limit` messages for this user+temple pair, oldest first."""
    with SessionLocal() as session:
        rows = session.execute(
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id, ChatMessage.temple_id == temple_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).scalars().all()

    return [ChatHistoryMessage(role=r.role, content=r.content) for r in reversed(rows)]
