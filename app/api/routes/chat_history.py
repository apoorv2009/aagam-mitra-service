from fastapi import APIRouter

from app.schemas.assistant import ChatHistoryMessage
from app.services.chat_history import load_history

router = APIRouter()


@router.get("/chat/{user_id}/{temple_id}/history", response_model=list[ChatHistoryMessage])
async def get_chat_history(user_id: str, temple_id: str, limit: int = 40) -> list[ChatHistoryMessage]:
    return load_history(user_id=user_id, temple_id=temple_id, limit=min(limit, 100))
