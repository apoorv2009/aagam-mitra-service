from fastapi import APIRouter

from app.schemas.assistant import TempleAssistantRequest, TempleAssistantResponse
from app.services.assistant import generate_assistant_reply

router = APIRouter()


@router.post("/temples/{temple_id}/chat", response_model=TempleAssistantResponse)
async def chat_with_temple_assistant(
    temple_id: str,
    payload: TempleAssistantRequest,
) -> TempleAssistantResponse:
    return await generate_assistant_reply(temple_id, payload)
