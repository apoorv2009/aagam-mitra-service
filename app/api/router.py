from fastapi import APIRouter

from app.api.routes import assistant, chat_history, health, knowledge

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"])
api_router.include_router(assistant.router, prefix="/api/v1/assistant", tags=["assistant"])
api_router.include_router(chat_history.router, prefix="/api/v1", tags=["chat-history"])
