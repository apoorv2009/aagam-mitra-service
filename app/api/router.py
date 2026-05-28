from fastapi import APIRouter

from app.api.routes import health, knowledge

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"])
