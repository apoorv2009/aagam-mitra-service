from fastapi import FastAPI

from app.api.router import api_router

app = FastAPI(
    title="Aagam Mitra",
    version="0.1.0",
    summary="Local AI assistant — RAG over Jain Agam texts.",
)
app.include_router(api_router)
