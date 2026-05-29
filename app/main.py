from fastapi import FastAPI

from app.api.router import api_router
from app.core.database import init_database

app = FastAPI(
    title="Aagam Mitra",
    version="0.2.0",
    summary="Temple AI — Jain scripture RAG and temple operations assistant.",
)
app.include_router(api_router)


@app.on_event("startup")
async def on_startup() -> None:
    init_database()
