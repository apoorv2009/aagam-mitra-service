from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "aagam-mitra-service"
    environment: str = "dev"

    # Google Gemini — embeddings only
    gemini_api_key: str = ""

    # Groq — answer generation (free, 14,400 req/day)
    groq_api_key: str = ""
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Pinecone — persistent vector store for Jain texts
    pinecone_api_key: str = ""
    pinecone_index_name: str = "jain-texts"

    # SQLite / Postgres — temple operational knowledge (synced from other services)
    database_url: str = "sqlite:///./temple_ai.db"

    # Upstream services
    admin_service_url: str = "http://localhost:8003"
    registration_service_url: str = "http://localhost:8002"
    upstream_timeout_seconds: float = 45.0
    upstream_retry_attempts: int = 4

    # RAG tuning
    retrieval_limit: int = 4
    chunk_size_characters: int = 800
    chunk_overlap_characters: int = 100
    sync_ttl_seconds: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
