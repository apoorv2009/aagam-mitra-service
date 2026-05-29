from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "aagam-mitra-service"
    environment: str = "dev"

    # Google Gemini — embeddings only
    gemini_api_key: str = ""

    # Groq — answer generation (free, 14,400 req/day)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Pinecone — persistent vector store
    pinecone_api_key: str = ""
    pinecone_index_name: str = "jain-texts"

    # RAG tuning
    retrieval_limit: int = 4
    chunk_size_characters: int = 800
    chunk_overlap_characters: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
