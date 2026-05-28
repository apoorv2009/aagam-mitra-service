import httpx

from app.core.config import get_settings

_BATCH_EMBED_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/text-embedding-004:batchEmbedContents"
)
_BATCH_SIZE = 100  # Gemini batch limit per request


async def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts using Gemini text-embedding-004.

    Use task_type="RETRIEVAL_DOCUMENT" when embedding PDF chunks for storage.
    Use task_type="RETRIEVAL_QUERY" when embedding a user question at query time.
    """
    settings = get_settings()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        payload = {
            "requests": [
                {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in batch
            ]
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_BATCH_EMBED_URL}?key={settings.gemini_api_key}",
                json=payload,
            )
            response.raise_for_status()

        all_embeddings.extend(e["values"] for e in response.json()["embeddings"])

    return all_embeddings
