import httpx

from app.core.config import get_settings
from app.services.embedder import embed_texts
from app.services.vector_store import get_index

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are Aagam Mitra, a knowledgeable guide for Jain Agam texts and temple matters. "
    "Answer using only the context passages provided below. "
    "If the answer cannot be found in the context, say so honestly rather than guessing. "
    "Keep your answer concise and cite the source text when relevant."
)


async def ask(question: str) -> dict:
    settings = get_settings()

    # Embed the question with QUERY task type
    query_embedding = (await embed_texts([question], task_type="RETRIEVAL_QUERY"))[0]

    # Retrieve top-k matching chunks from Pinecone
    index = get_index()
    results = index.query(
        vector=query_embedding,
        top_k=settings.retrieval_limit,
        include_metadata=True,
    )
    matches = results.matches if results.matches else []

    if not matches:
        return {
            "answer": "No relevant passages found. Please ingest some Jain text PDFs first.",
            "sources": [],
        }

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[{m.metadata.get('source', '?')}, p.{m.metadata.get('page', '?')}]\n{m.metadata.get('text', '')}"
        for m in matches
    )

    # Generate answer with Groq
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            _GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                "temperature": 0.1,
            },
        )
        response.raise_for_status()

    answer: str = response.json()["choices"][0]["message"]["content"]
    sources = [
        {"file": m.metadata.get("source"), "page": m.metadata.get("page")}
        for m in matches
    ]
    return {"answer": answer.strip(), "sources": sources}
