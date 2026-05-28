import httpx

from app.core.config import get_settings
from app.services.embedder import embed_texts
from app.services.vector_store import get_index

_GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.0-flash:generateContent"
)

_SYSTEM_PROMPT = (
    "You are Aagam Mitra, a knowledgeable guide for Jain Agam texts and temple matters. "
    "Answer using only the context passages provided below. "
    "If the answer cannot be found in the context, say so honestly rather than guessing. "
    "Keep your answer concise and cite the source text when relevant."
)


async def ask(question: str) -> dict:
    settings = get_settings()

    # Embed the question with the QUERY task type
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

    # Build context block from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[{m.metadata.get('source', '?')}, p.{m.metadata.get('page', '?')}]\n{m.metadata.get('text', '')}"
        for m in matches
    )
    prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    # Generate answer with Gemini
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{_GEMINI_GENERATE_URL}?key={settings.gemini_api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        response.raise_for_status()

    answer: str = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    sources = [
        {"file": m.metadata.get("source"), "page": m.metadata.get("page")}
        for m in matches
    ]
    return {"answer": answer.strip(), "sources": sources}
