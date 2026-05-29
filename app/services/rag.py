import httpx

from app.core.config import get_settings
from app.services.embedder import embed_texts
from app.services.vector_store import get_index

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = """You are Aagam Mitra — a deeply knowledgeable guide in Jain philosophy, Agam scriptures, and Jain history.

Your role is to give clear, well-structured, and insightful answers. You have access to reference passages from Jain texts. Use them as your source material, but present the information naturally — like a scholar explaining to a student, not like a machine copying text.

Guidelines:
- Synthesize information from the passages into a coherent, flowing answer
- Use numbered lists or sections when the answer has multiple parts (like listing bhavs, vows, or principles)
- Present names, terms, and concepts clearly — transliterate Sanskrit/Hindi terms and briefly explain them
- If the passages contain a list or sequence (like bhavs/lives), present it completely and in order
- Add brief context or explanation where it helps understanding
- Write in the same language the user asked in (Hindi question → Hindi answer, English → English)
- Be thorough for questions asking for complete lists or all items
- Do not make up information not present in the passages"""


async def ask(question: str) -> dict:
    settings = get_settings()

    query_embedding = (await embed_texts([question], task_type="RETRIEVAL_QUERY"))[0]

    index = get_index()
    results = index.query(
        vector=query_embedding,
        top_k=8,  # More chunks = more complete answers for multi-page topics
        include_metadata=True,
    )
    matches = results.matches if results.matches else []

    if not matches:
        return {
            "answer": "No relevant passages found. Please ingest some Jain text PDFs first.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {m.metadata.get('source', '?')}, Page {m.metadata.get('page', '?')}]\n{m.metadata.get('text', '')}"
        for m in matches
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            _GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Reference passages from Jain texts:\n\n{context}\n\nQuestion: {question}",
                    },
                ],
                "temperature": 0.3,
            },
        )
        response.raise_for_status()

    answer: str = response.json()["choices"][0]["message"]["content"]
    sources = [
        {"file": m.metadata.get("source"), "page": m.metadata.get("page")}
        for m in matches
    ]
    return {"answer": answer.strip(), "sources": sources}
