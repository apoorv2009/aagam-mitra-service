from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models import TempleKnowledgeChunk, TempleKnowledgeDocument, TempleKnowledgeSyncState
from app.schemas.assistant import (
    TempleAssistantActionCard,
    TempleAssistantCitation,
    TempleAssistantRequest,
    TempleAssistantResponse,
)
from app.services.chat_history import save_exchange
from app.services.embedder import embed_texts
from app.services.vector_store import get_index

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass
class SourceDocument:
    source_type: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, object]


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_type: str
    source_id: str
    title: str
    content: str
    similarity: float


@dataclass
class ToolResult:
    source_id: str
    title: str
    source_type: str
    excerpt: str
    fact: str


def _retry_delay_seconds(attempt: int) -> float:
    return min(8.0, float(1 + attempt))


async def _get_json(url: str) -> dict[str, object]:
    settings = get_settings()
    attempts = max(1, settings.upstream_retry_attempts)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
                response = await client.get(url)
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < attempts - 1:
                await asyncio.sleep(_retry_delay_seconds(attempt + 1))
                continue
            raise

        if response.status_code >= 500 and attempt < attempts - 1:
            await asyncio.sleep(_retry_delay_seconds(attempt + 1))
            continue

        response.raise_for_status()
        return response.json()

    if last_error:
        raise last_error
    raise RuntimeError("Unable to load upstream data")


def _content_checksum(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


def _build_citation_from_chunk(chunk: RetrievedChunk) -> TempleAssistantCitation:
    return TempleAssistantCitation(
        source_id=chunk.source_id,
        title=chunk.title,
        source_type=chunk.source_type,  # type: ignore[arg-type]
        excerpt=chunk.content[:220],
    )


def _build_citation_from_tool(result: ToolResult) -> TempleAssistantCitation:
    return TempleAssistantCitation(
        source_id=result.source_id,
        title=result.title,
        source_type=result.source_type,  # type: ignore[arg-type]
        excerpt=result.excerpt,
    )


def _action_cards_for_message(message: str, role: str) -> list[TempleAssistantActionCard]:
    lowered = message.lower()
    cards: list[TempleAssistantActionCard] = []
    if any(token in lowered for token in ["book", "shantidhara", "slot", "calendar"]):
        cards.append(TempleAssistantActionCard(
            action_id="open-book", title="Book Shantidhara",
            description="Open the booking flow and review available temple slots.",
            action_label="Open Book", action_target="book",
        ))
    if any(token in lowered for token in ["donate", "donation", "payment", "qr"]):
        cards.append(TempleAssistantActionCard(
            action_id="open-donate", title="Open Donate",
            description="Go to the donation flow and continue QR payment.",
            action_label="Open Donate", action_target="donate",
        ))
    if any(token in lowered for token in ["notice", "update", "news", "information", "wall of fame"]):
        cards.append(TempleAssistantActionCard(
            action_id="open-home", title="View Temple Updates",
            description="Review recent temple information and recognitions.",
            action_label="Open Home", action_target="home",
        ))
    if role == "admin" and any(token in lowered for token in ["draft", "publish", "notification"]):
        cards.append(TempleAssistantActionCard(
            action_id="open-admin", title="Open Admin Tools",
            description="Go to the admin tab and publish an update.",
            action_label="Open Admin", action_target="admin",
        ))
    if not cards:
        cards.append(TempleAssistantActionCard(
            action_id="stay-chat", title="Ask another temple question",
            description="You can ask about bookings, donations, notices, or membership status.",
            action_label="Keep chatting", action_target="chat",
        ))
    return cards


async def _load_temple_source_documents(temple_id: str) -> list[SourceDocument]:
    settings = get_settings()
    temple_data, news_data, fame_data, payment_data = await asyncio.gather(
        _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}"),
        _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/news-feed"),
        _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/wall-of-fame"),
        _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/payment-profile"),
    )

    documents: list[SourceDocument] = [
        SourceDocument(
            source_type="temple_profile",
            source_id=str(temple_data["temple_id"]),
            title=str(temple_data["temple_name"]),
            content=(
                f"Temple name: {temple_data['temple_name']}. "
                f"Location: {temple_data['temple_location']}. "
                f"Status: {temple_data['status']}."
            ),
            metadata={"temple_name": temple_data["temple_name"]},
        ),
        SourceDocument(
            source_type="payment_profile",
            source_id=f"{temple_id}-payment-profile",
            title=f"{temple_data['temple_name']} payment profile",
            content=(
                f"Account label: {payment_data['account_label']}. "
                f"Payment instructions: {payment_data['payment_instructions']}."
            ),
            metadata={},
        ),
        SourceDocument(
            source_type="temple_policy",
            source_id=f"{temple_id}-temple-policy",
            title=f"{temple_data['temple_name']} app guidance",
            content=(
                "Temple access is granted after admin approval. "
                "Shantidhara slots are open only for the next 30 days at 8:00 AM. "
                "Donations use the temple QR payment flow followed by screenshot submission."
            ),
            metadata={},
        ),
    ]

    for item in news_data.get("items", []):
        documents.append(SourceDocument(
            source_type="news_feed",
            source_id=str(item["news_item_id"]),
            title=str(item["headline"]),
            content=str(item["summary"]),
            metadata={"published_at": item["published_at"]},
        ))

    for item in fame_data.get("items", []):
        documents.append(SourceDocument(
            source_type="wall_of_fame",
            source_id=str(item["fame_item_id"]),
            title=str(item["title"]),
            content=f"Honoree: {item['honoree_name']}. {item['note']}",
            metadata={"created_at": item["created_at"]},
        ))

    return documents


async def _sync_temple_knowledge(temple_id: str) -> None:
    settings = get_settings()
    documents = await _load_temple_source_documents(temple_id)
    snapshot_checksum = _content_checksum([
        {"source_type": d.source_type, "source_id": d.source_id,
         "title": d.title, "content": d.content, "metadata": d.metadata}
        for d in documents
    ])

    with SessionLocal() as session:
        sync_state = session.get(TempleKnowledgeSyncState, temple_id)
        if sync_state and sync_state.last_checksum == snapshot_checksum:
            synced_at = sync_state.synced_at
            if synced_at.tzinfo is None:
                synced_at = synced_at.replace(tzinfo=UTC)
            if synced_at >= datetime.now(UTC) - timedelta(seconds=settings.sync_ttl_seconds):
                return

        session.execute(delete(TempleKnowledgeChunk).where(TempleKnowledgeChunk.temple_id == temple_id))
        session.execute(delete(TempleKnowledgeDocument).where(TempleKnowledgeDocument.temple_id == temple_id))

        chunk_texts: list[str] = []
        chunk_rows: list[tuple[TempleKnowledgeDocument, int, str]] = []

        for document in documents:
            doc_row = TempleKnowledgeDocument(
                document_id=f"{document.source_type}:{document.source_id}",
                temple_id=temple_id,
                source_type=document.source_type,
                source_id=document.source_id,
                title=document.title,
                content=document.content,
                content_checksum=_content_checksum({"title": document.title, "content": document.content}),
                metadata_json=json.dumps(document.metadata, ensure_ascii=True, sort_keys=True),
            )
            session.add(doc_row)
            for idx, chunk in enumerate(_chunk_text(
                document.content,
                chunk_size=settings.chunk_size_characters,
                overlap=settings.chunk_overlap_characters,
            )):
                chunk_rows.append((doc_row, idx, chunk))
                chunk_texts.append(f"{document.title}\n{chunk}")

        embeddings = await embed_texts(chunk_texts, task_type="RETRIEVAL_DOCUMENT")
        for (doc_row, chunk_idx, chunk_content), embedding in zip(chunk_rows, embeddings, strict=False):
            session.add(TempleKnowledgeChunk(
                chunk_id=f"{doc_row.document_id}:{chunk_idx}",
                document_id=doc_row.document_id,
                temple_id=temple_id,
                source_type=doc_row.source_type,
                chunk_index=chunk_idx,
                title=doc_row.title,
                content=chunk_content,
                embedding_json=json.dumps(embedding, ensure_ascii=True),
            ))

        if sync_state is None:
            session.add(TempleKnowledgeSyncState(
                temple_id=temple_id,
                synced_at=datetime.now(UTC),
                last_checksum=snapshot_checksum,
            ))
        else:
            sync_state.synced_at = datetime.now(UTC)
            sync_state.last_checksum = snapshot_checksum
        session.commit()


async def _retrieve_chunks(temple_id: str, query_embedding: list[float]) -> list[RetrievedChunk]:
    settings = get_settings()

    with SessionLocal() as session:
        rows = session.execute(
            select(TempleKnowledgeChunk, TempleKnowledgeDocument.source_id)
            .join(TempleKnowledgeDocument, TempleKnowledgeDocument.document_id == TempleKnowledgeChunk.document_id)
            .where(TempleKnowledgeChunk.temple_id == temple_id)
        ).all()

    ranked: list[RetrievedChunk] = []
    for chunk_row, source_id in rows:
        embedding = json.loads(chunk_row.embedding_json)
        ranked.append(RetrievedChunk(
            chunk_id=chunk_row.chunk_id,
            source_type=chunk_row.source_type,
            source_id=str(source_id),
            title=chunk_row.title,
            content=chunk_row.content,
            similarity=_cosine_similarity(query_embedding, embedding),
        ))

    ranked.sort(key=lambda item: item.similarity, reverse=True)
    return ranked[:settings.retrieval_limit]


async def _retrieve_jain_chunks(query_embedding: list[float]) -> list[dict]:
    """Query Pinecone for relevant Jain text chunks."""
    try:
        index = get_index()
        results = index.query(
            vector=query_embedding,
            top_k=8,  # More chunks for richer scripture answers
            include_metadata=True,
        )
        return results.matches if results.matches else []
    except Exception:
        return []


def _extract_iso_date(message: str) -> str | None:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    return match.group(1) if match else None


async def _run_membership_tool(*, temple_id: str, user_id: str) -> ToolResult | None:
    settings = get_settings()
    body = await _get_json(
        f"{settings.registration_service_url}/api/v1/temple-subscriptions/me?user_id={user_id}"
    )
    items = [i for i in body.get("items", []) if i.get("temple_id") == temple_id]
    if not items:
        return ToolResult(
            source_id=f"{temple_id}-membership", title="Membership status",
            source_type="membership_status",
            excerpt="No approved or pending membership exists for this temple yet.",
            fact="No temple membership record exists yet for this user in the current temple.",
        )
    latest = items[0]
    return ToolResult(
        source_id=str(latest["subscription_id"]), title="Membership status",
        source_type="membership_status",
        excerpt=f"Temple access is currently {latest['status']}.",
        fact=f"Temple membership status for this temple is {latest['status']}.",
    )


async def _run_booking_tool(*, temple_id: str, user_id: str, message: str) -> list[ToolResult]:
    settings = get_settings()
    results: list[ToolResult] = []

    booking_body = await _get_json(
        f"{settings.registration_service_url}/api/v1/temple-subscriptions/shantidhara-bookings/me"
        f"?user_id={user_id}&temple_id={temple_id}"
    )
    items = booking_body.get("items", [])
    if items:
        latest = items[0]
        results.append(ToolResult(
            source_id=str(latest["booking_id"]), title="Latest Shantidhara booking",
            source_type="booking_status",
            excerpt=f"{latest['slot_date']} {latest['slot_label']} - {latest['status']}",
            fact=f"Latest Shantidhara booking is for {latest['slot_date']} at {latest['slot_label']} with status {latest['status']}.",
        ))
    else:
        results.append(ToolResult(
            source_id=f"{temple_id}-booking-status", title="Latest Shantidhara booking",
            source_type="booking_status",
            excerpt="No Shantidhara booking exists yet.",
            fact="No Shantidhara booking exists yet for this user in the current temple.",
        ))

    slot_date = _extract_iso_date(message)
    slot_url = f"{settings.admin_service_url}/api/v1/temples/{temple_id}/shantidhara/slots"
    if slot_date:
        slot_url += f"?slot_date={slot_date}"
    slot_body = await _get_json(slot_url)
    available = [i for i in slot_body.get("items", []) if i.get("status") == "available"]
    if slot_date:
        fact = f"{len(available)} Shantidhara slot(s) available on {slot_date}." if available else f"No slot available on {slot_date}."
        results.append(ToolResult(
            source_id=f"{temple_id}-{slot_date}-slots", title="Shantidhara slot availability",
            source_type="booking_status", excerpt=fact, fact=fact,
        ))
    elif available:
        next_slot = available[0]
        results.append(ToolResult(
            source_id=str(next_slot["slot_id"]), title="Next available Shantidhara slot",
            source_type="booking_status",
            excerpt=f"{next_slot['slot_date']} at {next_slot['slot_label']} is available.",
            fact=f"The next available Shantidhara slot is on {next_slot['slot_date']} at {next_slot['slot_label']}.",
        ))
    return results


async def _run_donation_tool(*, temple_id: str, user_id: str) -> list[ToolResult]:
    settings = get_settings()
    results: list[ToolResult] = []

    donation_body = await _get_json(
        f"{settings.registration_service_url}/api/v1/temple-subscriptions/donations/me"
        f"?user_id={user_id}&temple_id={temple_id}"
    )
    items = donation_body.get("items", [])
    if items:
        latest = items[0]
        results.append(ToolResult(
            source_id=str(latest["donation_id"]), title="Latest donation",
            source_type="donation_status",
            excerpt=f"{latest['amount_label']} - {latest['status']}",
            fact=f"Latest donation is {latest['amount_label']} with status {latest['status']}.",
        ))
    else:
        results.append(ToolResult(
            source_id=f"{temple_id}-donation-status", title="Latest donation",
            source_type="donation_status",
            excerpt="No donation record exists yet.",
            fact="No donation record exists yet for this user in the current temple.",
        ))

    payment_profile = await _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/payment-profile")
    results.append(ToolResult(
        source_id=f"{temple_id}-payment-profile", title="Temple payment profile",
        source_type="payment_profile",
        excerpt=str(payment_profile["payment_instructions"]),
        fact=f"Donations use account {payment_profile['account_label']}. {payment_profile['payment_instructions']}",
    ))
    return results


async def _run_notification_tool(*, temple_id: str) -> list[ToolResult]:
    settings = get_settings()
    news_body, fame_body = await asyncio.gather(
        _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/news-feed"),
        _get_json(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/wall-of-fame"),
    )
    results: list[ToolResult] = []
    if news_body.get("items"):
        latest = news_body["items"][0]
        results.append(ToolResult(
            source_id=str(latest["news_item_id"]), title=str(latest["headline"]),
            source_type="news_feed", excerpt=str(latest["summary"]),
            fact=f"Latest information update: {latest['headline']}. {latest['summary']}",
        ))
    if fame_body.get("items"):
        latest = fame_body["items"][0]
        results.append(ToolResult(
            source_id=str(latest["fame_item_id"]), title=str(latest["title"]),
            source_type="wall_of_fame",
            excerpt=f"{latest['honoree_name']}. {latest['note']}",
            fact=f"Latest wall of fame: {latest['title']} for {latest['honoree_name']}.",
        ))
    return results


async def _run_tools(temple_id: str, request: TempleAssistantRequest) -> list[ToolResult]:
    lowered = request.message.lower()
    single_tasks = []
    list_tasks = []

    if any(t in lowered for t in ["membership", "subscribed", "enrolled", "approval", "status"]):
        single_tasks.append(_run_membership_tool(temple_id=temple_id, user_id=request.user_id))
    if any(t in lowered for t in ["book", "shantidhara", "slot", "calendar"]):
        list_tasks.append(_run_booking_tool(temple_id=temple_id, user_id=request.user_id, message=request.message))
    if any(t in lowered for t in ["donate", "donation", "payment", "qr"]):
        list_tasks.append(_run_donation_tool(temple_id=temple_id, user_id=request.user_id))
    if any(t in lowered for t in ["notice", "update", "news", "information", "wall of fame"]):
        list_tasks.append(_run_notification_tool(temple_id=temple_id))

    results: list[ToolResult] = []
    if single_tasks:
        singles = await asyncio.gather(*single_tasks)
        results.extend(s for s in singles if s is not None)
    if list_tasks:
        grouped = await asyncio.gather(*list_tasks)
        for batch in grouped:
            results.extend(batch)
    return results


async def _generate_with_groq(
    *,
    request: TempleAssistantRequest,
    retrieved_chunks: list[RetrievedChunk],
    tool_results: list[ToolResult],
    jain_matches: list,
) -> str:
    settings = get_settings()

    context_parts: list[str] = []

    # Temple operational context (bookings, donations, news, membership)
    temple_blocks = [f"[{c.source_type}] {c.title}: {c.content}" for c in retrieved_chunks[:4]]
    if tool_results:
        temple_blocks.append("Live temple facts:\n" + "\n".join(f"- {r.fact}" for r in tool_results))
    if temple_blocks:
        context_parts.append("--- TEMPLE INFORMATION ---\n" + "\n\n".join(temple_blocks))

    # Jain scripture context (Agam texts, philosophy, shlokas)
    if jain_matches:
        jain_blocks = [
            f"[Source: {m.metadata.get('source', '?')}, Page {m.metadata.get('page', '?')}]\n{m.metadata.get('text', '')}"
            for m in jain_matches[:6]
        ]
        context_parts.append("--- JAIN TEXTS ---\n" + "\n\n---\n\n".join(jain_blocks))

    context = "\n\n".join(context_parts) if context_parts else "No context available."

    system_prompt = """You are Aagam Mitra — a deeply knowledgeable guide in Jain philosophy, Agam scriptures, temple operations, and Jain history.

You have two knowledge sources:
1. TEMPLE INFORMATION — live data: bookings, donations, membership, news, payment instructions.
2. JAIN TEXTS — passages from Jain Agam scriptures and philosophical texts.

For temple operations questions (booking, donations, my status), use TEMPLE INFORMATION and be practical and direct.
For spiritual, philosophical, or scripture questions, use JAIN TEXTS and give a rich, well-structured answer.

When answering from Jain texts:
- Synthesize the passages into a clear, flowing answer — do not just copy the text
- Use numbered lists or sections for multi-part answers (bhavs, vows, principles, events)
- Explain Sanskrit/Hindi terms briefly where helpful
- Present complete lists in order when asked (e.g., all bhavs, all vows)
- Write in the same language the user asked in
- Be thorough — do not cut short a list or sequence
- Add brief context or explanation where it helps understanding

Do not make up information not present in the passages."""

    # Build message list: system + history + current question with context
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Inject last 10 turns of history so Groq can answer follow-up questions
    for h in request.history[-10:]:
        messages.append({"role": h.role, "content": h.content})

    messages.append({
        "role": "user",
        "content": (
            f"Temple: {request.temple_name or 'Temple'}\n"
            f"Role: {request.role}\n"
            f"Question: {request.message}\n\n"
            f"Context:\n{context}"
        ),
    })

    async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
        response = await client.post(
            _GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": messages,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _build_fallback_message(
    *,
    retrieved_chunks: list[RetrievedChunk],
    tool_results: list[ToolResult],
) -> tuple[str, str]:
    if tool_results:
        return " ".join(r.fact for r in tool_results[:3]), "agent"
    if retrieved_chunks:
        return (
            "Here is the most relevant temple information I found: "
            + " ".join(c.content for c in retrieved_chunks[:2]),
            "retrieval",
        )
    return (
        "I can help with temple notices, Shantidhara booking guidance, donation guidance, and membership status.",
        "fallback",
    )


async def generate_assistant_reply(
    temple_id: str,
    request: TempleAssistantRequest,
) -> TempleAssistantResponse:
    from app.services.agent import run_agent  # local import avoids circular deps

    try:
        final_message = await run_agent(temple_id, request)
        mode = "agent"
    except Exception:
        final_message = "Aagam Mitra is temporarily unavailable. Please try again shortly."
        mode = "fallback"

    # Persist the exchange — best-effort, never crash the response
    try:
        save_exchange(
            user_id=request.user_id,
            temple_id=temple_id,
            user_message=request.message,
            assistant_message=final_message,
        )
    except Exception:
        pass

    return TempleAssistantResponse(
        message=final_message,
        mode=mode,  # type: ignore[arg-type]
        citations=[],
        action_cards=_action_cards_for_message(request.message, request.role),
    )
