"""
Aagam Mitra Agent — Phase 1
Uses Groq function calling to handle all user requests:
  - Jain scripture Q&A (Pinecone RAG)
  - Temple info, news
  - Shantidhara slot availability + booking
  - My existing bookings
  - Membership status
"""

from __future__ import annotations

import asyncio
import json
from datetime import date

import httpx

from app.core.config import get_settings
from app.schemas.assistant import TempleAssistantRequest
from app.services.embedder import embed_texts
from app.services.vector_store import get_index

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_MAX_ITERATIONS = 6

_SYSTEM_PROMPT = f"""You are Aagam Mitra — a knowledgeable, thoughtful Jain temple assistant with deep expertise in Jain philosophy and temple operations. Today is {date.today().isoformat()}.

**CORE PRINCIPLES:**
1. Always use available tools to fetch REAL DATA before answering
2. SYNTHESIZE information — never copy passages verbatim
3. Provide CONTEXT before details (explain significance and deeper meaning)
4. Structure answers with clear sections, numbering, or bullet points
5. Include TRANSLITERATION for Sanskrit/Prakrit terms with English meanings
6. For spiritual/philosophical questions: explain concepts thoroughly with examples

**FOR JAIN SCRIPTURE QUESTIONS:**
- Lead with the concept/context (why this matters)
- Provide the relevant shloka, bhav, or sutra if applicable
- Explain the literal meaning and deeper philosophical significance
- Connect to practical application in spiritual life
- Use numbered lists when covering multiple bhavs/concepts
- Example structure:
  1. Context/Background
  2. The Sacred Text (with transliteration)
  3. Meaning Explained
  4. Deeper Spiritual Insight
  5. How to Apply This

**FOR TEMPLE OPERATIONS:**
- Be direct and actionable
- Always provide dates/times in clear format
- Give next steps explicitly
- Confirm understanding

**RESPONSE QUALITY GUIDELINES:**
- Minimum 100 words for scripture questions (depth matters)
- Use subheadings/numbering for clarity
- Include the why, not just the what
- Add practical wisdom when relevant

**LANGUAGE:**
- Match user's language: Hindi input → respond in Hindi, English input → respond in English
- Use Devanagari transliteration for Sanskrit terms
- Provide English translations inline

**FOR BOOKING/OPERATIONS:**
- For Shantidhara: first call get_shantidhara_slots, display options clearly, ask for karta_name, then book
- Always confirm details before final action
- Provide booking reference and next steps"""


def _build_tools() -> list[dict]:
    today = date.today().isoformat()
    return [
        {
            "type": "function",
            "function": {
                "name": "search_jain_texts",
                "description": (
                    "Search the Jain Agam scripture knowledge base. "
                    "Use for any question about Jain philosophy, shlokas, bhavs, sutras, vows, or stories."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_shantidhara_slots",
                "description": "Get available Shantidhara slots. Without slot_date returns all upcoming available slots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot_date": {
                            "type": "string",
                            "description": f"Specific date YYYY-MM-DD. Today is {today}. Omit to get all upcoming slots.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_shantidhara_slot",
                "description": (
                    "Book a Shantidhara slot. "
                    "Only call AFTER showing available slots AND receiving karta_name from the user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string", "description": "The slot_id from get_shantidhara_slots"},
                        "karta_name": {"type": "string", "description": "Full name of the person performing the ritual"},
                        "occasion": {"type": "string", "description": "Occasion or purpose (e.g. Varshitap Parana, Birthday). Optional."},
                    },
                    "required": ["slot_id", "karta_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_my_bookings",
                "description": "Get the user's existing Shantidhara bookings.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_membership_status",
                "description": "Check whether the user is an approved, pending, or non-member of this temple.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_temple_news",
                "description": "Get the latest news, announcements, and updates from the temple.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_temple_info",
                "description": "Get general temple information: name, location, status, and payment instructions.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        # --- Phase 2 tools ---
        {
            "type": "function",
            "function": {
                "name": "get_events",
                "description": "Get upcoming temple events.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_wall_of_fame",
                "description": "Get the temple wall of fame — recognition and honour posts for devotees.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_membership_request",
                "description": (
                    "Submit a request to join the temple as a member. "
                    "Only call after confirming the user's full name (requester_name)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requester_name": {
                            "type": "string",
                            "description": "Full name of the person requesting membership",
                        },
                    },
                    "required": ["requester_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_booking",
                "description": (
                    "Cancel an unpaid Shantidhara booking. "
                    "Only works for pending or proof_submitted bookings — not paid or confirmed ones. "
                    "First call get_my_bookings to find the booking_id if the user doesn't provide it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "booking_id": {"type": "string", "description": "The booking_id to cancel"},
                    },
                    "required": ["booking_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_feedback",
                "description": (
                    "Submit a rating and optional feedback comment for the temple. "
                    "Ask the user for a rating (1-5) and an optional comment before calling."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_name": {
                            "type": "string",
                            "description": "Display name of the user submitting feedback",
                        },
                        "overall_rating": {
                            "type": "integer",
                            "description": "Rating from 1 (poor) to 5 (excellent)",
                        },
                        "comment": {
                            "type": "string",
                            "description": "Optional comment or review text",
                        },
                    },
                    "required": ["user_name", "overall_rating"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _tool_search_jain_texts(query: str) -> dict:
    try:
        embedding = (await embed_texts([query], task_type="RETRIEVAL_QUERY"))[0]
        index = get_index()
        results = index.query(vector=embedding, top_k=8, include_metadata=True)
        matches = results.matches or []
        if not matches:
            return {"found": False, "message": "No relevant passages found in the Jain texts knowledge base."}
        return {
            "found": True,
            "passages": [
                {
                    "source": m.metadata.get("source", "?"),
                    "page": m.metadata.get("page"),
                    "text": m.metadata.get("text", ""),
                    "score": round(m.score, 3),
                }
                for m in matches
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_get_shantidhara_slots(temple_id: str, slot_date: str | None) -> dict:
    settings = get_settings()
    url = f"{settings.admin_service_url}/api/v1/temples/{temple_id}/shantidhara/slots"
    params = {"slot_date": slot_date} if slot_date else {}
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
        available = [s for s in r.json().get("items", []) if s.get("status") == "available"]
        if not available:
            return {"available": False, "message": "No available Shantidhara slots found."}
        return {
            "available": True,
            "slots": [
                {
                    "slot_id": s["slot_id"],
                    "date": s["slot_date"],
                    "label": s["slot_label"],
                    "amount": s.get("amount_label", ""),
                    "pratima": s.get("pratima_name", ""),
                }
                for s in available[:10]
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_book_shantidhara_slot(
    temple_id: str, user_id: str, slot_id: str, karta_name: str, occasion: str
) -> dict:
    settings = get_settings()
    url = f"{settings.registration_service_url}/api/v1/temple-subscriptions/shantidhara-bookings"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.post(url, json={
                "user_id": user_id,
                "temple_id": temple_id,
                "slot_id": slot_id,
                "karta_name": karta_name,
                "occasion": occasion or "",
            })
            r.raise_for_status()
        b = r.json()
        return {
            "success": True,
            "booking_id": b.get("booking_id"),
            "date": b.get("slot_date"),
            "label": b.get("slot_label"),
            "karta_name": b.get("karta_name"),
            "occasion": b.get("occasion"),
            "amount": b.get("amount_label"),
            "temple_name": b.get("temple_name"),
        }
    except httpx.HTTPStatusError as exc:
        detail = "Unable to complete booking."
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        return {"success": False, "error": detail}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _tool_get_my_bookings(temple_id: str, user_id: str) -> dict:
    settings = get_settings()
    url = (
        f"{settings.registration_service_url}/api/v1/temple-subscriptions"
        f"/shantidhara-bookings/me?user_id={user_id}&temple_id={temple_id}"
    )
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.get(url)
            r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return {"found": False, "message": "No Shantidhara bookings found."}
        return {
            "found": True,
            "bookings": [
                {
                    "booking_id": b.get("booking_id"),
                    "date": b.get("slot_date"),
                    "label": b.get("slot_label"),
                    "karta_name": b.get("karta_name"),
                    "occasion": b.get("occasion"),
                    "status": b.get("status"),
                    "amount": b.get("amount_label"),
                }
                for b in items[:10]
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_get_membership_status(temple_id: str, user_id: str) -> dict:
    settings = get_settings()
    url = f"{settings.registration_service_url}/api/v1/temple-subscriptions/me?user_id={user_id}"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.get(url)
            r.raise_for_status()
        items = [i for i in r.json().get("items", []) if i.get("temple_id") == temple_id]
        if not items:
            return {"status": "not_a_member", "message": "No membership found for this temple."}
        latest = items[0]
        return {"status": latest.get("status"), "temple_name": latest.get("temple_name")}
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_get_temple_news(temple_id: str) -> dict:
    settings = get_settings()
    url = f"{settings.admin_service_url}/api/v1/temples/{temple_id}/news-feed"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.get(url)
            r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return {"found": False, "message": "No news or announcements yet."}
        return {
            "found": True,
            "news": [
                {"headline": i.get("headline"), "summary": i.get("summary"), "date": i.get("published_at")}
                for i in items[:5]
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_get_temple_info(temple_id: str) -> dict:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            temple_r, payment_r = await asyncio.gather(
                client.get(f"{settings.admin_service_url}/api/v1/temples/{temple_id}"),
                client.get(f"{settings.admin_service_url}/api/v1/temples/{temple_id}/payment-profile"),
            )
        temple = temple_r.json() if temple_r.status_code == 200 else {}
        payment = payment_r.json() if payment_r.status_code == 200 else {}
        return {
            "name": temple.get("temple_name"),
            "location": temple.get("temple_location"),
            "status": temple.get("status"),
            "payment_instructions": payment.get("payment_instructions"),
            "account_label": payment.get("account_label"),
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_cancel_booking(user_id: str, booking_id: str) -> dict:
    settings = get_settings()
    url = (
        f"{settings.registration_service_url}/api/v1/temple-subscriptions"
        f"/shantidhara-bookings/{booking_id}/cancel?user_id={user_id}"
    )
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.post(url)
            r.raise_for_status()
        return {"success": True, "message": "Booking cancelled successfully. The slot has been freed."}
    except httpx.HTTPStatusError as exc:
        detail = "Unable to cancel booking."
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        return {"success": False, "error": detail}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _tool_get_events(temple_id: str) -> dict:
    settings = get_settings()
    url = f"{settings.admin_service_url}/api/v1/temples/{temple_id}/events"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.get(url)
            r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return {"found": False, "message": "No upcoming events found."}
        return {
            "found": True,
            "events": [
                {
                    "title": e.get("title"),
                    "summary": e.get("summary"),
                    "date": e.get("event_date"),
                    "location": e.get("location"),
                }
                for e in items[:5]
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_get_wall_of_fame(temple_id: str) -> dict:
    settings = get_settings()
    url = f"{settings.admin_service_url}/api/v1/temples/{temple_id}/wall-of-fame"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.get(url)
            r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return {"found": False, "message": "No wall of fame entries yet."}
        return {
            "found": True,
            "entries": [
                {
                    "title": e.get("title"),
                    "honoree": e.get("honoree_name"),
                    "note": e.get("note"),
                    "date": e.get("created_at"),
                }
                for e in items[:5]
            ],
        }
    except Exception as exc:
        return {"error": str(exc)}


async def _tool_submit_membership_request(temple_id: str, user_id: str, requester_name: str) -> dict:
    settings = get_settings()
    url = f"{settings.registration_service_url}/api/v1/temple-subscriptions"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.post(url, json={
                "user_id": user_id,
                "temple_id": temple_id,
                "requester_name": requester_name,
            })
            r.raise_for_status()
        result = r.json()
        return {
            "success": True,
            "subscription_id": result.get("subscription_id"),
            "status": result.get("status"),
            "message": "Your membership request has been submitted. An admin will review it shortly.",
        }
    except httpx.HTTPStatusError as exc:
        detail = "Unable to submit membership request."
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        return {"success": False, "error": detail}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _tool_submit_feedback(
    temple_id: str, user_id: str, user_name: str, overall_rating: int, comment: str
) -> dict:
    settings = get_settings()
    url = f"{settings.admin_service_url}/api/v1/temples/{temple_id}/feedback"
    try:
        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            r = await client.post(url, json={
                "user_id": user_id,
                "user_name": user_name,
                "overall_rating": max(1, min(5, overall_rating)),
                "comment": comment or "",
            })
            r.raise_for_status()
        return {
            "success": True,
            "message": f"Thank you for your feedback! Your rating of {overall_rating}/5 has been submitted.",
        }
    except httpx.HTTPStatusError as exc:
        detail = "Unable to submit feedback."
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        return {"success": False, "error": detail}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _execute_tool(temple_id: str, user_id: str, tool_call: dict) -> dict:
    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except (json.JSONDecodeError, KeyError):
        args = {}

    dispatch = {
        "search_jain_texts":    lambda: _tool_search_jain_texts(args.get("query", "")),
        "get_shantidhara_slots": lambda: _tool_get_shantidhara_slots(temple_id, args.get("slot_date")),
        "book_shantidhara_slot": lambda: _tool_book_shantidhara_slot(
            temple_id, user_id,
            slot_id=args.get("slot_id", ""),
            karta_name=args.get("karta_name", ""),
            occasion=args.get("occasion", ""),
        ),
        "get_my_bookings":        lambda: _tool_get_my_bookings(temple_id, user_id),
        "get_membership_status":  lambda: _tool_get_membership_status(temple_id, user_id),
        "get_temple_news":           lambda: _tool_get_temple_news(temple_id),
        "get_temple_info":           lambda: _tool_get_temple_info(temple_id),
        # Phase 2
        "cancel_booking":            lambda: _tool_cancel_booking(user_id, args.get("booking_id", "")),
        "get_events":                lambda: _tool_get_events(temple_id),
        "get_wall_of_fame":          lambda: _tool_get_wall_of_fame(temple_id),
        "submit_membership_request": lambda: _tool_submit_membership_request(
            temple_id, user_id, args.get("requester_name", "")
        ),
        "submit_feedback":           lambda: _tool_submit_feedback(
            temple_id, user_id,
            user_name=args.get("user_name", ""),
            overall_rating=int(args.get("overall_rating", 3)),
            comment=args.get("comment", ""),
        ),
    }

    handler = dispatch.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return await handler()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def _call_groq(messages: list[dict], tools: list[dict]) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            _GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": getattr(settings, 'groq_temperature', 0.5),
            },
        )
        r.raise_for_status()
    return r.json()


def _format_tool_result_for_synthesis(tool_name: str, result: dict) -> str:
    """Format tool results to be more synthesizable by the LLM

    Provides context and structure that helps the model understand and
    synthesize the data rather than copy it verbatim.
    """
    if "error" in result:
        return f"[Error from {tool_name}: {result['error']}]"

    if tool_name == "search_jain_texts":
        if not result.get("found"):
            return "[No relevant Jain scripture passages found]"

        passages = result.get("passages", [])
        formatted = f"[Retrieved {len(passages)} relevant passages from Jain texts for synthesis]\n"
        for i, p in enumerate(passages[:8], 1):  # Max 8 chunks
            formatted += f"\n{i}. From '{p['source']}':\n"
            formatted += f"   Text: {p['text']}\n"
            formatted += f"   Relevance: {p['score']:.2f}\n"
        formatted += "\n[Use these passages to synthesize a comprehensive, insightful answer]"
        return formatted

    elif tool_name == "get_shantidhara_slots":
        if not result.get("available"):
            return "[No available Shantidhara slots found]"

        slots = result.get("slots", [])
        formatted = f"[Found {len(slots)} available Shantidhara slots]\n"
        for s in slots:
            formatted += f"\n• {s['label']} - {s.get('amount', 'N/A')}\n"
            formatted += f"  Pratima: {s.get('pratima', 'Not specified')}"
        return formatted

    elif tool_name == "get_my_bookings":
        if not result.get("found"):
            return "[No Shantidhara bookings found]"

        bookings = result.get("bookings", [])
        formatted = f"[Found {len(bookings)} bookings]\n"
        for b in bookings:
            formatted += f"\n• {b['label']} - Status: {b['status']}\n"
            formatted += f"  Karta: {b['karta_name']}, Amount: {b['amount']}"
        return formatted

    elif tool_name == "get_temple_news":
        if not result.get("found"):
            return "[No temple news available]"

        news_items = result.get("news", [])
        formatted = f"[Found {len(news_items)} temple news items]\n"
        for item in news_items[:5]:
            formatted += f"\n• {item['title']}\n"
            formatted += f"  {item['content'][:200]}...\n"
        return formatted

    elif tool_name == "get_temple_info":
        if "error" in result:
            return f"[Could not retrieve temple info: {result['error']}]"

        formatted = f"[Temple Information]\n"
        formatted += f"• Name: {result.get('name', 'N/A')}\n"
        formatted += f"• Address: {result.get('address', 'N/A')}\n"
        formatted += f"• Hours: {result.get('hours', 'N/A')}\n"
        formatted += f"• Phone: {result.get('phone', 'N/A')}\n"
        formatted += f"• Email: {result.get('email', 'N/A')}"
        return formatted

    # Default: pretty-print JSON for other tools
    return json.dumps(result, indent=2, ensure_ascii=False)


async def run_agent(temple_id: str, request: TempleAssistantRequest) -> str:
    tools = _build_tools()
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Add few-shot examples to teach the model what good synthesis looks like
    messages.extend([
        {
            "role": "user",
            "content": "What are the key bhavs of Aadinath Bhagwan?"
        },
        {
            "role": "assistant",
            "content": """Aadinath Bhagwan, the first Tirthankara, represents the pinnacle of spiritual evolution in Jainism. His bhavs (divine qualities/spiritual states) exemplify the path to liberation:

**1. Samata Bhav (Equanimity)**
The foundational quality of perfect balance and non-attachment. Aadinath embodies complete detachment from worldly attractions and repulsions, maintaining perfect equanimity in all circumstances.

**2. Uttama Kshama Bhav (Supreme Forbearance)**
Transcendent patience and forgiveness. He accepts all trials without anger or resentment, never retaliating even when provoked.

**3. Uttama Satya Bhav (Perfect Truthfulness)**
Complete adherence to truth in thought, word, and deed. His every utterance aligns with absolute reality and dharma.

**4. Anukula Bhav (Favorable Dispositions)**
Actions perfectly aligned with Jain principles. Every action contributes to spiritual progress and never causes harm.

**5. Kevala Darshan (Infinite Vision)**
Supreme omniscient perception - ability to see all past, present, and future simultaneously without any obstruction.

**6. Kevala Jnana (Infinite Knowledge)**
Perfect, all-encompassing knowledge of every aspect of existence. This is the ultimate fruit of spiritual practice.

These bhavs are not merely abstract concepts but stages of spiritual transformation that culminate in becoming a Kevalin (fully liberated soul). They guide devotees on their own paths to enlightenment."""
        }
    ])

    # Inject conversation history for follow-up question support
    for h in request.history[-10:]:
        messages.append({"role": h.role, "content": h.content})

    messages.append({"role": "user", "content": request.message})

    for _ in range(_MAX_ITERATIONS):
        response = await _call_groq(messages, tools)
        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason")
        assistant_msg = choice["message"]

        if finish_reason == "tool_calls":
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls", [])
            # Execute all tool calls in parallel
            results = await asyncio.gather(
                *[_execute_tool(temple_id, request.user_id, tc) for tc in tool_calls]
            )
            for tc, result in zip(tool_calls, results):
                # Format tool result to help LLM synthesize better
                tool_name = tc["function"]["name"]
                formatted_content = _format_tool_result_for_synthesis(tool_name, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": formatted_content,
                })

        elif finish_reason in ("stop", "length", None):
            return (assistant_msg.get("content") or "").strip()
        else:
            break

    return "I was unable to complete your request. Please try again."
