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

_SYSTEM_PROMPT = (
    f"You are Aagam Mitra — a knowledgeable Jain temple assistant. Today is {date.today().isoformat()}. "
    "Always use the available tools to fetch real data before answering. "
    "For Shantidhara booking: first call get_shantidhara_slots, show the slots, ask for karta_name if missing, then book. "
    "Match the user's language (Hindi → Hindi, English → English). "
    "For lists like bhavs or shlokas, use numbered format and be thorough."
)


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
                "temperature": 0.3,
            },
        )
        r.raise_for_status()
    return r.json()


async def run_agent(temple_id: str, request: TempleAssistantRequest) -> str:
    tools = _build_tools()
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

        elif finish_reason in ("stop", "length", None):
            return (assistant_msg.get("content") or "").strip()
        else:
            break

    return "I was unable to complete your request. Please try again."
