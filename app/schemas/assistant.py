from typing import Literal

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TempleAssistantRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=32)
    role: Literal["devotee", "admin"]
    message: str = Field(..., min_length=2, max_length=2000)
    temple_name: str | None = Field(default=None, max_length=160)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=20)


class TempleAssistantCitation(BaseModel):
    source_id: str
    title: str
    source_type: Literal[
        "temple_profile",
        "news_feed",
        "wall_of_fame",
        "payment_profile",
        "temple_policy",
        "booking_status",
        "donation_status",
        "membership_status",
        "jain_text",
    ]
    excerpt: str


class TempleAssistantActionCard(BaseModel):
    action_id: str
    title: str
    description: str
    action_label: str
    action_target: Literal["home", "book", "donate", "chat", "admin"]


class TempleAssistantResponse(BaseModel):
    message: str
    mode: Literal["retrieval", "tool", "agent", "fallback"]
    citations: list[TempleAssistantCitation]
    action_cards: list[TempleAssistantActionCard]
    phase: str = "temple_ai"
