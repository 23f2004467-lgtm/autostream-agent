"""Pydantic schemas and typed literals used across the graph."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Phase = Literal["browsing", "qualifying", "confirming", "captured"]
Intent = Literal[
    "greeting",
    "product_inquiry",
    "high_intent",
    "objection",
    "correction",
    "other",
]


class IntentLabel(BaseModel):
    """Single-label intent classification output."""

    intent: Intent


class LeadInfo(BaseModel):
    """Fields extracted from a user message. Any absent field is None."""

    name: str | None = None
    email: str | None = None
    platform: str | None = None


class AgentReply(BaseModel):
    """Response-node output: the reply text plus 0–4 quick-reply buttons."""

    reply_text: str
    quick_replies: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 short button labels (<=25 chars) for likely next actions. "
            "Empty when asking for name or email (free text only)."
        ),
    )
