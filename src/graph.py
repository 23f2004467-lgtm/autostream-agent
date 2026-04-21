"""LangGraph: state schema, nodes, routing, compiled graph.

Routing is driven by `phase`, not intent. Intent is detected once per
turn and used by `respond` to compose the reply. See plan §2.2 for the
diagram this mirrors.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.llm_util import structured_call
from src.prompts import (
    INTENT_CLASSIFIER_PROMPT,
    LEAD_EXTRACTOR_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    RESPONSE_USER_PROMPT,
)
from src.rag import retrieve_context
from src.schemas import AgentReply, Intent, IntentLabel, LeadInfo, Phase
from src.tools import is_valid_email, mock_lead_capture

logger = logging.getLogger(__name__)

EMPTY_SLOTS: dict[str, str | None] = {"name": None, "email": None, "platform": None}

YES_RE = re.compile(
    r"\b(yes|yep|yup|yeah|sure|submit|confirm|go ahead|go|correct|"
    r"ready|sounds good|looks good|let'?s go)\b",
    re.IGNORECASE,
)
FIX_RE = re.compile(
    r"\b(no|nope|fix|wrong|change|edit|update|actually|not quite|hold on)\b",
    re.IGNORECASE,
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: Phase
    intent: Intent
    lead_slots: dict
    retrieved_context: str
    quick_replies: list[str]
    pending_confirmation: bool


def _last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _format_history(messages: list[BaseMessage], n: int = 6) -> str:
    recent = messages[-n:]
    lines: list[str] = []
    for m in recent:
        role = "User" if getattr(m, "type", None) == "human" else "Agent"
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {text}")
    return "\n".join(lines) or "(none)"


def _slots_with_defaults(state: AgentState) -> dict:
    slots = dict(state.get("lead_slots") or EMPTY_SLOTS)
    for k in EMPTY_SLOTS:
        slots.setdefault(k, None)
    return slots


# ---------- Nodes ----------

def classify_intent_node(state: AgentState) -> dict:
    msgs = state.get("messages", [])
    msg = _last_user_text(msgs)
    phase = state.get("phase", "browsing")
    history = _format_history(msgs[:-1])  # exclude current turn to give true "recent"
    prompt = INTENT_CLASSIFIER_PROMPT.format(
        message=msg, recent_history=history, phase=phase
    )
    result = structured_call(
        IntentLabel, prompt, fallback=IntentLabel(intent="other")
    )
    logger.debug("classify_intent: %r -> %s", msg, result.intent)
    return {"intent": result.intent}


def extract_lead_node(state: AgentState) -> dict:
    msg = _last_user_text(state.get("messages", []))
    intent = state.get("intent", "other")
    current_phase: Phase = state.get("phase", "browsing")
    slots = _slots_with_defaults(state)

    prompt = LEAD_EXTRACTOR_PROMPT.format(
        message=msg, current_slots=slots
    )
    extracted = structured_call(LeadInfo, prompt, fallback=LeadInfo())

    if intent == "correction":
        any_overwrite = False
        for k in ("name", "email", "platform"):
            v = getattr(extracted, k)
            if v is not None:
                slots[k] = v
                any_overwrite = True
        if not any_overwrite:
            # User signalled correction but didn't give the new value yet.
            # Drop back to qualifying so respond asks what to fix.
            return {"lead_slots": slots, "phase": "qualifying"}
    else:
        for k in ("name", "email", "platform"):
            v = getattr(extracted, k)
            if v is not None and slots.get(k) is None:
                slots[k] = v

    # Email validation gate: invalid email → clear so we re-ask politely.
    if slots.get("email") and not is_valid_email(slots["email"]):
        slots["email"] = None

    all_filled = all(slots.get(k) for k in ("name", "email", "platform"))
    email_ok = is_valid_email(slots.get("email"))

    new_phase: Phase = current_phase
    if all_filled and email_ok:
        new_phase = "confirming"
    elif current_phase == "browsing" and intent == "high_intent":
        new_phase = "qualifying"
    elif current_phase == "confirming" and not (all_filled and email_ok):
        # Confirming was destabilised (e.g. correction cleared a field).
        new_phase = "qualifying"

    logger.debug(
        "extract_lead: slots=%s phase %s -> %s (intent=%s)",
        slots, current_phase, new_phase, intent,
    )
    return {"lead_slots": slots, "phase": new_phase}


def retrieve_node(state: AgentState) -> dict:
    query = _last_user_text(state.get("messages", []))
    context = retrieve_context(query, k=3)
    return {"retrieved_context": context}


def capture_node(state: AgentState) -> dict:
    intent = state.get("intent", "other")
    slots = _slots_with_defaults(state)
    msg = _last_user_text(state.get("messages", []))

    # "no, fix this" without an explicit value → drop back to qualifying
    # so respond can ask what they want to fix. Correction-with-value is
    # already handled by extract_lead (it overwrites and leaves phase
    # in confirming so respond re-confirms with the new values).
    if intent == "other" and FIX_RE.search(msg):
        return {"phase": "qualifying"}

    confirms = intent == "high_intent" or (
        intent == "other" and YES_RE.search(msg)
    )
    if confirms and all(slots.get(k) for k in ("name", "email", "platform")) \
            and is_valid_email(slots.get("email")):
        mock_lead_capture(slots["name"], slots["email"], slots["platform"])
        return {"phase": "captured"}

    # Ambiguous or product question mid-confirm — stay, respond re-prompts.
    return {}


def respond_node(state: AgentState) -> dict:
    msgs = state.get("messages", [])
    msg = _last_user_text(msgs)
    phase = state.get("phase", "browsing")
    intent = state.get("intent", "other")
    slots = _slots_with_defaults(state)
    retrieved = state.get("retrieved_context", "") or "(no retrieval this turn)"
    history = _format_history(msgs[:-1])

    # Best-effort flags that help the reply prompt compose tone correctly.
    just_captured = phase == "captured"
    entering_confirming = phase == "confirming"

    user_prompt = RESPONSE_USER_PROMPT.format(
        phase=phase,
        intent=intent,
        lead_slots=slots,
        entering_confirming=entering_confirming,
        just_captured=just_captured,
        retrieved_context=retrieved,
        recent_history=history,
        message=msg,
    )

    fallback = AgentReply(
        reply_text=(
            "Sorry — I had a hiccup. Could you say that again?"
        ),
        quick_replies=[],
    )
    reply = structured_call(
        AgentReply,
        [
            ("system", RESPONSE_SYSTEM_PROMPT),
            ("human", user_prompt),
        ],
        fallback=fallback,
        temperature=0.4,
    )

    # Enforce free-text-only when asking for name or email.
    quick_replies = list(reply.quick_replies or [])
    if phase == "qualifying":
        if slots.get("name") is None or slots.get("email") is None:
            quick_replies = []

    updates: dict = {
        "messages": [AIMessage(content=reply.reply_text)],
        "quick_replies": quick_replies,
    }
    # After a successful capture, drop back to browsing so follow-up
    # questions route normally without trapping the user.
    if phase == "captured":
        updates["phase"] = "browsing"
    return updates


# ---------- Routing ----------

def after_classify(state: AgentState) -> str:
    phase = state.get("phase", "browsing")
    intent = state.get("intent", "other")
    if phase in ("qualifying", "confirming"):
        return "extract_lead"
    if intent == "high_intent":
        return "extract_lead"
    if intent == "product_inquiry":
        return "retrieve"
    return "respond"


def after_extract(state: AgentState) -> str:
    intent = state.get("intent", "other")
    phase = state.get("phase", "browsing")
    if intent == "product_inquiry":
        return "retrieve"
    if phase == "confirming":
        return "capture"
    return "respond"


def after_retrieve(state: AgentState) -> str:
    if state.get("phase") == "confirming":
        return "capture"
    return "respond"


# ---------- Graph assembly ----------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("extract_lead", extract_lead_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("capture", capture_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        after_classify,
        {"extract_lead": "extract_lead", "retrieve": "retrieve", "respond": "respond"},
    )
    g.add_conditional_edges(
        "extract_lead",
        after_extract,
        {"retrieve": "retrieve", "capture": "capture", "respond": "respond"},
    )
    g.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"capture": "capture", "respond": "respond"},
    )
    g.add_edge("capture", "respond")
    g.add_edge("respond", END)

    return g.compile(checkpointer=MemorySaver())


def default_state(user_message: str) -> dict:
    """Initial state dict for a brand-new thread."""
    return {
        "messages": [HumanMessage(content=user_message)],
        "phase": "browsing",
        "intent": "other",
        "lead_slots": dict(EMPTY_SLOTS),
        "retrieved_context": "",
        "quick_replies": [],
        "pending_confirmation": False,
    }


# Compile at import time so callers can `from src.graph import graph`.
graph = build_graph()
