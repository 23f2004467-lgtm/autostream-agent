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

# Regex augmentation runs AFTER the LLM extractor as a safety net: Llama 8B
# occasionally returns malformed tool calls that parse to empty LeadInfo
# even when the message clearly contains an email or platform name.
# Regexes are deterministic, so slot fills never silently fail for these
# two fields. (Names still rely on the LLM.)
_EMAIL_PAT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# (regex-literal pattern, canonical value). Patterns are applied with
# re.search over the lowercased message. Ordered from most specific to
# most generic so e.g. "tiktok" doesn't accidentally match inside
# something else. We deliberately DON'T regex for bare "x" because
# it's too ambiguous (shows up inside "xy.com", "Xbox", etc.) —
# "x.com", "X (Twitter)", and "twitter" cover the real cases.
_PLATFORM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\byoutube\b"), "YouTube"),
    (re.compile(r"\byt\b"), "YouTube"),
    (re.compile(r"\binstagram\b"), "Instagram"),
    (re.compile(r"\binsta\b"), "Instagram"),
    (re.compile(r"\big\b"), "Instagram"),
    (re.compile(r"\btiktok\b"), "TikTok"),
    (re.compile(r"\btik\s*tok\b"), "TikTok"),
    (re.compile(r"\btwitch\b"), "Twitch"),
    (re.compile(r"\btwitter\b"), "X (Twitter)"),
    (re.compile(r"(?<![a-z])x\.com\b"), "X (Twitter)"),
    (re.compile(r"\bkick\b"), "Kick"),
    (re.compile(r"\bfacebook\b"), "Facebook"),
    (re.compile(r"\bfb\b"), "Facebook"),
    (re.compile(r"\blinkedin\b"), "LinkedIn"),
    (re.compile(r"\bsnapchat\b"), "Snapchat"),
    (re.compile(r"\bsnap\b"), "Snapchat"),
    (re.compile(r"\bpinterest\b"), "Pinterest"),
    (re.compile(r"\breddit\b"), "Reddit"),
]

# Fallback for name extraction when the LLM returns None. The 8B model
# occasionally ships malformed tool-calls on first-person sentences.
# These patterns catch the most common phrasings.
_NAME_PREFIX_PATTERN = re.compile(
    r"(?:call\s+me|i'?m\s+called|my\s+name\s+is|this\s+is|i\s+am|i'?m)\s*[,:\-]?\s*"
    r"([A-Za-z][A-Za-z'\-]{1,24})",
    re.IGNORECASE,
)
# Bare name pattern: 1-2 capitalized/lowercase alphabetic words, used only
# when phase=qualifying and name slot is still None (i.e., we JUST asked).
_BARE_NAME_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z'\-]{1,24}(?:\s+[A-Za-z][A-Za-z'\-]{1,24})?)\s*[.!]?\s*$")

_NAME_FILLERS = {
    "a", "an", "the", "on", "at", "for", "with", "just", "and", "or",
    "going", "planning", "trying", "sure", "ok", "okay", "yes", "yeah",
    "yep", "yup", "no", "nope", "hi", "hello", "hey", "thanks", "thank",
    "good", "great", "cool", "nice", "basic", "pro", "youtube", "tiktok",
    "instagram", "twitch", "twitter", "submit", "confirm", "done",
    "ready", "sign", "up", "here", "there", "now", "later",
}


def _extract_name_from_message(msg: str, *, phase: str) -> str | None:
    """Pull a name from the user's message, returning None if unsure.

    First tries explicit prefixes ("I'm X", "call me X"). If the user is
    in qualifying phase AND the message is just 1-2 alpha words (a bare
    "Dheeraj" reply to "what should I call you?"), accept that too. We
    skip common non-name fillers so "yes" or "basic" aren't stored as
    names.
    """
    m = _NAME_PREFIX_PATTERN.search(msg)
    if m:
        candidate = m.group(1).strip(" ,.!")
    elif phase == "qualifying":
        bare = _BARE_NAME_PATTERN.match(msg)
        if not bare:
            return None
        candidate = bare.group(1).strip()
    else:
        return None

    first_word = candidate.split()[0].lower()
    if first_word in _NAME_FILLERS:
        return None
    # Title-case if the user typed all upper ("DHEERAJ") or all lower
    # ("dheeraj"). Keep mixed case ("Jainam Desai") as-is.
    if candidate.isupper() or candidate.islower():
        candidate = candidate.title()
    return candidate

# Deterministic fallbacks for quick-reply chips. The responder LLM
# usually generates its own, but it occasionally returns an empty list
# even outside the "asking for name/email" case — in which case the UI
# looks barren. These defaults follow plan §2.4 and kick in only when
# the LLM returned nothing AND the state is one where chips should show.
_DEFAULT_REPLIES: dict[str, list[str]] = {
    "browsing.greeting": [
        "Tell me about pricing",
        "Compare plans",
        "I want to sign up",
    ],
    "browsing.product_inquiry": [
        "Sign me up",
        "Compare plans",
        "What about refunds?",
    ],
    "browsing.objection": [
        "Tell me more",
        "Compare plans",
        "I'll sign up anyway",
    ],
    "browsing.other": [
        "Tell me about pricing",
        "Compare plans",
        "I want to sign up",
    ],
    "qualifying.platform": [
        "YouTube",
        "Instagram",
        "TikTok",
        "Twitch",
    ],
    "confirming": [
        "Yes, submit",
        "Fix something",
    ],
    "captured": [
        "Ask another question",
        "Talk to sales",
        "I'm good, thanks",
    ],
}


# Phrases the LLM frequently falls back to even when they don't answer
# the question it just asked. When the reply ends in "?" and every chip
# is from this set, we blank them out so the UI shows no chips rather
# than disconnected ones.
_GENERIC_CHIP_PHRASES: set[str] = {
    "tell me about pricing",
    "tell me more",
    "compare plans",
    "i want to sign up",
    "sign me up",
    "sign up",
    "pricing",
    "plans",
    "i'll sign up anyway",
    "go pro",
    "go with pro",
    "tell me about pro",
    "tell me about basic",
    "what about refunds?",
    "ask another question",
    "talk to sales",
}


def _default_quick_replies(phase: Phase, intent: Intent, slots: dict) -> list[str]:
    """Pick a sensible chip set when the LLM didn't supply one."""
    if phase == "qualifying":
        if slots.get("name") is None or slots.get("email") is None:
            return []  # free-text-only for unique fields
        if slots.get("platform") is None:
            return _DEFAULT_REPLIES["qualifying.platform"]
        return []
    if phase == "confirming":
        return _DEFAULT_REPLIES["confirming"]
    if phase == "captured":
        return _DEFAULT_REPLIES["captured"]
    key = f"browsing.{intent}"
    return _DEFAULT_REPLIES.get(key, _DEFAULT_REPLIES["browsing.other"])


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
    # Snapshot of `phase` before any node modifies it this turn.
    # Lets capture_node distinguish "user responding to our confirmation
    # prompt" (fire tool) from "extract_lead just entered confirming"
    # (don't fire — respond with the prompt so the user has a chance to
    # confirm or correct first).
    phase_at_turn_start: Phase
    # Populated by capture_node on a successful fire. The UI layer
    # reads this to update its floating "leads captured" inspector in
    # real time. Each entry is a dict of name/email/platform/timestamp.
    last_capture: dict


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
    return {"intent": result.intent, "phase_at_turn_start": phase}


def extract_lead_node(state: AgentState) -> dict:
    msg = _last_user_text(state.get("messages", []))
    intent = state.get("intent", "other")
    current_phase: Phase = state.get("phase", "browsing")
    slots = _slots_with_defaults(state)

    prompt = LEAD_EXTRACTOR_PROMPT.format(
        message=msg, current_slots=slots
    )
    extracted = structured_call(LeadInfo, prompt, fallback=LeadInfo())

    # Regex safety net: the 8B LLM occasionally returns empty LeadInfo
    # even when the message clearly contains these fields (malformed
    # tool-calls). Regexes are deterministic — fill only if the LLM
    # returned None so we never overwrite a good LLM extraction.
    if extracted.email is None:
        m = _EMAIL_PAT.search(msg)
        if m:
            extracted = extracted.model_copy(update={"email": m.group()})
    if extracted.platform is None:
        msg_lc = msg.lower()
        for pattern, canonical in _PLATFORM_PATTERNS:
            if pattern.search(msg_lc):
                extracted = extracted.model_copy(update={"platform": canonical})
                break
    if extracted.name is None:
        candidate = _extract_name_from_message(msg, phase=current_phase)
        if candidate:
            extracted = extracted.model_copy(update={"name": candidate})

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
        import datetime
        return {
            "phase": "captured",
            "last_capture": {
                "name": slots["name"],
                "email": slots["email"],
                "platform": slots["platform"],
                "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
        }

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

    # Post-hoc filter: Llama 8B doesn't always respect the prompt rule
    # "chips must answer the question you just asked". If the reply ends
    # in a question mark AND the chips are all from the generic
    # top-funnel set (pricing, plans, sign up...), drop them — an empty
    # chip row is better than chips that feel disconnected from the
    # question the agent just asked.
    if quick_replies and reply.reply_text.rstrip().endswith("?"):
        lowered = {c.lower().strip() for c in quick_replies}
        if lowered.issubset(_GENERIC_CHIP_PHRASES):
            quick_replies = []

    # Fallback: deterministic chips ONLY where they always make sense
    # regardless of what the LLM asked. Specifically: platform chips
    # during qualifying, and yes/fix chips during confirming. Generic
    # "Tell me about pricing" chips are intentionally NOT used as a
    # fallback anymore — they override whatever question the LLM just
    # asked and make the buttons feel disconnected from the reply.
    if not quick_replies:
        if phase == "confirming":
            quick_replies = _DEFAULT_REPLIES["confirming"]
        elif phase == "captured":
            quick_replies = _DEFAULT_REPLIES["captured"]
        elif (
            phase == "qualifying"
            and slots.get("name") is not None
            and slots.get("email") is not None
            and slots.get("platform") is None
        ):
            quick_replies = _DEFAULT_REPLIES["qualifying.platform"]

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
    """Always route through extract_lead first.

    It's a cheap safety pass that catches names/emails/platforms the
    user may slip into any message ("im dheeraj i do on youtube" was
    classified as 'other' and previously bypassed extraction, leaving
    the slots empty in the inspector even after the user had clearly
    identified themselves). Extraction overwrites nothing when there's
    nothing to find, so running it on every turn is safe. The next
    router (after_extract) handles retrieval / capture / respond.
    """
    return "extract_lead"


def _should_capture(state: AgentState) -> bool:
    """Capture is only valid when the user is answering our confirmation
    prompt — i.e., the turn started in confirming AND we're still there.
    If extract_lead JUST transitioned the phase to confirming, we respond
    with the confirmation prompt first and wait for the next turn.
    """
    return (
        state.get("phase_at_turn_start") == "confirming"
        and state.get("phase") == "confirming"
    )


def after_extract(state: AgentState) -> str:
    if state.get("intent", "other") == "product_inquiry":
        return "retrieve"
    if _should_capture(state):
        return "capture"
    return "respond"


def after_retrieve(state: AgentState) -> str:
    if _should_capture(state):
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
        "phase_at_turn_start": "browsing",
    }


# Compile at import time so callers can `from src.graph import graph`.
graph = build_graph()
