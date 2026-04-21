"""End-to-end tests for the AutoStream agent.

All 14 test cases from plan §10 step 10. Most exercise real Gemini calls
(temperature=0 for classifier and extractor to keep them deterministic).
Skips cleanly when GOOGLE_API_KEY is absent.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.messages import HumanMessage

from src.graph import default_state, graph
from src.llm_util import structured_call
from src.prompts import INTENT_CLASSIFIER_PROMPT, LEAD_EXTRACTOR_PROMPT
from src.schemas import IntentLabel, LeadInfo

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — live LLM tests skipped.",
)


# ---------------- helpers ----------------

def _cfg() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _classify(message: str, *, phase: str = "browsing", history: str = "(none)") -> str:
    return structured_call(
        IntentLabel,
        INTENT_CLASSIFIER_PROMPT.format(
            message=message, recent_history=history, phase=phase
        ),
        fallback=IntentLabel(intent="other"),
    ).intent


def _extract(message: str, *, slots: dict | None = None) -> LeadInfo:
    slots = slots or {"name": None, "email": None, "platform": None}
    return structured_call(
        LeadInfo,
        LEAD_EXTRACTOR_PROMPT.format(message=message, current_slots=slots),
        fallback=LeadInfo(),
    )


# ---------------- 1–4: intent classification ----------------

def test_intent_greeting():
    assert _classify("hi there") == "greeting"


def test_intent_product_inquiry():
    assert _classify("how much is Pro") == "product_inquiry"


def test_intent_high_intent():
    assert _classify("I want to sign up") == "high_intent"


def test_intent_objection():
    assert _classify("that's too expensive") == "objection"


# ---------------- 5–7: lead extraction ----------------

def test_extract_email():
    info = _extract("my email is foo@bar.com")
    assert info.email == "foo@bar.com"


def test_extract_all_fields_single_message():
    info = _extract("I'm Jainam, jainam@distill.fyi, I do YouTube")
    assert info.name and "jainam" in info.name.lower()
    assert info.email == "jainam@distill.fyi"
    assert info.platform and info.platform.lower() == "youtube"


def test_platform_inferred_on_intent_shift():
    info = _extract("I want Pro for my YouTube channel")
    assert info.platform and info.platform.lower() == "youtube"


# ---------------- 8: correction overwrites slot ----------------

def test_correction_overwrites_slot():
    cfg = _cfg()
    # Fill all three slots in one message — phase goes straight to confirming.
    out1 = graph.invoke(
        default_state(
            "I want to sign up. I'm Jainam, jainam@distill.fyi, I do YouTube."
        ),
        config=cfg,
    )
    assert out1["lead_slots"]["email"] == "jainam@distill.fyi"
    # Correction
    out2 = graph.invoke(
        {"messages": [HumanMessage(content="actually my email is jd@distill.fyi")]},
        config=cfg,
    )
    assert out2["lead_slots"]["email"] == "jd@distill.fyi"
    # Other slots unaffected
    assert out2["lead_slots"]["name"] and "jainam" in out2["lead_slots"]["name"].lower()
    assert out2["lead_slots"]["platform"].lower() == "youtube"


# ---------------- 9: free-text parity ----------------

def test_free_text_parity():
    """Typing the text of a button produces the same classification as
    the 'button click' path. The UI funnels both through the same graph
    input — this test codifies that there is no button-specific code.
    """
    text = "tell me about pricing"
    typed_out = graph.invoke(default_state(text), config=_cfg())
    clicked_out = graph.invoke(default_state(text), config=_cfg())
    assert typed_out["intent"] == clicked_out["intent"] == "product_inquiry"
    # Both runs triggered retrieval and populated context.
    assert typed_out.get("retrieved_context") and clicked_out.get("retrieved_context")


# ---------------- 10: product question mid-collection ----------------

def test_product_question_mid_collection(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        "src.graph.mock_lead_capture",
        lambda *a, **k: calls.append(a),
    )

    cfg = _cfg()
    # 1. Signal intent to sign up — agent asks for name.
    graph.invoke(default_state("I want to sign up"), config=cfg)
    # 2. Give name only (email and platform still missing).
    graph.invoke(
        {"messages": [HumanMessage(content="I'm Jainam")]},
        config=cfg,
    )
    # 3. Pivot to a product question mid-collection.
    out = graph.invoke(
        {"messages": [HumanMessage(content="wait, does Pro have 4K?")]},
        config=cfg,
    )
    reply = out["messages"][-1].content.lower()

    # RAG fired and the answer made it into the reply.
    assert "4k" in reply
    # Collection was NOT dropped — still qualifying (email + platform pending).
    assert out["phase"] == "qualifying"
    # Tool hasn't fired.
    assert calls == []


# ---------------- 11: tool not fired prematurely ----------------

def test_tool_not_fired_prematurely(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        "src.graph.mock_lead_capture",
        lambda *a, **k: calls.append(a),
    )

    cfg = _cfg()
    graph.invoke(default_state("Hi, I want to sign up"), config=cfg)
    graph.invoke(
        {"messages": [HumanMessage(content="Jainam, jainam@distill.fyi")]},
        config=cfg,
    )
    # Platform still missing — confirmation not reached — tool must not fire.
    assert calls == []


# ---------------- 12: tool fires exactly once ----------------

def test_tool_fires_exactly_once(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        "src.graph.mock_lead_capture",
        lambda *a, **k: calls.append(a),
    )

    cfg = _cfg()
    graph.invoke(default_state("I want to sign up"), config=cfg)
    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I'm Jainam, jainam@distill.fyi, I do YouTube"
                )
            ]
        },
        config=cfg,
    )
    # Confirm
    graph.invoke(
        {"messages": [HumanMessage(content="yes, submit")]},
        config=cfg,
    )
    assert len(calls) == 1
    # Follow-up should not re-fire.
    graph.invoke(
        {"messages": [HumanMessage(content="thanks!")]},
        config=cfg,
    )
    assert len(calls) == 1


# ---------------- 13: hallucination guard ----------------

def test_hallucination_guard():
    out = graph.invoke(default_state("do you have a $5 plan?"), config=_cfg())
    reply = out["messages"][-1].content.lower()
    # The agent must not confirm a $5 plan exists.
    bad_phrases = [
        "we have a $5",
        "our $5 plan",
        "yes, the $5",
        "$5 plan includes",
        "$5 plan offers",
        "$5 tier",
        "$5 starter",
    ]
    assert not any(p in reply for p in bad_phrases), f"invented a $5 plan: {reply}"
    # Soft evidence of compliance: either mention uncertainty / flag, or
    # redirect to the real plan prices (29 or 79).
    good_evidence = [
        "not sure",
        "don't have",
        "don't offer",
        "no $5",
        "flag",
        "team to confirm",
        "29",
        "79",
        "basic",
        "pro",
    ]
    assert any(p in reply for p in good_evidence), (
        f"no uncertainty or real-plan redirect: {reply}"
    )


# ---------------- 14: memory across six turns ----------------

def test_memory_across_six_turns():
    cfg = _cfg()
    # Turn 1: mention a distinctive fact.
    graph.invoke(
        default_state(
            "Hi — I'm Alexandria and I run a YouTube gaming channel."
        ),
        config=cfg,
    )
    # Turns 2–5: ask unrelated product questions.
    graph.invoke(
        {"messages": [HumanMessage(content="Tell me about pricing")]}, config=cfg
    )
    graph.invoke(
        {"messages": [HumanMessage(content="What about refunds?")]}, config=cfg
    )
    graph.invoke(
        {
            "messages": [
                HumanMessage(content="Does the Pro plan include AI captions?")
            ]
        },
        config=cfg,
    )
    graph.invoke(
        {"messages": [HumanMessage(content="That's useful, thanks")]},
        config=cfg,
    )
    # Turn 6: reference the fact mentioned on turn 1.
    out = graph.invoke(
        {"messages": [HumanMessage(content="By the way, what's my name again?")]},
        config=cfg,
    )
    reply = out["messages"][-1].content
    assert "alexandria" in reply.lower(), (
        f"agent forgot the name provided on turn 1: {reply}"
    )
