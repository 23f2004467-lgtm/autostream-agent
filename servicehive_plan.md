# ServiceHive ML Intern Assignment — Build Plan

**Project:** AutoStream Social-to-Lead Agentic Workflow
**Deadline:** 2 days from receipt
**Stack decisions (locked):**
- LLM: **Gemini 1.5 Flash** (free tier, solid tool-calling)
- Framework: **LangGraph** (checkpointer = 5–6 turn memory for free)
- RAG: **FAISS + sentence-transformers** (all-MiniLM-L6-v2, local, no embedding API costs)
- UI: **Streamlit chat with quick-reply buttons** (matches WhatsApp/IG interaction model; free text always also accepted)
- Scope: MVP + polish (validation, logging, architecture diagram, clean README)

This doc is the handoff spec for Claude Code. Follow the build order at the bottom.

## Design philosophy (read before coding)

**The agent is an LLM-driven conversationalist first.** Intent classification, slot extraction, RAG, and response generation are all done by Gemini with structured output. The agent must handle nuance, corrections, off-script messages, typos, and multi-field inputs gracefully. A reviewer who tries to "break" it by going off the happy path must still feel like they're talking to something intelligent.

**Quick-reply buttons are a convenience affordance, not a routing mechanism.** They speed up the common path (two taps beats twenty characters of typing) and mirror how WhatsApp/Instagram actually render messages. But the agent never *depends* on a button click. Every state accepts free text. A clicked button is treated identically to the same text being typed — it runs through the classifier like any other message. If buttons vanished entirely, the agent would still work. That's the test.

**Cold-start latency is unacceptable.** The demo must open instantly. The FAISS index and embedding model are pre-built via a warm-up script and the cache is checked into the repo so `streamlit run app.py` is ready in under 2 seconds on first launch. No "loading model…" screens.

---

## 1. What the rubric actually rewards

The six evaluation criteria map to six concrete things the code must show:

| Rubric item | How we win it |
|---|---|
| Agent reasoning & intent detection | Pydantic-structured classifier with a "sticky" lead-collection state so "Jainam" isn't misread as a greeting |
| Correct use of RAG | FAISS index over chunked KB; retrieval only fires on product_inquiry, not on greetings — shows judgment |
| Clean state management | LangGraph `TypedDict` + `MemorySaver` checkpointer keyed by `thread_id` |
| Proper tool calling logic | `mock_lead_capture` only fires when all three slots are filled and email passes regex |
| Code clarity & structure | Modular `src/` layout, type hints, docstrings, separate prompts file |
| Real-world deployability | Streamlit UI + detailed WhatsApp webhook design in README + `.env.example` |

---

## 2. Architecture

### 2.1 State schema

Single source of truth for "where are we in the flow" is `phase`. Intent is still detected per turn (so the respond node knows what the user just asked) but doesn't double as routing state.

```python
Phase = Literal["browsing", "qualifying", "confirming", "captured"]
Intent = Literal["greeting", "product_inquiry", "high_intent", "objection", "correction", "other"]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: Phase                      # "browsing" → "qualifying" → "confirming" → "captured"
    intent: Intent                    # last detected intent (informational, not routing)
    lead_slots: dict                  # {"name": str|None, "email": str|None, "platform": str|None}
    retrieved_context: str            # last RAG snippet
    quick_replies: list[str]          # 2-4 button labels for next turn
    pending_confirmation: bool        # True when all slots filled, awaiting user "yes"
```

**Phase transitions:**
- `browsing` → `qualifying` when intent = `high_intent`
- `qualifying` → `confirming` when all 3 slots are filled and email passes regex
- `confirming` → `captured` when user confirms (via button tap OR free-text "yes"/"submit"/"correct")
- `confirming` → `qualifying` if user says "no" / "fix" / "actually…" (correction intent)
- `captured` → `browsing` after confirmation response is sent (so follow-up questions route normally)

### 2.2 Graph (nodes + routing)

Routing is driven by `phase`, not intent. Intent is detected once at the top and used by `respond` to compose the reply.

```
user message ──▶ classify_intent ──▶ (phase router)
                                          │
       ┌──────────────┬─────────────┬─────┴──────────┐
       ▼              ▼             ▼                ▼
  [browsing]    [qualifying]   [confirming]     [captured]
       │              │             │                │
       │         extract_lead       │                │
       │              │             │                │
       │              ▼             ▼                ▼
   (maybe         (if all 3   (check user intent:  respond
    retrieve if    filled +    yes → capture_lead    │
    product_       email       no/correction →       │
    inquiry)       valid →     back to qualifying)   │
       │           → confirming)       │             │
       ▼              │                ▼             │
     respond       respond          respond          │
       │              │                │             │
       └──────────────┴────────────────┴─────────────┘
                             ▼
                            END
```

**Key routing rules:**
1. When `phase == "qualifying"` we skip the classifier's authority over routing — regardless of intent label, the turn runs through `extract_lead_info` first. This prevents "Jainam" from being misread as a greeting and dropping the flow. Intent is still detected so the response prompt knows whether the user is providing info, raising an objection, or going off-topic during collection.
2. `extract_lead_info` only overwrites slots that are `None` UNLESS intent = `correction` — in which case it overwrites the specified slot with the new value.
3. `retrieve` fires only on `phase == "browsing"` AND intent = `product_inquiry`. Never on greetings. Never mid-collection.
4. `capture_lead` fires exactly once, on the transition `confirming → captured` after user consent.

### 2.3 Node responsibilities

Every LLM call uses a **structured-output helper** (`src/llm_util.py: structured_call(schema, prompt)`) that wraps `with_structured_output` in a try/except. On `ValidationError` or malformed JSON, it falls back to raw JSON mode + manual Pydantic parse. On second failure, it returns a sensible default (intent=`other`, empty slot extraction, etc.) and logs. One flaky call never crashes a turn.

- **classify_intent** — always runs. Returns one of `{greeting, product_inquiry, high_intent, objection, correction, other}`. Passes recent 4 messages for context so "yes" after "should I try Pro?" resolves to high_intent.
- **retrieve** — runs only when `phase == "browsing"` and intent = `product_inquiry`. FAISS top-3 cosine similarity (embeddings normalized at index build time). Concatenates hits into `retrieved_context`.
- **extract_lead_info** — runs when `phase == "qualifying"` OR intent = `high_intent` (to catch the intent-shift message that mentions platform). Extracts all three fields in one pass — critical for "I'm Jainam, jainam@distill.fyi, YouTube" working on a single turn. Handles corrections (overwrite on intent = `correction`).
- **capture_lead** — calls `mock_lead_capture(name, email, platform)` with the exact signature from the brief. Only reachable from `confirming` phase after user consent. Sets `phase = "captured"`.
- **respond** — composes the reply AND generates 2–4 contextual quick-reply button labels in one structured call. Inputs: phase, intent, retrieved context (if any), current slots, whether we're entering `confirming` (so it renders the "confirm or fix" prompt with buttons), whether we just captured (so it renders the success confirmation).

### 2.4 Quick-reply button sets (generated per turn, optional)

Buttons are generated by the LLM in the `respond` node as a hint for the user. They're rendered in Streamlit as clickable chips, but every button label is just text — clicking it is equivalent to typing that label. The classifier handles it the same way either way.

| Phase / last intent | Typical button set |
|---|---|
| `browsing`, after greeting | "Tell me about pricing", "Compare plans", "I want to sign up" |
| `browsing`, after RAG fired on pricing | "Go with Pro", "Tell me about Basic", "What about refunds?", "Sign me up" |
| `browsing`, after objection | "Tell me more", "Compare plans", "I'll sign up anyway" |
| `qualifying`, asking for name | *empty list — free text only* |
| `qualifying`, asking for email | *empty list — free text only* |
| `qualifying`, asking for platform | "YouTube", "Instagram", "TikTok", "Twitch" (free text still accepted) |
| `confirming` | "Yes, submit", "Fix something" |
| `captured` | "Ask another question", "Talk to sales", "I'm good, thanks" |

**Hard rule enforced by test:** every state must also accept free text. Test case `test_free_text_parity` types "pro plan please" into a state where buttons are shown and asserts intent = `high_intent`. If that test fails, we've slipped into decision-tree territory and the build blocks.

---

## 3. Project structure

```
autostream-agent/
├── README.md                  # setup + architecture (200 words) + WhatsApp answer
├── requirements.txt
├── .env.example               # GOOGLE_API_KEY=
├── .gitignore                 # .env, __pycache__ (NOT .faiss_cache — that's committed)
├── architecture.png           # exported from mermaid
├── data/
│   └── knowledge_base.md      # AutoStream pricing, features, policies
├── .faiss_cache/              # COMMITTED — pre-built index for instant cold start
│   ├── index.faiss
│   └── index.pkl
├── scripts/
│   └── warmup.py              # pre-builds FAISS index; re-run only if KB changes
├── src/
│   ├── __init__.py
│   ├── config.py              # env loading, model name, paths
│   ├── llm_util.py            # structured_call() wrapper with JSON-mode fallback
│   ├── knowledge_base.py      # loads + chunks the MD file
│   ├── rag.py                 # FAISS loader + retriever (loads from .faiss_cache/)
│   ├── prompts.py             # intent, extraction, response templates
│   ├── schemas.py             # Pydantic: Intent, Phase, LeadInfo, AgentReply
│   ├── tools.py               # mock_lead_capture + email validation
│   └── graph.py               # LangGraph state + nodes + compiled graph
├── app.py                     # Streamlit chat UI with quick-reply buttons
├── main.py                    # CLI fallback
└── tests/
    └── test_flow.py           # see §10 for exact test list
```

---

## 4. Knowledge base (data/knowledge_base.md)

Structure each fact as its own chunk so retrieval stays precise:

```markdown
# AutoStream Knowledge Base

## Basic Plan
AutoStream's Basic Plan costs $29/month. It includes 10 videos per month
at up to 720p resolution. Best for hobbyist creators testing the platform.

## Pro Plan
AutoStream's Pro Plan costs $79/month. It includes unlimited video edits,
4K resolution export, and AI-generated captions. 24/7 customer support is
included. Best for full-time content creators and small studios.

## Refund Policy
AutoStream does not issue refunds after 7 days from purchase. Within the
first 7 days, full refunds are available on request.

## Support Policy
24/7 customer support is available exclusively on the Pro plan. Basic plan
users have access to email support during business hours.

## Product Overview
AutoStream is a SaaS platform for automated video editing aimed at content
creators on YouTube, TikTok, Instagram, and similar platforms.
```

Chunker: split on `##` headings, one chunk per section. Gives 5 clean chunks.

---

## 5. Prompt designs (prompts.py)

### 5.1 Intent classifier

```
You are an intent classifier for a sales agent at AutoStream
(a SaaS for automated video editing).

Classify the user's latest message into exactly one label:
- greeting: social pleasantries ("hi", "hello", "how are you")
- product_inquiry: questions about features, pricing, plans, refunds, support
- high_intent: user signals readiness to sign up / buy / try / subscribe,
  or specifies a plan for their use case ("let me try Pro for YouTube")
- objection: user pushes back on price, commitment, or a feature
  ("that's expensive", "I need to think", "not sure it's worth it")
- correction: user updates info they previously gave
  ("actually my email is X", "no wait, I meant Instagram")
- other: off-topic, garbled, or unclear

Return only the label. If ambiguous, use the recent conversation as context:
a short message like "yes" after the agent asked "ready to sign up?" is high_intent.

User message: {message}
Recent conversation: {recent_history}
Current phase: {phase}
```

### 5.2 Lead info extractor

```
Extract any of {name, email, platform} present in the user's message.
Return null for fields not explicitly stated. Do NOT guess.

Important:
- Extract ALL fields present in a single message. If the user writes
  "I'm Jainam, jainam@distill.fyi, I do YouTube", extract all three.
- Platform means the user's creator platform (YouTube, Instagram, TikTok,
  Twitch, Twitter, Kick, etc.) — NOT a plan name like "Pro" or "Basic".
- If the user is correcting a previous value ("actually my email is X",
  "I meant Instagram"), still extract the new value. The caller decides
  whether to overwrite based on intent label.
- Names can be first-name-only ("Jainam") or full ("Jainam Desai"). Both
  are valid. Do not infer a last name.

Message: {message}
Current slots (for context only): {current_slots}
```

### 5.3 Response generator (structured output)

Single prompt with conditional sections injected based on state. Tone: warm,
concise, no emojis unless the user uses them. Returns:

```python
class AgentReply(BaseModel):
    reply_text: str
    quick_replies: list[str] = Field(
        default_factory=list,
        description="2-4 short button labels (<=25 chars) for likely next actions. "
                    "Empty when asking for name or email (free text only)."
    )
```

**Hallucination guard (mandatory in the system prompt):**
> Only state facts that appear in the retrieved context or that the user
> has told you. If asked about a feature, price, or policy not in the
> context, say: "I'm not sure off the top — let me flag this for our team
> to confirm." Never invent numbers, plan names, or features.

**Phase-specific response guidance injected into the prompt:**
- `browsing` + `greeting` → warm greeting, invite them to ask about pricing/features. Buttons: top-funnel options.
- `browsing` + `product_inquiry` → answer using `retrieved_context`; never go beyond it.
- `browsing` + `objection` → acknowledge the concern, respond using KB facts (e.g., price objection → mention 7-day refund; commitment objection → mention Basic plan as try-before-upgrade). Buttons: include "I'll sign up anyway" to convert.
- `browsing` + `other` → "I didn't quite catch that — I can help with AutoStream's plans, features, or getting you set up. What sounds useful?" + top-funnel buttons. Never dead-end.
- `qualifying` → ask naturally for whichever slot is `None`. One at a time feels human. If user just provided info but a slot is still missing, acknowledge + ask for the next. Buttons only when asking for platform.
- `confirming` → "Just to confirm — {name}, {email}, {platform}. Ready to submit?" Buttons: "Yes, submit", "Fix something".
- `captured` → brief success message acknowledging the capture, invite further questions. Buttons: follow-up options.

**Quick-reply generation rule:** 2–4 labels, ≤25 chars each, action-oriented, mutually distinct. Empty list when asking for name or email (typing is faster than tapping for unique fields).

---

## 6. Lead-capture state machine details

Slots: `{"name": None, "email": None, "platform": None}`. Phase transitions live on top of these.

### Per-turn logic (when phase is `qualifying` or `confirming`)

1. **Extract** — always run `extract_lead_info` on the message. Even if intent is `objection` or `other`, the user might have slipped in their name. Update slots:
   - If intent ≠ `correction`: only fill `None` slots (never overwrite).
   - If intent = `correction`: overwrite the specific slot mentioned in the new extraction.
2. **Validate email** — regex `r"^[\w.+-]+@[\w-]+\.[\w.-]+$"`. If email field is filled but regex fails, clear it and the response will ask again with the correction reason ("that doesn't look quite right, mind double-checking?").
3. **Check completeness:**
   - If any slot still `None` → stay in `qualifying`, ask for what's missing (one at a time). Response should acknowledge what we just received if applicable.
   - If all three filled + email valid → transition `qualifying → confirming`, respond with the confirmation prompt + "Yes, submit" / "Fix something" buttons.
4. **Handle confirming phase:**
   - User intent = `high_intent` or message matches /yes|submit|confirm|correct|go/ → `confirming → captured`, call `mock_lead_capture(...)`, respond with success.
   - User intent = `correction` or message matches /no|fix|wrong|change/ → `confirming → qualifying`, response asks what they want to fix (buttons: "Name", "Email", "Platform").
   - Any other intent during `confirming` → stay in `confirming`, gently re-prompt the confirmation question. Don't fire the tool.
5. **After capture:** `captured → browsing` so follow-up product questions route normally. Agent does NOT re-enter qualifying unless new high_intent is detected (don't trap the user).

### Graceful off-script handling

- If intent = `other` during `qualifying` ("what's the weather?") → acknowledge briefly + redirect: "Haha, I'll stick to AutoStream — still need your email to wrap this up." Maintain slots, don't reset.
- If user goes silent mid-collection (irrelevant in Streamlit but relevant for WhatsApp) → no action; state persists on next message thanks to the checkpointer.
- If user asks a product question mid-collection ("wait, does Pro include AI captions?") → intent classifier catches it. We run retrieve + answer the question, THEN re-prompt the missing slot. This is the hardest case and the one that proves the agent is "smart" not "decision tree". Include it as a test.

### The tool call

```python
# src/tools.py — exact signature and print string from the brief
def mock_lead_capture(name: str, email: str, platform: str) -> None:
    print(f"Lead captured successfully: {name}, {email}, {platform}")
```

Called exactly once per session, from the `capture_lead` node, on the `confirming → captured` transition.

---

## 7. Memory (5–6 turn requirement)

LangGraph's `MemorySaver` checkpointer keyed by a `thread_id` gives us full history for free. No manual buffer.

Streamlit: generate `thread_id = str(uuid.uuid4())` on `st.session_state` init. Pass as `config={"configurable": {"thread_id": st.session_state.thread_id}}` on every graph invocation.

The `AgentState.messages` list with `add_messages` reducer accumulates history across turns. Include the last 6 messages in intent/response prompts.

---

## 8. README.md outline

1. **Project title + 2-line description**
2. **Demo GIF** (optional, from screen recording)
3. **Setup** — `python -m venv`, `pip install -r requirements.txt`, `cp .env.example .env`, add `GOOGLE_API_KEY`, `streamlit run app.py`
4. **Architecture (≈200 words)** — draft below
5. **Project structure** (tree)
6. **WhatsApp deployment answer** — draft below
7. **Evaluation notes** — brief pointer to where each rubric item is handled

### 8.1 Architecture section (draft, ~200 words)

> AutoStream Agent is a LangGraph-based conversational agent that moves users
> from product questions to captured leads through a single state machine.
> LangGraph was chosen over plain LangChain chains because the task is
> inherently stateful — the agent must remember partial lead info across
> multiple turns and decide each turn whether to retrieve, extract, call a
> tool, or just reply. LangGraph's typed state and conditional edges make that
> routing explicit and debuggable, versus burying it in implicit chain logic.
>
> State is a `TypedDict` holding the message history, current intent, a
> sticky `in_lead_collection` flag, three lead slots, and the last retrieved
> RAG context. LangGraph's `MemorySaver` checkpointer persists this state
> under a per-session `thread_id`, giving us multi-turn memory (far beyond
> the required 5–6 turns) without any manual buffer code.
>
> Intent classification uses Gemini 1.5 Flash with Pydantic-structured
> output to produce a single label; routing then dispatches to `retrieve`
> (FAISS + all-MiniLM-L6-v2 over a chunked knowledge base), to
> `extract_lead_info`, or directly to `respond`. The mock lead-capture tool
> is gated on all three slots being filled and the email passing regex
> validation, so it cannot fire prematurely.

### 8.2 WhatsApp deployment answer (draft)

> **Integration outline:**
>
> 1. **Meta WhatsApp Business Cloud API** — register a business phone number,
>    get `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`, set a webhook URL and a
>    `VERIFY_TOKEN`.
> 2. **Webhook endpoint** — expose a FastAPI server with two handlers on
>    `/webhook`:
>    - `GET` — verification handshake: echo back `hub.challenge` when
>      `hub.verify_token` matches our `VERIFY_TOKEN`.
>    - `POST` — incoming message handler. Parse `entry[].changes[].value.messages[]`,
>      extract the sender's phone number and message text.
> 3. **Map phone → thread_id** — use the sender's `wa_id` (phone number) as
>    the LangGraph `thread_id`. This gives each user a persistent conversation
>    state with no extra code.
> 4. **Run the graph** — call `graph.invoke({"messages":[HumanMessage(text)]},
>    config={"configurable": {"thread_id": wa_id}})` and take the final
>    assistant message.
> 5. **Reply with interactive buttons** — `POST` to
>    `https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages`. For plain
>    text: `{"messaging_product":"whatsapp","to":wa_id,"type":"text",
>    "text":{"body":reply}}`. For quick replies, use the `interactive` type:
>    ```json
>    {"messaging_product":"whatsapp","to":wa_id,"type":"interactive",
>     "interactive":{"type":"button","body":{"text":reply},
>       "action":{"buttons":[
>         {"type":"reply","reply":{"id":"btn_1","title":"Tell me more"}},
>         {"type":"reply","reply":{"id":"btn_2","title":"Compare plans"}}
>       ]}}}
>    ```
>    For sets of 4+ options, use `type:"list"` instead of `type:"button"`.
>    Incoming button taps arrive as webhook events with `messages[].type ==
>    "interactive"` and `interactive.button_reply.title` containing the
>    tapped label — feed that `title` into the graph as a normal user
>    message. Our `quick_replies` state field maps one-to-one onto this
>    WhatsApp contract with zero agent-code changes.
> 6. **Security** — validate the `X-Hub-Signature-256` header (HMAC-SHA256 of
>    the raw body using the app secret) on every POST; reject mismatches.
> 7. **Scale & reliability** — move graph invocation behind a Celery/RQ
>    worker so the webhook returns `200` in <5s (Meta retries on timeout);
>    swap `MemorySaver` for `PostgresSaver`/`RedisSaver` so state survives
>    restarts; log every inbound/outbound message.
>
> No change to the agent code itself — only the transport layer.

---

## 9. requirements.txt (pinned to compatible ranges)

```
langgraph~=0.2.50
langchain~=0.3.10
langchain-google-genai~=2.0.5
langchain-community~=0.3.10
langchain-huggingface~=0.1.0
faiss-cpu~=1.9.0
sentence-transformers~=3.3.0
pydantic~=2.9
python-dotenv~=1.0
streamlit~=1.40
pytest~=8.3
```

The `~=` operator pins to a minor range, protecting against LangGraph's frequent breaking minor releases while still allowing patch updates. Verify with `pip install -r requirements.txt` in a fresh venv before pushing.

---

## 10. Build order (hand this to Claude Code as the task list)

1. **Scaffold** — create dir tree, `requirements.txt` (§9, pinned), `.env.example`, `.gitignore` (excludes `.env`, `__pycache__` — NOT `.faiss_cache`). Stub `README.md`. Verify `pip install -r requirements.txt` completes in a fresh venv.
2. **Knowledge base** — `data/knowledge_base.md` from §4. `src/knowledge_base.py`: `load_chunks()` splits on `## ` and returns `list[Document]` with heading in metadata.
3. **LLM util** — `src/llm_util.py`: `structured_call(schema, prompt, fallback)` wrapping `with_structured_output` with try/except → JSON mode fallback → fallback value. All LLM calls go through this.
4. **RAG + warm-up** — `src/rag.py`: `load_retriever()` reads from `.faiss_cache/`, returns a retriever. If cache missing, raise with a clear "run `python scripts/warmup.py`" message (never builds on import). Use `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})`. `scripts/warmup.py` builds the FAISS index and saves to `.faiss_cache/`. Run it once and commit the cache files. Result: `streamlit run` is instant.
5. **Schemas + prompts** — `src/schemas.py` (`Intent`, `Phase`, `LeadInfo`, `AgentReply`), `src/prompts.py` (content from §5 including hallucination guard and phase-specific guidance).
6. **Tools** — `src/tools.py`: `mock_lead_capture(name, email, platform)` with the exact signature and print string from the brief. `is_valid_email(s)`.
7. **Graph** — `src/graph.py`: `AgentState` TypedDict with `phase`, nodes from §2.3, routing from §2.2, `MemorySaver` checkpointer. Export compiled `graph`.
8. **Streamlit UI** — `app.py`: session-scoped `thread_id` via `uuid4()`. Chat history from `st.session_state.messages`. `st.chat_input` always visible (free text never disabled). Quick-reply buttons render as `st.columns` of `st.button`s only beneath the LAST assistant message. Clicking a button writes label to `st.session_state.pending_input`; on next rerun, consumed as the user message. `st.session_state.processing` flag guards against button double-clicks while the graph is running. Show `st.status("Thinking…")` while graph runs.
9. **CLI** — `main.py`: REPL against the same graph. Same `thread_id` for the whole session.
10. **Tests** — `tests/test_flow.py` with these cases (each asserted):
    - `test_intent_greeting` — "hi there" → greeting
    - `test_intent_product_inquiry` — "how much is Pro" → product_inquiry
    - `test_intent_high_intent` — "I want to sign up" → high_intent
    - `test_intent_objection` — "that's too expensive" → objection
    - `test_extract_email` — "my email is foo@bar.com" → email extracted
    - `test_extract_all_fields_single_message` — "I'm Jainam, jainam@distill.fyi, I do YouTube" → all 3 slots filled in one call (**critical test**)
    - `test_platform_inferred_on_intent_shift` — "I want Pro for my YouTube channel" extracts platform=YouTube (matches §4 of brief)
    - `test_correction_overwrites_slot` — after filling email, "actually it's x@y.com" overwrites
    - `test_free_text_parity` — typing "tell me about pricing" works identically to clicking a button with that label
    - `test_product_question_mid_collection` — during qualifying, asking "does Pro have 4K?" is answered, then collection resumes
    - `test_tool_not_fired_prematurely` — graph does not call `mock_lead_capture` until all 3 slots filled + user confirmed
    - `test_tool_fires_exactly_once` — confirmed happy path fires the tool exactly once
    - `test_hallucination_guard` — "do you have a $5 plan?" → response does NOT invent a plan
    - `test_memory_across_six_turns` — a fact mentioned on turn 1 is referenced correctly on turn 6
11. **README polish** — all sections from §8, Mermaid architecture diagram. Add "Beyond the brief" section calling out objection intent, quick-replies, confirmation step as production-oriented extensions.
12. **Record demo** — script in §12.
13. **Push to GitHub** — public repo, incremental commits (NOT one giant commit). Share repo URL + video link via email to ServiceHive.

Estimated time for a focused session: 5–7 hours (extra hour vs. prior estimate for the added tests and warm-up).

---

## 11. Gotchas / things that will bite

- **Cold start is solved via committed cache.** The `.faiss_cache/` directory is checked into git (it's ~5MB). `scripts/warmup.py` rebuilds it whenever the KB changes. Streamlit never builds the index on import — if the cache is missing it raises with a clear error telling you to run warmup. This guarantees sub-2-second startup.
- **All LLM calls go through `structured_call()`** which handles malformed JSON gracefully. Never call `with_structured_output` directly from a node.
- **MemorySaver needs a `thread_id`** on every `.invoke()` / `.stream()` call. Missing it silently loses history.
- **Phase, not intent, drives routing.** When phase is `qualifying`, the turn runs through extract_lead_info first regardless of intent label.
- **Email regex validates before the tool fires.** Invalid email → clear slot → ask again. Tool is never called with bad data.
- **FAISS on Apple Silicon** — `faiss-cpu` sometimes needs `pip install faiss-cpu --no-cache-dir`. Flag in README troubleshooting.
- **API key hygiene** — `.env` in `.gitignore`, only `.env.example` committed, no key in code or in the recorded video.
- **Button click race** — `st.session_state.processing` guard prevents double-submit while graph is running.

---

## 12. Demo video script (2:30)

| Time | Beat | Dialogue / action |
|---|---|---|
| 0:00–0:10 | Title card | "AutoStream Agent — LangGraph + Gemini Flash + FAISS RAG" |
| 0:10–0:25 | Show project tree in VS Code | Narrate key files: graph.py, rag.py, tools.py |
| 0:25–0:40 | Streamlit open, greet | Type "Hi there" → friendly greeting + 3 quick-reply buttons render |
| 0:40–1:00 | Product inquiry (via button tap) | Click "Tell me about pricing" button → RAG retrieves both plans → new buttons render |
| 1:00–1:20 | Free-text still works | Type "what about refunds?" manually → shows free text path still works alongside buttons |
| 1:20–1:40 | Intent shift (button tap) | Click "Sign me up" → agent asks for name (no buttons — free text field focused) |
| 1:40–2:00 | Provide name + email | Type "Jainam, jainam@distill.fyi" → extractor pulls both → agent asks for platform with buttons |
| 2:00–2:15 | Platform (button tap) | Click "YouTube" → all 3 slots filled |
| 2:15–2:25 | Tool fires | Terminal shows: `Lead captured successfully: Jainam, jainam@distill.fyi, YouTube` + chat confirmation |
| 2:25–2:30 | Close | Quick flash of README WhatsApp section (showing interactive message JSON) |

---

## 13. Sample conversation (paste into tests)

```
User: Hi
Agent: Hey! I'm AutoStream's assistant. Want to hear about our plans, features, or pricing?

User: What does the Pro plan include?
Agent: The Pro Plan is $79/month and includes unlimited video edits, 4K export, AI captions, and 24/7 customer support — it's built for full-time creators.

User: That sounds great, I want to sign up for my YouTube channel.
Agent: Love that! I just need a few details to get you set up. What's your name and email?

User: Jainam, jainam@distill.fyi
Agent: Perfect. And just to confirm — YouTube is your primary platform, right?

User: Yes
Agent: All set, Jainam — we've captured your details and someone from the AutoStream team will reach out shortly.
[terminal] Lead captured successfully: Jainam, jainam@distill.fyi, YouTube
```

---

## 14. Submission checklist

- [ ] GitHub repo public (or access granted to ServiceHive reviewer)
- [ ] README has setup, architecture (~200 words), WhatsApp answer, project structure, "Beyond the brief" section
- [ ] `.env.example` present; real `.env` gitignored; `.faiss_cache/` IS committed
- [ ] `requirements.txt` pinned — verify with `pip install -r requirements.txt` in a fresh venv
- [ ] `scripts/warmup.py` runs successfully; cache committed; `streamlit run app.py` opens in <2s
- [ ] `mock_lead_capture(name, email, platform)` signature and print string match brief exactly
- [ ] Tool only fires after all 3 slots filled, email valid, AND user confirms
- [ ] All 14 tests in §10 pass
- [ ] Intent classifier handles the 3 required categories (greeting, product_inquiry, high_intent) plus objection/correction/other
- [ ] RAG answers "Basic vs Pro" and "refund" questions from the KB without hallucinating
- [ ] Multi-turn memory demonstrated across 6+ turns, with a turn that references earlier context
- [ ] Free-text parity test passes — agent never depends on button clicks
- [ ] Demo video 2–3 min shows all 4 required beats + a free-text path + tool firing in terminal
- [ ] Incremental commit history (not one giant initial commit)
- [ ] Record demo on the pushed version; share repo URL + video link via email to ServiceHive
