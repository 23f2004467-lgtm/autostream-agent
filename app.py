"""Streamlit chat UI for the AutoStream agent.

Styled to match the AutoStream design system (see `design/shared/tokens.css`):
dark #0A0A0A surfaces, electric-orange #FF5A1F accent, Space Grotesk /
Inter / JetBrains Mono type stack, monospace micro-caps labels.

- `st.chat_input` is ALWAYS visible; free text is never blocked.
- Quick-reply buttons render only beneath the last assistant message.
  Clicking a button is equivalent to typing that label (free-text parity).
- A `processing` flag guards against double-submits while the graph runs.
- A new `thread_id` is minted per session, so the LangGraph checkpointer
  persists conversation state across turns for the whole session.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st

# Streamlit Community Cloud stores secrets in st.secrets (TOML-backed),
# not in os.environ. Promote any secrets to env vars BEFORE importing
# anything that reads GROQ_API_KEY via os.getenv — our config module
# loads env at import time. Local dev still works via .env.
try:
    for _k, _v in dict(st.secrets).items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from langchain_core.messages import HumanMessage  # noqa: E402

from src.graph import default_state, graph  # noqa: E402

st.set_page_config(
    page_title="AutoStream Agent",
    page_icon=":material/bolt:",
    layout="centered",
)


_THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0A0A0A;
  --bg-elev: #141414;
  --bg-elev-2: #1C1C1C;
  --line: #262626;
  --fg: #F5F1EA;
  --fg-dim: #A8A29E;
  --fg-muted: #6B6660;
  --accent: #FF5A1F;
  --accent-ink: #0A0A0A;
  --font-display: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  --font-body: "Inter", "Helvetica Neue", Arial, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
  background: var(--bg) !important;
  color: var(--fg) !important;
  font-family: var(--font-body) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Leave room for the floating inspector panel (top-right, 260px wide). */
@media (min-width: 1100px) {
  [data-testid="stMain"] .block-container {
    max-width: calc(100% - 320px) !important;
    margin-right: 300px !important;
  }
}

::selection { background: var(--accent); color: var(--bg); }

/* Title — big display type */
h1, [data-testid="stHeading"] h1 {
  font-family: var(--font-display) !important;
  font-weight: 600 !important;
  letter-spacing: -0.03em !important;
  color: var(--fg) !important;
}

/* Caption under title */
[data-testid="stCaptionContainer"], .stCaption, [data-testid="stMarkdownContainer"] p small {
  font-family: var(--font-mono) !important;
  color: var(--fg-dim) !important;
  letter-spacing: 0.02em !important;
}

/* Chat messages — user orange bubble, assistant dark card */
[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 6px 0 !important;
  border: none !important;
  gap: 10px !important;
}
[data-testid="stChatMessageContent"] {
  background: var(--bg-elev) !important;
  border: 1px solid var(--line) !important;
  color: var(--fg) !important;
  border-radius: 4px 14px 14px 14px !important;
  padding: 11px 16px !important;
  font-family: var(--font-body) !important;
  font-size: 15px !important;
  line-height: 1.55 !important;
  letter-spacing: -0.005em !important;
  max-width: 78% !important;
  flex-grow: 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: var(--accent) !important;
  color: var(--accent-ink) !important;
  border-color: var(--accent) !important;
  border-radius: 14px 4px 14px 14px !important;
  font-weight: 500 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {
  color: var(--accent-ink) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--accent) !important;
  border-radius: 4px !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,.08), 0 4px 14px rgba(255,90,31,.35) !important;
}
[data-testid="stChatMessageAvatarAssistant"] svg {
  fill: var(--accent-ink) !important;
  color: var(--accent-ink) !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: linear-gradient(140deg, #3a3a3a, #1a1a1a) !important;
  border: 1px solid #333 !important;
  color: var(--fg-dim) !important;
  font-family: var(--font-mono) !important;
}

/* Quick-reply chips — ghost buttons w/ orange outline */
[data-testid="stBaseButton-secondary"] {
  background: var(--bg) !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 999px !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.02em !important;
  padding: 6px 12px !important;
  transition: all .15s !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  background: var(--accent) !important;
  color: var(--accent-ink) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
  background: var(--bg-elev) !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(255,90,31,.15) !important;
}
[data-testid="stChatInput"] textarea {
  color: var(--fg) !important;
  font-family: var(--font-body) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--fg-muted) !important;
}

/* Status spinner ("Thinking…") */
[data-testid="stStatus"] {
  background: var(--bg-elev) !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  color: var(--fg-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

/* Micro-caps kicker above the title */
.kicker {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--accent);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.kicker::before {
  content: '';
  width: 28px;
  height: 2px;
  background: var(--accent);
  display: inline-block;
}

/* ===== Floating inspector panel ===== */
#as-inspector {
  position: fixed;
  top: 72px;
  right: 24px;
  width: 260px;
  z-index: 9999;
  background: #141414;
  border: 1px solid #262626;
  border-radius: 4px;
  box-shadow: 0 24px 60px rgba(0,0,0,.5);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  color: #F5F1EA;
}
#as-inspector .as-toggle {
  width: 100%;
  border: none;
  background: linear-gradient(180deg, #FF5A1F, #E04714);
  color: #0A0A0A;
  padding: 10px 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  font-family: inherit;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
}
#as-inspector .as-pulse {
  width: 6px; height: 6px; border-radius: 50%;
  background: #0A0A0A;
  box-shadow: 0 0 0 3px rgba(10,10,10,.2);
  animation: as-pulse 2s ease-in-out infinite;
}
@keyframes as-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}
#as-inspector .as-title { flex: 1; text-align: left; }
#as-inspector .as-count { font-weight: 500; opacity: .8; letter-spacing: 0.06em; }
#as-inspector .as-arrow { font-size: 10px; margin-left: 4px; }
#as-inspector.collapsed .as-body { display: none; }
#as-inspector .as-body {
  padding: 14px 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
#as-inspector .as-section-label {
  font-size: 9px; letter-spacing: 0.16em; color: #6B6660; margin-bottom: 6px;
}
#as-inspector .as-phase-pill {
  display: inline-block;
  padding: 4px 10px;
  background: rgba(255,90,31,.12);
  color: #FF5A1F;
  border: 1px solid #FF5A1F;
  font-size: 11px; letter-spacing: 0.14em; font-weight: 600;
}
#as-inspector .as-intent { font-size: 12px; color: #F5F1EA; }
#as-inspector .as-slot {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 8px;
  border: 1px solid #262626;
  background: #0A0A0A;
  font-size: 10px;
  margin-top: 3px;
  color: #6B6660;
}
#as-inspector .as-slot.on { border-color: #FF5A1F; color: #FF5A1F; }
#as-inspector .as-slot-label { letter-spacing: 0.12em; }
#as-inspector .as-slot-value {
  max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  text-align: right;
}
#as-inspector .as-leads { display: flex; flex-direction: column; gap: 4px; }
#as-inspector .as-lead-row {
  display: flex; gap: 8px; align-items: baseline;
  padding: 5px 8px; background: #0A0A0A; border: 1px solid #262626;
  font-size: 10px;
}
#as-inspector .as-lead-idx { color: #FF5A1F; font-weight: 600; letter-spacing: 0.1em; }
#as-inspector .as-lead-name { color: #F5F1EA; flex: 1; }
#as-inspector .as-lead-plat { color: #6B6660; letter-spacing: 0.08em; font-size: 9px; }
#as-inspector .as-lead-empty {
  padding: 8px; color: #6B6660; font-size: 10px; font-style: italic;
  border: 1px dashed #262626;
}
@media (max-width: 900px) {
  #as-inspector {
    top: auto; bottom: 90px; right: 12px; width: 220px;
  }
}
</style>
"""


def _escape_dollar(text: str) -> str:
    """Escape `$` so Streamlit's markdown doesn't run MathJax on prices.

    Otherwise "$29/month" between two dollars parses as inline LaTeX and
    renders as an italic fraction — what the user saw on screenshot 1.
    """
    return text.replace("$", r"\$")


def _render_inspector(
    phase: str,
    intent: str,
    slots: dict,
    captured_leads: list[dict],
) -> None:
    """Floating, collapsible state inspector — top-right corner.

    Shows current phase, detected intent, the three lead slots with
    fill state, and a running list of leads captured this session.
    Toggle is pure JS (no Streamlit rerun), so collapsing doesn't
    trigger a graph invocation.
    """
    slot_row = lambda label, value: (  # noqa: E731
        f'<div class="as-slot {"on" if value else ""}">'
        f'<span class="as-slot-label">{label}</span>'
        f'<span class="as-slot-value">{_esc(value) if value else "—"}</span>'
        "</div>"
    )
    leads_rows = ""
    for i, lead in enumerate(reversed(captured_leads[-5:])):
        leads_rows += (
            '<div class="as-lead-row">'
            f'<span class="as-lead-idx">{len(captured_leads) - i:02d}</span>'
            f'<span class="as-lead-name">{_esc(lead.get("name", ""))}</span>'
            f'<span class="as-lead-plat">{_esc(lead.get("platform", ""))}</span>'
            "</div>"
        )
    if not leads_rows:
        leads_rows = '<div class="as-lead-empty">no captures yet this session</div>'

    # CSS is injected once in _inject_theme() — here we only emit the
    # HTML div. Using st.markdown(unsafe_allow_html=True) because
    # st.html sandboxes position:fixed elements into an iframe where
    # they can't escape to the page viewport.
    html = (
        '<div id="as-inspector">'
        '<button class="as-toggle" onclick="'
        "var el = document.getElementById('as-inspector');"
        "el.classList.toggle('collapsed');"
        "var arr = document.getElementById('as-arrow');"
        "arr.textContent = el.classList.contains('collapsed') ? '\\u25B8' : '\\u25BE';"
        '">'
        '<span class="as-pulse"></span>'
        '<span class="as-title">INSPECTOR</span>'
        f'<span class="as-count">{len(captured_leads)} '
        f'{"lead" if len(captured_leads) == 1 else "leads"}</span>'
        '<span id="as-arrow" class="as-arrow">&#9662;</span>'
        '</button>'
        '<div class="as-body">'
        '<div><div class="as-section-label">PHASE</div>'
        f'<div class="as-phase-pill">{phase.upper()}</div></div>'
        '<div><div class="as-section-label">LAST INTENT</div>'
        f'<div class="as-intent">{_esc(intent)}</div></div>'
        '<div><div class="as-section-label">SLOTS</div>'
        + slot_row("NAME", slots.get("name"))
        + slot_row("EMAIL", slots.get("email"))
        + slot_row("PLATFORM", slots.get("platform"))
        + '</div>'
        f'<div><div class="as-section-label">CAPTURED LEADS &middot; {len(captured_leads)}</div>'
        f'<div class="as-leads">{leads_rows}</div></div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _esc(val) -> str:
    import html as _h
    return _h.escape(str(val)) if val is not None else ""


def _inject_theme() -> None:
    # st.html bypasses markdown parsing — critical here because our CSS
    # contains [data-testid="..."] selectors that st.markdown would
    # otherwise interpret as markdown reference-style links and render
    # as visible text.
    st.html(_THEME_CSS)


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role, content)]
    if "quick_replies" not in st.session_state:
        st.session_state.quick_replies = []
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "first_turn" not in st.session_state:
        st.session_state.first_turn = True
    if "phase" not in st.session_state:
        st.session_state.phase = "browsing"
    if "intent" not in st.session_state:
        st.session_state.intent = "other"
    if "slots" not in st.session_state:
        st.session_state.slots = {"name": None, "email": None, "platform": None}
    if "captured_leads" not in st.session_state:
        st.session_state.captured_leads = []  # list[dict]
    if "last_capture_ts" not in st.session_state:
        st.session_state.last_capture_ts = None


def _run_turn(user_text: str) -> dict:
    """Invoke the graph for one user message and return the new state."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    if st.session_state.first_turn:
        payload = default_state(user_text)
        st.session_state.first_turn = False
    else:
        payload = {"messages": [HumanMessage(content=user_text)]}
    return graph.invoke(payload, config=config)


def _queue_input(text: str) -> None:
    if st.session_state.processing:
        return
    st.session_state.pending_input = text
    st.session_state.processing = True


def main() -> None:
    _inject_theme()
    _init_session()

    # Floating inspector — always present, collapsible in-place.
    _render_inspector(
        phase=st.session_state.phase,
        intent=st.session_state.intent,
        slots=st.session_state.slots,
        captured_leads=st.session_state.captured_leads,
    )

    st.html('<div class="kicker">AutoStream · Agent</div>')
    st.title("Ship videos. Not edits.")
    st.caption(
        "Ask about plans, features, pricing, or sign up. "
        "Chips under each reply are shortcuts — typing any free text works the same way."
    )

    # Render chat history. Escape "$" so "$29/month" doesn't hit the
    # Streamlit MathJax renderer and turn into a LaTeX fraction.
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(_escape_dollar(m["content"]))

    # Quick-reply chips under the last assistant message
    if (
        not st.session_state.processing
        and st.session_state.quick_replies
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "assistant"
    ):
        cols = st.columns(min(len(st.session_state.quick_replies), 4))
        for i, label in enumerate(st.session_state.quick_replies):
            with cols[i % len(cols)]:
                st.button(
                    label,
                    key=f"qr_{len(st.session_state.messages)}_{i}",
                    use_container_width=True,
                    on_click=_queue_input,
                    args=(label,),
                )
        st.caption(
            "↑ Shortcuts. Skip the chips and type your own answer anytime."
        )

    # Free-text input — always available
    if typed := st.chat_input(
        "Type anything — or tap a suggestion above",
        disabled=st.session_state.processing,
    ):
        _queue_input(typed)

    # Process the queued input (from either button or free text)
    if st.session_state.pending_input is not None:
        user_text = st.session_state.pending_input
        st.session_state.pending_input = None
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.status("Thinking…", expanded=False):
                try:
                    new_state = _run_turn(user_text)
                    reply = new_state["messages"][-1].content
                    st.session_state.quick_replies = new_state.get(
                        "quick_replies", []
                    )
                    # Mirror graph state into session_state so the
                    # floating inspector reflects reality on the next
                    # rerun. respond_node resets captured → browsing,
                    # so we record the snapshot PRE-reset by reading
                    # last_capture (set by capture_node).
                    st.session_state.phase = new_state.get("phase", "browsing")
                    st.session_state.intent = new_state.get("intent", "other")
                    st.session_state.slots = dict(
                        new_state.get("lead_slots")
                        or {"name": None, "email": None, "platform": None}
                    )
                    capture = new_state.get("last_capture")
                    if (
                        capture
                        and capture.get("ts") != st.session_state.last_capture_ts
                    ):
                        st.session_state.captured_leads.append(capture)
                        st.session_state.last_capture_ts = capture.get("ts")
                except Exception as exc:  # noqa: BLE001
                    reply = (
                        "Sorry — something broke on my end. Try again in a "
                        f"moment. ({type(exc).__name__})"
                    )
                    st.session_state.quick_replies = []
            st.markdown(_escape_dollar(reply))
            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
        st.session_state.processing = False
        st.rerun()


if __name__ == "__main__":
    main()
