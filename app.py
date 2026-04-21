"""Streamlit chat UI for the AutoStream agent.

- `st.chat_input` is ALWAYS visible; free text is never blocked.
- Quick-reply buttons render only beneath the last assistant message.
  Clicking a button is equivalent to typing that label (free-text parity).
- A `processing` flag guards against double-submits while the graph runs.
- A new `thread_id` is minted per session, so the LangGraph checkpointer
  persists conversation state across turns for the whole session.
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from src.graph import default_state, graph

st.set_page_config(page_title="AutoStream Agent", page_icon=":speech_balloon:")


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
    _init_session()

    st.title("AutoStream Agent")
    st.caption(
        "Ask about plans, features, pricing, or sign up. "
        "Quick-reply buttons are shortcuts — typing works identically."
    )

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

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

    # Free-text input — always available
    if typed := st.chat_input(
        "Type a message…", disabled=st.session_state.processing
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
                except Exception as exc:  # noqa: BLE001
                    reply = (
                        "Sorry — something broke on my end. Try again in a "
                        f"moment. ({type(exc).__name__})"
                    )
                    st.session_state.quick_replies = []
            st.markdown(reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
        st.session_state.processing = False
        st.rerun()


if __name__ == "__main__":
    main()
