"""CLI REPL for the AutoStream agent. Useful for quick smoke tests."""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage

from src.graph import default_state, graph


def _format_quick_replies(labels: list[str]) -> str:
    if not labels:
        return ""
    return "  (quick replies: " + " | ".join(labels) + ")"


def run() -> None:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print("AutoStream Agent — CLI. Ctrl-C or `quit` to exit.\n")

    first_turn = True
    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"quit", "exit"}:
                break

            payload = (
                default_state(user_text)
                if first_turn
                else {"messages": [HumanMessage(content=user_text)]}
            )
            first_turn = False
            state = graph.invoke(payload, config=config)
            reply = state["messages"][-1].content
            qr = state.get("quick_replies") or []
            print(f"Agent: {reply}{_format_quick_replies(qr)}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")


if __name__ == "__main__":
    run()
