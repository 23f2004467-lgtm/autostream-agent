"""Gemini wrapper with graceful structured-output handling.

Every LLM call in this project goes through `structured_call`. It layers:

1. `with_structured_output(schema)` — the happy path; LangChain parses.
2. Raw JSON mode — on failure, re-issue the prompt with a JSON
   instruction and parse the response string with Pydantic.
3. Hardcoded fallback value — if both attempts fail, return the caller's
   default so a single flaky LLM call never crashes a turn.

Never raises. The caller always gets a valid schema instance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from src.config import GEMINI_MODEL

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_LLM: ChatGoogleGenerativeAI | None = None


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Lazy-init and cache the Gemini client."""
    global _LLM
    if _LLM is None or _LLM.temperature != temperature:
        _LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
    return _LLM


def _normalize(prompt: str | list[BaseMessage]) -> list[BaseMessage]:
    if isinstance(prompt, str):
        return [HumanMessage(content=prompt)]
    return list(prompt)


def _strip_json_fences(text: str) -> str:
    """Pull the JSON body out of markdown fences if Gemini wrapped the output."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _extract_text(raw) -> str:
    """Gemini's .content is usually str, but occasionally a list of parts."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        buf: list[str] = []
        for part in raw:
            if isinstance(part, dict):
                buf.append(part.get("text", ""))
            else:
                buf.append(str(part))
        return "".join(buf)
    return str(raw)


def structured_call(
    schema: Type[T],
    prompt: str | list[BaseMessage],
    fallback: T,
    temperature: float = 0.0,
) -> T:
    """Invoke Gemini with Pydantic-structured output, falling back through JSON mode.

    Args:
        schema: Pydantic model to parse the response into.
        prompt: Either a plain user prompt string, or a list of BaseMessages.
        fallback: Returned verbatim if both the structured and JSON-mode
            paths fail. Must already be a valid instance of `schema`.
        temperature: Sampling temperature. Defaults to 0 for classifiers
            and extractors; bump for creative responses.

    Returns:
        A `schema` instance — never raises.
    """
    llm = get_llm(temperature=temperature)
    messages = _normalize(prompt)

    try:
        return llm.with_structured_output(schema).invoke(messages)
    except (ValidationError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning(
            "structured_call: with_structured_output failed for %s: %s",
            schema.__name__,
            exc,
        )

    try:
        json_instruction = (
            "\n\nIMPORTANT: Respond with a single valid JSON object matching "
            "this schema (field names and types must match exactly):\n"
            f"{json.dumps(schema.model_json_schema())}\n"
            "Return ONLY the JSON object. No prose. No markdown fences."
        )
        augmented = messages[:-1] + [
            messages[-1].__class__(content=messages[-1].content + json_instruction)
        ]
        raw_message = llm.invoke(augmented)
        text = _strip_json_fences(_extract_text(raw_message.content))
        return schema.model_validate_json(text)
    except (ValidationError, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        logger.warning(
            "structured_call: JSON-mode fallback failed for %s: %s",
            schema.__name__,
            exc,
        )

    logger.error(
        "structured_call: returning fallback for %s after both attempts",
        schema.__name__,
    )
    return fallback
