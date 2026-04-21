"""Groq (Llama 3.3) wrapper with graceful structured-output handling.

Every LLM call in this project goes through `structured_call`. It layers:

1. `with_structured_output(schema)` — the happy path; LangChain parses.
2. Raw JSON mode — re-issue the prompt with a JSON instruction and
   parse the response string with Pydantic.
3. Hardcoded fallback value — if both attempts fail, return the caller's
   default so a single flaky LLM call never crashes a turn.

Rate limits (429s) are handled separately from parsing errors: we
honour the retry hint, back off, and retry the primary path without
falling through to JSON mode. Falling through would just double the
quota burn.

Never raises. The caller always gets a valid schema instance.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Type, TypeVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from src.config import LLM_MODEL

logger = logging.getLogger(__name__)

# Groq's free tier on Llama 3.3 70B is 30 req/min. We throttle to
# ~20 req/min (3s/call) to stay safely below the burst ceiling when
# the graph fans out multiple LLM calls per turn. Override via env var
# (set to "0" to disable on paid tiers).
_RATE_LIMIT_SEC = float(os.getenv("LLM_MIN_INTERVAL", "3.0"))
_last_call_ts = 0.0
_rate_lock = threading.Lock()
_MAX_429_RETRIES = 4


def _throttle() -> None:
    """Block until at least _RATE_LIMIT_SEC have passed since the last call."""
    global _last_call_ts
    with _rate_lock:
        now = time.monotonic()
        wait = _RATE_LIMIT_SEC - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def _parse_retry_delay(exc: Exception, default: float = 10.0) -> float:
    """Pull the provider's recommended retry delay out of a 429 error."""
    m = re.search(r"retry[_ -]?after\D+(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*seconds?", str(exc), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


T = TypeVar("T", bound=BaseModel)

_LLM: ChatGroq | None = None


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Lazy-init and cache the Groq client."""
    global _LLM
    if _LLM is None or _LLM.temperature != temperature:
        _LLM = ChatGroq(model=LLM_MODEL, temperature=temperature)
    return _LLM


_ROLE_TO_MSG = {
    "system": SystemMessage,
    "human": HumanMessage,
    "user": HumanMessage,
    "ai": AIMessage,
    "assistant": AIMessage,
}


def _normalize(prompt: str | list) -> list[BaseMessage]:
    """Accept str, BaseMessage list, or (role, text) tuple list."""
    if isinstance(prompt, str):
        return [HumanMessage(content=prompt)]
    out: list[BaseMessage] = []
    for item in prompt:
        if isinstance(item, BaseMessage):
            out.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            role, text = item
            cls = _ROLE_TO_MSG.get(role.lower(), HumanMessage)
            out.append(cls(content=text))
        else:
            raise TypeError(f"Unsupported prompt item: {item!r}")
    return out


def _strip_json_fences(text: str) -> str:
    """Pull the JSON body out of markdown fences if the model wrapped output."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _extract_text(raw) -> str:
    """.content is usually str, but occasionally a list of parts."""
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


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "rate_limit" in msg.lower() or "quota" in msg.lower()


def _invoke_with_backoff(call, *args, **kwargs):
    """Run a callable, retrying on 429 with honoured delay."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_429_RETRIES):
        _throttle()
        try:
            return call(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit(exc):
                raise
            last_exc = exc
            delay = _parse_retry_delay(exc, default=10.0) + 1.0
            logger.warning(
                "LLM 429 (attempt %d/%d); sleeping %.1fs",
                attempt + 1, _MAX_429_RETRIES, delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def structured_call(
    schema: Type[T],
    prompt: str | list,
    fallback: T,
    temperature: float = 0.0,
) -> T:
    """Invoke the LLM with Pydantic-structured output, with JSON-mode fallback.

    Args:
        schema: Pydantic model to parse the response into.
        prompt: Either a plain user prompt string, a list of BaseMessages,
            or a list of (role, text) tuples.
        fallback: Returned verbatim if both paths fail. Must already be a
            valid instance of `schema`.
        temperature: Sampling temperature. 0 for classifiers/extractors;
            bump for creative responses.

    Returns:
        A `schema` instance — never raises.
    """
    llm = get_llm(temperature=temperature)
    messages = _normalize(prompt)

    try:
        return _invoke_with_backoff(
            llm.with_structured_output(schema).invoke, messages
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "structured_call: with_structured_output failed for %s: %s",
            schema.__name__, exc,
        )

    try:
        json_instruction = (
            "\n\nIMPORTANT: Respond with a single valid JSON object matching "
            "this schema (field names and types must match exactly):\n"
            f"{json.dumps(schema.model_json_schema())}\n"
            "Return ONLY the JSON object. No prose. No markdown fences."
        )
        last = messages[-1]
        last_text = last.content if isinstance(last.content, str) else str(last.content)
        augmented = messages[:-1] + [
            last.__class__(content=last_text + json_instruction)
        ]
        raw_message = _invoke_with_backoff(llm.invoke, augmented)
        text = _strip_json_fences(_extract_text(raw_message.content))
        return schema.model_validate_json(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "structured_call: JSON-mode fallback failed for %s: %s",
            schema.__name__, exc,
        )

    logger.error(
        "structured_call: returning fallback for %s after both attempts",
        schema.__name__,
    )
    return fallback
