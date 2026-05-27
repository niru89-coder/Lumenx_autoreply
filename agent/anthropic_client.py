"""Wrapped Anthropic client — every call records tokens, USD, latency, feature tag.

Hard rule: never import the `anthropic` SDK directly from feature code.
Import `call_llm` from here so the cost log is always populated.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from agent.config import settings

logger = logging.getLogger(__name__)

# USD per 1M tokens. Update when Anthropic publishes new pricing.
# Unknown models log a warning and bill at 0 — better than guessing wrong.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5":          {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00},
}

# Convenience aliases used by feature code so model IDs live in one place.
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-7"


@dataclass
class LLMCallResult:
    text: str
    model: str
    feature: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float
    latency_ms: int
    raw: Any = field(repr=False, default=None)


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model)
    if not p:
        logger.warning(
            "No PRICING entry for model %r — cost will be recorded as 0", model
        )
        return 0.0
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def _log_call(record: dict) -> None:
    log_path = Path(settings.LLM_CALLS_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def call_llm(
    *,
    feature: str,
    model: str,
    messages: list[dict],
    system: str | list[dict] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **extra: Any,
) -> LLMCallResult:
    """Single chokepoint for all Anthropic Messages API calls.

    `feature` is a free-form tag used by the cost dashboard to attribute spend
    (e.g. "intent_router", "drafter", "customer_history_summary").
    """
    client = _get_client()
    started = time.perf_counter()

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system is not None:
        payload["system"] = system
    payload.update(extra)

    resp = client.messages.create(**payload)
    latency_ms = int((time.perf_counter() - started) * 1000)

    usage = resp.usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = _compute_cost(model, input_tokens, output_tokens)

    text = "".join(
        getattr(block, "text", "")
        for block in resp.content
        if getattr(block, "type", None) == "text"
    )

    _log_call(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "feature": feature,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation,
            "cache_read_tokens": cache_read,
            "cost_usd": round(cost, 6),
            "latency_ms": latency_ms,
        }
    )

    return LLMCallResult(
        text=text,
        model=model,
        feature=feature,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
        cost_usd=cost,
        latency_ms=latency_ms,
        raw=resp,
    )
