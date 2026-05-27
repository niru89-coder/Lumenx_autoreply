"""Per-customer history summarizer with a tiny disk cache.

For a customer with 0 prior threads: returns empty.
For 1–2 prior threads: returns a hand-formatted inline summary (no LLM).
For 3+ prior threads: calls Haiku once, caches the result keyed by the
set of summarised thread ids. When new threads appear, the cache key
mismatches and we re-summarise.

Source of prior threads: the cached export at data/raw/export.json. In
production this would come from a live LumenX query, but the admin API
has no "threads-by-username" filter. Phase 10 deploy task.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.anthropic_client import MODEL_HAIKU, call_llm
from agent.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "customer_summaries"
EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"


@dataclass
class CustomerHistory:
    customer_username: str
    prior_thread_count: int
    summary: str  # may be empty if no prior threads
    summarised_thread_ids: list[str]  # threads whose content is in `summary`
    from_cache: bool


def _load_threads_for(username: str, current_thread_id: str) -> list[dict[str, Any]]:
    if not EXPORT_PATH.exists():
        return []
    raw = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    threads = [
        t for t in raw.get("threads", [])
        if t.get("customer_username") == username
        and t.get("id") != current_thread_id
    ]
    # Sort newest-first so the LLM sees the most recent context first
    threads.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return threads


def _format_thread_one_liner(thread: dict[str, Any]) -> str:
    msgs = thread.get("messages") or []
    customer_msg = next(
        (m for m in msgs if m.get("role") == "customer"), None
    )
    product = thread.get("product_id") or "?"
    intent = thread.get("intent") or "?"
    when = (thread.get("created_at") or "")[:10]
    snippet = ""
    if customer_msg:
        text = (customer_msg.get("text") or "").strip().replace("\n", " ")
        snippet = text[:120] + ("…" if len(text) > 120 else "")
    return f"- {when}  product={product}  intent={intent}  — {snippet}"


def _inline_summary(threads: list[dict[str, Any]]) -> str:
    lines = [_format_thread_one_liner(t) for t in threads]
    return "Previous thread(s):\n" + "\n".join(lines)


def _llm_summary(username: str, threads: list[dict[str, Any]]) -> str:
    """Compact Haiku summary across many prior threads."""
    formatted = "\n".join(_format_thread_one_liner(t) for t in threads)
    system = (
        "You compress a customer-support history into a 2–4 sentence note "
        "the next agent can read in a glance. Focus on the customer's product "
        "mix, recurring intent themes, and any unresolved issues. Do NOT invent "
        "details — only summarise what's in the list. No bullet points, just "
        "a short paragraph."
    )
    user = (
        f"Customer username: {username}\n\n"
        f"Prior threads (newest first):\n{formatted}\n\n"
        "Write the summary now."
    )
    result = call_llm(
        feature="customer_history_summary",
        model=MODEL_HAIKU,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=200,
        temperature=0,
    )
    return result.text.strip()


def _cache_path(username: str) -> Path:
    # Username may contain odd chars — keep it safe.
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in username)
    return CACHE_DIR / f"{safe}.json"


def build_history(
    customer_username: str,
    current_thread_id: str,
) -> CustomerHistory:
    threads = _load_threads_for(customer_username, current_thread_id)
    n = len(threads)

    if n == 0:
        return CustomerHistory(
            customer_username=customer_username,
            prior_thread_count=0,
            summary="",
            summarised_thread_ids=[],
            from_cache=False,
        )

    if n <= 2:
        return CustomerHistory(
            customer_username=customer_username,
            prior_thread_count=n,
            summary=_inline_summary(threads),
            summarised_thread_ids=[t["id"] for t in threads],
            from_cache=False,
        )

    thread_ids = sorted(t["id"] for t in threads)
    cache_file = _cache_path(customer_username)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if sorted(cached.get("summarised_thread_ids", [])) == thread_ids:
                return CustomerHistory(
                    customer_username=customer_username,
                    prior_thread_count=n,
                    summary=cached.get("summary", ""),
                    summarised_thread_ids=cached.get("summarised_thread_ids", []),
                    from_cache=True,
                )
        except (json.JSONDecodeError, OSError):
            pass

    summary = _llm_summary(customer_username, threads)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "customer_username": customer_username,
                "prior_thread_count": n,
                "summary": summary,
                "summarised_thread_ids": thread_ids,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return CustomerHistory(
        customer_username=customer_username,
        prior_thread_count=n,
        summary=summary,
        summarised_thread_ids=thread_ids,
        from_cache=False,
    )
