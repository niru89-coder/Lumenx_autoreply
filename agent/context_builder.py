"""Assemble the prompt context that the drafter sees.

Sections (in render order):
  system           — persona + hallucination rules + output format
  wiki             — intent-aware top-k product / company-policy snippets
  past_qa          — top-k similar already-resolved Q&A pairs (from
                     historical pool today; from the feedback log in Phase 5)
  customer_history — short note about this customer's prior threads (Haiku,
                     cached per username)
  company_stats    — one-line "what's happening this week" anchor
  current_thread   — the live conversation, coalesced into alternating
                     user / assistant turns ready for the Anthropic API

Token budgeting is heuristic (≈ 4 chars per token). Sections that exceed
their budget are truncated from the bottom; current_thread truncates from
the FRONT (drop oldest messages first).

Caching: a full ContextWindow is persisted to data/context_cache/<sha>.json
keyed by (thread_id, last_customer_msg_id, intent, wiki_version). Lookups
return immediately when keys match.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import REPO_ROOT
from agent.customer_summary import build_history
from agent.historical_qa import HistoricalQAPool, QAHit
from agent.intent_router import IntentResult
from agent.llm_wiki.retriever import CHROMA_DIR as WIKI_CHROMA_DIR
from agent.llm_wiki.retriever import WikiHit, WikiRetriever

logger = logging.getLogger(__name__)

CACHE_DIR = REPO_ROOT / "data" / "context_cache"

# Approximate budgets per section, in tokens. ≈ 4 chars per token.
DEFAULT_BUDGETS: dict[str, int] = {
    "system": 500,
    "wiki": 900,
    "past_qa": 700,
    "customer_history": 250,
    "company_stats": 120,
    "current_thread": 1200,
}

# ----- system persona (Phase 4 will reuse this directly) -----

SYSTEM_PROMPT = """You are a customer-support assistant for LumenX, a SaaS company that makes 20 small productivity tools (EmailPilot, InvoiceFlow, TaskGrid, FormCraft, CalendarSync, NoteHub, ChatRelay, SignPath, PollWise, ReceiptVault, TimeMark, DocuMerge, KanbanLite, MeetMinutes, PixelDeck, InboxClean, TeamPulse, LinkVault, BillSplit, AuditTrail). You reply to customer messages on behalf of the support team.

VOICE
- Professional and empathetic. Acknowledge the customer's concern before answering.
- Concise: 1–4 sentences for simple questions, up to a short paragraph for complex ones.
- Use the customer's display name when known.

HARD RULES
- NEVER invent pricing, refund windows, cancellation terms, or discount eligibility. If the KNOWLEDGE section below does not contain the answer, say something like "Let me check that and get back to you" or "I want to confirm that with the team before I quote a number."
- ALWAYS cite your sources using [source: <chunk_id>] when you reference a wiki snippet. Example: "Our refund window is 14 days [source: _company__refund_window]."
- Do NOT promise specific timelines, custom discounts, or refunds beyond what the wiki states.
- Do NOT discuss internal LumenX operations, employees, or systems beyond what's in the wiki.

OUTPUT
- One reply only. No prose about your reasoning, just the reply text the customer would receive."""


# ----- data classes -----


@dataclass
class ContextSection:
    name: str
    title: str
    body: str
    token_count: int
    citations: list[str] = field(default_factory=list)


@dataclass
class ContextWindow:
    thread_id: str
    intent: str
    intent_confidence: float
    sensitivity: str
    requires_wiki: bool
    sections: list[ContextSection]
    total_tokens: int
    cache_key: str
    cached_hit: bool = False
    coalesced_messages: list[dict[str, str]] = field(default_factory=list)

    def section(self, name: str) -> ContextSection | None:
        return next((s for s in self.sections if s.name == name), None)

    def render_for_anthropic(self) -> tuple[str, list[dict[str, str]]]:
        """Returns (system_content, messages) ready for Anthropic Messages API.

        System content = persona + everything except current_thread.
        Messages = the coalesced conversation; last turn is the customer's
        most recent message, which is what the drafter must respond to.
        """
        persona = (self.section("system") or _empty("system")).body
        knowledge_parts: list[str] = []
        for s in self.sections:
            if s.name in {"wiki", "past_qa", "customer_history", "company_stats"}:
                if s.body.strip():
                    knowledge_parts.append(f"## {s.title}\n{s.body}")
        system_content = persona
        if knowledge_parts:
            system_content += "\n\n# KNOWLEDGE\n\n" + "\n\n".join(knowledge_parts)
        return system_content, self.coalesced_messages

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _empty(name: str) -> ContextSection:
    return ContextSection(name=name, title=name, body="", token_count=0)


# ----- helpers -----


def _est_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _truncate_to_tokens(text: str, budget: int) -> tuple[str, bool]:
    """Truncate from the bottom on a line boundary. Returns (text, truncated)."""
    if _est_tokens(text) <= budget:
        return text, False
    max_chars = budget * 4
    cut = text[:max_chars].rstrip()
    nl = cut.rfind("\n")
    if nl > max_chars // 2:  # only chop on a newline if it doesn't lose too much
        cut = cut[:nl].rstrip()
    return cut + "\n…[truncated]", True


def _wiki_version() -> str:
    """Stable across queries; only changes when `scripts/build_wiki.py` runs.

    Earlier version used the Chroma store's mtime, but Chroma writes to its
    sqlite on every PersistentClient open, breaking the cache key. Source
    markdown is the stable signal — it only changes on a rebuild.
    """
    wiki_md_dir = REPO_ROOT / "data" / "wiki"
    if not wiki_md_dir.exists():
        return "no-wiki"
    try:
        files = [p for p in wiki_md_dir.rglob("*.md") if p.is_file()]
        if not files:
            return "empty-wiki"
        mtime = max(p.stat().st_mtime for p in files)
        return f"mtime-{int(mtime)}"
    except OSError:
        return "unknown-wiki"


def _last_customer_message(thread: dict) -> dict | None:
    for m in reversed(thread.get("messages") or []):
        if m.get("role") == "customer" and (m.get("text") or "").strip():
            return m
    return None


def _cache_key(thread_id: str, last_msg_id: str, intent: str, wiki_version: str) -> str:
    raw = f"{thread_id}|{last_msg_id}|{intent}|{wiki_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_load(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_store(key: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["_cached_at"] = datetime.now(timezone.utc).isoformat()
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ----- per-section builders -----


def _build_wiki_section(
    query: str,
    intent: IntentResult,
    budget_tokens: int,
    product_id: str | None = None,
    retriever: WikiRetriever | None = None,
) -> ContextSection:
    if not intent.requires_wiki:
        return _empty("wiki")
    retriever = retriever or WikiRetriever()
    hits: list[WikiHit] = retriever.query_intent_aware(
        query,
        intent=intent.intent,
        product_id=product_id,
        k=5,
        k_company=2,
        k_product=2,
    )
    blocks: list[str] = []
    cites: list[str] = []
    for h in hits:
        d = f"{h.distance:.3f}" if h.distance is not None else "?"
        block = (
            f"[{h.chunk_id}]  ({h.title}, similarity-distance={d})\n"
            f"{h.text}"
        )
        blocks.append(block)
        cites.append(h.chunk_id)
    body = "\n\n---\n\n".join(blocks)
    body, _ = _truncate_to_tokens(body, budget_tokens)
    return ContextSection(
        name="wiki",
        title="Product / policy knowledge (verbatim from wiki)",
        body=body,
        token_count=_est_tokens(body),
        citations=cites,
    )


def _build_past_qa_section(
    query: str,
    intent: IntentResult,
    budget_tokens: int,
    pool: HistoricalQAPool | None = None,
) -> ContextSection:
    pool = pool or HistoricalQAPool()
    hits: list[QAHit] = pool.query(query, k=3, intent=intent.intent or None)
    if not hits:
        # Fall back to a no-intent query so we don't return empty just because
        # intent isn't a known LumenX label (e.g. chitchat).
        hits = pool.query(query, k=3)
    blocks: list[str] = []
    cites: list[str] = []
    for h in hits:
        e = h.entry
        d = f"{h.distance:.3f}" if h.distance is not None else "?"
        block = (
            f"[past:{e.thread_id}]  (intent={e.intent}, product={e.product_id}, distance={d})\n"
            f"Customer: {e.question}\n"
            f"Admin reply: {e.answer}"
        )
        blocks.append(block)
        cites.append(f"past:{e.thread_id}")
    body = "\n\n---\n\n".join(blocks)
    body, _ = _truncate_to_tokens(body, budget_tokens)
    return ContextSection(
        name="past_qa",
        title="Similar already-resolved cases",
        body=body,
        token_count=_est_tokens(body),
        citations=cites,
    )


def _build_customer_history_section(
    customer_username: str,
    current_thread_id: str,
    budget_tokens: int,
) -> ContextSection:
    hist = build_history(customer_username, current_thread_id)
    if hist.prior_thread_count == 0:
        return _empty("customer_history")
    body = (
        f"Username: {customer_username} — {hist.prior_thread_count} previous thread(s)\n"
        f"{hist.summary}"
    )
    body, _ = _truncate_to_tokens(body, budget_tokens)
    return ContextSection(
        name="customer_history",
        title="This customer's prior history with LumenX",
        body=body,
        token_count=_est_tokens(body),
        citations=hist.summarised_thread_ids,
    )


def _build_company_stats_section(
    stats: dict | None,
    budget_tokens: int,
) -> ContextSection:
    if not stats:
        return _empty("company_stats")
    intents = stats.get("intents_by_count") or []
    top = ", ".join(f"{i['intent']}({i['n']})" for i in intents[:3]) if intents else "—"
    body = (
        f"Threads (total={stats.get('threads', {}).get('total', '?')}, "
        f"unread={stats.get('threads', {}).get('unread_for_admin', '?')}). "
        f"Top intents this period: {top}."
    )
    body, _ = _truncate_to_tokens(body, budget_tokens)
    return ContextSection(
        name="company_stats",
        title="Recent volume",
        body=body,
        token_count=_est_tokens(body),
        citations=[],
    )


def _build_current_thread_section(
    thread: dict,
    budget_tokens: int,
) -> tuple[ContextSection, list[dict[str, str]]]:
    """Coalesce + render. Truncates from the OLDEST end if over budget."""
    msgs = thread.get("messages") or []
    # Coalesce consecutive same-role messages into one entry.
    coalesced: list[dict[str, str]] = []
    for m in msgs:
        if not (m.get("text") or "").strip():
            continue
        role = "user" if m.get("role") == "customer" else "assistant"
        text = m["text"]
        if coalesced and coalesced[-1]["role"] == role:
            coalesced[-1]["content"] += "\n\n" + text
        else:
            coalesced.append({"role": role, "content": text})

    # If first message is assistant, drop until first user.
    while coalesced and coalesced[0]["role"] != "user":
        coalesced.pop(0)

    # The Anthropic Messages API requires the conversation to end with a user
    # (customer) turn. Drop trailing assistant turns so the drafter is always
    # responding to the last customer message, not trying to continue an
    # already-completed exchange.
    while coalesced and coalesced[-1]["role"] != "user":
        coalesced.pop()

    # Token budget. Keep most recent turns; drop oldest until we fit.
    def total_tokens(msgs: list[dict[str, str]]) -> int:
        return sum(_est_tokens(m["content"]) for m in msgs)

    truncated = False
    while coalesced and total_tokens(coalesced) > budget_tokens and len(coalesced) > 1:
        coalesced.pop(0)
        truncated = True
        # Re-check first-role invariant
        while coalesced and coalesced[0]["role"] != "user":
            coalesced.pop(0)
            truncated = True

    body_lines = [
        f"{'Customer' if m['role']=='user' else 'You (admin)'}: {m['content']}"
        for m in coalesced
    ]
    body = "\n\n".join(body_lines)
    if truncated:
        body = "…[older turns truncated]\n\n" + body
    section = ContextSection(
        name="current_thread",
        title="Current thread",
        body=body,
        token_count=_est_tokens(body),
        citations=[thread.get("id", "")],
    )
    return section, coalesced


# ----- top-level -----


def build_context(
    thread: dict,
    intent: IntentResult,
    *,
    company_stats: dict | None = None,
    budgets: dict[str, int] | None = None,
    use_cache: bool = True,
) -> ContextWindow:
    budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
    thread_id = thread["id"]
    last_msg = _last_customer_message(thread) or {}
    last_msg_id = last_msg.get("id", "")
    query = last_msg.get("text", "")
    wiki_version = _wiki_version()
    key = _cache_key(thread_id, last_msg_id, intent.intent, wiki_version)

    if use_cache:
        cached = _cache_load(key)
        if cached:
            sections = [ContextSection(**s) for s in cached["sections"]]
            return ContextWindow(
                thread_id=cached["thread_id"],
                intent=cached["intent"],
                intent_confidence=cached.get("intent_confidence", 0.0),
                sensitivity=cached.get("sensitivity", "standard"),
                requires_wiki=cached.get("requires_wiki", False),
                sections=sections,
                total_tokens=cached["total_tokens"],
                cache_key=key,
                cached_hit=True,
                coalesced_messages=cached.get("coalesced_messages", []),
            )

    # System section is a fixed string but we still budget it for parity.
    sys_body, _ = _truncate_to_tokens(SYSTEM_PROMPT, budgets["system"])
    sys_section = ContextSection(
        name="system",
        title="Persona and rules",
        body=sys_body,
        token_count=_est_tokens(sys_body),
    )

    wiki_section = _build_wiki_section(
        query,
        intent,
        budgets["wiki"],
        product_id=thread.get("product_id"),
    )
    qa_section = _build_past_qa_section(query, intent, budgets["past_qa"])
    history_section = _build_customer_history_section(
        thread.get("customer_username", ""),
        thread_id,
        budgets["customer_history"],
    )
    stats_section = _build_company_stats_section(company_stats, budgets["company_stats"])
    thread_section, coalesced = _build_current_thread_section(thread, budgets["current_thread"])

    sections = [
        sys_section,
        wiki_section,
        qa_section,
        history_section,
        stats_section,
        thread_section,
    ]
    total = sum(s.token_count for s in sections)
    window = ContextWindow(
        thread_id=thread_id,
        intent=intent.intent,
        intent_confidence=intent.confidence,
        sensitivity=intent.sensitivity,
        requires_wiki=intent.requires_wiki,
        sections=sections,
        total_tokens=total,
        cache_key=key,
        cached_hit=False,
        coalesced_messages=coalesced,
    )
    if use_cache:
        _cache_store(key, window.to_jsonable())
    return window
