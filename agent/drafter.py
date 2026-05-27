"""LLM Drafter — Phase 4.

Turns a ContextWindow into a candidate reply using Claude Sonnet 4.6.

Output schema (parsed from model response):
  {
    "reply":             str,   # text the customer would receive
    "cited_sources":     list,  # chunk_ids / past:thread_ids the model drew on
    "uncertainty_flags": list   # non-empty when information is missing / uncertain
  }

Hard guardrails (pre-MLP, enforced mechanically):
  1. Sensitive intent (pricing / discount / cancellation / billing) + empty
     wiki citations → confidence_label="blocked", auto_sendable=False.
  2. JSON parse failure after 1 retry → confidence_label="blocked",
     reply set to a safe human-escalation message.

Confidence labels:
  - "high"    → well-sourced, no uncertainty flags (Phase 8 adds MLP on top)
  - "low"     → uncertainty flags present; needs human review
  - "blocked" → guardrail tripped or parse failed; must not auto-send

Drafts are persisted to data/drafts/<draft_id>.json so the Phase 5 feedback
log can pick them up and Phase 9 dashboard can display them.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.anthropic_client import MODEL_SONNET, LLMCallResult, call_llm
from agent.config import REPO_ROOT
from agent.context_builder import ContextWindow
from agent.intent_router import SENSITIVE_INTENTS

logger = logging.getLogger(__name__)

DRAFTS_DIR = REPO_ROOT / "data" / "drafts"

# ─────────────────────────────────────────────────────────────────────────────
# JSON output instruction appended to the context system prompt.
# Appending (rather than replacing) lets us keep the full persona and hard
# rules from context_builder.SYSTEM_PROMPT while overriding the OUTPUT block.
# ─────────────────────────────────────────────────────────────────────────────
_JSON_OUTPUT_INSTRUCTION = """

# OUTPUT FORMAT (MANDATORY — overrides any previous OUTPUT instruction)
Reply with ONLY a valid JSON object — no prose, no code fences, no extra text.
The object must have exactly these three keys:

{
  "reply": "<the full reply text the customer will receive>",
  "cited_sources": ["<chunk_id_1>", "<chunk_id_2>"],
  "uncertainty_flags": ["<reason if something is uncertain or missing>"]
}

Key rules:
- `reply`: the exact text sent to the customer. Use their display name when
  available. Empathetic and concise (1–4 sentences for simple questions;
  one short paragraph for complex ones).
- `cited_sources`: list every wiki chunk_id or past:thread_id you drew on.
  Examples: "emailpilot__refund_policy", "billsplit__pricing", "past:conv-042".
  Use [] if you drew on no specific sources.
- `uncertainty_flags`: list reasons you could not answer with confidence.
  Add one entry whenever the knowledge base lacks the requested information
  or you are uncertain which product the customer means.
  Examples: "No pricing found for TeamPulse in knowledge base",
            "Customer asked about enterprise custom plan — not in wiki".
  Use [] when fully confident.
- Do NOT include any text outside the JSON object. No commentary, no apologies,
  no code-fence markers. The output must parse with json.loads()."""

_JSON_STRICT_RETRY = (
    "\n\nCRITICAL: Your previous response was not valid JSON. "
    "Reply with ONLY the JSON object. Nothing before it, nothing after it."
)

_ESCALATION_REPLY = (
    "Thank you for reaching out! I want to make sure I give you the most "
    "accurate information, so I'm passing this to our support team who will "
    "follow up with you shortly."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Draft:
    draft_id: str
    thread_id: str
    intent: str
    sensitivity: str
    reply: str
    cited_sources: list[str]
    uncertainty_flags: list[str]
    confidence_label: str       # "high" | "low" | "blocked"
    auto_sendable: bool
    guardrail_triggered: bool
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    context_cache_key: str
    created_at: str
    parse_attempts: int
    raw_llm_response: str = field(repr=False, default="")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json(text: str) -> dict | None:
    """Best-effort extract a JSON object from the model response.

    Tries four increasingly lenient strategies:
      1. Strict: the whole response is a bare JSON object.
      2. Code-fenced: wrapped in ```json ... ``` or ``` ... ```.
      3. First {...} block that contains the key "reply".
      4. Outermost {...} block (greedy DOTALL match containing "reply").
    """
    text = text.strip()

    # 1. Strict bare object
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 2. Code-fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First {...} containing "reply" without nested braces
    m = re.search(r'\{[^{}]*"reply"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Greedy: outermost {...} containing "reply" (handles arrays inside)
    m = re.search(r'\{.*?"reply".*?\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _valid(parsed: dict | None) -> bool:
    """Returns True if `parsed` has all required keys in expected types."""
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("reply"), str)
        and len(parsed.get("reply", "").strip()) > 0
        and isinstance(parsed.get("cited_sources"), list)
        and isinstance(parsed.get("uncertainty_flags"), list)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core drafter
# ─────────────────────────────────────────────────────────────────────────────


def draft_reply(ctx: ContextWindow) -> Draft:
    """Generate a candidate reply for the thread described by `ctx`.

    Two LLM attempts:
      • Attempt 1 — temperature=0.3, normal instruction.
      • Attempt 2 — temperature=0.0, strict retry suffix (if attempt 1 fails
        to produce valid JSON).
    After both attempts fail, returns a "blocked" escalation draft.
    """
    system_base, messages = ctx.render_for_anthropic()
    system = system_base + _JSON_OUTPUT_INSTRUCTION

    # Accumulators for token / cost totals across attempts
    total_in = 0
    total_out = 0
    total_cost = 0.0
    total_latency = 0
    raw_response = ""
    parse_attempts = 0
    parsed: dict | None = None

    for attempt in (1, 2):
        temperature = 0.3 if attempt == 1 else 0.0
        sys_for_attempt = system if attempt == 1 else system + _JSON_STRICT_RETRY

        try:
            result: LLMCallResult = call_llm(
                feature="drafter",
                model=MODEL_SONNET,
                system=sys_for_attempt,
                messages=messages,
                max_tokens=1024,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("drafter attempt %d errored: %s", attempt, exc)
            if attempt == 2:
                break
            continue

        total_in += result.input_tokens
        total_out += result.output_tokens
        total_cost += result.cost_usd
        total_latency += result.latency_ms
        raw_response = result.text
        parse_attempts += 1

        parsed = _parse_json(result.text)
        if _valid(parsed):
            logger.info("drafter: valid JSON on attempt %d", attempt)
            break

        logger.warning(
            "drafter attempt %d returned invalid / non-JSON (first 300 chars): %r",
            attempt,
            result.text[:300],
        )

    # ── assemble the Draft ──────────────────────────────────────────────────
    draft_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    if not _valid(parsed):
        # Both attempts produced no usable JSON → escalation draft
        logger.error(
            "drafter: parse failed after %d attempt(s) for thread %s — escalating",
            parse_attempts,
            ctx.thread_id,
        )
        return _persist_and_return(
            Draft(
                draft_id=draft_id,
                thread_id=ctx.thread_id,
                intent=ctx.intent,
                sensitivity=ctx.sensitivity,
                reply=_ESCALATION_REPLY,
                cited_sources=[],
                uncertainty_flags=[
                    f"JSON parse failed after {parse_attempts} attempt(s) — escalated to human review"
                ],
                confidence_label="blocked",
                auto_sendable=False,
                guardrail_triggered=True,
                model=MODEL_SONNET,
                input_tokens=total_in,
                output_tokens=total_out,
                cost_usd=total_cost,
                latency_ms=total_latency,
                context_cache_key=ctx.cache_key,
                created_at=now,
                parse_attempts=parse_attempts,
                raw_llm_response=raw_response,
            )
        )

    # ── extract fields ──────────────────────────────────────────────────────
    assert parsed is not None  # narrowing: _valid() returned True
    reply: str = parsed["reply"].strip()
    cited_sources: list[str] = [str(s) for s in (parsed.get("cited_sources") or [])]
    uncertainty_flags: list[str] = [str(f) for f in (parsed.get("uncertainty_flags") or [])]

    # ── hard guardrail ──────────────────────────────────────────────────────
    # Sensitive intents MUST cite at least one wiki chunk (not a past: reference)
    # to be eligible for auto-send. "past:" references are historical Q&A pairs,
    # not the canonical wiki, so they don't count as verified policy citations.
    is_sensitive = ctx.intent in SENSITIVE_INTENTS
    has_wiki_citation = any(not s.startswith("past:") for s in cited_sources)
    guardrail_triggered = is_sensitive and not has_wiki_citation

    if guardrail_triggered:
        confidence_label = "blocked"
        auto_sendable = False
        logger.warning(
            "drafter guardrail: intent=%r is sensitive but cited_sources has no wiki chunk — "
            "draft blocked from auto-send  (thread=%s)",
            ctx.intent,
            ctx.thread_id,
        )
    elif uncertainty_flags:
        confidence_label = "low"
        auto_sendable = False
    else:
        confidence_label = "high"
        auto_sendable = True

    return _persist_and_return(
        Draft(
            draft_id=draft_id,
            thread_id=ctx.thread_id,
            intent=ctx.intent,
            sensitivity=ctx.sensitivity,
            reply=reply,
            cited_sources=cited_sources,
            uncertainty_flags=uncertainty_flags,
            confidence_label=confidence_label,
            auto_sendable=auto_sendable,
            guardrail_triggered=guardrail_triggered,
            model=MODEL_SONNET,
            input_tokens=total_in,
            output_tokens=total_out,
            cost_usd=total_cost,
            latency_ms=total_latency,
            context_cache_key=ctx.cache_key,
            created_at=now,
            parse_attempts=parse_attempts,
            raw_llm_response=raw_response,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def _persist_and_return(draft: Draft) -> Draft:
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAFTS_DIR / f"{draft.draft_id}.json"
    path.write_text(
        json.dumps(draft.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Draft saved → %s  (thread=%s  label=%s  auto_send=%s)",
        path,
        draft.thread_id,
        draft.confidence_label,
        draft.auto_sendable,
    )
    return draft
