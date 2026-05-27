"""Intent classification for incoming customer messages.

Uses Haiku for cost. Returns a structured IntentResult with a sensitivity
flag and a requires_wiki flag so downstream stages (context builder,
drafter, router) can branch correctly.

Hard rule: sensitive intents (pricing / discount / cancellation / billing)
ALWAYS have requires_wiki=True. The drafter and router both depend on this.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from agent.anthropic_client import MODEL_HAIKU, call_llm

logger = logging.getLogger(__name__)


# LumenX-native labels are kept exactly as the platform uses them (some use
# hyphens, some don't) so the eval can compare apples to apples. The two
# tail labels (chitchat, out_of_scope) are ours — LumenX never seeded them.
VALID_INTENTS: set[str] = {
    "greeting",
    "pricing",
    "discount",
    "feature",
    "integration",
    "technical",
    "billing",
    "cancellation",
    "multi-product",
    "compare-competitor",
    "chitchat",
    "out_of_scope",
}

# Sensitive intents demand wiki citation; the drafter will refuse to auto-send
# without retrieved sources. Hallucinated pricing/refund is the blast radius.
SENSITIVE_INTENTS: set[str] = {"pricing", "discount", "cancellation", "billing"}

# These intents need no product wiki — the agent can respond from persona alone.
NON_PRODUCT_INTENTS: set[str] = {"greeting", "chitchat", "out_of_scope"}


_SYSTEM_PROMPT = """You are an intent classifier for LumenX customer support.

LumenX makes 20 small productivity tools: EmailPilot, InvoiceFlow, TaskGrid,
FormCraft, CalendarSync, NoteHub, ChatRelay, SignPath, PollWise, ReceiptVault,
TimeMark, DocuMerge, KanbanLite, MeetMinutes, PixelDeck, InboxClean, TeamPulse,
LinkVault, BillSplit, AuditTrail.

Classify the customer's message into exactly ONE of these intents. Use the
label EXACTLY as written (note hyphens vs underscores):

- greeting             — hi, hello, thanks, brief social
- pricing              — prices, plans, tier costs, "how much"
- discount             — any kind of discount including: deals, coupons, education / student / non-profit / startup / founder discount, AND annual-billing discount ("is annual cheaper")
- feature              — what does X do, how does Y work, asking about a capability
- integration          — connects to X, works with Y, exporting to Z
- technical            — bug, error, broken, doesn't work, crash, AND login problems (can't log in, spinning, password not accepted)
- billing              — invoice, charge, payment, subscription state, AND currency questions (what currency, are prices in USD)
- cancellation         — cancel, refund, money back, terminate plan, end subscription
- multi-product        — a single question that names TWO OR MORE LumenX products, including bundle requests
- compare-competitor   — comparison to other tools (Asana, Trello, Notion, Slack, etc.)
- chitchat             — small talk that has nothing to do with any LumenX product
- out_of_scope         — unrelated to LumenX entirely (politics, weather, world events)

Disambiguation rules — apply in this order:
1. If the customer asks about a BUNDLE / PACKAGE / multiple products together
   for one purchase ("can I buy A, B, and C together", "is there a bundle of
   N products") → `multi-product`.
2. If the customer asks how TWO LumenX products CONNECT / WORK TOGETHER /
   sync / integrate → `integration` (not multi-product), even when both
   products are named.
3. "Is annual cheaper" / "annual vs monthly savings" → `discount` (annual
   billing IS a discount), NOT `pricing`.
4. "What currency" / "are prices in USD" → `billing`, NOT `pricing`.
5. "Can't log in" / "login spinning" / "password not accepted" → `technical`,
   NOT `billing`.
6. "Where do I find/download my invoices/receipts/account info" → `billing`,
   NOT `technical`.
7. Comparison to a non-LumenX product (Asana, Trello, etc.) →
   `compare-competitor`, even if it touches on price or features.

Reply with ONLY this JSON, no prose, no code fences:
{"intent": "<one-label>", "confidence": <0.0-1.0>}

Confidence should reflect ambiguity. A clear, on-pattern message earns 0.9+.
An ambiguous message that could fit two labels earns 0.5–0.7."""


@dataclass
class IntentResult:
    intent: str
    confidence: float
    requires_wiki: bool
    sensitivity: str  # "standard" | "high"
    raw_response: str


def _parse_json(text: str) -> dict | None:
    """Best-effort extract a JSON object from a model response."""
    text = text.strip()
    # Strict
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Greedy — first {...} containing "intent"
    m = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _fallback(reason: str) -> IntentResult:
    """Fallback when classification fails. Low confidence → forces human review."""
    logger.warning("intent_router falling back (%s)", reason)
    return IntentResult(
        intent="feature",
        confidence=0.3,
        requires_wiki=True,
        sensitivity="standard",
        raw_response="",
    )


def classify_intent(message: str) -> IntentResult:
    """Classify a single customer message. Two attempts, then fallback."""
    for attempt in (1, 2):
        try:
            result = call_llm(
                feature="intent_router",
                model=MODEL_HAIKU,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": message}],
                max_tokens=60,
                temperature=0,
            )
        except Exception as e:
            logger.warning("intent_router attempt %d errored: %s", attempt, e)
            if attempt == 2:
                return _fallback(f"llm error: {e}")
            continue

        parsed = _parse_json(result.text)
        intent = parsed.get("intent") if parsed else None
        if intent in VALID_INTENTS:
            try:
                conf = float(parsed.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            return IntentResult(
                intent=intent,
                confidence=conf,
                requires_wiki=intent not in NON_PRODUCT_INTENTS,
                sensitivity="high" if intent in SENSITIVE_INTENTS else "standard",
                raw_response=result.text,
            )

        logger.warning(
            "intent_router attempt %d returned invalid JSON or intent: %r",
            attempt,
            result.text[:200],
        )

    return _fallback("parse failed twice")
