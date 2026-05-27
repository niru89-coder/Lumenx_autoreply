"""End-to-end CLI: fetch thread -> classify intent -> build context -> draft reply.

Exercises the full Phase 4 pipeline in a single command and prints the draft,
cited sources, uncertainty flags, and cost breakdown.

Usage:
  python -m scripts.draft_reply <thread_id>
  python -m scripts.draft_reply <thread_id> --no-cache
  python -m scripts.draft_reply <thread_id> --raw     # also dump raw LLM response

Examples:
  python -m scripts.draft_reply conv-001
  python -m scripts.draft_reply conv-042 --no-cache
"""
from __future__ import annotations

import argparse
import io
import sys
import textwrap

# Force stdout to UTF-8 so box-drawing / arrow characters from model replies
# don't crash on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from agent.context_builder import build_context
from agent.drafter import draft_reply
from agent.intent_router import classify_intent
from agent.lumenx_client import LumenXClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_customer_message(thread: dict) -> str:
    for m in reversed(thread.get("messages", [])):
        if m.get("role") == "customer" and (m.get("text") or "").strip():
            return m["text"]
    return "(no customer message found)"


def _hr(label: str = "", width: int = 72) -> str:
    """ASCII horizontal rule, optionally with a centred label."""
    if label:
        pad = max(0, width - len(label) - 4)
        return f"  --  {label}  {'-' * pad}"
    return "-" * width


def _wrap(text: str, indent: str = "  ", width: int = 70) -> str:
    """Word-wrap `text` with an indent prefix."""
    lines = text.split("\n")
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            for sub in textwrap.wrap(line, width=width):
                wrapped.append(indent + sub)
    return "\n".join(wrapped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Draft a reply for a LumenX support thread (Phase 4 pipeline)."
    )
    ap.add_argument("thread_id", help="Thread ID, e.g. conv-001")
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the context cache and rebuild from scratch",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Also print the raw LLM response text (for debugging)",
    )
    args = ap.parse_args()

    client = LumenXClient()

    # 1. Fetch thread
    print(f"\nFetching thread {args.thread_id!r} ...")
    try:
        raw = client.get_thread(args.thread_id)
    except Exception as exc:
        print(f"  ERROR fetching thread: {exc}")
        return 1

    if raw is None:
        print(f"  ERROR: thread {args.thread_id!r} not found.")
        return 1

    # The API wraps the payload: {"thread": {...}}. Unwrap if needed.
    thread: dict = raw.get("thread", raw) if isinstance(raw, dict) else raw

    last_msg = _last_customer_message(thread)
    product_id = thread.get("product_id") or "(unknown product)"
    customer = thread.get("customer_username") or "(unknown customer)"
    print(f"  product:  {product_id}")
    print(f"  customer: {customer}")
    print(f"  message:  {last_msg[:120]!r}")

    # 2. Classify intent
    print("\nClassifying intent ...")
    intent = classify_intent(last_msg)
    print(
        f"  intent={intent.intent!r}  confidence={intent.confidence:.2f}"
        f"  sensitivity={intent.sensitivity!r}  requires_wiki={intent.requires_wiki}"
    )

    # 3. Build context
    print("\nBuilding context ...")
    ctx = build_context(thread, intent, use_cache=not args.no_cache)
    cache_status = "HIT" if ctx.cached_hit else "MISS (built fresh)"
    print(
        f"  {ctx.total_tokens:,} tokens across {len(ctx.sections)} sections"
        f"  (cache {cache_status})"
    )
    for s in ctx.sections:
        if s.token_count > 0:
            print(f"    {s.name:<20} {s.token_count:>5} tok")

    # 4. Draft reply
    print("\nDrafting reply with Sonnet ...")
    draft = draft_reply(ctx)

    # 5. Pretty output
    print()
    print("=" * 72)
    label_map = {"high": "[HIGH]", "low": "[LOW]", "blocked": "[BLOCKED]"}
    label_str = label_map.get(draft.confidence_label, draft.confidence_label.upper())
    print(f"  DRAFT   [{draft.draft_id[:8]}...]   confidence: {label_str}")
    print("=" * 72)
    print(f"  thread:    {draft.thread_id}")
    print(f"  intent:    {draft.intent}  ({draft.sensitivity})")
    print(f"  auto-send: {'YES' if draft.auto_sendable else 'NO'}")

    if draft.guardrail_triggered:
        print("  !! GUARDRAIL TRIGGERED -- sensitive intent without wiki citations")

    # Reply text
    print()
    print(_hr("REPLY TO CUSTOMER"))
    print()
    print(_wrap(draft.reply))
    print()

    # Cited sources
    print(_hr("CITED SOURCES"))
    if draft.cited_sources:
        for src in draft.cited_sources:
            print(f"  * {src}")
    else:
        print("  (none)")
    print()

    # Uncertainty flags
    if draft.uncertainty_flags:
        print(_hr("UNCERTAINTY FLAGS"))
        for flag in draft.uncertainty_flags:
            print(f"  !! {flag}")
        print()

    # Cost summary
    print(_hr("COST & TOKENS"))
    print(f"  model:      {draft.model}")
    print(f"  in tokens:  {draft.input_tokens:,}")
    print(f"  out tokens: {draft.output_tokens:,}")
    print(f"  cost:       ${draft.cost_usd:.5f}")
    print(f"  latency:    {draft.latency_ms:,} ms")
    print(f"  attempts:   {draft.parse_attempts}")
    print()

    print(f"  Draft persisted -> data/drafts/{draft.draft_id}.json")
    print("=" * 72)
    print()

    # Optional: raw LLM response
    if args.raw and draft.raw_llm_response:
        print(_hr("RAW LLM RESPONSE"))
        print()
        print(_wrap(draft.raw_llm_response, indent="  "))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
