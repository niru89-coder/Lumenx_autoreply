"""Render the assembled context window for a given thread id.

  python -m scripts.show_context conv-007
  python -m scripts.show_context conv-007 --live   # fetch thread from LumenX API instead of export cache

Prints each section with token counts plus the final (system, messages)
pair ready for the Anthropic API. Also reports cumulative token estimate
and whether the result was served from the cache.
"""
from __future__ import annotations

import argparse
import json
import sys

from agent.config import REPO_ROOT
from agent.context_builder import build_context
from agent.intent_router import classify_intent
from agent.lumenx_client import LumenXClient

EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"


def _load_thread_from_export(thread_id: str) -> dict | None:
    if not EXPORT_PATH.exists():
        return None
    raw = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    return next((t for t in raw.get("threads", []) if t.get("id") == thread_id), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("thread_id")
    ap.add_argument(
        "--live",
        action="store_true",
        help="fetch the thread and stats from the deployed LumenX API "
             "(default: use the cached export at data/raw/export.json)",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the context cache and rebuild from scratch",
    )
    args = ap.parse_args()

    # 1. Load thread + stats
    if args.live:
        with LumenXClient() as client:
            thread = client.get_thread(args.thread_id)
            stats = client.get_stats()
    else:
        thread = _load_thread_from_export(args.thread_id)
        if thread is None:
            print(
                f"thread {args.thread_id!r} not found in {EXPORT_PATH}. "
                "Try --live or run scripts/pull_export.py first."
            )
            return 1
        stats = None  # company-stats section will be empty; harmless

    # 2. Classify intent on the LAST customer message
    last_cust = next(
        (m for m in reversed(thread.get("messages") or []) if m.get("role") == "customer"),
        None,
    )
    if last_cust is None:
        print(f"thread {thread['id']} has no customer message; nothing to classify")
        return 1
    intent = classify_intent(last_cust["text"])

    # 3. Build context
    window = build_context(
        thread,
        intent,
        company_stats=stats,
        use_cache=not args.no_cache,
    )

    # 4. Print summary
    print("=" * 72)
    print(f" thread:    {window.thread_id}")
    print(f" customer:  {thread.get('customer_username')}")
    print(f" product:   {thread.get('product_id')}  (LumenX label: {thread.get('intent')!r})")
    print(f" intent:    {window.intent}  (confidence={window.intent_confidence:.2f}, "
          f"sensitivity={window.sensitivity}, requires_wiki={window.requires_wiki})")
    print(f" cache_key: {window.cache_key[:16]}…  (hit={window.cached_hit})")
    print(f" tokens:    ~{window.total_tokens} total (heuristic)")
    print("=" * 72)
    for s in window.sections:
        if not s.body.strip():
            print(f"\n[ {s.name:<18} ]  (empty)")
            continue
        print(f"\n[ {s.name:<18} ]  ~{s.token_count} tokens  title={s.title!r}")
        if s.citations:
            print(f"  citations: {', '.join(s.citations[:6])}"
                  + ("" if len(s.citations) <= 6 else f" (+{len(s.citations)-6} more)"))
        body = s.body
        lines = body.splitlines()
        for line in lines[:25]:
            print(f"  {line}")
        if len(lines) > 25:
            print(f"  …[{len(lines) - 25} more lines]")
    print()
    print("=" * 72)
    print(" coalesced messages (what the Anthropic API will see):")
    print("=" * 72)
    for m in window.coalesced_messages:
        print(f"  [{m['role']}] {m['content'][:200]}"
              + ("…" if len(m["content"]) > 200 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
