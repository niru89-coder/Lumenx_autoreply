"""Demo / smoke-test for the Phase 8 auto-send router.

Runs the full pipeline for one or more threads and prints the routing
decision.  By default uses dry_run=True — it will NOT actually call the
LumenX POST /reply endpoint.

Usage
-----
  python -m scripts.demo_route conv-001
  python -m scripts.demo_route conv-001 --live           # actually sends if eligible
  python -m scripts.demo_route conv-001 --threshold 0.5  # lower threshold for testing
  python -m scripts.demo_route --all-intents             # one thread per intent

Phase 8 success criteria (manual check):
  - A greeting/feature thread → MLP scores high → action=auto_send (if enabled)
  - A pricing thread with no wiki → action=human_review (guardrail)
  - Any thread with uncertainty_flags → action=human_review
"""
from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import json
import textwrap
from pathlib import Path

from agent.config import REPO_ROOT, settings
from agent.context_builder import build_context, _last_customer_message
from agent.drafter import draft_reply
from agent.intent_router import classify_intent
from agent.lumenx_client import LumenXClient
from agent.router import route

EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"

_SEP = "-" * 72


def _one_per_intent() -> list[str]:
    """Return one thread_id per intent from the export (for variety demo)."""
    data = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    for t in data.get("threads") or []:
        intent = t.get("intent")
        if intent and intent not in seen:
            seen[intent] = t["id"]
    return list(seen.values())


def run_thread(thread_id: str, *, dry_run: bool, threshold: float) -> dict:
    """Full pipeline for one thread. Returns result dict."""
    with LumenXClient() as client:
        raw = client.get_thread(thread_id)

    if raw is None:
        return {"thread_id": thread_id, "error": "thread not found"}

    thread: dict = raw.get("thread", raw) if isinstance(raw, dict) else raw
    last_msg = _last_customer_message(thread)
    if last_msg is None:
        return {"thread_id": thread_id, "error": "no customer message"}

    intent = classify_intent(last_msg["text"])
    ctx = build_context(thread, intent, use_cache=True)
    draft = draft_reply(ctx)
    decision = route(draft, threshold=threshold, dry_run=dry_run)

    return {
        "thread_id": thread_id,
        "intent": intent.intent,
        "intent_confidence": intent.confidence,
        "draft_id": draft.draft_id[:8],
        "confidence_label": draft.confidence_label,
        "guardrail_triggered": draft.guardrail_triggered,
        "uncertainty_flags": draft.uncertainty_flags,
        "cited_sources": draft.cited_sources,
        "reply_preview": draft.reply[:200],
        "cost_usd": draft.cost_usd,
        "routing": {
            "action": decision.action,
            "reason": decision.reason,
            "score": decision.score,
            "threshold": decision.threshold,
            "sent": decision.sent,
        },
    }


def print_result(r: dict) -> None:
    if "error" in r:
        print(f"  ERROR  {r['thread_id']}: {r['error']}")
        return

    action = r["routing"]["action"]
    score = r["routing"]["score"]
    action_icon = "AUTO-SEND" if action == "auto_send" else "HUMAN-REVIEW"

    print(_SEP)
    print(f"  Thread : {r['thread_id']}   intent={r['intent']}  ({r['intent_confidence']:.2f})")
    print(f"  Draft  : {r['draft_id']}...  label={r['confidence_label']}")
    if r.get("guardrail_triggered"):
        print("  !! GUARDRAIL TRIGGERED")
    if r.get("uncertainty_flags"):
        print(f"  uncertainty_flags: {', '.join(r['uncertainty_flags'])}")
    print(f"  cited  : {', '.join(r['cited_sources']) or '(none)'}")
    print()
    print(f"  ROUTING DECISION: [{action_icon}]")
    print(f"    reason    : {r['routing']['reason']}")
    print(f"    score     : {f'{score:.4f}' if score is not None else 'n/a'} "
          f"(threshold={r['routing']['threshold']:.2f})")
    print(f"    sent      : {'YES' if r['routing']['sent'] else 'NO (dry_run or human_review)'}")
    print()
    print("  Reply preview:")
    for line in textwrap.wrap(r["reply_preview"], 68):
        print(f"    {line}")
    print()
    print(f"  Cost: ${r['cost_usd']:.5f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 8 router demo")
    ap.add_argument("thread_ids", nargs="*", help="Thread IDs to process")
    ap.add_argument("--live",       action="store_true", help="Actually send (not dry-run)")
    ap.add_argument("--threshold",  type=float, default=settings.AUTO_SEND_THRESHOLD,
                    help=f"MLP threshold (default: {settings.AUTO_SEND_THRESHOLD})")
    ap.add_argument("--all-intents", action="store_true",
                    help="Run one thread per intent from the export")
    args = ap.parse_args()

    thread_ids = list(args.thread_ids)
    if args.all_intents:
        thread_ids += _one_per_intent()
    if not thread_ids:
        ap.print_help()
        return 1

    dry_run = not args.live
    print(f"\n=== Phase 8 Router Demo ===")
    print(f"  Threads          : {len(thread_ids)}")
    print(f"  dry_run          : {dry_run}")
    print(f"  threshold        : {args.threshold:.2f}")
    print(f"  AUTO_SEND_ENABLED: {settings.AUTO_SEND_ENABLED}")
    print()

    results = []
    for tid in thread_ids:
        print(f"Processing {tid} ...", flush=True)
        try:
            r = run_thread(tid, dry_run=dry_run, threshold=args.threshold)
            results.append(r)
            print_result(r)
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")

    # Summary
    print(_SEP)
    print(f"Summary: {len(results)} thread(s) processed")
    auto_send = sum(1 for r in results if r.get("routing", {}).get("action") == "auto_send")
    human = sum(1 for r in results if r.get("routing", {}).get("action") == "human_review")
    print(f"  auto_send    : {auto_send}")
    print(f"  human_review : {human}")
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"  total_cost   : ${total_cost:.4f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
