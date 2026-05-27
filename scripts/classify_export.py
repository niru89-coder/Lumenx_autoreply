"""Evaluate the intent router against LumenX's ground-truth labels.

Strategy:
  * Load data/raw/export.json (run scripts/pull_export.py first).
  * Pick seed threads (id starts with "conv-") where LumenX has set `intent`.
  * Classify the FIRST customer message of each thread.
  * Print per-intent comparison (true count vs pred count vs delta), overall
    accuracy, sensitivity-flag sanity check, and total estimated cost.

Run: python -m scripts.classify_export
Optional: --limit N to cap thread count while iterating.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from agent.config import REPO_ROOT
from agent.intent_router import (
    SENSITIVE_INTENTS,
    NON_PRODUCT_INTENTS,
    classify_intent,
)

EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"


def _first_customer_message(thread: dict) -> str | None:
    for m in thread.get("messages", []):
        if m.get("role") == "customer" and (m.get("text") or "").strip():
            return m["text"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of threads")
    args = ap.parse_args()

    if not EXPORT_PATH.exists():
        print(f"ERROR: {EXPORT_PATH} not found. Run `python -m scripts.pull_export` first.")
        return 1

    export = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    threads = export["threads"]
    seed = [
        t for t in threads
        if str(t.get("id", "")).startswith("conv-") and t.get("intent")
    ]
    if args.limit:
        seed = seed[: args.limit]

    print(f"Classifying first customer message of {len(seed)} seed threads")
    print()

    rows: list[dict] = []
    total_in_tokens = 0
    total_out_tokens = 0
    total_cost = 0.0
    bad_sensitivity = 0  # sanity check counter

    for i, t in enumerate(seed, 1):
        msg = _first_customer_message(t)
        if msg is None:
            continue
        res = classify_intent(msg)
        # The wrapped client logged this call to data/llm_calls.jsonl — we
        # don't get the call's token counts back through IntentResult, so
        # we estimate after the loop by tailing that log.
        rows.append({
            "thread": t["id"],
            "true": t["intent"],
            "pred": res.intent,
            "confidence": res.confidence,
            "sensitivity": res.sensitivity,
            "requires_wiki": res.requires_wiki,
        })
        # Sanity: sensitive intent must require wiki
        if res.sensitivity == "high" and not res.requires_wiki:
            bad_sensitivity += 1

        if i % 20 == 0:
            print(f"  ... {i}/{len(seed)} done")

    if not rows:
        print("No threads with a customer message + labeled intent. Nothing to score.")
        return 1

    true_dist = Counter(r["true"] for r in rows)
    pred_dist = Counter(r["pred"] for r in rows)
    correct = sum(1 for r in rows if r["true"] == r["pred"])
    total = len(rows)

    # Pull cost from the JSONL log — the last `total` intent_router entries
    log_path = REPO_ROOT / "data" / "llm_calls.jsonl"
    if log_path.exists():
        entries = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        intent_entries = [e for e in entries if e.get("feature") == "intent_router"][-total:]
        total_in_tokens = sum(e.get("input_tokens", 0) for e in intent_entries)
        total_out_tokens = sum(e.get("output_tokens", 0) for e in intent_entries)
        total_cost = sum(e.get("cost_usd", 0.0) for e in intent_entries)

    # --- output ---
    print()
    print("=" * 72)
    print(f"Per-intent distribution (true vs predicted)")
    print("=" * 72)
    all_intents = sorted(set(true_dist) | set(pred_dist))
    print(f"  {'intent':<22} {'true':>6} {'pred':>6} {'delta':>8} {'flag':>6}")
    over_threshold_misses: list[str] = []
    for intent in all_intents:
        t = true_dist.get(intent, 0)
        p = pred_dist.get(intent, 0)
        delta = p - t
        # ±20% rule, with a small absolute tolerance to forgive ±1 on tiny buckets
        tolerance = max(1, int(round(t * 0.20)))
        flag = " " if t == 0 or abs(delta) <= tolerance else "MISS"
        if flag == "MISS":
            over_threshold_misses.append(intent)
        print(f"  {intent:<22} {t:>6} {p:>6} {delta:+6d}   {flag:>6}")

    print()
    print("=" * 72)
    print(f"Accuracy:     {correct}/{total} = {correct/total:.1%}")
    print(f"Distribution: {'within ±20% on every populated intent' if not over_threshold_misses else 'OUT of ±20% on: ' + ', '.join(over_threshold_misses)}")
    print(f"Sensitivity:  sensitive-without-wiki count = {bad_sensitivity} (must be 0)")
    print()
    print(f"Cost:    ${total_cost:.4f}  ({total_in_tokens:,} in + {total_out_tokens:,} out tokens, Haiku)")
    print(f"Per-thread avg: ${total_cost/total:.5f}")

    # Show 5 wrong predictions for qualitative inspection
    wrong = [r for r in rows if r["true"] != r["pred"]]
    if wrong:
        print()
        print("=" * 72)
        print(f"Sample mispredictions (showing up to 5 of {len(wrong)}):")
        print("=" * 72)
        for r in wrong[:5]:
            print(f"  {r['thread']}")
            print(f"    true={r['true']!r:<22} pred={r['pred']!r:<22} conf={r['confidence']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
