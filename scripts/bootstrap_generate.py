"""Bootstrap label generation — Phase 6.

Generates drafts under 4 deliberately-varied conditions for the 100 seed
threads in data/raw/export.json (those with a non-None intent label).  The
quality spread lets the Confidence Net learn to distinguish strong drafts
from weak ones.

Conditions
----------
  A  Sonnet,  wiki ON,   temp=0.3  (normal, best-quality baseline)
  B  Haiku,   wiki ON,   temp=0.3  (lower-quality, cheaper model)
  C  Sonnet,  wiki OFF,  temp=0.3  (hallucination-prone for sensitive intents)
  D  Sonnet,  wiki ON,   temp=0.9  (creative / variable, higher entropy)

Each draft is persisted to data/drafts/<uuid>.json AND recorded in
data/feedback.db (same path as live drafts, so bootstrap_review.py and
the dashboard see them all).

Progress is tracked in data/bootstrap_progress.json so the script is safe
to interrupt and resume.

Usage
-----
  python -m scripts.bootstrap_generate                    # all 100 threads × 4 conditions
  python -m scripts.bootstrap_generate --dry-run          # print plan, no LLM calls
  python -m scripts.bootstrap_generate --conditions A B   # run only conditions A and B
  python -m scripts.bootstrap_generate --limit 10         # cap at N threads (for testing)
  python -m scripts.bootstrap_generate --thread conv-042  # single thread, all conditions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── stdout ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

from agent.anthropic_client import MODEL_HAIKU, MODEL_SONNET
from agent.config import REPO_ROOT
from agent.context_builder import build_context
from agent.drafter import draft_reply
from agent.feedback_log.db import ensure_tables
from agent.intent_router import (
    SENSITIVE_INTENTS,
    NON_PRODUCT_INTENTS,
    IntentResult,
)

EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"
PROGRESS_PATH = REPO_ROOT / "data" / "bootstrap_progress.json"

# ──────────────────────────────────────────────────────────────────────────────
# Condition definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    label: str          # A / B / C / D
    model: str
    use_wiki: bool      # False = override requires_wiki to False
    temperature: float
    description: str


CONDITIONS: dict[str, Condition] = {
    "A": Condition("A", MODEL_SONNET, True,  0.3, "Sonnet, wiki ON,  temp=0.3 (baseline)"),
    "B": Condition("B", MODEL_HAIKU,  True,  0.3, "Haiku,  wiki ON,  temp=0.3 (lower quality)"),
    "C": Condition("C", MODEL_SONNET, False, 0.3, "Sonnet, wiki OFF, temp=0.3 (hallucination-prone)"),
    "D": Condition("D", MODEL_SONNET, True,  0.9, "Sonnet, wiki ON,  temp=0.9 (hot / variable)"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"done": [], "errors": []}


def _save_progress(prog: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(prog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prog_key(thread_id: str, condition: str) -> str:
    return f"{thread_id}:{condition}"


# ──────────────────────────────────────────────────────────────────────────────
# IntentResult reconstruction from export label
# (saves 100 Haiku classification calls)
# ──────────────────────────────────────────────────────────────────────────────

def _intent_from_label(label: str) -> IntentResult:
    """Build an IntentResult from the export's pre-labelled intent string.

    We reconstruct requires_wiki and sensitivity using the same logic as
    classify_intent(), so no Haiku call is needed for seed threads.
    """
    intent = label.strip().lower() if label else "feature"
    requires_wiki = intent not in NON_PRODUCT_INTENTS
    sensitivity = "high" if intent in SENSITIVE_INTENTS else "standard"
    return IntentResult(
        intent=intent,
        confidence=1.0,  # ground-truth label from export
        requires_wiki=requires_wiki,
        sensitivity=sensitivity,
        raw_response=f"[from export label: {label}]",
    )


def _intent_nowiki(base: IntentResult) -> IntentResult:
    """Return a copy of `base` with requires_wiki forced to False (condition C)."""
    return IntentResult(
        intent=base.intent,
        confidence=base.confidence,
        requires_wiki=False,
        sensitivity=base.sensitivity,
        raw_response=base.raw_response + " [wiki-disabled for bootstrap condition C]",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Seed thread selection
# ──────────────────────────────────────────────────────────────────────────────

def _load_seed_threads(limit: int | None, single: str | None) -> list[dict]:
    """Load threads from the export that have a non-None intent label."""
    data = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    all_threads: list[dict] = data.get("threads") or []

    if single:
        found = [t for t in all_threads if t.get("id") == single]
        if not found:
            print(f"[ERROR] Thread {single!r} not found in export.")
            sys.exit(1)
        return found

    # Keep only threads with a labelled intent AND at least one customer message
    labelled = [
        t for t in all_threads
        if t.get("intent") is not None
        and any(m.get("role") == "customer" for m in (t.get("messages") or []))
    ]

    if limit:
        labelled = labelled[:limit]

    return labelled


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def run(
    conditions: list[str],
    limit: int | None,
    dry_run: bool,
    single: str | None,
    pause_between: float = 0.5,
) -> None:
    threads = _load_seed_threads(limit, single)
    prog = _load_progress()
    done_keys: set[str] = set(prog.get("done_keys", []))

    # Migrate old format (done was a list of dicts) to a set of keys
    for item in prog.get("done", []):
        if isinstance(item, dict):
            done_keys.add(_prog_key(item.get("thread_id", ""), item.get("condition", "")))
        elif isinstance(item, str):
            done_keys.add(item)

    total_tasks = len(threads) * len(conditions)
    already_done = sum(
        1 for t in threads for c in conditions
        if _prog_key(t["id"], c) in done_keys
    )
    remaining = total_tasks - already_done

    print(f"\n=== Bootstrap Generate ===")
    print(f"  Seed threads : {len(threads)}")
    print(f"  Conditions   : {', '.join(conditions)}")
    print(f"  Total tasks  : {total_tasks}  ({already_done} already done, {remaining} to run)")
    for cname in conditions:
        c = CONDITIONS[cname]
        print(f"    [{c.label}] {c.description}")
    print()

    if dry_run:
        print("[DRY RUN] No LLM calls will be made.\n")
        for thread in threads:
            tid = thread["id"]
            intent_label = thread.get("intent", "?")
            for cname in conditions:
                key = _prog_key(tid, cname)
                status = "DONE" if key in done_keys else "PENDING"
                print(f"  {status:7s}  {tid}  [{cname}]  intent={intent_label}")
        print()
        return

    if not dry_run:
        ensure_tables()

    # ── per-condition cost estimate (rough) ──
    # Haiku ~$1/M input + $5/M output; Sonnet ~$3/M + $15/M
    # Assume ~800 input tokens, ~200 output tokens per call
    # A: 0.0024 + 0.003 = $0.0054; B: 0.0008 + 0.001 = $0.0018; C/D same as A
    # Per thread (A+B+C+D): ~$0.0054*3 + $0.0018 ≈ $0.018
    # 100 threads: ~$1.80 total (rough upper bound)

    generated = 0
    errors = 0

    for t_idx, thread in enumerate(threads):
        tid = thread["id"]
        intent_label = thread.get("intent", "feature")
        base_intent = _intent_from_label(intent_label)

        for cname in conditions:
            key = _prog_key(tid, cname)
            if key in done_keys:
                print(f"  SKIP   {tid} [{cname}] (already done)")
                continue

            cond = CONDITIONS[cname]
            intent = _intent_nowiki(base_intent) if not cond.use_wiki else base_intent

            print(
                f"  [{cname}] {tid}  intent={intent_label}"
                f"  model={cond.model.split('-')[1][:6]}"
                f"  temp={cond.temperature}"
                f"  wiki={'ON' if cond.use_wiki else 'OFF'}",
                end="  ... ",
                flush=True,
            )

            try:
                ctx = build_context(
                    thread,
                    intent,
                    use_cache=False,   # each condition is its own context window
                )
                draft = draft_reply(
                    ctx,
                    model=cond.model,
                    temperature=cond.temperature,
                    feature=f"bootstrap_{cname.lower()}",
                )

                status = f"{draft.confidence_label:<7}  cost=${draft.cost_usd:.4f}"
                print(status)

                # Record progress
                done_keys.add(key)
                prog.setdefault("done", []).append({
                    "thread_id": tid,
                    "condition": cname,
                    "draft_id": draft.draft_id,
                    "intent": intent_label,
                    "confidence_label": draft.confidence_label,
                    "cost_usd": round(draft.cost_usd, 6),
                })
                prog["done_keys"] = list(done_keys)
                _save_progress(prog)
                generated += 1

            except Exception as exc:
                print(f"ERROR: {exc}")
                logger.exception("bootstrap_generate: %s [%s] failed", tid, cname)
                prog.setdefault("errors", []).append({
                    "thread_id": tid,
                    "condition": cname,
                    "error": str(exc),
                })
                _save_progress(prog)
                errors += 1

            # Polite pause — avoid hammering the API
            if pause_between > 0:
                time.sleep(pause_between)

        # Progress marker every 10 threads
        if (t_idx + 1) % 10 == 0:
            total_done = len([k for k in done_keys if any(
                _prog_key(th["id"], c) == k for th in threads for c in conditions
            )])
            print(f"\n  --- [{t_idx+1}/{len(threads)} threads done, {generated} new drafts] ---\n")

    # ── summary ──
    print()
    print("=== Bootstrap Generate Complete ===")
    print(f"  Generated : {generated} new drafts")
    print(f"  Errors    : {errors}")
    total_done_now = sum(
        1 for t in threads for c in conditions
        if _prog_key(t["id"], c) in done_keys
    )
    print(f"  Total done: {total_done_now}/{total_tasks}")
    print(f"  Progress  : {PROGRESS_PATH}")
    if errors:
        print(f"\n  [!] {errors} error(s) — re-run to retry. Errors logged in {PROGRESS_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6: bootstrap draft generation")
    ap.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
        metavar="COND",
        help="Conditions to run (A B C D). Default: all.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap at N seed threads (default: all 100).",
    )
    ap.add_argument(
        "--thread",
        default=None,
        metavar="THREAD_ID",
        help="Run only this specific thread (e.g. conv-042).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making any LLM calls.",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.5,
        metavar="SECS",
        help="Seconds to sleep between LLM calls (default: 0.5).",
    )
    args = ap.parse_args()

    run(
        conditions=args.conditions,
        limit=args.limit,
        dry_run=args.dry_run,
        single=args.thread,
        pause_between=args.pause,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
