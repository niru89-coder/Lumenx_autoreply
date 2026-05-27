"""Interactive terminal review tool for bootstrap drafts — Phase 6.

Shows each unlabelled draft alongside the ground-truth admin reply from the
export.  You decide:

  a  approve  — draft is good as-is, would send it
  e  edit     — draft needs a fix; you type the corrected reply
  r  reject   — draft is bad / wrong; would not send at all
  s  skip     — come back to this one later
  q  quit     — stop the session (progress is saved)

Labels are written to human_actions via record_human_action().

By default the tool shows one draft at a time, sorted so that condition-A
(Sonnet baseline) drafts for each thread are reviewed first, giving you a
ground-truth anchor before seeing the degraded-condition versions.

Usage
-----
  python -m scripts.bootstrap_review
  python -m scripts.bootstrap_review --batch 50    # stop after N labels this session
  python -m scripts.bootstrap_review --random      # shuffle order (no anchor)
  python -m scripts.bootstrap_review --stats       # show label stats and exit
  python -m scripts.bootstrap_review --thread conv-042   # review only one thread
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

# ── stdout ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from agent.config import REPO_ROOT
from agent.feedback_log.reader import (
    get_feedback_stats,
    list_pending_drafts,
    get_draft_with_actions,
)
from agent.feedback_log.writer import record_human_action

EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"
PROGRESS_PATH = REPO_ROOT / "data" / "bootstrap_progress.json"

# ──────────────────────────────────────────────────────────────────────────────
# Export ground-truth map
# ──────────────────────────────────────────────────────────────────────────────

def _load_ground_truth() -> dict[str, str]:
    """Returns {thread_id: last_admin_reply_text} for all threads in the export."""
    if not EXPORT_PATH.exists():
        return {}
    try:
        data = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    gt: dict[str, str] = {}
    for thread in data.get("threads") or []:
        tid = thread.get("id", "")
        # Find the last admin reply
        admin_reply = ""
        for msg in reversed(thread.get("messages") or []):
            if msg.get("role") == "admin" and (msg.get("text") or "").strip():
                admin_reply = msg["text"].strip()
                break
        if tid and admin_reply:
            gt[tid] = admin_reply
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap progress: condition map
# ──────────────────────────────────────────────────────────────────────────────

def _load_condition_map() -> dict[str, str]:
    """Returns {draft_id: condition_label} from bootstrap_progress.json."""
    if not PROGRESS_PATH.exists():
        return {}
    try:
        prog = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        item["draft_id"]: item["condition"]
        for item in prog.get("done", [])
        if isinstance(item, dict) and "draft_id" in item and "condition" in item
    }


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

_COL = 72  # wrap width


def _wrap(text: str, indent: str = "  ") -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para, _COL, initial_indent=indent, subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


def _sep(char: str = "-", width: int = 78) -> str:
    return char * width


def _display_draft(draft: dict, ground_truth: str, condition: str, n: int, total: int) -> None:
    print()
    print(_sep("="))
    print(
        f"  Draft {n}/{total}   thread={draft['thread_id']}   intent={draft['intent']}"
        f"   condition=[{condition}]   label={draft['confidence_label']}"
    )
    print(_sep("="))

    # Confidence details
    flags = draft.get("uncertainty_flags") or []
    sources = draft.get("cited_sources") or []
    guardrail = draft.get("guardrail_triggered", False)

    if guardrail:
        print("  !! GUARDRAIL TRIGGERED (no wiki citation for sensitive intent)")
    if flags:
        print(f"  uncertainty_flags: {', '.join(flags)}")
    if sources:
        print(f"  cited_sources    : {', '.join(sources)}")
    print(
        f"  model={draft.get('model','?')}   "
        f"cost=${draft.get('cost_usd', 0):.4f}   "
        f"tokens={draft.get('input_tokens',0)}+{draft.get('output_tokens',0)}"
    )
    print()

    # Ground truth side
    print(_sep("-"))
    print("  GROUND-TRUTH admin reply (from export):")
    print(_sep("-"))
    if ground_truth:
        print(_wrap(ground_truth))
    else:
        print("  (no admin reply found in export for this thread)")
    print()

    # Agent draft side
    print(_sep("-"))
    print("  AGENT DRAFT:")
    print(_sep("-"))
    print(_wrap(draft.get("reply") or "(empty)"))
    print()
    print(_sep("-"))


def _prompt_action() -> tuple[str, str]:
    """Prompt until the user enters a valid action. Returns (action, final_text)."""
    VALID = {"a", "e", "r", "s", "q"}
    while True:
        raw = input("  Action [a=approve / e=edit / r=reject / s=skip / q=quit]: ").strip().lower()
        if raw not in VALID:
            print(f"  Invalid input {raw!r}. Use a / e / r / s / q.")
            continue

        if raw == "e":
            print("  Enter your corrected reply (press Enter twice to finish):")
            lines = []
            blank = 0
            while blank < 1:
                try:
                    line = input()
                except EOFError:
                    break
                if not line.strip():
                    blank += 1
                else:
                    blank = 0
                    lines.append(line)
            final_text = "\n".join(lines).strip()
            if not final_text:
                print("  No text entered — switching to reject.")
                return "rejected", ""
            return "edited", final_text

        action_map = {"a": "approved", "r": "rejected", "s": "skip", "q": "quit"}
        return action_map.get(raw, raw), ""


# ──────────────────────────────────────────────────────────────────────────────
# Stats display
# ──────────────────────────────────────────────────────────────────────────────

def _show_stats() -> None:
    stats = get_feedback_stats()
    print("\n=== Feedback DB Stats ===")
    print(f"  Total drafts   : {stats.get('total_drafts', 0)}")
    print(f"  Pending review : {stats.get('pending_review', 0)}")
    print(f"  Total actions  : {stats.get('total_actions', 0)}")
    ab = stats.get("action_breakdown") or {}
    for action, count in sorted(ab.items()):
        print(f"    {action:<12}: {count}")
    print(f"  Total cost     : ${stats.get('total_cost_usd', 0):.4f}")

    # Phase 6 goal: >=300 labelled with >=25% in each bucket
    total_actions = stats.get("total_actions", 0)
    approved = ab.get("approved", 0)
    edited   = ab.get("edited", 0)
    rejected = ab.get("rejected", 0)
    print()
    print("=== Phase 6 Progress ===")
    goal = 300
    pct = (total_actions / goal * 100) if goal else 0
    print(f"  Labelled: {total_actions}/{goal}  ({pct:.0f}%)")
    for label, count in [("approved", approved), ("edited", edited), ("rejected", rejected)]:
        share = count / total_actions * 100 if total_actions else 0
        ok = "OK" if share >= 25 or total_actions < 30 else "LOW"
        print(f"    {label:<12}: {count:4d}  ({share:.0f}%)  {ok}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main review loop
# ──────────────────────────────────────────────────────────────────────────────

def run(
    batch: int | None,
    randomise: bool,
    stats_only: bool,
    single_thread: str | None,
) -> None:
    if stats_only:
        _show_stats()
        return

    ground_truth = _load_ground_truth()
    condition_map = _load_condition_map()

    # Load pending drafts (those with no human_action yet)
    pending = list_pending_drafts(limit=10_000)

    # Filter by thread if requested
    if single_thread:
        pending = [d for d in pending if d.get("thread_id") == single_thread]
        if not pending:
            print(f"No pending drafts for thread {single_thread!r}.")
            return

    # Sort: condition A first (anchor), then B, C, D — within each condition by thread
    CONDITION_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "": 9}

    def _sort_key(d: dict) -> tuple:
        cond = condition_map.get(d["id"], "")
        return (CONDITION_ORDER.get(cond, 9), d.get("thread_id", ""))

    if randomise:
        random.shuffle(pending)
    else:
        pending.sort(key=_sort_key)

    if batch:
        pending = pending[:batch]

    if not pending:
        print("No pending drafts to review. Run bootstrap_generate.py first.")
        _show_stats()
        return

    print(f"\n=== Bootstrap Review ===")
    print(f"  {len(pending)} pending draft(s) to review (this session).")
    print("  Commands: a=approve  e=edit  r=reject  s=skip  q=quit")
    _show_stats()
    print(_sep("="))

    labelled = 0
    skipped = 0

    for i, draft_summary in enumerate(pending, 1):
        draft_id = draft_summary["id"]
        thread_id = draft_summary.get("thread_id", "")
        condition = condition_map.get(draft_id, "?")

        # Fetch full draft record (includes reply text)
        draft = get_draft_with_actions(draft_id)
        if not draft:
            continue

        gt = ground_truth.get(thread_id, "")
        _display_draft(draft, gt, condition, i, len(pending))

        try:
            action, final_text = _prompt_action()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  [interrupted — progress saved]")
            break

        if action == "quit":
            print("\n  Quitting. Progress saved.")
            break

        if action == "skip":
            skipped += 1
            print("  Skipped.\n")
            continue

        # Record the action
        try:
            record_human_action(
                draft_id=draft_id,
                action=action,
                final_text=final_text or None,
                reviewer="bootstrap",
            )
            labelled += 1
            print(f"  Recorded: {action}.\n")
        except Exception as exc:
            print(f"  [ERROR] Could not record action: {exc}\n")

    # ── end-of-session summary ──
    print()
    print("=== Session Summary ===")
    print(f"  Labelled this session : {labelled}")
    print(f"  Skipped               : {skipped}")
    _show_stats()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6: interactive bootstrap review")
    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="Stop after reviewing N drafts this session.",
    )
    ap.add_argument(
        "--random",
        dest="randomise",
        action="store_true",
        help="Shuffle review order instead of condition-A-first.",
    )
    ap.add_argument(
        "--stats",
        dest="stats_only",
        action="store_true",
        help="Print label statistics and exit without reviewing.",
    )
    ap.add_argument(
        "--thread",
        default=None,
        metavar="THREAD_ID",
        help="Review only pending drafts for this thread (e.g. conv-042).",
    )
    args = ap.parse_args()

    run(
        batch=args.batch,
        randomise=args.randomise,
        stats_only=args.stats_only,
        single_thread=args.thread,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
