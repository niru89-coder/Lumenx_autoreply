"""Cost summary from data/llm_calls.jsonl — Phase 11 (extends Phase 9 CLI).

Run as a CLI (prints to stdout):
    python -m scripts.cost_summary
    python -m scripts.cost_summary --days 7

Run as the weekly Railway cron (also delivers to Slack/email):
    python -m scripts.cost_summary --notify --days 7

``--notify`` routes the same text through agent.notifier so it lands in
Slack (or email / stdout fallback).  The window defaults to all-time for the
CLI and is typically set to 7 for the weekly job.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from agent.config import REPO_ROOT

# UTF-8 stdout so the 📊 emoji in --notify output is safe on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_ts(raw: str | None):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def build_summary(days: int | None = None) -> tuple[str, float]:
    """Return (text, total_cost_usd) for the last `days` (None = all-time)."""
    log = REPO_ROOT / "data" / "llm_calls.jsonl"
    if not log.exists():
        return "no calls logged yet", 0.0

    entries = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cutoff = None
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        entries = [
            e for e in entries
            if (ts := _parse_ts(e.get("ts"))) is not None and ts >= cutoff
        ]

    if not entries:
        window = "ever" if days is None else f"in the last {days} day(s)"
        return f"no calls logged {window}", 0.0

    by_feature: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0}
    )
    by_day: dict[str, float] = defaultdict(float)
    for e in entries:
        f = e.get("feature", "?")
        by_feature[f]["calls"] += 1
        by_feature[f]["in_tokens"] += e.get("input_tokens", 0)
        by_feature[f]["out_tokens"] += e.get("output_tokens", 0)
        by_feature[f]["cost"] += e.get("cost_usd", 0.0)
        ts = _parse_ts(e.get("ts"))
        if ts is not None:
            by_day[ts.date().isoformat()] += e.get("cost_usd", 0.0)

    total_cost = sum(v["cost"] for v in by_feature.values())
    total_calls = sum(v["calls"] for v in by_feature.values())

    window = "all time" if days is None else f"last {days} day(s)"
    lines = [
        f"LumenX agent LLM spend ({window})",
        f"Total: ${total_cost:.4f} across {total_calls} calls",
        "",
        f"  {'feature':<22} {'calls':>7} {'in_tok':>10} {'out_tok':>10} {'cost USD':>10}",
    ]
    for f, v in sorted(by_feature.items(), key=lambda kv: -kv[1]["cost"]):
        lines.append(
            f"  {f:<22} {int(v['calls']):>7} "
            f"{int(v['in_tokens']):>10,} {int(v['out_tokens']):>10,} "
            f"{v['cost']:>10.4f}"
        )
    if by_day:
        lines.append("")
        lines.append("  by day:")
        for day in sorted(by_day):
            lines.append(f"    {day}   ${by_day[day]:.4f}")

    return "\n".join(lines), total_cost


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM cost summary")
    ap.add_argument("--days", type=int, default=None,
                    help="restrict to the last N days (default: all time)")
    ap.add_argument("--notify", action="store_true",
                    help="also deliver via Slack/email (agent.notifier)")
    args = ap.parse_args()

    text, _total = build_summary(days=args.days)
    print(text)

    if args.notify:
        from agent.notifier import notify
        window = "weekly" if args.days else "all-time"
        sent = notify(f"📊 LumenX agent {window} cost summary", text)
        print(f"\n[delivered via: {', '.join(sent)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
