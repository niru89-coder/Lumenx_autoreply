"""Quick cost summary from data/llm_calls.jsonl.

Run: python -m scripts.cost_summary
"""
from __future__ import annotations

import json
from collections import defaultdict

from agent.config import REPO_ROOT


def main() -> int:
    log = REPO_ROOT / "data" / "llm_calls.jsonl"
    if not log.exists():
        print("no calls logged yet")
        return 0

    entries = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        print("no calls logged yet")
        return 0

    by_feature: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0}
    )
    for e in entries:
        f = e.get("feature", "?")
        by_feature[f]["calls"] += 1
        by_feature[f]["in_tokens"] += e.get("input_tokens", 0)
        by_feature[f]["out_tokens"] += e.get("output_tokens", 0)
        by_feature[f]["cost"] += e.get("cost_usd", 0.0)

    total_cost = sum(v["cost"] for v in by_feature.values())
    total_calls = sum(v["calls"] for v in by_feature.values())

    print(f"Total LLM spend: ${total_cost:.4f} across {total_calls} calls")
    print()
    print(f"  {'feature':<22} {'calls':>7} {'in_tok':>10} {'out_tok':>10} {'cost USD':>10}")
    for f, v in sorted(by_feature.items(), key=lambda kv: -kv[1]["cost"]):
        print(
            f"  {f:<22} {int(v['calls']):>7} "
            f"{int(v['in_tokens']):>10,} {int(v['out_tokens']):>10,} "
            f"{v['cost']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
