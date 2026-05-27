"""Extract (customer Q, admin A) pairs from the cached export and index them
into Chroma as a historical Q&A pool. The context builder retrieves from
this pool to give the drafter "similar already-resolved" examples.

Phase 5's feedback log will replace this with approved drafts; same interface.

Run: python -m scripts.build_qa_pool
"""
from __future__ import annotations

import sys

from agent.historical_qa import HistoricalQAPool, entries_from_export


def main() -> int:
    print("Building historical Q&A pool from data/raw/export.json")
    entries = entries_from_export()
    print(f"  extracted {len(entries)} thread Q&A pairs")
    pool = HistoricalQAPool()
    pool.reset()
    n = pool.index(entries)
    print(f"  indexed {n} entries into Chroma collection at data/qa_chroma/")
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
