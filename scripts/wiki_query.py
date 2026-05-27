"""Query the LLM Wiki from the command line.

  python -m scripts.wiki_query "what is emailpilot's refund window"
  python -m scripts.wiki_query --k 5 "non-profit discount"
"""
from __future__ import annotations

import argparse
import sys

from agent.llm_wiki.retriever import WikiRetriever


def main() -> int:
    ap = argparse.ArgumentParser(description="Query the LumenX LLM Wiki")
    ap.add_argument("query", nargs="+", help="free-form question")
    ap.add_argument("--k", type=int, default=3, help="number of hits to return")
    args = ap.parse_args()

    q = " ".join(args.query)
    retriever = WikiRetriever()
    hits = retriever.query(q, k=args.k)

    print(f"Query: {q!r}")
    print(f"Top {len(hits)} hits:")
    for i, h in enumerate(hits, 1):
        d = f"{h.distance:.3f}" if h.distance is not None else "?"
        print()
        print(f"  [{i}] distance={d}  product={h.product_id or '(company)'}  section={h.section}")
        print(f"      source: {h.source_path}")
        body = h.text.strip().splitlines()
        for line in body[:12]:
            print(f"      {line}")
        if len(body) > 12:
            print(f"      ... ({len(body) - 12} more lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
