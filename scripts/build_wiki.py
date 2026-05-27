"""Build the LLM Wiki end-to-end:

  1. Load cached raw data (run scripts/pull_export.py first).
  2. Distil each product + the company into markdown under data/wiki/.
  3. Reset and re-populate the Chroma collection at data/wiki_chroma/.

Run: python -m scripts.build_wiki
"""
from __future__ import annotations

import sys

from agent.llm_wiki.builder import WIKI_DIR, build_wiki
from agent.llm_wiki.retriever import CHROMA_DIR, WikiRetriever


def main() -> int:
    print("Building LumenX LLM Wiki")
    chunks = build_wiki()
    print(f"  wrote {len(chunks)} chunks across {WIKI_DIR}")

    retriever = WikiRetriever()
    print(f"  resetting Chroma collection at {CHROMA_DIR}")
    retriever.reset()
    n = retriever.index(chunks)
    print(f"  indexed {n} chunks")
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
