"""End-to-end sanity check for Phase 0.

Verifies:
  1. LumenX /api/admin/stats is reachable and the admin token works.
  2. LumenX /api/admin/products returns the product catalogue.
  3. The Anthropic API is reachable via the wrapped client (logs tokens + cost).

Prints a clean summary. Secrets never appear in stdout.

Run: python -m scripts.healthcheck
"""
from __future__ import annotations

import sys

from agent.anthropic_client import MODEL_HAIKU, call_llm
from agent.lumenx_client import LumenXClient


def _section(label: str, ok: bool) -> str:
    return f"[{'OK  ' if ok else 'FAIL'}] {label}"


def main() -> int:
    print("=" * 64)
    print(" LumenX Auto-Reply Agent — Phase 0 healthcheck")
    print("=" * 64)
    all_ok = True

    # 1) LumenX stats
    try:
        with LumenXClient() as client:
            stats = client.get_stats()
        print()
        print(_section("LumenX /api/admin/stats", True))
        for k, v in (stats or {}).items():
            print(f"        {k}: {v}")
    except Exception as e:
        all_ok = False
        print()
        print(_section("LumenX /api/admin/stats", False))
        print(f"        error: {type(e).__name__}: {e}")

    # 2) LumenX products
    try:
        with LumenXClient() as client:
            data = client.get_products()
        products = data["products"] if isinstance(data, dict) and "products" in data else data
        count = len(products) if isinstance(products, list) else "?"
        print()
        print(_section(f"LumenX /api/admin/products  ({count} products)", True))
        if isinstance(products, list) and products:
            preview = ", ".join(str(p.get("id", "?")) for p in products[:5])
            print(f"        first 5 ids: {preview}")
    except Exception as e:
        all_ok = False
        print()
        print(_section("LumenX /api/admin/products", False))
        print(f"        error: {type(e).__name__}: {e}")

    # 3) Anthropic Haiku echo
    try:
        result = call_llm(
            feature="healthcheck",
            model=MODEL_HAIKU,
            max_tokens=20,
            temperature=0,
            messages=[
                {"role": "user", "content": "Reply with exactly the two characters: OK"}
            ],
        )
        print()
        print(_section(f"Anthropic Haiku  ({result.model})", True))
        print(f"        reply:   {result.text.strip()!r}")
        print(f"        tokens:  in={result.input_tokens}  out={result.output_tokens}")
        print(f"        cost:    ${result.cost_usd:.6f}")
        print(f"        latency: {result.latency_ms} ms")
    except Exception as e:
        all_ok = False
        print()
        print(_section("Anthropic Haiku", False))
        print(f"        error: {type(e).__name__}: {e}")

    print()
    print("=" * 64)
    print(f" RESULT: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
