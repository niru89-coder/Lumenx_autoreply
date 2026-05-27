"""Cache LumenX raw data locally so wiki building and bootstrap labelling
can iterate offline.

Writes:
  data/raw/products.json  — /api/admin/products  (20 products + company policies)
  data/raw/export.json    — /api/admin/export    (all threads, all messages)

Idempotent. Re-run to refresh.

Run: python -m scripts.pull_export
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.config import REPO_ROOT
from agent.lumenx_client import LumenXClient

RAW_DIR = REPO_ROOT / "data" / "raw"


def _save(obj: object, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return len(text)


def _top_level(o: object) -> str:
    if isinstance(o, dict):
        return "dict with keys: " + ", ".join(sorted(o.keys()))
    if isinstance(o, list):
        return f"list of {len(o)}"
    return type(o).__name__


def main() -> int:
    print("Pulling LumenX raw data into", RAW_DIR)
    with LumenXClient() as client:
        products = client.get_products()
        export = client.get_export()

    n1 = _save(products, RAW_DIR / "products.json")
    n2 = _save(export, RAW_DIR / "export.json")

    print(f"  products.json  {n1:>10,} bytes  ({_top_level(products)})")
    print(f"  export.json    {n2:>10,} bytes  ({_top_level(export)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
