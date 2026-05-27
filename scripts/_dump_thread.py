"""Dump full content of a thread for inspection."""
import json, sys
from agent.config import REPO_ROOT

tid = sys.argv[1]
e = json.loads((REPO_ROOT / "data" / "raw" / "export.json").read_text(encoding="utf-8"))
t = next((th for th in e["threads"] if th["id"] == tid), None)
if not t:
    print(f"{tid}: not found"); sys.exit(1)
print(f"intent={t['intent']!r}  product={t['product_id']}  customer={t['customer_username']}")
for m in t.get("messages", []):
    print()
    print(f"[{m['role']:8s} {m['ts']}]")
    print(m["text"])
