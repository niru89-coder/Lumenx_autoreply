"""One-off: dump the customer messages for a set of mispredicted threads."""
import json, sys
from agent.config import REPO_ROOT

ids = sys.argv[1:] or ["conv-007", "conv-099", "conv-096", "conv-054", "conv-025"]
export = json.loads((REPO_ROOT / "data" / "raw" / "export.json").read_text(encoding="utf-8"))
threads = {t["id"]: t for t in export["threads"]}
for tid in ids:
    t = threads.get(tid)
    if not t:
        print(f"{tid}: not found"); continue
    msg = next((m for m in t["messages"] if m.get("role") == "customer"), None)
    print(f"{tid}  intent={t['intent']!r}  product={t['product_id']}")
    if msg:
        print(f"  msg: {msg['text']}")
    print()
