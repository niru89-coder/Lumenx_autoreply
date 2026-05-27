"""Quick check: print feedback DB stats and pending drafts."""
from agent.feedback_log.reader import get_feedback_stats, list_drafts, list_pending_drafts
import json

stats = get_feedback_stats()
print("=== Feedback Stats ===")
print(json.dumps(stats, indent=2))

print()
print("=== All Drafts ===")
drafts = list_drafts(limit=20)
print(f"  {len(drafts)} draft(s) in DB")
for d in drafts:
    print(f"  {d['id'][:8]}  thread={d['thread_id']}  intent={d['intent']}  label={d['confidence_label']}  auto_sendable={d['auto_sendable']}")

print()
print("=== Pending (no action) ===")
pending = list_pending_drafts()
print(f"  {len(pending)} pending")
for d in pending:
    print(f"  {d['id'][:8]}  thread={d['thread_id']}  label={d['confidence_label']}")
