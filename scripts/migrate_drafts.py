"""One-time migration: import all existing data/drafts/*.json files into SQLite.

Safe to re-run — uses INSERT OR REPLACE so duplicate IDs are skipped or updated.

Usage:
  python -m scripts.migrate_drafts
  python -m scripts.migrate_drafts --dry-run   # print what would be imported
"""
from __future__ import annotations

import argparse
import json
import sys

from agent.config import REPO_ROOT
from agent.feedback_log.db import ensure_tables
from agent.feedback_log.writer import record_draft

DRAFTS_DIR = REPO_ROOT / "data" / "drafts"


def main() -> int:
    ap = argparse.ArgumentParser(description="Import data/drafts/*.json into feedback.db")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be imported without writing")
    args = ap.parse_args()

    if not DRAFTS_DIR.exists():
        print(f"No drafts directory at {DRAFTS_DIR} — nothing to migrate.")
        return 0

    files = sorted(DRAFTS_DIR.glob("*.json"))
    if not files:
        print("No draft JSON files found.")
        return 0

    print(f"Found {len(files)} draft file(s) in {DRAFTS_DIR}")

    if not args.dry_run:
        ensure_tables()

    imported = 0
    errors = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP  {path.name}  -- read error: {e}")
            errors += 1
            continue

        draft_id = data.get("draft_id", "(no id)")

        if args.dry_run:
            print(f"  DRY   {path.name}  draft_id={draft_id[:8]}  thread={data.get('thread_id')}  label={data.get('confidence_label')}")
            imported += 1
            continue

        try:
            record_draft(data)
            print(f"  OK    {path.name}  draft_id={draft_id[:8]}  thread={data.get('thread_id')}  label={data.get('confidence_label')}")
            imported += 1
        except Exception as e:
            print(f"  ERROR {path.name}  -- {e}")
            errors += 1

    print()
    if args.dry_run:
        print(f"Dry run: would import {imported} draft(s).  Errors: {errors}")
    else:
        print(f"Imported {imported} draft(s) into data/feedback.db.  Errors: {errors}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
