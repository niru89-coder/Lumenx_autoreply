"""Write helpers for the feedback log.

Every function is idempotent where possible:
  - record_draft uses INSERT OR REPLACE keyed on draft.id.
  - record_human_action always appends (a draft can have multiple actions
    over its lifetime, e.g. rejected then re-generated and approved).
  - record_confidence_prediction always appends.

Callers pass domain objects (drafter.Draft, plain dicts) — this module owns
the mapping to DB rows.
"""
from __future__ import annotations

import difflib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from agent.feedback_log.db import get_session
from agent.feedback_log.models import ConfidencePredictionRow, DraftRow, HumanActionRow

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _approx_edit_distance(a: str, b: str) -> int:
    """Fast approximation of character-level edit distance using difflib.

    Exact Levenshtein is O(n*m). For reply texts (typically < 2 000 chars)
    this is fine, but we cap at 5 000 chars to be safe.
    """
    a, b = a[:5000], b[:5000]
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    # ratio = 2*M / T  where M=matches, T=total chars
    # edit_distance ≈ (1 - ratio) * max_len  (lower bound approximation)
    return int((1.0 - ratio) * max(len(a), len(b)))


def _parse_dt(val: Any) -> datetime:
    """Parse ISO string or return datetime as-is."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# record_draft
# ─────────────────────────────────────────────────────────────────────────────


def record_draft(draft: Any, context_snapshot: dict | None = None) -> None:
    """Insert or replace a draft record from a drafter.Draft object or dict.

    Accepts both:
      • drafter.Draft dataclass instances (accessed via getattr)
      • plain dicts (from JSON migration)

    `context_snapshot` is the full ContextWindow.to_jsonable() dict; stored
    as JSON text for the dashboard drill-down view. Pass None to skip.
    """
    # Normalise to dict — works for both dataclasses and plain dicts
    if hasattr(draft, "to_dict"):
        d: dict = draft.to_dict()
    elif isinstance(draft, dict):
        d = draft
    else:
        raise TypeError(f"record_draft: unsupported type {type(draft)}")

    cited = d.get("cited_sources", [])
    flags = d.get("uncertainty_flags", [])
    snapshot_text = json.dumps(context_snapshot, ensure_ascii=False) if context_snapshot else None

    row_data = {
        "id": d["draft_id"],
        "thread_id": d["thread_id"],
        "intent": d["intent"],
        "sensitivity": d["sensitivity"],
        "draft_text": d["reply"],
        "cited_sources": json.dumps(cited, ensure_ascii=False),
        "uncertainty_flags": json.dumps(flags, ensure_ascii=False),
        "confidence_label": d["confidence_label"],
        "auto_sendable": bool(d["auto_sendable"]),
        "guardrail_triggered": bool(d.get("guardrail_triggered", False)),
        "model": d["model"],
        "input_tokens": int(d.get("input_tokens", 0)),
        "output_tokens": int(d.get("output_tokens", 0)),
        "cost_usd": float(d.get("cost_usd", 0.0)),
        "latency_ms": int(d.get("latency_ms", 0)),
        "parse_attempts": int(d.get("parse_attempts", 1)),
        "context_cache_key": d.get("context_cache_key"),
        "context_snapshot_json": snapshot_text,
        "created_at": _parse_dt(d.get("created_at")),
    }

    stmt = sqlite_insert(DraftRow).values(**row_data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={k: v for k, v in row_data.items() if k != "id"},
    )

    with get_session() as session:
        session.execute(stmt)

    logger.debug(
        "recorded draft %s (thread=%s intent=%s label=%s)",
        d["draft_id"][:8], d["thread_id"], d["intent"], d["confidence_label"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# record_human_action
# ─────────────────────────────────────────────────────────────────────────────

VALID_ACTIONS = frozenset({"approved", "edited", "rejected", "auto_sent"})


def record_human_action(
    draft_id: str,
    action: str,
    *,
    final_text: str | None = None,
    reviewer: str | None = "human",
) -> int:
    """Record a human (or agent) decision on a draft.

    Returns the auto-assigned `id` of the new HumanActionRow.

    For "approved" and "auto_sent": pass `final_text` = the text that was sent
    (often the same as draft_text, but may differ if lightly tweaked).

    For "edited": `final_text` is the human's revised version; edit_distance
    is computed automatically.

    For "rejected": `final_text` should be None (draft was discarded).

    Raises ValueError on invalid action.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid action {action!r}. Must be one of: {sorted(VALID_ACTIONS)}"
        )

    edit_dist: int | None = None
    if final_text is not None and action == "edited":
        # Fetch original draft_text to compute distance
        with get_session() as session:
            row = session.get(DraftRow, draft_id)
            if row and row.draft_text:
                edit_dist = _approx_edit_distance(row.draft_text, final_text)

    ha = HumanActionRow(
        draft_id=draft_id,
        action=action,
        final_text=final_text,
        edit_distance=edit_dist,
        reviewer=reviewer,
        decided_at=datetime.now(timezone.utc),
    )
    with get_session() as session:
        session.add(ha)
        session.flush()
        row_id = ha.id

    logger.info(
        "human_action: draft=%s action=%s reviewer=%s edit_dist=%s",
        draft_id[:8], action, reviewer, edit_dist,
    )
    return row_id


# ─────────────────────────────────────────────────────────────────────────────
# record_confidence_prediction
# ─────────────────────────────────────────────────────────────────────────────


def record_confidence_prediction(
    draft_id: str,
    features: dict,
    score: float,
    *,
    threshold: float = 0.90,
    model_version: str | None = None,
) -> int:
    """Record an MLP confidence score for a draft.

    `features` is the feature dict (will be JSON-serialised).
    `score` is the sigmoid output from the MLP (0.0–1.0).
    `threshold` is the decision boundary used at inference time.
    `model_version` is the checkpoint filename (e.g. "model_v1.pt").

    Returns the auto-assigned row id.
    """
    cp = ConfidencePredictionRow(
        draft_id=draft_id,
        features_json=json.dumps(features, ensure_ascii=False),
        score=float(score),
        threshold=float(threshold),
        would_auto_send=score >= threshold,
        model_version=model_version,
        created_at=datetime.now(timezone.utc),
    )
    with get_session() as session:
        session.add(cp)
        session.flush()
        row_id = cp.id

    logger.debug(
        "confidence_prediction: draft=%s score=%.3f threshold=%.2f auto_send=%s",
        draft_id[:8], score, threshold, score >= threshold,
    )
    return row_id
