"""Read helpers for the feedback log.

All functions return plain dicts (JSON-serialisable) so the API layer doesn't
need to import SQLAlchemy models directly.

Naming conventions
  _row_to_dict   — converts a single ORM row to a dict
  get_*          — fetches one record by id
  list_*         — fetches a page of records
  get_*_stats    — aggregation / summary
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from agent.feedback_log.db import get_session
from agent.feedback_log.models import ConfidencePredictionRow, DraftRow, HumanActionRow

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Row → dict converters
# ─────────────────────────────────────────────────────────────────────────────


def _draft_to_dict(row: DraftRow, include_snapshot: bool = False) -> dict[str, Any]:
    d = {
        "id": row.id,
        "thread_id": row.thread_id,
        "intent": row.intent,
        "sensitivity": row.sensitivity,
        "draft_text": row.draft_text,
        "cited_sources": _load_json(row.cited_sources, []),
        "uncertainty_flags": _load_json(row.uncertainty_flags, []),
        "confidence_label": row.confidence_label,
        "auto_sendable": row.auto_sendable,
        "guardrail_triggered": row.guardrail_triggered,
        "model": row.model,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "cost_usd": row.cost_usd,
        "latency_ms": row.latency_ms,
        "parse_attempts": row.parse_attempts,
        "context_cache_key": row.context_cache_key,
        "created_at": _iso(row.created_at),
    }
    if include_snapshot and row.context_snapshot_json:
        d["context_snapshot"] = _load_json(row.context_snapshot_json, None)
    return d


def _action_to_dict(row: HumanActionRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "draft_id": row.draft_id,
        "action": row.action,
        "final_text": row.final_text,
        "edit_distance": row.edit_distance,
        "reviewer": row.reviewer,
        "decided_at": _iso(row.decided_at),
    }


def _prediction_to_dict(row: ConfidencePredictionRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "draft_id": row.draft_id,
        "features": _load_json(row.features_json, {}),
        "score": row.score,
        "threshold": row.threshold,
        "would_auto_send": row.would_auto_send,
        "model_version": row.model_version,
        "created_at": _iso(row.created_at),
    }


def _load_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Single-record fetchers
# ─────────────────────────────────────────────────────────────────────────────


def get_draft(draft_id: str, include_snapshot: bool = False) -> dict | None:
    """Return one draft by its UUID, or None if not found."""
    with get_session() as session:
        row = session.get(DraftRow, draft_id)
        if row is None:
            return None
        return _draft_to_dict(row, include_snapshot=include_snapshot)


def get_draft_with_actions(draft_id: str) -> dict | None:
    """Return a draft plus all its human_actions and confidence_predictions."""
    with get_session() as session:
        row = session.get(DraftRow, draft_id)
        if row is None:
            return None
        d = _draft_to_dict(row, include_snapshot=True)
        d["human_actions"] = [_action_to_dict(a) for a in row.human_actions]
        d["confidence_predictions"] = [_prediction_to_dict(p) for p in row.confidence_predictions]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# List fetchers
# ─────────────────────────────────────────────────────────────────────────────


def list_drafts(
    *,
    intent: str | None = None,
    confidence_label: str | None = None,
    auto_sendable: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Paginated list of drafts, most recent first.

    All filter arguments are optional. combine freely.
    """
    with get_session() as session:
        stmt = select(DraftRow).order_by(DraftRow.created_at.desc())
        if intent:
            stmt = stmt.where(DraftRow.intent == intent)
        if confidence_label:
            stmt = stmt.where(DraftRow.confidence_label == confidence_label)
        if auto_sendable is not None:
            stmt = stmt.where(DraftRow.auto_sendable == auto_sendable)
        stmt = stmt.limit(limit).offset(offset)
        rows = session.scalars(stmt).all()
        return [_draft_to_dict(r) for r in rows]


def list_pending_drafts(limit: int = 50) -> list[dict]:
    """Drafts that have no human_action yet — the reviewer's inbox.

    Returns the oldest first so reviewers work through them in order.
    """
    with get_session() as session:
        # LEFT JOIN with human_actions, keep only those with no match
        stmt = (
            select(DraftRow)
            .outerjoin(DraftRow.human_actions)
            .where(HumanActionRow.id.is_(None))
            .order_by(DraftRow.created_at.asc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()
        return [_draft_to_dict(r) for r in rows]


def list_actions(draft_id: str) -> list[dict]:
    """All human actions for a single draft, oldest first."""
    with get_session() as session:
        stmt = (
            select(HumanActionRow)
            .where(HumanActionRow.draft_id == draft_id)
            .order_by(HumanActionRow.decided_at.asc())
        )
        return [_action_to_dict(r) for r in session.scalars(stmt).all()]


# ─────────────────────────────────────────────────────────────────────────────
# Stats / aggregation
# ─────────────────────────────────────────────────────────────────────────────


def get_feedback_stats() -> dict[str, Any]:
    """Summary counts for the dashboard Costs / Overview page."""
    with get_session() as session:
        total_drafts = session.scalar(select(func.count()).select_from(DraftRow)) or 0
        total_actions = session.scalar(select(func.count()).select_from(HumanActionRow)) or 0

        # Pending = drafts without any action
        pending = session.scalar(
            select(func.count())
            .select_from(DraftRow)
            .outerjoin(DraftRow.human_actions)
            .where(HumanActionRow.id.is_(None))
        ) or 0

        # Action breakdown
        action_rows = session.execute(
            select(HumanActionRow.action, func.count().label("n"))
            .group_by(HumanActionRow.action)
        ).all()
        action_counts = {r.action: r.n for r in action_rows}

        # Confidence label breakdown
        label_rows = session.execute(
            select(DraftRow.confidence_label, func.count().label("n"))
            .group_by(DraftRow.confidence_label)
        ).all()
        label_counts = {r.confidence_label: r.n for r in label_rows}

        # Intent breakdown
        intent_rows = session.execute(
            select(DraftRow.intent, func.count().label("n"))
            .group_by(DraftRow.intent)
            .order_by(func.count().desc())
        ).all()
        intent_counts = {r.intent: r.n for r in intent_rows}

        # Cost totals
        cost_result = session.execute(
            select(
                func.sum(DraftRow.cost_usd).label("total_cost"),
                func.sum(DraftRow.input_tokens).label("total_in"),
                func.sum(DraftRow.output_tokens).label("total_out"),
            )
        ).one()

        # Guardrail hits
        guardrail_count = (
            session.scalar(
                select(func.count()).select_from(DraftRow).where(DraftRow.guardrail_triggered.is_(True))
            ) or 0
        )

    return {
        "total_drafts": total_drafts,
        "pending_review": pending,
        "total_actions": total_actions,
        "action_breakdown": action_counts,
        "confidence_breakdown": label_counts,
        "intent_breakdown": intent_counts,
        "guardrail_hits": guardrail_count,
        "total_cost_usd": round(float(cost_result.total_cost or 0), 6),
        "total_input_tokens": int(cost_result.total_in or 0),
        "total_output_tokens": int(cost_result.total_out or 0),
    }
