"""Feedback log: SQLite-backed store for drafts, human actions, and MLP scores.

Public API (re-exported for convenience):
  from agent.feedback_log import record_draft, record_human_action, record_confidence_prediction
  from agent.feedback_log import get_draft, list_drafts, list_pending_drafts, get_feedback_stats
"""
from agent.feedback_log.writer import (
    record_draft,
    record_human_action,
    record_confidence_prediction,
)
from agent.feedback_log.reader import (
    get_draft,
    list_drafts,
    list_pending_drafts,
    get_draft_with_actions,
    get_feedback_stats,
)

__all__ = [
    "record_draft",
    "record_human_action",
    "record_confidence_prediction",
    "get_draft",
    "list_drafts",
    "list_pending_drafts",
    "get_draft_with_actions",
    "get_feedback_stats",
]
