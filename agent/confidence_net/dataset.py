"""Dataset loader for Confidence Net training — Phase 7.

Queries the feedback DB for all labelled drafts, encodes labels, extracts
features, and returns numpy arrays ready for PyTorch DataLoader.

Label encoding
--------------
  approved                          → 1.0
  rejected                          → 0.0
  edited, edit_distance > threshold → 0.0  (heavy rewrite ≈ rejected)
  edited, edit_distance ≤ threshold → DROPPED  (minor tweak, ambiguous)
  skip / no action                  → DROPPED  (not a label)

The drop-ambiguous policy keeps the training signal clean and is reversed
once we have enough data for soft-labelling in a future version.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from agent.confidence_net.features import extract, N_FEATURES
from agent.feedback_log.db import get_session
from agent.feedback_log.models import DraftRow, HumanActionRow
from sqlalchemy import select

logger = logging.getLogger(__name__)

# Edits with fewer than this many character changes are considered ambiguous.
# We drop rather than label them, following the PLAN.md decision.
EDIT_DISTANCE_THRESHOLD: int = 30


def load_labelled_dataset(
    edit_distance_threshold: int = EDIT_DISTANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load all labelled drafts from the feedback DB.

    Returns
    -------
    X : float32 ndarray, shape (n_samples, N_FEATURES)
    y : float32 ndarray, shape (n_samples,)  — 0.0 or 1.0
    meta : list of dicts with 'draft_id', 'thread_id', 'intent', 'action',
           'label', 'edit_distance'  (for diagnostics)
    """
    rows = _query_labelled_rows()
    X_list: list[list[float]] = []
    y_list: list[float] = []
    meta_list: list[dict[str, Any]] = []

    dropped_ambiguous = 0
    label_counts: Counter = Counter()

    for draft_dict, action_dict in rows:
        action = action_dict["action"]
        edit_dist = action_dict.get("edit_distance") or 0

        # ── encode label ──────────────────────────────────────────────────────
        if action == "approved":
            label = 1.0
        elif action == "rejected":
            label = 0.0
        elif action == "edited":
            if edit_dist > edit_distance_threshold:
                label = 0.0   # heavy rewrite → treat as rejected
            else:
                dropped_ambiguous += 1
                continue      # minor edit → drop
        else:
            # "skip" or anything unexpected
            continue

        vec = extract(draft_dict)
        X_list.append(vec)
        y_list.append(label)
        label_counts[action] += 1

        meta_list.append({
            "draft_id":      draft_dict["id"],
            "thread_id":     draft_dict.get("thread_id", ""),
            "intent":        draft_dict.get("intent", ""),
            "model":         draft_dict.get("model", ""),
            "action":        action,
            "label":         label,
            "edit_distance": edit_dist,
            "confidence_label": draft_dict.get("confidence_label", ""),
        })

    logger.info(
        "dataset: %d samples  (approved=%d rejected=%d edited-heavy=%d  dropped_ambiguous=%d)",
        len(X_list),
        sum(1 for m in meta_list if m["action"] == "approved"),
        sum(1 for m in meta_list if m["action"] == "rejected"),
        sum(1 for m in meta_list if m["action"] == "edited"),
        dropped_ambiguous,
    )

    if not X_list:
        return (
            np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta_list


def _query_labelled_rows() -> list[tuple[dict, dict]]:
    """Return [(draft_dict, action_dict)] for every draft that has at least
    one human action, using the first action only (assumes one per draft).
    """
    with get_session() as session:
        stmt = (
            select(DraftRow, HumanActionRow)
            .join(HumanActionRow, DraftRow.id == HumanActionRow.draft_id)
            .order_by(DraftRow.created_at.asc(), HumanActionRow.decided_at.asc())
        )
        rows = session.execute(stmt).all()

    # Keep only the first action per draft (in case multiple were recorded)
    seen: set[str] = set()
    result: list[tuple[dict, dict]] = []
    for draft_row, action_row in rows:
        if draft_row.id in seen:
            continue
        seen.add(draft_row.id)
        draft_dict = _draft_row_to_dict(draft_row)
        action_dict = _action_row_to_dict(action_row)
        result.append((draft_dict, action_dict))
    return result


def _draft_row_to_dict(row: DraftRow) -> dict[str, Any]:
    import json
    def _j(v):
        if not v:
            return []
        try:
            return json.loads(v)
        except Exception:
            return []

    return {
        "id":                 row.id,
        "thread_id":          row.thread_id,
        "intent":             row.intent,
        "sensitivity":        row.sensitivity,
        "draft_text":         row.draft_text or "",
        "cited_sources":      _j(row.cited_sources),
        "uncertainty_flags":  _j(row.uncertainty_flags),
        "confidence_label":   row.confidence_label or "",
        "auto_sendable":      bool(row.auto_sendable),
        "guardrail_triggered":bool(row.guardrail_triggered),
        "model":              row.model or "",
        "input_tokens":       row.input_tokens or 0,
        "output_tokens":      row.output_tokens or 0,
        "cost_usd":           float(row.cost_usd or 0),
        "latency_ms":         row.latency_ms or 0,
        "parse_attempts":     row.parse_attempts or 1,
    }


def _action_row_to_dict(row: HumanActionRow) -> dict[str, Any]:
    return {
        "id":            row.id,
        "draft_id":      row.draft_id,
        "action":        row.action,
        "final_text":    row.final_text or "",
        "edit_distance": row.edit_distance or 0,
        "reviewer":      row.reviewer or "",
    }


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply per-feature z-score normalisation. Exported for eval/scorer use."""
    return ((X - mean) / std).astype(np.float32)


def dataset_summary(X: np.ndarray, y: np.ndarray, meta: list[dict]) -> str:
    """Human-readable summary for the training CLI."""
    n = len(y)
    if n == 0:
        return "Dataset is empty — no labelled drafts found."
    n_pos = int(y.sum())
    n_neg = n - n_pos
    pct_pos = 100.0 * n_pos / n
    intents = Counter(m["intent"] for m in meta)
    top = ", ".join(f"{k}:{v}" for k, v in intents.most_common(5))
    return (
        f"n={n}  positive(approved)={n_pos} ({pct_pos:.0f}%)  "
        f"negative={n_neg} ({100-pct_pos:.0f}%)\n"
        f"  top intents: {top}"
    )
