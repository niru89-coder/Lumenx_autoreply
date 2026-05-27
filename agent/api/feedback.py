"""FastAPI router for the feedback log endpoints.

Mounted at /api by agent/main.py.

Endpoints
---------
GET  /api/drafts                       list drafts (filters: intent, label, auto_sendable, limit, offset)
GET  /api/drafts/pending               drafts awaiting human review (no action yet)
GET  /api/drafts/{draft_id}            single draft + actions + confidence predictions
GET  /api/drafts/{draft_id}/context    full context snapshot JSON
POST /api/drafts/{draft_id}/action     record a human or agent action
GET  /api/stats                        aggregated feedback stats

Auth
----
All endpoints check the X-Admin-Token header against LUMENX_ADMIN_TOKEN from
the environment. The dashboard sends this header on every request.
"""
from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from agent.config import settings
from agent.feedback_log.reader import (
    get_draft,
    get_draft_with_actions,
    get_feedback_stats,
    list_drafts,
    list_pending_drafts,
)
from agent.feedback_log.writer import (
    VALID_ACTIONS,
    record_human_action,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["feedback"])


# ─────────────────────────────────────────────────────────────────────────────
# Auth dependency
# ─────────────────────────────────────────────────────────────────────────────


def verify_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    """Raise 401 if the X-Admin-Token header is missing or wrong."""
    if x_admin_token != settings.LUMENX_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


AdminAuth = Depends(verify_admin)


# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────


class ActionRequest(BaseModel):
    action: Literal["approved", "edited", "rejected", "auto_sent"] = Field(
        ...,
        description="Reviewer decision. 'edited' requires final_text.",
    )
    final_text: str | None = Field(
        None,
        description="Text actually sent to the customer. Required for 'approved', 'edited', 'auto_sent'.",
    )
    reviewer: str | None = Field(
        "human",
        description="Reviewer username, or 'agent' for automated actions.",
    )


class ActionResponse(BaseModel):
    action_id: int
    draft_id: str
    action: str
    edit_distance: int | None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/drafts", dependencies=[AdminAuth])
def route_list_drafts(
    intent: str | None = Query(None, description="Filter by intent label"),
    label: str | None = Query(None, description="Filter by confidence_label (high/low/blocked)"),
    auto_sendable: bool | None = Query(None, description="Filter by auto_sendable flag"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    return list_drafts(
        intent=intent,
        confidence_label=label,
        auto_sendable=auto_sendable,
        limit=limit,
        offset=offset,
    )


@router.get("/drafts/pending", dependencies=[AdminAuth])
def route_pending_drafts(
    limit: int = Query(50, ge=1, le=200),
) -> list[dict]:
    """Drafts that have no human_action yet — the reviewer's inbox."""
    return list_pending_drafts(limit=limit)


@router.get("/drafts/{draft_id}", dependencies=[AdminAuth])
def route_get_draft(draft_id: str) -> dict:
    """Full draft record including all actions and confidence predictions."""
    d = get_draft_with_actions(draft_id)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id!r} not found")
    return d


@router.get("/drafts/{draft_id}/context", dependencies=[AdminAuth])
def route_get_context(draft_id: str) -> dict:
    """Return the serialised ContextWindow snapshot for a draft."""
    d = get_draft(draft_id, include_snapshot=True)
    if d is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id!r} not found")
    snapshot = d.get("context_snapshot")
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail="No context snapshot stored for this draft (run with context_snapshot=True)",
        )
    return snapshot


@router.post("/drafts/{draft_id}/action", dependencies=[AdminAuth])
def route_record_action(draft_id: str, body: ActionRequest) -> ActionResponse:
    """Record a human or agent decision on a draft.

    For 'approved' / 'auto_sent': provide final_text (the text that was sent).
    For 'edited': provide final_text (the revised version); edit_distance is computed.
    For 'rejected': final_text may be omitted (draft was discarded).
    """
    if body.action in {"approved", "auto_sent"} and body.final_text is None:
        raise HTTPException(
            status_code=422,
            detail=f"final_text is required for action={body.action!r}",
        )

    # Verify draft exists
    existing = get_draft(draft_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id!r} not found")

    action_id = record_human_action(
        draft_id=draft_id,
        action=body.action,
        final_text=body.final_text,
        reviewer=body.reviewer,
    )

    # Retrieve the newly created action to get edit_distance
    from agent.feedback_log.reader import list_actions
    actions = list_actions(draft_id)
    last_action = next((a for a in reversed(actions) if a["id"] == action_id), {})

    return ActionResponse(
        action_id=action_id,
        draft_id=draft_id,
        action=body.action,
        edit_distance=last_action.get("edit_distance"),
    )


@router.get("/stats", dependencies=[AdminAuth])
def route_stats() -> dict:
    """Aggregated feedback stats: draft counts, action breakdown, cost totals."""
    return get_feedback_stats()
