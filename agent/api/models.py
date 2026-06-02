"""Model registry HTTP endpoints — Phase 11.

Endpoints
---------
GET  /api/models                       list all checkpoints with metadata + is_active
GET  /api/models/active                metadata for the currently-served version
POST /api/models/{version}/promote     pin active.json to `version`; hot-reload scorer

Promoting a candidate is a deliberate human action — `weekly_retrain.py`
saves new versions but never promotes.  The reviewer inspects the val AUC
on the dashboard Models page and clicks Promote when satisfied.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from agent.api.feedback import AdminAuth
from agent.confidence_net import registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", dependencies=[AdminAuth])
def route_list_models() -> dict:
    """List every Confidence Net checkpoint on disk, newest first."""
    checkpoints = registry.list_checkpoints()
    return {
        "active_version": registry.get_active_version(),
        "active_record": registry.active_record(),
        "checkpoints": checkpoints,
    }


@router.get("/active", dependencies=[AdminAuth])
def route_active() -> dict:
    """Metadata for the version the scorer is (or would be) serving."""
    serving = registry.resolve_serving_version()
    if serving is None:
        raise HTTPException(status_code=404, detail="No checkpoint trained yet")
    for entry in registry.list_checkpoints():
        if entry["version"] == serving:
            return {
                "serving_version": serving,
                "active_record": registry.active_record(),
                "checkpoint": entry,
            }
    raise HTTPException(status_code=500, detail="Serving version not found on disk")


@router.post("/{version}/promote", dependencies=[AdminAuth])
def route_promote(version: int, reviewer: str = "human") -> dict:
    """Make `version` the active checkpoint and hot-reload the router's scorer."""
    try:
        record = registry.set_active(version, promoted_by=reviewer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Hot-reload so the next routing decision uses the new checkpoint.
    try:
        from agent.router import _get_scorer, _reset_scorer
        _reset_scorer()
        scorer = _get_scorer()
        loaded = (
            {
                "version": scorer.version,
                "val_auc": scorer.val_auc,
                "temperature": scorer.temperature,
            }
            if scorer is not None
            else None
        )
    except Exception as exc:
        logger.warning("Scorer hot-reload failed after promote: %s", exc)
        loaded = None

    return {"promoted": record, "loaded": loaded}
