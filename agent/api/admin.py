"""Admin trigger endpoints — Phase 11 cron hooks.

Railway cron services can't mount the agent's data volume (one volume → one
service). So instead of running the Phase 11 jobs in a separate service, a
cron service makes an authenticated HTTP call here and the work runs *inside*
the agent process, which already has the volume (feedback DB, cost log, model
checkpoints).

Endpoints (all require X-Admin-Token)
-------------------------------------
POST /api/admin/retrain        train a candidate Confidence Net checkpoint (never auto-promotes)
POST /api/admin/cost-summary   build the spend summary and deliver it (Slack/email/stdout)
POST /api/admin/run-alerts     run cost/error-rate checks; alert if a threshold trips

Each reuses the exact logic the CLI scripts use, run in a worker thread so the
event loop isn't blocked.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from fastapi import APIRouter, Query

from agent.api.feedback import AdminAuth
from agent.config import REPO_ROOT

# Ensure the repo root is importable so `scripts.*` resolves regardless of how
# uvicorn was launched (the editable install usually covers this; belt-and-suspenders).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/retrain", dependencies=[AdminAuth])
async def trigger_retrain() -> dict:
    """Train a new candidate checkpoint from all current labels.

    Does NOT promote — a human still promotes via the dashboard Models page.
    """
    from scripts.weekly_retrain import run_retrain

    entry = await asyncio.to_thread(run_retrain)
    return {"ok": entry.get("status") != "error", "result": entry}


@router.post("/cost-summary", dependencies=[AdminAuth])
async def trigger_cost_summary(
    days: int = Query(7, ge=1, le=365, description="Window in days"),
    deliver: bool = Query(True, description="Also send via Slack/email/stdout"),
) -> dict:
    """Build the LLM spend summary and (by default) deliver it via the notifier."""
    from scripts.cost_summary import build_summary

    text, total = await asyncio.to_thread(build_summary, days)
    channels: list[str] = []
    if deliver:
        from agent.notifier import notify

        channels = await asyncio.to_thread(
            notify, f"📊 LumenX agent {days}-day cost summary", text
        )
    return {
        "ok": True,
        "days": days,
        "total_usd": round(total, 6),
        "delivered_via": channels,
        "summary": text,
    }


@router.post("/run-alerts", dependencies=[AdminAuth])
async def trigger_alerts() -> dict:
    """Run the cost/day + poller error-rate checks; alert if a threshold trips."""
    from scripts.monitor_alerts import run_checks

    result = await asyncio.to_thread(run_checks)
    return {"ok": True, **result}


@router.post("/build-wiki", dependencies=[AdminAuth])
async def trigger_build_wiki() -> dict:
    """(Re)build the LLM Wiki on the volume: pull LumenX export -> distil
    markdown -> embed into Chroma. Needed once on a fresh deploy, since the
    wiki/Chroma index isn't baked into the image — it lives on the data volume.
    Local embeddings (no Anthropic cost). Idempotent: resets then re-indexes.
    """
    def _build() -> dict:
        from agent.llm_wiki.builder import build_wiki
        from agent.llm_wiki.retriever import WikiRetriever
        from scripts.pull_export import main as pull_export_main

        pull_export_main()                 # cache /api/admin/export + /products
        chunks = build_wiki()              # distil markdown from cached raw
        retriever = WikiRetriever()
        retriever.reset()                  # clear stale collection
        n = retriever.index(chunks)        # embed + index locally
        return {"chunks_built": len(chunks), "chunks_indexed": n}

    result = await asyncio.to_thread(_build)
    return {"ok": True, **result}
