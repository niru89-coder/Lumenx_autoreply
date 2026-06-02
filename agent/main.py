"""FastAPI application entry point — LumenX Auto-Reply Agent.

Phase 5: HTTP API for the feedback log (drafts, actions, stats).
Phase 8: Background inbox poller + auto-send router.

Run locally:
  uvicorn agent.main:app --reload --port 8000

Environment variables (.env):
  ANTHROPIC_API_KEY            required
  LUMENX_ADMIN_TOKEN           required
  LUMENX_BASE_URL              default: https://lumenx-demo.up.railway.app
  LUMENX_POLL_INTERVAL_SECONDS default: 5
  AUTO_SEND_ENABLED            default: false  (must opt-in after Phase 7)
  AUTO_SEND_THRESHOLD          default: 0.90
  AUTO_SEND_BLOCKED_INTENTS    default: pricing,cancellation
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.config import REPO_ROOT, settings
from agent.feedback_log.db import ensure_tables

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("agent.main")

# ── Poller state ──────────────────────────────────────────────────────────────

_POLLER_STATE_PATH = REPO_ROOT / "data" / "poller_state.json"

_poller_stats: dict = {
    "cycles": 0,
    "threads_processed": 0,
    "auto_sent": 0,
    "human_review": 0,
    "errors": 0,
    "last_poll_ts": None,
}


def _load_last_poll_ts() -> str | None:
    if _POLLER_STATE_PATH.exists():
        try:
            d = json.loads(_POLLER_STATE_PATH.read_text(encoding="utf-8"))
            return d.get("last_poll_ts")
        except Exception:
            pass
    return None


def _save_last_poll_ts(ts: str) -> None:
    _POLLER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _POLLER_STATE_PATH.write_text(
        json.dumps({"last_poll_ts": ts}, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Pipeline (one thread) ─────────────────────────────────────────────────────

def _process_thread(thread_id: str) -> str:
    """Run the full pipeline for one thread.  Returns "auto_send" | "human_review" | "skip".

    Runs in a threadpool (called via asyncio.to_thread) so sync httpx and
    Chroma queries don't block the event loop.
    """
    from agent.context_builder import build_context, _last_customer_message
    from agent.drafter import draft_reply
    from agent.intent_router import classify_intent
    from agent.lumenx_client import LumenXClient
    from agent.router import route

    with LumenXClient() as client:
        # 1. Fetch full thread
        try:
            raw = client.get_thread(thread_id)
        except Exception as exc:
            logger.warning("fetch thread %s failed: %s", thread_id, exc)
            return "skip"

        if raw is None:
            return "skip"
        thread: dict = raw.get("thread", raw) if isinstance(raw, dict) else raw

        # 2. Guard: only process threads where the last message is from a customer
        last_msg = _last_customer_message(thread)
        if last_msg is None:
            logger.debug("thread %s: no customer message — skipping", thread_id)
            return "skip"

        # 3. Guard: skip threads where we already have a recent draft pending
        #    (avoids generating duplicate drafts on every poll cycle)
        from agent.feedback_log.db import get_session
        from agent.feedback_log.models import DraftRow, HumanActionRow
        from sqlalchemy import select
        with get_session() as sess:
            existing = sess.scalar(
                select(DraftRow)
                .outerjoin(DraftRow.human_actions)
                .where(DraftRow.thread_id == thread_id)
                .where(HumanActionRow.id.is_(None))  # no action yet
                .limit(1)
            )
        if existing:
            logger.debug(
                "thread %s: pending draft %s already exists — skipping",
                thread_id, existing.id[:8],
            )
            return "skip"

        # 4. Classify intent (Haiku — cheap)
        try:
            intent = classify_intent(last_msg["text"])
        except Exception as exc:
            logger.warning("classify_intent failed for %s: %s", thread_id, exc)
            return "skip"

        # 5. Build context
        try:
            ctx = build_context(thread, intent)
        except Exception as exc:
            logger.warning("build_context failed for %s: %s", thread_id, exc)
            return "skip"

        # 6. Draft (Sonnet)
        try:
            draft = draft_reply(ctx)
        except Exception as exc:
            logger.error("draft_reply failed for %s: %s", thread_id, exc)
            return "skip"

        # 7. Route (hard vetoes + MLP gate)
        try:
            decision = route(draft, client=client)
        except Exception as exc:
            logger.error("route failed for %s: %s", thread_id, exc)
            return "human_review"

        logger.info(
            "thread=%-12s intent=%-18s action=%s  score=%s  reason=%s",
            thread_id, intent.intent, decision.action,
            f"{decision.score:.3f}" if decision.score is not None else "n/a",
            decision.reason,
        )
        return decision.action


# ── Poller loop ───────────────────────────────────────────────────────────────

async def _poll_once() -> None:
    """Single poll cycle: get inbox, process new threads."""
    from agent.lumenx_client import LumenXClient

    now_ts = datetime.now(timezone.utc).isoformat()
    last_ts = _poller_stats.get("last_poll_ts")

    try:
        with LumenXClient() as client:
            inbox = client.get_inbox(since=last_ts)
    except Exception as exc:
        logger.warning("inbox poll failed: %s", exc)
        _poller_stats["errors"] += 1
        return

    threads = inbox if isinstance(inbox, list) else inbox.get("threads") or []
    _poller_stats["cycles"] += 1
    _poller_stats["last_poll_ts"] = now_ts
    _save_last_poll_ts(now_ts)

    for thread_meta in threads:
        tid = thread_meta.get("id") or thread_meta.get("thread_id")
        if not tid:
            continue
        try:
            result = await asyncio.to_thread(_process_thread, tid)
            _poller_stats["threads_processed"] += 1
            if result == "auto_send":
                _poller_stats["auto_sent"] += 1
            elif result == "human_review":
                _poller_stats["human_review"] += 1
        except Exception as exc:
            logger.error("_process_thread(%s) raised: %s", tid, exc)
            _poller_stats["errors"] += 1


async def _poll_loop() -> None:
    """Background coroutine: polls the LumenX inbox every POLL_INTERVAL seconds."""
    interval = settings.LUMENX_POLL_INTERVAL_SECONDS
    logger.info(
        "Inbox poller started  interval=%ds  auto_send=%s  threshold=%.2f",
        interval, settings.AUTO_SEND_ENABLED, settings.AUTO_SEND_THRESHOLD,
    )
    # Load last poll timestamp from persistent state so restarts don't reprocess
    _poller_stats["last_poll_ts"] = _load_last_poll_ts()

    while True:
        try:
            await _poll_once()
        except Exception as exc:
            logger.exception("_poll_once raised unexpectedly: %s", exc)
        await asyncio.sleep(interval)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LumenX Auto-Reply Agent",
    description="Internal API: drafts, feedback, costs, and auto-send router.",
    version="0.8.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_cors_origins = [
    o.strip()
    for o in (
        ["http://localhost:3000", "http://localhost:3001"]
        + [o for o in settings.DASHBOARD_URL.split(",") if o.strip()]
    )
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    ensure_tables()
    logger.info("Feedback DB ready")
    # Start background poller
    asyncio.create_task(_poll_loop())
    logger.info("Background poller scheduled")


# ── Routers ───────────────────────────────────────────────────────────────────

from agent.api.feedback import router as feedback_router  # noqa: E402
from agent.api.models import router as models_router  # noqa: E402

app.include_router(feedback_router)
app.include_router(models_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    from agent.feedback_log.reader import get_feedback_stats
    stats = get_feedback_stats()
    return {
        "status": "ok",
        "version": "0.8.0",
        "drafts": stats["total_drafts"],
        "pending_review": stats["pending_review"],
        "auto_send_enabled": settings.AUTO_SEND_ENABLED,
        "auto_send_threshold": settings.AUTO_SEND_THRESHOLD,
        "poller": _poller_stats,
    }


# ── Router control ────────────────────────────────────────────────────────────

@app.post("/api/router/reload-scorer")
async def reload_scorer() -> dict:
    """Force the ConfidenceScorer to reload from the latest checkpoint.
    Call this after retraining the Confidence Net without restarting uvicorn.
    """
    from agent.api.feedback import verify_admin
    from agent.router import _reset_scorer, _get_scorer
    _reset_scorer()
    scorer = _get_scorer()
    if scorer is None:
        return {"status": "no_checkpoint", "message": "No trained checkpoint found"}
    return {
        "status": "ok",
        "version": scorer.version,
        "val_auc": scorer.val_auc,
        "temperature": scorer.temperature,
    }


@app.post("/api/router/process-thread/{thread_id}")
async def process_thread_endpoint(thread_id: str, dry_run: bool = True) -> dict:
    """Run the full pipeline for a single thread (for testing/demo).

    dry_run=true (default): evaluates routing but does NOT send or write to DB.
    dry_run=false: full live run — auto-sends if eligible.

    Auth: X-Admin-Token header required.
    """
    from agent.api.feedback import verify_admin
    from agent.context_builder import build_context, _last_customer_message
    from agent.drafter import draft_reply
    from agent.intent_router import classify_intent
    from agent.lumenx_client import LumenXClient
    from agent.router import route

    try:
        with LumenXClient() as client:
            raw = client.get_thread(thread_id)

        if raw is None:
            return {"error": f"Thread {thread_id!r} not found"}

        thread: dict = raw.get("thread", raw) if isinstance(raw, dict) else raw
        last_msg = _last_customer_message(thread)
        if last_msg is None:
            return {"error": "No customer message found in thread"}

        intent = await asyncio.to_thread(classify_intent, last_msg["text"])
        ctx = await asyncio.to_thread(build_context, thread, intent)
        draft = await asyncio.to_thread(draft_reply, ctx)

        decision = route(draft, dry_run=dry_run)

        return {
            "thread_id": thread_id,
            "intent": intent.intent,
            "confidence_label": draft.confidence_label,
            "draft_id": draft.draft_id,
            "reply_preview": draft.reply[:200],
            "routing": {
                "action": decision.action,
                "reason": decision.reason,
                "score": decision.score,
                "threshold": decision.threshold,
                "sent": decision.sent,
                "dry_run": dry_run,
            },
            "cost_usd": draft.cost_usd,
        }
    except Exception as exc:
        logger.exception("process_thread_endpoint(%s) failed", thread_id)
        return {"error": str(exc)}


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("agent.main:app", host="0.0.0.0", port=8000, reload=True)
