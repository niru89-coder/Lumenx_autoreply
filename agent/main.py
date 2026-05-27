"""FastAPI application entry point for the LumenX auto-reply agent.

Phase 5: HTTP API for the feedback log (drafts, actions, stats).
Phase 8: Background poller + auto-send router (added later).

Run locally:
  uvicorn agent.main:app --reload --port 8000

Environment variables (loaded from .env):
  ANTHROPIC_API_KEY        — Anthropic API key
  LUMENX_ADMIN_TOKEN       — used to authenticate inbound API requests
  LUMENX_BASE_URL          — LumenX deployment URL
"""
from __future__ import annotations

import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.feedback_log.db import ensure_tables

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("agent.main")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LumenX Auto-Reply Agent",
    description="Internal API for the auto-reply agent: drafts, feedback, costs.",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the dashboard (localhost:3000 in dev, same domain in prod) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def on_startup() -> None:
    ensure_tables()
    logger.info("Feedback DB ready")


# ── Routers ───────────────────────────────────────────────────────────────────

from agent.api.feedback import router as feedback_router  # noqa: E402

app.include_router(feedback_router)


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    from agent.feedback_log.reader import get_feedback_stats
    stats = get_feedback_stats()
    return {
        "status": "ok",
        "drafts": stats["total_drafts"],
        "pending_review": stats["pending_review"],
    }


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("agent.main:app", host="0.0.0.0", port=8000, reload=True)
