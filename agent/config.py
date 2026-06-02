"""Central config — loads .env once, exposes typed settings.

Import `settings` from here; do not read os.environ directly elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env only when it exists (local development).  In production (Railway)
# the variables are already injected into os.environ, so we skip the file and
# use override=False to ensure Railway's values are never overwritten by a
# stale local .env that somehow ends up in the image.
_dotenv_path = REPO_ROOT / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path, override=False)


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required env var {name!r} is not set. "
            f"Set it in Railway service variables (production) or copy "
            f".env.example to .env and fill it in (local development)."
        )
    return val


class Settings:
    ANTHROPIC_API_KEY: str = _required("ANTHROPIC_API_KEY")
    LUMENX_BASE_URL: str = os.environ.get(
        "LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app"
    )
    LUMENX_ADMIN_TOKEN: str = _required("LUMENX_ADMIN_TOKEN")
    LLM_CALLS_LOG: Path = REPO_ROOT / os.environ.get(
        "LLM_CALLS_LOG", "data/llm_calls.jsonl"
    )
    LUMENX_POLL_INTERVAL_SECONDS: int = int(
        os.environ.get("LUMENX_POLL_INTERVAL_SECONDS", "5")
    )

    # ── Phase 8: Auto-send router ─────────────────────────────────────────────
    # Auto-send is OFF by default.  Enable only after Phase 7 AUC >= 0.75 and
    # after manually reviewing a batch of auto-sent replies.
    AUTO_SEND_ENABLED: bool = (
        os.environ.get("AUTO_SEND_ENABLED", "false").strip().lower() == "true"
    )
    # MLP probability threshold to gate auto-send.  Start strict at 0.90.
    AUTO_SEND_THRESHOLD: float = float(
        os.environ.get("AUTO_SEND_THRESHOLD", "0.90")
    )
    # Intents that must NEVER auto-send regardless of MLP score.
    AUTO_SEND_BLOCKED_INTENTS: frozenset[str] = frozenset(
        os.environ.get("AUTO_SEND_BLOCKED_INTENTS", "pricing,cancellation").split(",")
    )

    # ── Phase 10: Deployment ──────────────────────────────────────────────────
    # Comma-separated list of origins allowed to call this API.
    # Defaults include localhost for dev; add the Railway dashboard URL in prod.
    # Example: "http://localhost:3000,https://dashboard-xxxx.up.railway.app"
    DASHBOARD_URL: str = os.environ.get("DASHBOARD_URL", "")

    # ── Phase 11: Monitoring, alerts & weekly summary ─────────────────────────
    # Slack incoming-webhook URL.  If unset, notifications fall back to stdout
    # logging so the weekly cron still produces a visible record in Railway logs.
    SLACK_WEBHOOK_URL: str = os.environ.get("SLACK_WEBHOOK_URL", "")
    # Optional SMTP fallback for the weekly cost summary / alerts.
    SMTP_HOST: str = os.environ.get("SMTP_HOST", "")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER: str = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
    ALERT_EMAIL_TO: str = os.environ.get("ALERT_EMAIL_TO", "")
    # Alert thresholds (Phase 11 task 3).
    COST_ALERT_USD_PER_DAY: float = float(
        os.environ.get("COST_ALERT_USD_PER_DAY", "5.0")
    )
    ERROR_RATE_ALERT_THRESHOLD: float = float(
        os.environ.get("ERROR_RATE_ALERT_THRESHOLD", "0.01")
    )
    # Where the monitor script reaches the running agent to read poller stats.
    AGENT_HEALTH_URL: str = os.environ.get(
        "AGENT_HEALTH_URL", "http://localhost:8000/health"
    )


settings = Settings()
