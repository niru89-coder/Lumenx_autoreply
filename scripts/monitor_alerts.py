"""Monitoring alerts — Phase 11 task 3.

Fires a notification when either:
  * today's LLM spend exceeds COST_ALERT_USD_PER_DAY, or
  * the poller error rate exceeds ERROR_RATE_ALERT_THRESHOLD (default 1%).

Cost is read locally from data/llm_calls.jsonl.  Error rate is read from the
running agent's /health endpoint (AGENT_HEALTH_URL).  Designed to run as a
frequent Railway cron, e.g. hourly:

    0 * * * *  cd /app && python -m scripts.monitor_alerts

Exit codes:  0 = checked OK (alert or not),  1 = unexpected failure.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import httpx

from agent.config import REPO_ROOT, settings
from agent.notifier import notify

# Make stdout UTF-8 so emoji in alert text don't crash logging on Windows
# (cp1252) consoles. No-op on Linux/Railway where stdout is already UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("monitor_alerts")


def _cost_today() -> float:
    log = REPO_ROOT / "data" / "llm_calls.jsonl"
    if not log.exists():
        return 0.0
    today = datetime.now(timezone.utc).date()
    total = 0.0
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            ts = datetime.fromisoformat(e["ts"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        if ts.date() == today:
            total += e.get("cost_usd", 0.0)
    return total


def _poller_error_rate() -> tuple[float | None, dict]:
    """Return (error_rate, poller_stats). error_rate is None if unreachable."""
    try:
        resp = httpx.get(settings.AGENT_HEALTH_URL, timeout=10.0)
        resp.raise_for_status()
        poller = resp.json().get("poller", {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not reach agent health: %s", exc)
        return None, {}

    errors = poller.get("errors", 0)
    processed = poller.get("threads_processed", 0)
    denom = processed + errors
    rate = (errors / denom) if denom else 0.0
    return rate, poller


def main() -> int:
    alerts: list[str] = []

    try:
        cost_today = _cost_today()
        logger.info(
            "Cost today: $%.4f  (threshold $%.2f)",
            cost_today, settings.COST_ALERT_USD_PER_DAY,
        )
        if cost_today > settings.COST_ALERT_USD_PER_DAY:
            alerts.append(
                f"💸 Daily cost ${cost_today:.4f} exceeds "
                f"threshold ${settings.COST_ALERT_USD_PER_DAY:.2f}"
            )

        rate, poller = _poller_error_rate()
        if rate is None:
            alerts.append("⚠️ Agent /health unreachable — poller may be down")
        else:
            logger.info(
                "Poller error rate: %.2f%%  (threshold %.2f%%)",
                rate * 100, settings.ERROR_RATE_ALERT_THRESHOLD * 100,
            )
            if rate > settings.ERROR_RATE_ALERT_THRESHOLD:
                alerts.append(
                    f"🚨 Poller error rate {rate * 100:.1f}% exceeds "
                    f"threshold {settings.ERROR_RATE_ALERT_THRESHOLD * 100:.1f}% "
                    f"({poller.get('errors')} errors / "
                    f"{poller.get('threads_processed')} processed)"
                )

        if alerts:
            body = "\n".join(alerts)
            sent = notify("🔔 LumenX agent alert", body)
            logger.warning("Alert sent via %s:\n%s", ", ".join(sent), body)
        else:
            logger.info("All checks within thresholds — no alert sent")
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("monitor_alerts failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
