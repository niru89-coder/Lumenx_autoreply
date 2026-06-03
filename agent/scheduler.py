"""In-process job scheduler — Phase 11.

Runs the monitoring / retrain jobs inside the always-on agent process (which
already has the data volume mounted), instead of separate Railway cron services
(blocked on the trial plan, and unable to share the volume anyway).

Schedules (all UTC):
  * alerts        hourly         — cost/day + poller error-rate checks
  * retrain       Mon 04:00      — train a candidate checkpoint (never promotes)
  * cost_summary  Mon 08:00      — build + deliver the weekly spend summary

Blocking work (torch training, HTTP, file IO) runs in a worker thread so the
event loop stays responsive. Toggle the whole scheduler with SCHEDULER_ENABLED.
Each job swallows its own exceptions so one failure never kills the scheduler.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _job_alerts() -> None:
    from scripts.monitor_alerts import run_checks
    try:
        await asyncio.to_thread(run_checks)
    except Exception:  # noqa: BLE001 — never let a job crash the scheduler
        logger.exception("scheduled job 'alerts' failed")


async def _job_retrain() -> None:
    from scripts.weekly_retrain import run_retrain
    try:
        await asyncio.to_thread(run_retrain)
    except Exception:  # noqa: BLE001
        logger.exception("scheduled job 'retrain' failed")


async def _job_cost_summary() -> None:
    from agent.notifier import notify
    from scripts.cost_summary import build_summary
    try:
        text, _total = await asyncio.to_thread(build_summary, 7)
        await asyncio.to_thread(notify, "📊 LumenX agent weekly cost summary", text)
    except Exception:  # noqa: BLE001
        logger.exception("scheduled job 'cost_summary' failed")


def start_scheduler() -> AsyncIOScheduler:
    """Create + start the AsyncIO scheduler. Idempotent."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    utc = timezone.utc
    sched = AsyncIOScheduler(timezone=utc)
    sched.add_job(
        _job_alerts, CronTrigger(minute=0, timezone=utc),
        id="alerts", replace_existing=True, misfire_grace_time=300, coalesce=True,
    )
    sched.add_job(
        _job_retrain, CronTrigger(day_of_week="mon", hour=4, minute=0, timezone=utc),
        id="retrain", replace_existing=True, misfire_grace_time=3600, coalesce=True,
    )
    sched.add_job(
        _job_cost_summary, CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=utc),
        id="cost_summary", replace_existing=True, misfire_grace_time=3600, coalesce=True,
    )
    sched.start()
    _scheduler = sched
    jobs = ", ".join(f"{j.id} [{j.trigger}]" for j in sched.get_jobs())
    logger.info("In-process scheduler started (UTC): %s", jobs)
    return sched


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("In-process scheduler stopped")
