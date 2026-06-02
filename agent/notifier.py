"""Outbound notifications — Phase 11.

A single chokepoint for sending the weekly cost summary (task 2) and the
cost/error-rate alerts (task 3) to the user.  Delivery order:

  1. Slack incoming webhook  (SLACK_WEBHOOK_URL)
  2. SMTP email              (SMTP_HOST + ALERT_EMAIL_TO)
  3. stdout                  (always — so Railway cron logs keep a record)

The function never raises: a failed Slack/SMTP send is logged and the
function returns the set of channels that succeeded.  Callers (cron jobs)
should not crash just because Slack is down.
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

import httpx

from agent.config import settings

logger = logging.getLogger(__name__)


def _send_slack(subject: str, body: str) -> bool:
    if not settings.SLACK_WEBHOOK_URL:
        return False
    # Slack renders the leading line in bold via mrkdwn.
    text = f"*{subject}*\n```{body}```"
    try:
        resp = httpx.post(
            settings.SLACK_WEBHOOK_URL,
            json={"text": text},
            timeout=10.0,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 — never let a notifier crash a cron
        logger.warning("Slack notify failed: %s", exc)
        return False


def _send_email(subject: str, body: str) -> bool:
    if not (settings.SMTP_HOST and settings.ALERT_EMAIL_TO):
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_USER or "agent@lumenx"
    msg["To"] = settings.ALERT_EMAIL_TO
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as s:
            s.starttls()
            if settings.SMTP_USER:
                s.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("SMTP notify failed: %s", exc)
        return False


def notify(subject: str, body: str) -> list[str]:
    """Send `subject`/`body` to every configured channel.

    Returns the list of channels that accepted the message (e.g.
    ``["slack", "stdout"]``).  Always includes ``"stdout"`` because the
    message is logged unconditionally.
    """
    sent: list[str] = []
    if _send_slack(subject, body):
        sent.append("slack")
    if _send_email(subject, body):
        sent.append("email")

    # stdout fallback is unconditional — keeps a record in Railway logs even
    # when no channel is configured.
    logger.info("[notify] %s\n%s", subject, body)
    sent.append("stdout")
    return sent
