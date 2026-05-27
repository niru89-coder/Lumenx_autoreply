"""Auto-send router — Phase 8.

Decision flow (applied in order — first match wins):
  1. confidence_label == "blocked"                 → human_review
  2. guardrail_triggered                           → human_review
  3. uncertainty_flags non-empty                   → human_review
  4. intent in AUTO_SEND_BLOCKED_INTENTS           → human_review
  5. no trained MLP checkpoint                     → human_review (warn)
  6. MLP score < threshold                         → human_review
  7. AUTO_SEND_ENABLED is False                    → human_review (scored, not sent)
  8. all checks pass                               → auto_send

When action == "auto_send":
  - POSTs reply to LumenX via draft_source="agent" + confidence field
  - Records human_action(action="auto_sent") in feedback DB
  - Records confidence_prediction in feedback DB

When action == "human_review":
  - Draft stays in the pending inbox (no action recorded by the router)
  - Records confidence_prediction if MLP was evaluated (for dashboard visibility)

Hard safety net: any exception in the send path is caught; the router
returns action="human_review" so the draft is never silently lost.

Configuration (via .env / Settings):
  AUTO_SEND_ENABLED         = false   (must opt-in)
  AUTO_SEND_THRESHOLD       = 0.90
  AUTO_SEND_BLOCKED_INTENTS = pricing,cancellation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from agent.config import settings
from agent.drafter import Draft
from agent.feedback_log.writer import record_confidence_prediction, record_human_action
from agent.intent_router import SENSITIVE_INTENTS
from agent.lumenx_client import LumenXClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """The outcome of routing a single draft."""

    draft_id: str
    thread_id: str
    action: Literal["auto_send", "human_review"]
    reason: str                      # human-readable explanation for audit log
    score: float | None              # MLP P(approved); None if not evaluated
    threshold: float                 # threshold used at decision time
    sent: bool                       # True iff POST /reply was called successfully
    send_error: str | None = field(default=None)   # error message if send failed

    @property
    def auto_sent(self) -> bool:
        return self.action == "auto_send" and self.sent


# ──────────────────────────────────────────────────────────────────────────────
# Singleton scorer (lazy-loaded on first call)
# ──────────────────────────────────────────────────────────────────────────────

_scorer = None
_scorer_loaded = False


def _get_scorer():
    """Lazy-load ConfidenceScorer.  Returns None if no checkpoint exists."""
    global _scorer, _scorer_loaded
    if _scorer_loaded:
        return _scorer
    _scorer_loaded = True
    try:
        from agent.confidence_net.scorer import ConfidenceScorer
        _scorer = ConfidenceScorer()
        logger.info(
            "ConfidenceScorer loaded: v%d  val_auc=%.3f  T=%.3f",
            _scorer.version, _scorer.val_auc, _scorer.temperature,
        )
    except FileNotFoundError:
        logger.warning(
            "No Confidence Net checkpoint — all drafts will go to human review. "
            "Train the model with: python -m agent.confidence_net.train"
        )
        _scorer = None
    except Exception as exc:
        logger.error("Failed to load ConfidenceScorer: %s", exc)
        _scorer = None
    return _scorer


def _reset_scorer() -> None:
    """Force scorer reload on next call (used after retraining)."""
    global _scorer, _scorer_loaded
    _scorer = None
    _scorer_loaded = False


# ──────────────────────────────────────────────────────────────────────────────
# Main routing function
# ──────────────────────────────────────────────────────────────────────────────

def route(
    draft: Draft,
    *,
    client: LumenXClient | None = None,
    threshold: float | None = None,
    dry_run: bool = False,
) -> RoutingDecision:
    """Decide whether to auto-send `draft` or queue it for human review.

    Parameters
    ----------
    draft     : the Draft produced by drafter.draft_reply()
    client    : LumenXClient to use for the actual send.  If None, a new one
                is created when needed (auto-send path only).
    threshold : MLP score threshold.  Defaults to settings.AUTO_SEND_THRESHOLD.
    dry_run   : If True, evaluate all rules and compute MLP score but do NOT
                call LumenX or write any DB records.  Useful for testing.

    Returns
    -------
    RoutingDecision
    """
    thr = threshold if threshold is not None else settings.AUTO_SEND_THRESHOLD

    # ── Hard vetoes ───────────────────────────────────────────────────────────

    if draft.confidence_label == "blocked":
        return _human_review(draft, thr, "confidence_label=blocked", score=None, dry_run=dry_run)

    if draft.guardrail_triggered:
        return _human_review(draft, thr, "guardrail_triggered: sensitive intent without wiki citation",
                             score=None, dry_run=dry_run)

    if draft.uncertainty_flags:
        n = len(draft.uncertainty_flags)
        return _human_review(draft, thr, f"uncertainty_flags: {n} flag(s) present",
                             score=None, dry_run=dry_run)

    if draft.intent in settings.AUTO_SEND_BLOCKED_INTENTS:
        return _human_review(
            draft, thr,
            f"intent '{draft.intent}' is in AUTO_SEND_BLOCKED_INTENTS",
            score=None, dry_run=dry_run,
        )

    # ── MLP gate ─────────────────────────────────────────────────────────────

    scorer = _get_scorer()
    if scorer is None:
        return _human_review(draft, thr, "no MLP checkpoint — auto-send disabled until trained",
                             score=None, dry_run=dry_run)

    try:
        score = scorer.score(draft.to_dict())
    except Exception as exc:
        logger.error("MLP scoring failed for draft %s: %s", draft.draft_id[:8], exc)
        return _human_review(draft, thr, f"MLP scoring error: {exc}",
                             score=None, dry_run=dry_run)

    # Record confidence prediction (best-effort, skip in dry_run)
    if not dry_run:
        try:
            from agent.confidence_net.features import extract
            features_vec = extract(draft.to_dict())
            features_dict = {f"f{i}": v for i, v in enumerate(features_vec)}
            record_confidence_prediction(
                draft_id=draft.draft_id,
                features=features_dict,
                score=score,
                threshold=thr,
                model_version=f"v{scorer.version}",
            )
        except Exception as exc:
            logger.warning("record_confidence_prediction failed (non-fatal): %s", exc)

    if score < thr:
        return _human_review(
            draft, thr,
            f"MLP score {score:.3f} < threshold {thr:.2f}",
            score=score, dry_run=dry_run,
        )

    # Score is above threshold — check if auto-send is globally enabled
    if not settings.AUTO_SEND_ENABLED:
        return _human_review(
            draft, thr,
            f"MLP score {score:.3f} >= {thr:.2f} but AUTO_SEND_ENABLED=false "
            f"(set in .env to enable)",
            score=score, dry_run=dry_run,
        )

    # ── Auto-send path ────────────────────────────────────────────────────────

    if dry_run:
        logger.info(
            "[DRY RUN] would auto-send draft %s (thread=%s score=%.3f)",
            draft.draft_id[:8], draft.thread_id, score,
        )
        return RoutingDecision(
            draft_id=draft.draft_id,
            thread_id=draft.thread_id,
            action="auto_send",
            reason=f"[DRY RUN] score {score:.3f} >= {thr:.2f}",
            score=score,
            threshold=thr,
            sent=False,
        )

    return _do_auto_send(draft, score, thr, client=client)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _human_review(
    draft: Draft,
    threshold: float,
    reason: str,
    *,
    score: float | None,
    dry_run: bool,
) -> RoutingDecision:
    logger.info(
        "human_review  draft=%s thread=%s intent=%s  reason=%s  score=%s",
        draft.draft_id[:8], draft.thread_id, draft.intent,
        reason, f"{score:.3f}" if score is not None else "n/a",
    )
    return RoutingDecision(
        draft_id=draft.draft_id,
        thread_id=draft.thread_id,
        action="human_review",
        reason=reason,
        score=score,
        threshold=threshold,
        sent=False,
    )


def _do_auto_send(
    draft: Draft,
    score: float,
    threshold: float,
    *,
    client: LumenXClient | None,
) -> RoutingDecision:
    """Call LumenX POST /reply and record the action. Catches all send errors."""
    reason = f"MLP score {score:.3f} >= threshold {threshold:.2f}"
    sent = False
    send_error: str | None = None

    try:
        _client = client or LumenXClient()
        _client.post_reply(
            thread_id=draft.thread_id,
            text=draft.reply,
            draft_source="agent",
            confidence=round(score, 4),
        )
        sent = True
        logger.info(
            "AUTO-SENT  draft=%s thread=%s intent=%s  score=%.3f",
            draft.draft_id[:8], draft.thread_id, draft.intent, score,
        )
    except Exception as exc:
        send_error = str(exc)
        logger.error(
            "Auto-send FAILED  draft=%s thread=%s: %s",
            draft.draft_id[:8], draft.thread_id, exc,
        )
        # Fall back to human review on send failure — never silently discard
        return RoutingDecision(
            draft_id=draft.draft_id,
            thread_id=draft.thread_id,
            action="human_review",
            reason=f"send_error: {exc}  (scored {score:.3f}, would have auto-sent)",
            score=score,
            threshold=threshold,
            sent=False,
            send_error=send_error,
        )
    finally:
        if client is None and "_client" in dir():
            try:
                _client.close()
            except Exception:
                pass

    # Record auto_sent action in feedback DB (best-effort)
    try:
        record_human_action(
            draft_id=draft.draft_id,
            action="auto_sent",
            final_text=draft.reply,
            reviewer="agent",
        )
    except Exception as exc:
        logger.warning("record_human_action(auto_sent) failed (non-fatal): %s", exc)

    return RoutingDecision(
        draft_id=draft.draft_id,
        thread_id=draft.thread_id,
        action="auto_send",
        reason=reason,
        score=score,
        threshold=threshold,
        sent=sent,
    )
