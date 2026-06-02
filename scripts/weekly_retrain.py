"""Weekly Confidence Net retrain — Phase 11.

Trains a new candidate checkpoint from every label currently in the
feedback DB. Saves it as ``data/models/confidence_v{N+1}/`` but does
NOT promote it — a human reviewer must POST /api/models/{N+1}/promote
(via the dashboard) after inspecting val metrics.

Designed to run as a Railway cron job, e.g.:

    0 4 * * 1  cd /app && python -m scripts.weekly_retrain

Exit codes
----------
  0  candidate saved (or skipped because dataset is too small)
  1  unexpected failure during training

The script never raises — failures are logged and surfaced as exit-code 1
so the Railway scheduler sees them.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from agent.confidence_net import registry
from agent.confidence_net.train import MIN_SAMPLES_DEFAULT, train
from agent.config import REPO_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("weekly_retrain")

RETRAIN_LOG = REPO_ROOT / "data" / "retrain_log.jsonl"


def _append_log(entry: dict) -> None:
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RETRAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    active_before = registry.get_active_version()
    latest_before = registry.latest_version()

    logger.info(
        "Weekly retrain starting  active=v%s  latest=v%s",
        active_before, latest_before,
    )

    entry: dict = {
        "started_at": started,
        "active_before": active_before,
        "latest_before": latest_before,
        "promoted": False,  # weekly retrain NEVER auto-promotes
    }

    try:
        ckpt_dir: Path = train(min_samples=MIN_SAMPLES_DEFAULT)
        new_version = int(ckpt_dir.name.replace("confidence_v", ""))

        # Pull val_auc out of the checkpoint meta for the log
        meta = next(
            (e for e in registry.list_checkpoints() if e["version"] == new_version),
            {},
        )
        entry.update({
            "status": "saved_candidate",
            "new_version": new_version,
            "val_auc": meta.get("val_auc"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
        })
        logger.info(
            "Saved candidate v%d  val_auc=%s  (active still v%s — promote via dashboard)",
            new_version, meta.get("val_auc"), active_before,
        )
        _append_log(entry)
        return 0

    except SystemExit as exc:
        # train() calls sys.exit(1) when dataset is below min_samples.
        entry.update({"status": "skipped_too_few_samples", "exit_code": int(exc.code or 0)})
        logger.warning("Retrain skipped — not enough labelled data yet")
        _append_log(entry)
        return 0  # not a failure — just nothing to learn from

    except Exception as exc:
        entry.update({
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        })
        logger.exception("Weekly retrain failed: %s", exc)
        _append_log(entry)
        return 1


if __name__ == "__main__":
    sys.exit(main())
