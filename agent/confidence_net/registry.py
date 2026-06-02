"""Model registry — Phase 11.

Tracks which Confidence Net checkpoint is *active* (served by the router)
versus which are *candidates* (trained, calibrated, awaiting a human promote).

Layout on disk
--------------
  data/models/
    confidence_v1/  model.pt  features_v1.json  train_log.json
    confidence_v2/  ...
    active.json     {"active_version": 2, "promoted_at": "...", ...}

If `active.json` is absent or invalid, the registry falls back to "latest" —
this keeps the previous Phase 7/8 behaviour for fresh installs.

Promote flow
------------
1. `weekly_retrain.py` trains and saves `confidence_v{N+1}` (candidate).
2. Reviewer inspects metrics in the dashboard Models page.
3. POST /api/models/{N+1}/promote → `set_active(N+1)` writes active.json.
4. `_reset_scorer()` is called so the next route() reloads.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from agent.config import REPO_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = REPO_ROOT / "data" / "models"
ACTIVE_FILE = MODELS_DIR / "active.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _version_dirs() -> list[Path]:
    """Return checkpoint dirs sorted by version ascending."""
    if not MODELS_DIR.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in MODELS_DIR.glob("confidence_v*"):
        if not p.is_dir():
            continue
        suffix = p.name.replace("confidence_v", "")
        if not suffix.isdigit():
            continue
        if not (p / "model.pt").exists():
            continue
        out.append((int(suffix), p))
    return [p for _, p in sorted(out)]


def _read_meta(version_dir: Path) -> dict[str, Any]:
    """Load checkpoint metadata without instantiating the model."""
    ckpt_path = version_dir / "model.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", ckpt_path, exc)
        return {}
    return {k: v for k, v in ckpt.items() if k != "model_state"}


def latest_version() -> int | None:
    dirs = _version_dirs()
    if not dirs:
        return None
    return int(dirs[-1].name.replace("confidence_v", ""))


# ─────────────────────────────────────────────────────────────────────────────
# active.json read / write
# ─────────────────────────────────────────────────────────────────────────────

def get_active_version() -> int | None:
    """Return the version listed in active.json, or None if missing/invalid.

    Callers that want "active if set, else latest" should use
    :func:`resolve_serving_version`.
    """
    if not ACTIVE_FILE.exists():
        return None
    try:
        data = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
        v = int(data.get("active_version"))
    except Exception as exc:
        logger.warning("active.json unreadable (%s) — ignoring", exc)
        return None
    if (MODELS_DIR / f"confidence_v{v}" / "model.pt").exists():
        return v
    logger.warning("active.json points to v%d which is missing on disk", v)
    return None


def resolve_serving_version() -> int | None:
    """Version the scorer should load: active if set, else latest."""
    v = get_active_version()
    if v is not None:
        return v
    return latest_version()


def set_active(version: int, *, promoted_by: str = "system") -> dict[str, Any]:
    """Write active.json pointing at `version`.  Returns the new record."""
    target = MODELS_DIR / f"confidence_v{version}" / "model.pt"
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint v{version} not found at {target}")

    previous = get_active_version()
    record = {
        "active_version": int(version),
        "previous_version": previous,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted_by": promoted_by,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic-ish write: temp + replace
    tmp = ACTIVE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, ACTIVE_FILE)
    logger.info(
        "Promoted Confidence Net v%d (was v%s) by %s",
        version, previous, promoted_by,
    )
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Public listing API
# ─────────────────────────────────────────────────────────────────────────────

def list_checkpoints() -> list[dict[str, Any]]:
    """Return every checkpoint on disk with metadata + active/candidate flag.

    Newest first.  Each entry has:
      version, val_auc, val_bce, n_train, n_val, n_pos_train, n_pos_val,
      n_epochs, temperature, feature_version, train_seed, is_active,
      created_at (mtime of model.pt as ISO), path.
    """
    active = get_active_version()
    out: list[dict[str, Any]] = []
    for d in reversed(_version_dirs()):
        version = int(d.name.replace("confidence_v", ""))
        meta = _read_meta(d)
        ckpt_path = d / "model.pt"
        try:
            created_at = datetime.fromtimestamp(
                ckpt_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
        except OSError:
            created_at = None
        out.append({
            "version": version,
            "val_auc": meta.get("val_auc"),
            "val_bce": meta.get("val_bce"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "n_pos_train": meta.get("n_pos_train"),
            "n_pos_val": meta.get("n_pos_val"),
            "n_epochs": meta.get("n_epochs"),
            "temperature": meta.get("temperature"),
            "feature_version": meta.get("feature_version"),
            "train_seed": meta.get("train_seed"),
            "is_active": (version == active),
            "created_at": created_at,
            "path": str(d.relative_to(REPO_ROOT).as_posix()),
        })
    return out


def active_record() -> dict[str, Any] | None:
    """Return the active.json contents, or None if not set."""
    if not ACTIVE_FILE.exists():
        return None
    try:
        return json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
