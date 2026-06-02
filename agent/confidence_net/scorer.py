"""Lightweight inference wrapper — Phase 7 / Phase 8.

Used by the Phase 8 router to score a draft in-process without spawning
a subprocess.  Loads the latest checkpoint once, then scores any number
of drafts with a single forward pass.

Usage (Phase 8 router)
-----------------------
  from agent.confidence_net.scorer import ConfidenceScorer

  scorer = ConfidenceScorer()           # loads latest checkpoint
  p = scorer.score(draft_dict)          # float P(approved) in [0, 1]
  ok, reason = scorer.should_auto_send(draft_dict, threshold=0.90)

The scorer is designed to be instantiated once and reused.  Thread-safe
after __init__ (all state is immutable after loading).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.confidence_net.features import (
    FEATURE_VERSION,
    N_FEATURES,
    extract,
)
from agent.confidence_net.model import ConfidenceNet, load_checkpoint
from agent.config import REPO_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = REPO_ROOT / "data" / "models"


class ConfidenceScorer:
    """Load a Confidence Net checkpoint and score draft dicts.

    Parameters
    ----------
    version : int | None
        Checkpoint version to load.  Pass None (default) to load the latest
        available version.  Raises RuntimeError if none exist.
    """

    def __init__(self, version: int | None = None) -> None:
        self._version, self._model_path = self._resolve(version)
        self._model, self._temperature, self._meta = load_checkpoint(
            str(self._model_path)
        )
        self._model.eval()

        # Normalisation parameters saved during training
        self._norm_mean = np.array(
            self._meta.get("norm_mean", [0.0] * N_FEATURES), dtype=np.float32
        )
        self._norm_std = np.array(
            self._meta.get("norm_std", [1.0] * N_FEATURES), dtype=np.float32
        )

        # Assert feature version matches
        loaded_fv = self._meta.get("feature_version", 0)
        if loaded_fv != FEATURE_VERSION:
            raise RuntimeError(
                f"ConfidenceScorer: checkpoint feature_version={loaded_fv} "
                f"!= code FEATURE_VERSION={FEATURE_VERSION}. "
                f"Retrain the model with the current feature extractor."
            )

        logger.info(
            "ConfidenceScorer: loaded v%d  val_auc=%.3f  T=%.3f",
            self._version,
            self._meta.get("val_auc", 0.0),
            self._temperature,
        )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def val_auc(self) -> float:
        return float(self._meta.get("val_auc", 0.0))

    @property
    def temperature(self) -> float:
        return self._temperature

    def score(self, draft: dict[str, Any]) -> float:
        """Return P(draft_would_be_approved) in [0, 1].

        Parameters
        ----------
        draft : dict   as returned by ``reader.get_draft()``
        """
        vec = extract(draft)
        return self._score_vec(vec)

    def score_vector(self, vec: list[float]) -> float:
        """Score a pre-extracted feature vector (22 floats)."""
        return self._score_vec(vec)

    def should_auto_send(
        self,
        draft: dict[str, Any],
        threshold: float = 0.90,
    ) -> tuple[bool, str]:
        """Returns (ok, reason) for use by the Phase 8 router.

        Hard vetoes (from PLAN.md) are applied before the MLP threshold:
          1. guardrail_triggered → never auto-send
          2. uncertainty_flags non-empty → never auto-send
          3. MLP score < threshold → human review

        reason is a human-readable string for the audit log.
        """
        # Hard veto 1: guardrail
        if draft.get("guardrail_triggered"):
            return False, "guardrail_triggered: sensitive intent without wiki citation"

        # Hard veto 2: uncertainty flags
        if draft.get("uncertainty_flags"):
            n = len(draft["uncertainty_flags"])
            return False, f"uncertainty_flags: {n} flag(s) present"

        # MLP threshold
        p = self.score(draft)
        if p >= threshold:
            return True, f"MLP score {p:.3f} >= threshold {threshold:.2f}"
        return False, f"MLP score {p:.3f} < threshold {threshold:.2f}"

    # ── internal ──────────────────────────────────────────────────────────────

    def _score_vec(self, vec: list[float]) -> float:
        arr = np.array([vec], dtype=np.float32)
        arr = (arr - self._norm_mean) / self._norm_std
        x = torch.tensor(arr, dtype=torch.float32)
        with torch.no_grad():
            logit = float(self._model(x).item())
        return float(1.0 / (1.0 + np.exp(-logit / self._temperature)))

    @staticmethod
    def _resolve(version: int | None) -> tuple[int, Path]:
        if version is not None:
            path = MODELS_DIR / f"confidence_v{version}" / "model.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"No checkpoint at {path}. "
                    f"Run: python -m agent.confidence_net.train"
                )
            return version, path

        # Phase 11: serve the version pinned by active.json; if no active
        # is set yet, fall back to the latest checkpoint on disk.
        from agent.confidence_net.registry import resolve_serving_version

        v = resolve_serving_version()
        if v is None:
            raise FileNotFoundError(
                "No Confidence Net checkpoint found. "
                "Run: python -m agent.confidence_net.train"
            )
        return v, MODELS_DIR / f"confidence_v{v}" / "model.pt"
