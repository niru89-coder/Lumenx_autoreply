"""Confidence Net MLP architecture — Phase 7.

Architecture:  n_features → 64 → 32 → 1 (raw logit)

The output is a raw logit.  During training use BCEWithLogitsLoss for
numerical stability.  During inference call sigmoid() to get P(approved).
Temperature calibration divides the logit by scalar T before sigmoid.

Checkpoint layout (saved by train.py):
  {
    "model_state": model.state_dict(),
    "temperature": float T,           # from calibration
    "n_features":  int,
    "n_epochs":    int,
    "val_auc":     float,
    "val_bce":     float,
    "feature_version": int,
  }
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceNet(nn.Module):
    """Tiny two-hidden-layer MLP for P(draft_approved).

    Parameters
    ----------
    n_features : int   input dimension (must match feature schema)
    dropout    : float applied after each hidden activation
    """

    def __init__(self, n_features: int = 22, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),   # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_features) → logit: (batch,)"""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return calibrated P(approved) in [0, 1].

        temperature > 1 softens the distribution (model was over-confident).
        temperature < 1 sharpens it.
        Pass temperature=1.0 (default) to skip calibration.
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits / temperature)


def load_checkpoint(path: str) -> tuple["ConfidenceNet", float, dict]:
    """Load a saved checkpoint.

    Returns
    -------
    model       : ConfidenceNet in eval mode
    temperature : float for calibration (divide logit before sigmoid)
    meta        : dict with training metadata
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n_features = ckpt.get("n_features", 22)
    model = ConfidenceNet(n_features=n_features)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    temperature = float(ckpt.get("temperature", 1.0))
    meta = {k: v for k, v in ckpt.items() if k not in {"model_state"}}
    return model, temperature, meta
