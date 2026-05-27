"""Train the Confidence Net MLP — Phase 7.

Loads labelled drafts from the feedback DB, trains an MLP with BCE loss,
applies temperature calibration, and saves a versioned checkpoint.

Usage
-----
  python -m agent.confidence_net.train
  python -m agent.confidence_net.train --epochs 300 --lr 3e-3 --seed 42
  python -m agent.confidence_net.train --min-samples 50   # override minimum

Output
------
  data/models/confidence_v{N}/
    model.pt           — state dict + temperature + metadata
    features_v1.json   — feature schema (must stay in sync with model)
    train_log.json     — per-epoch loss/auc history
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Suppress "only one class in y_true" warning — expected during tiny dev runs.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ── stdout ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

from agent.confidence_net.dataset import (
    EDIT_DISTANCE_THRESHOLD,
    apply_norm,
    dataset_summary,
    load_labelled_dataset,
)
from agent.confidence_net.features import N_FEATURES, FEATURE_VERSION, schema as feature_schema
from agent.confidence_net.model import ConfidenceNet
from agent.config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "data" / "models"
MIN_SAMPLES_DEFAULT = 30   # hard floor — don't train on fewer


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fit_normaliser(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std from training set (per feature)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0   # don't divide by ~zero for constant features
    return mean.astype(np.float32), std.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Temperature calibration
# ──────────────────────────────────────────────────────────────────────────────

def _find_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the scalar T that minimises NLL on the validation set.

    Calibrated probability: sigmoid(logit / T).
    T > 1 → softer (model was over-confident).
    T < 1 → sharper.
    """
    eps = 1e-7

    def nll(T: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, eps, 1 - eps)
        return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T = float(result.x)
    logger.info("Temperature calibration: T=%.4f  (NLL before=%.4f, after=%.4f)",
                T, nll(1.0), nll(T))
    return T


# ──────────────────────────────────────────────────────────────────────────────
# Next checkpoint version
# ──────────────────────────────────────────────────────────────────────────────

def _next_version() -> int:
    existing = [
        int(p.name.replace("confidence_v", ""))
        for p in MODELS_DIR.glob("confidence_v*")
        if p.is_dir() and p.name.replace("confidence_v", "").isdigit()
    ]
    return max(existing, default=0) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.3,
    val_fraction: float = 0.2,
    patience: int = 30,
    seed: int = 42,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    edit_distance_threshold: int = EDIT_DISTANCE_THRESHOLD,
) -> Path:
    """Full training run. Returns path to saved checkpoint directory."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── load data ────────────────────────────────────────────────────────────
    print("\n=== Confidence Net Training ===")
    print("Loading labelled dataset ...")
    X, y, meta = load_labelled_dataset(edit_distance_threshold=edit_distance_threshold)
    print(f"  {dataset_summary(X, y, meta)}")

    if len(y) < min_samples:
        print(
            f"\n[!] Only {len(y)} labelled samples — need at least {min_samples}.\n"
            f"    Label more drafts with:  python -m scripts.bootstrap_review\n"
            f"    Override with:           --min-samples {len(y)}"
        )
        sys.exit(1)

    # ── stratified train / val split ─────────────────────────────────────────
    # Need at least 2 samples per class for stratification.  Fall back to a
    # plain random split when data is small (only happens during dev/testing).
    n_per_class_min = int(np.bincount(y.astype(int)).min())
    can_stratify = n_per_class_min >= 2 and len(y) >= 10
    if can_stratify:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(sss.split(X, y))
    else:
        logger.warning("Too few samples per class for stratification — using random split.")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_val = max(1, int(len(y) * val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    mean, std = _fit_normaliser(X_tr)
    X_tr_n  = apply_norm(X_tr, mean, std)
    X_val_n = apply_norm(X_val, mean, std)

    print(
        f"  Split: train={len(y_tr)} (pos={int(y_tr.sum())})  "
        f"val={len(y_val)} (pos={int(y_val.sum())})"
    )

    # ── tensors ──────────────────────────────────────────────────────────────
    X_tr_t  = torch.tensor(X_tr_n,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,    dtype=torch.float32)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32)
    y_val_t = torch.tensor(y_val,   dtype=torch.float32)

    # Positive weight for BCE to handle imbalance
    n_pos = float(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    # ── model + optimiser ────────────────────────────────────────────────────
    model = ConfidenceNet(n_features=N_FEATURES, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=10
    )

    # ── training loop ────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_val_bce = float("inf")
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    log_history: list[dict] = []

    t0 = time.perf_counter()
    print(f"\n  Training {epochs} epochs  lr={lr}  wd={weight_decay}  dropout={dropout}  patience={patience}")
    print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>9}  {'ValAUC':>8}  {'Best':>6}")
    print(f"  {'------':>6}  {'----------':>10}  {'---------':>9}  {'--------':>8}  {'------':>6}")

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        logits_tr = model(X_tr_t)
        loss_tr = criterion(logits_tr, y_tr_t)
        optimiser.zero_grad()
        loss_tr.backward()
        optimiser.step()

        # -- val --
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            loss_val = float(criterion(logits_val, y_val_t).item())
            probs_val = torch.sigmoid(logits_val).numpy()

        try:
            val_auc = float(roc_auc_score(y_val, probs_val))
        except ValueError:
            val_auc = 0.5   # only one class in val — can't compute AUC

        scheduler.step(val_auc)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_val_bce = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1

        log_history.append({
            "epoch": epoch,
            "train_loss": round(float(loss_tr.item()), 5),
            "val_loss":   round(loss_val, 5),
            "val_auc":    round(val_auc, 4),
            "best":       is_best,
        })

        if epoch % 25 == 0 or is_best:
            marker = " <-- best" if is_best else ""
            print(
                f"  {epoch:>6}  {float(loss_tr.item()):>10.4f}  {loss_val:>9.4f}"
                f"  {val_auc:>8.4f}  {best_val_auc:>6.4f}{marker}"
            )

        if patience_ctr >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no AUC improvement for {patience} epochs).")
            break

    elapsed = time.perf_counter() - t0
    print(f"\n  Best val AUC = {best_val_auc:.4f}  at epoch {best_epoch}")
    print(f"  Training time: {elapsed:.1f}s")

    # ── calibration on val set using best weights ─────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw_logits_val = model(X_val_t).numpy()
    temperature = _find_temperature(raw_logits_val, y_val)

    # ── save checkpoint ───────────────────────────────────────────────────────
    version = _next_version()
    ckpt_dir = MODELS_DIR / f"confidence_v{version}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "model_state":      model.state_dict(),
            "temperature":      temperature,
            "n_features":       N_FEATURES,
            "feature_version":  FEATURE_VERSION,
            "norm_mean":        mean.tolist(),
            "norm_std":         std.tolist(),
            "n_epochs":         best_epoch,
            "val_auc":          round(best_val_auc, 4),
            "val_bce":          round(best_val_bce, 5),
            "n_train":          len(y_tr),
            "n_val":            len(y_val),
            "n_pos_train":      int(y_tr.sum()),
            "n_pos_val":        int(y_val.sum()),
            "train_seed":       seed,
        },
        model_path,
    )

    schema_path = ckpt_dir / f"features_v{FEATURE_VERSION}.json"
    schema_path.write_text(
        json.dumps(feature_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_path = ckpt_dir / "train_log.json"
    log_path.write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    size_kb = model_path.stat().st_size / 1024
    print(f"\n  Checkpoint saved -> {ckpt_dir}")
    print(f"  model.pt: {size_kb:.1f} KB")
    print(f"  Val AUC  : {best_val_auc:.4f}   (target >= 0.75)")
    print(f"  Temp T   : {temperature:.4f}")

    goal_ok = "PASS" if best_val_auc >= 0.75 else "FAIL (need more data or labels)"
    print(f"  Phase 7 goal: {goal_ok}")

    return ckpt_dir


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7: train Confidence Net MLP")
    ap.add_argument("--epochs",     type=int,   default=300,  help="Max training epochs")
    ap.add_argument("--lr",         type=float, default=3e-3, help="Learning rate")
    ap.add_argument("--dropout",    type=float, default=0.3,  help="Dropout probability")
    ap.add_argument("--wd",         type=float, default=1e-3, help="Weight decay")
    ap.add_argument("--patience",   type=int,   default=30,   help="Early stopping patience")
    ap.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    ap.add_argument("--min-samples",type=int,   default=MIN_SAMPLES_DEFAULT,
                    help="Minimum labelled samples to proceed (default: 30)")
    ap.add_argument("--edit-threshold", type=int, default=EDIT_DISTANCE_THRESHOLD,
                    help="Max edit_distance to DROP edited drafts (default: 30)")
    args = ap.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.wd,
        patience=args.patience,
        seed=args.seed,
        min_samples=args.min_samples,
        edit_distance_threshold=args.edit_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
