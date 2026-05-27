"""Evaluate a Confidence Net checkpoint and score individual drafts — Phase 7.

Usage
-----
  # Evaluate the latest checkpoint on all labelled data:
  python -m agent.confidence_net.eval

  # Evaluate a specific version:
  python -m agent.confidence_net.eval --version 2

  # Score a specific draft by UUID:
  python -m agent.confidence_net.eval --draft <uuid>

  # Save a calibration plot PNG:
  python -m agent.confidence_net.eval --plot

Output
------
  - Val AUC, BCE, calibration table printed to stdout
  - Calibration plot saved to data/models/confidence_v{N}/calibration.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── stdout ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from agent.confidence_net.dataset import load_labelled_dataset, apply_norm
from agent.confidence_net.features import N_FEATURES, FEATURE_NAMES, extract
from agent.confidence_net.model import ConfidenceNet, load_checkpoint
from agent.config import REPO_ROOT
from agent.feedback_log.reader import get_draft

import torch

MODELS_DIR = REPO_ROOT / "data" / "models"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _latest_version() -> int:
    versions = [
        int(p.name.replace("confidence_v", ""))
        for p in MODELS_DIR.glob("confidence_v*")
        if p.is_dir() and p.name.replace("confidence_v", "").isdigit()
    ]
    if not versions:
        return 0
    return max(versions)


def _ckpt_path(version: int) -> Path:
    return MODELS_DIR / f"confidence_v{version}" / "model.pt"


def _load_norm(ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array(ckpt.get("norm_mean", [0.0] * N_FEATURES), dtype=np.float32)
    std  = np.array(ckpt.get("norm_std",  [1.0] * N_FEATURES), dtype=np.float32)
    return mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _calibration_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> str:
    """Return an ASCII reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = ["  Predicted   Actual    Count  (reliability diagram)"]
    rows.append("  ---------   ------   ------  " + "-" * 30)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        pred_mean = probs[mask].mean()
        actual_frac = labels[mask].mean()
        count = mask.sum()
        bar = "#" * int(actual_frac * 30)
        rows.append(
            f"  {pred_mean:6.2f}→{hi:5.2f}   {actual_frac:5.2f}   {count:5d}  |{bar}"
        )
    return "\n".join(rows)


def _save_calibration_plot(
    probs: np.ndarray,
    labels: np.ndarray,
    ckpt_dir: Path,
    title: str = "Calibration",
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "b-o", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive (actual)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = ckpt_dir / "calibration.png"
        fig.savefig(str(out), dpi=120)
        plt.close(fig)
        return out
    except Exception as exc:
        print(f"  [calibration plot skipped: {exc}]")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(version: int | None = None, save_plot: bool = False) -> None:
    from sklearn.metrics import roc_auc_score, log_loss

    v = version if version is not None else _latest_version()
    if v == 0:
        print("No trained checkpoint found. Run: python -m agent.confidence_net.train")
        return

    ckpt_dir = MODELS_DIR / f"confidence_v{v}"
    model_path = ckpt_dir / "model.pt"
    if not model_path.exists():
        print(f"Checkpoint not found: {model_path}")
        return

    model, temperature, ckpt = load_checkpoint(str(model_path))
    mean, std = _load_norm(ckpt)

    print(f"\n=== Confidence Net v{v} ===")
    print(f"  n_features   : {ckpt.get('n_features', N_FEATURES)}")
    print(f"  best epoch   : {ckpt.get('n_epochs', '?')}")
    print(f"  val AUC (train) : {ckpt.get('val_auc', '?'):.4f}")
    print(f"  temperature T   : {temperature:.4f}")

    # Load all labelled data and score everything
    X, y, meta = load_labelled_dataset()
    if len(y) == 0:
        print("No labelled drafts found.")
        return

    X_n = apply_norm(X, mean, std)
    X_t = torch.tensor(X_n, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_t).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits / temperature))

    auc = float(roc_auc_score(y, probs))
    bce = float(log_loss(y, probs))

    print(f"\n  Full dataset (n={len(y)}):")
    print(f"    AUC      : {auc:.4f}   (goal >= 0.75)")
    print(f"    Log-loss : {bce:.4f}")
    print(f"    Pos rate : {y.mean():.2%}")

    # Score distribution
    thresholds = [0.70, 0.80, 0.90, 0.95]
    print("\n  Auto-send eligibility at various thresholds:")
    for t in thresholds:
        above = (probs >= t).sum()
        print(f"    >= {t:.0%} : {above}/{len(probs)} ({above/len(probs):.0%}) would auto-send")

    print(f"\n  Reliability diagram:\n{_calibration_table(probs, y)}")

    if save_plot:
        out = _save_calibration_plot(
            probs, y, ckpt_dir,
            title=f"Confidence Net v{v} calibration  (AUC={auc:.3f})"
        )
        if out:
            print(f"\n  Calibration plot saved -> {out}")

    goal_ok = "PASS" if auc >= 0.75 else "BELOW TARGET (label more data or retrain)"
    print(f"\n  Phase 7 AUC goal: {goal_ok}")


# ──────────────────────────────────────────────────────────────────────────────
# Score a single draft
# ──────────────────────────────────────────────────────────────────────────────

def score_draft(draft_id: str, version: int | None = None) -> None:
    v = version if version is not None else _latest_version()
    if v == 0:
        print("No trained checkpoint found.")
        return

    model_path = _ckpt_path(v)
    model, temperature, ckpt = load_checkpoint(str(model_path))
    mean, std = _load_norm(ckpt)

    draft = get_draft(draft_id)
    if draft is None:
        print(f"Draft {draft_id!r} not found in feedback DB.")
        return

    vec = extract(draft)
    vec_n = apply_norm(np.array([vec], dtype=np.float32), mean, std)
    x_t = torch.tensor(vec_n, dtype=torch.float32)

    with torch.no_grad():
        logit = float(model(x_t).item())
    p = float(1.0 / (1.0 + np.exp(-logit / temperature)))

    print(f"\n  Draft  : {draft_id}")
    print(f"  Thread : {draft.get('thread_id')}  intent={draft.get('intent')}")
    print(f"  Label  : {draft.get('confidence_label')}  model={draft.get('model', '?').split('-')[1][:6]}")
    print(f"  P(approved) = {p:.4f}  (raw logit={logit:.3f}  T={temperature:.3f})")
    print(f"  Auto-sendable at 0.90 threshold: {'YES' if p >= 0.90 else 'NO'}")
    print()
    print("  Feature vector:")
    for name, val in zip(FEATURE_NAMES, vec):
        if abs(val) > 1e-6:
            print(f"    {name:<30}: {val:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7: evaluate / score Confidence Net")
    ap.add_argument("--version", type=int, default=None,
                    help="Checkpoint version to load (default: latest)")
    ap.add_argument("--draft",   type=str, default=None,
                    help="Score a specific draft UUID from the feedback DB")
    ap.add_argument("--plot",    action="store_true",
                    help="Save a calibration plot PNG alongside the checkpoint")
    args = ap.parse_args()

    if args.draft:
        score_draft(args.draft, version=args.version)
    else:
        evaluate(version=args.version, save_plot=args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
