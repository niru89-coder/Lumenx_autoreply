---
name: confidence-net
description: Train, evaluate, and serve the tiny PyTorch MLP that decides whether a draft is safe to auto-send. Use when adding features, retraining on new feedback, debugging score calibration, or touching anything under agent/confidence_net/.
---

# Confidence Net skill

The Confidence Net is a deliberately tiny MLP. It scores a draft reply `0..1` for "would a human have approved this as-is?". The threshold at which we auto-send is a separate, configurable knob — the model itself is just a calibrated probability.

## What it is NOT

- Not the LLM. It does not generate text.
- Not a judge of "is this answer factually correct" — that's the wiki + guardrails' job.
- Not a black box: a feature vector explanation is shown in the dashboard for every score.

## Architecture (locked in)

- Input: ~20-dim feature vector defined in `agent/confidence_net/features.py`. Versioned schema — `features_v1.json`, `features_v2.json`, ...
- Layers: `Linear(20→64) → ReLU → Dropout(0.2) → Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1) → Sigmoid`
- Loss: `BCELoss`
- Optimizer: `AdamW`, lr=1e-3, weight_decay=1e-4
- Calibration: temperature scaling on the validation split
- Checkpoint format: `model.pt` + sibling `features_v{n}.json` describing input order. Inference refuses to load a checkpoint whose schema version doesn't match the running code.

## Feature list (initial)

See `PLAN.md` Phase 7. Headlines:
- intent one-hot
- draft length + length-ratio
- top-k feedback-log similarity
- wiki coverage fraction
- sensitive-topic flag × wiki-citation flag
- count of drafter `uncertainty_flags`
- customer-thread length, returning-customer flag

## Conventions

- Features go in `features.py`. Adding a new feature → bump the schema version, retrain.
- Never train on data that is currently in the validation split. Split is deterministic by `draft_id % 10` (8/2 train/val), stored in a `splits.json` alongside the dataset.
- Save calibration plot alongside the checkpoint. Visual sanity check beats trusting AUC alone.
- All training logs go to `data/confidence_net/runs/{timestamp}/`.

## Bootstrap → live transition

Phase 6 produces ~300 bootstrap labels. After that the live human review starts producing fresh labels continuously. Retrain weekly. The first live retrain that beats the bootstrap-only model on val AUC promotes to production.

## Quick CLI surface

```
python -m agent.confidence_net.train          # train + save checkpoint
python -m agent.confidence_net.eval           # eval current production checkpoint
python -m agent.confidence_net.score <draft_id>  # score one draft, print feature breakdown
```

## Common failure modes

- **Score always near 0.5** → likely a feature normalisation bug. Print feature stats per class first.
- **Val AUC suspiciously high (> 0.95) on bootstrap data** → likely leakage (e.g. a feature derived from the human action itself). Audit features.
- **Production score doesn't match offline eval** → feature schema mismatch. Check `features_v{n}.json` versioning.
