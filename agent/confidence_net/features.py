"""Feature extraction for the Confidence Net — Phase 7.

Feature schema v1 (22 dimensions):

  [0-11]  Intent one-hot (12 dims, sorted alphabetically)
           billing / cancellation / chitchat / compare-competitor /
           discount / feature / greeting / integration / multi-product /
           out_of_scope / pricing / technical

  [12]    log1p_draft_chars       log1p(len(reply text))
  [13]    log1p_output_tokens     log1p(LLM output_tokens)
  [14]    log1p_cited_count       log1p(len(cited_sources))
  [15]    has_wiki_citation       1 if any source NOT starting with "past:"
  [16]    log1p_uncertainty_count log1p(len(uncertainty_flags))
  [17]    guardrail_triggered     1/0
  [18]    sensitive_no_wiki       1 if sensitive intent AND no wiki citation
  [19]    parse_attempts_needed   1 if parse_attempts >= 2
  [20]    is_haiku                1 if model name contains "haiku"
  [21]    self_label_high         1 if drafter's own confidence_label == "high"

The schema dict is versioned and saved alongside model.pt so that the
inference path can assert it hasn't drifted.

All features are in [0, ∞) before any normalisation.  The log1p features
are already reasonably bounded.  Binary features are exactly 0 or 1.
Normalisation (zero-mean, unit-std) is applied during training and stored
in the checkpoint so inference applies identical scaling.
"""
from __future__ import annotations

import math
from typing import Any

from agent.intent_router import SENSITIVE_INTENTS

# ──────────────────────────────────────────────────────────────────────────────
# Schema constants
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_VERSION: int = 1

# All 12 known LumenX intents, alphabetically sorted for a stable one-hot order.
INTENTS_SORTED: list[str] = sorted([
    "billing",
    "cancellation",
    "chitchat",
    "compare-competitor",
    "discount",
    "feature",
    "greeting",
    "integration",
    "multi-product",
    "out_of_scope",
    "pricing",
    "technical",
])

SCALAR_FEATURE_NAMES: list[str] = [
    "log1p_draft_chars",        # 12
    "log1p_output_tokens",      # 13
    "log1p_cited_count",        # 14
    "has_wiki_citation",        # 15
    "log1p_uncertainty_count",  # 16
    "guardrail_triggered",      # 17
    "sensitive_no_wiki",        # 18
    "parse_attempts_needed",    # 19
    "is_haiku",                 # 20
    "self_label_high",          # 21
]

N_INTENT_DIMS: int = len(INTENTS_SORTED)        # 12
N_SCALAR_DIMS: int = len(SCALAR_FEATURE_NAMES)  # 10
N_FEATURES: int = N_INTENT_DIMS + N_SCALAR_DIMS  # 22

FEATURE_NAMES: list[str] = (
    [f"intent_{i}" for i in INTENTS_SORTED] + SCALAR_FEATURE_NAMES
)

# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────

_INTENT_IDX: dict[str, int] = {intent: i for i, intent in enumerate(INTENTS_SORTED)}


def extract(draft: dict[str, Any]) -> list[float]:
    """Extract a 22-dimensional feature vector from a draft dict.

    `draft` is the plain dict returned by ``reader.get_draft()`` or
    ``reader.get_draft_with_actions()``.  Missing / None fields are treated
    as zero / empty rather than raising — robust to schema evolution.
    """
    vec: list[float] = [0.0] * N_FEATURES

    # ── intent one-hot (indices 0-11) ────────────────────────────────────────
    intent: str = (draft.get("intent") or "").strip().lower()
    if intent in _INTENT_IDX:
        vec[_INTENT_IDX[intent]] = 1.0
    # Unknown intents → all zeros (fine; the model has an "other" implicit class)

    # ── scalar features (indices 12-21) ──────────────────────────────────────
    reply_text: str = draft.get("draft_text") or ""
    output_tokens: int = int(draft.get("output_tokens") or 0)
    cited: list = draft.get("cited_sources") or []
    flags: list = draft.get("uncertainty_flags") or []
    guardrail: bool = bool(draft.get("guardrail_triggered"))
    parse_att: int = int(draft.get("parse_attempts") or 1)
    model_name: str = (draft.get("model") or "").lower()
    conf_label: str = (draft.get("confidence_label") or "").lower()

    is_sensitive = intent in SENSITIVE_INTENTS
    has_wiki = any(not str(s).startswith("past:") for s in cited) if cited else False

    vec[12] = math.log1p(len(reply_text))
    vec[13] = math.log1p(output_tokens)
    vec[14] = math.log1p(len(cited))
    vec[15] = 1.0 if has_wiki else 0.0
    vec[16] = math.log1p(len(flags))
    vec[17] = 1.0 if guardrail else 0.0
    vec[18] = 1.0 if (is_sensitive and not has_wiki) else 0.0
    vec[19] = 1.0 if parse_att >= 2 else 0.0
    vec[20] = 1.0 if "haiku" in model_name else 0.0
    vec[21] = 1.0 if conf_label == "high" else 0.0

    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Schema serialisation
# ──────────────────────────────────────────────────────────────────────────────

def schema() -> dict[str, Any]:
    """Return the versioned feature schema for saving alongside model.pt.

    The inference path loads this and asserts the version matches, preventing
    model + feature-extractor drift.
    """
    return {
        "version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "n_intent_dims": N_INTENT_DIMS,
        "n_scalar_dims": N_SCALAR_DIMS,
        "intent_order": INTENTS_SORTED,
        "scalar_feature_names": SCALAR_FEATURE_NAMES,
        "feature_names": FEATURE_NAMES,
        "sensitive_intents": sorted(SENSITIVE_INTENTS),
    }
