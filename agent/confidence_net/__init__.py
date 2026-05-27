"""Confidence Net — Phase 7.

Small MLP that takes a draft's feature vector and outputs
P(would_be_approved), used by the Phase 8 router to gate auto-send.

Public surface:
  from agent.confidence_net.scorer import ConfidenceScorer
  scorer = ConfidenceScorer()
  p = scorer.score(draft_dict)   # float in [0, 1]

Training:
  python -m agent.confidence_net.train

Evaluation / scoring CLI:
  python -m agent.confidence_net.eval
  python -m agent.confidence_net.eval --draft <uuid>
"""
