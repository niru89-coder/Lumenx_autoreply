"""Smoke test for intent-aware wiki retrieval."""
from agent.llm_wiki.retriever import WikiRetriever


def show(label: str, hits):
    print(label)
    for h in hits:
        d = f"{h.distance:.3f}" if h.distance is not None else "?"
        product = h.product_id if h.product_id else "(company)"
        print(f"  d={d}  product={product:<20}  section={h.section}")
    print()


r = WikiRetriever()
show(
    "[A] non profit discount eligibility (intent=discount, company-aware):",
    r.query_intent_aware("non profit discount eligibility", intent="discount", k=5, k_company=2),
)
show(
    "[B] emailpilot refund window (intent=cancellation):",
    r.query_intent_aware("emailpilot refund window", intent="cancellation", k=5, k_company=2),
)
show(
    "[C] purple dragon (intent=feature, no company bias):",
    r.query_intent_aware("purple dragon", intent="feature", k=3),
)
