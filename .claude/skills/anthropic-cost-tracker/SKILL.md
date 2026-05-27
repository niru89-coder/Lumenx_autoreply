---
name: anthropic-cost-tracker
description: Make Anthropic API calls in a way that always logs tokens and USD. Use whenever you're about to call the Anthropic SDK from agent code — never call it raw. Also use when investigating cost spikes, picking a model, or wiring a new LLM-driven feature.
---

# Anthropic cost tracker skill

Every Anthropic call from this project goes through `agent/anthropic_client.py`. The wrapper exists so the dashboard's cost page is always truthful and so that no LLM call is invisible.

## Hard rules

1. **Never import the `anthropic` SDK directly from feature code.** Import from `agent.anthropic_client` instead. The only file allowed to import the raw SDK is `anthropic_client.py` itself.
2. **Every call records:** model, input_tokens, output_tokens, latency_ms, computed USD, request hash, feature-name tag (e.g. `"intent_router"`, `"drafter"`, `"customer_history_summary"`). Stored in the `llm_calls` table.
3. **Model defaults:**
   - Intent routing, summarisation, utility tasks → `claude-haiku-4-5`
   - Draft generation → `claude-sonnet-4-6`
   - Only use `claude-opus-4-7` if explicitly justified in the call site comment AND the user has approved it for that feature.
4. **Prompt caching** is on by default for any prompt with a stable prefix > 1024 tokens (system + wiki snippets typically qualify). The wrapper sets the cache control marker; feature code provides the segments in order: stable → variable.

## Wrapper surface

```python
from agent.anthropic_client import call_llm

result = call_llm(
    feature="drafter",
    model="claude-sonnet-4-6",
    system=system_prompt,
    messages=[...],
    cache_segments=["wiki", "system"],   # which parts to cache
    response_format="json",              # optional
)
# result.text, result.usage, result.cost_usd, result.cached_input_tokens, ...
```

## Cost pricing

The wrapper holds a `PRICING` table per model (USD per 1M input tokens, USD per 1M output tokens, cache write/read multipliers). Update this table when Anthropic publishes new prices — it lives in `agent/anthropic_client.py`, one place to change.

## When investigating a cost spike

1. Open the dashboard's Costs page.
2. Sort by feature-name tag descending — the spike will localise to one feature.
3. Drill into a specific draft via the Draft Detail page; the full context window and per-call cost is there.
4. Common culprits:
   - Forgot to cache a stable system prompt
   - Customer-history summary regenerated per draft instead of cached
   - Used Sonnet where Haiku would do
   - A retry loop firing on a transient API error

## When adding a new LLM-driven feature

- Pick the cheapest model that plausibly works. Measure before upgrading.
- Tag the calls with a unique feature name.
- Add an entry to the Costs page's feature filter list.
- If the feature is going to be hot (called per-message), check that caching is wired before merging.
