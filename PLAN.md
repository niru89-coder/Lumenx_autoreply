# Execution Plan — LumenX Auto-Reply Agent

This is the canonical phased plan. Each phase has:
- **Goal** — what we're trying to achieve
- **Deliverable** — concrete thing the user can poke at to confirm it works
- **Tasks** — implementation steps
- **Risks / decisions** — what could trip us up and what choice we're locking in
- **Success criteria** — how we'll know to move to the next phase

We will NOT start Phase 0 work until the user has read this file and given an explicit go-ahead.

Time estimates are rough — what matters is the order and the success criteria, not the calendar.

---

## Phase 0 — Setup & API connectivity (≈ 0.5 day)

**Goal:** A clean repo with secrets isolated and a verified ability to talk to LumenX.

**Deliverable:** A single script that lists the 20 LumenX products and prints inbox count. Runs locally.

**Tasks:**
1. Initialise git repo (`git init`), add `.gitignore` for `.env`, `data/`, `Antrphic_API_Key.txt`, `__pycache__`, `.next`, `node_modules`.
2. Move the Anthropic key out of `Antrphic_API_Key.txt` into `.env` (and delete the txt). Create `.env.example` listing required vars without values:
   - `ANTHROPIC_API_KEY`
   - `LUMENX_ADMIN_TOKEN`
   - `LUMENX_BASE_URL` (default `https://lumenx-demo.up.railway.app`)
3. Initialise Python project with `uv` (or `poetry`): `pyproject.toml` with `httpx`, `pydantic`, `python-dotenv`, `fastapi`, `uvicorn`, `sqlalchemy`, `anthropic`, `pytest`, `chromadb`, `torch`, `numpy`.
4. Write `agent/lumenx_client.py` — typed wrapper around the LumenX admin API. Auth header pulled from env. Retries with backoff for 5xx.
5. Write `agent/anthropic_client.py` — wraps `anthropic.Anthropic`, logs every call (model, input_tokens, output_tokens, latency, computed USD, request hash). No raw call goes around this wrapper.
6. Write `scripts/healthcheck.py` — calls `/api/admin/stats`, `/api/admin/products` (count = 20 expected), and a 1-token Haiku call. Prints summary.

**Risks / decisions:**
- *Decision:* `uv` over `poetry` for speed. Defer to user if they object.
- *Risk:* the deployed LumenX seed may have moved on from the documented 20 products. Healthcheck just logs the count; it isn't asserted.

**Success criteria:** `python -m scripts.healthcheck` prints LumenX stats + product count + a Haiku echo, with no secrets in stdout.

---

## Phase 1 — LLM Wiki (≈ 1–2 days)

**Goal:** A structured, retrievable knowledge base of every product and company policy. This is the agent's source of truth for anything pricing- or policy-related.

The Karpathy "context wiki" pattern (from the gist) calls for: **distil raw source material into compact, retrievable markdown pages**, then index them. We do exactly that.

**Deliverable:** A `data/wiki/` directory of one markdown file per product + `company_policies.md`, plus a Chroma index. A CLI query: `python -m scripts.wiki_query "what is emailpilot's refund window"` returns top-3 wiki chunks.

**Tasks:**
1. `scripts/pull_export.py` — caches `/api/admin/export` and `/api/admin/products` to `data/raw/`. Re-runnable.
2. `agent/llm_wiki/builder.py` — for each product, distil the raw JSON into a markdown page with stable sections:
   - Identity (name, id, one-liner)
   - Target audience
   - Pricing tiers (verbatim, never paraphrased)
   - Refund policy (verbatim)
   - Cancellation (verbatim)
   - Features (bulleted)
   - Integrations
   - Support SLA
   - Known FAQs (mined from past conversations in the export)
3. `agent/llm_wiki/retriever.py` — chunk by section, embed via a small open-source model (default: `sentence-transformers/all-MiniLM-L6-v2`, runs locally, no API cost), store in Chroma. Retrieval returns chunks with their source product id and section name so the drafter can cite.
4. `scripts/build_wiki.py` — orchestrates 1+2+3.

**Risks / decisions:**
- *Decision:* embeddings are local, not Anthropic. Cheaper and avoids tying us to a single provider for retrieval.
- *Decision:* pricing/refund text is stored **verbatim**, not LLM-paraphrased. Paraphrase is where hallucination starts.
- *Risk:* the FAQ-mining step could itself hallucinate. Mitigation: cite the source thread/message id for each FAQ entry, and skip if no exact-match evidence found.

**Success criteria:** wiki query for "emailpilot refund window" returns the actual refund window string from the product data, and a query for "purple dragon" returns nothing relevant (no hallucinated hits).

---

## Phase 2 — Intent Router (≈ 0.5–1 day)

**Goal:** Cheaply classify every incoming customer message so downstream stages can branch correctly.

**Deliverable:** A function `route_intent(thread) -> IntentResult` and a CLI that classifies the last 100 messages from the export and prints the distribution.

**Tasks:**
1. Define intent taxonomy. Phase 0's healthcheck revealed LumenX is already
   tagging threads with the following intents (counts from 100 seed threads):
   `pricing` (20), `feature` (15), `technical` (10), `integration` (10),
   `discount` (10), `cancellation` (10), `billing` (10), `multi-product` (5),
   `greeting` (5), `compare-competitor` (5). We adopt these as-is for
   compatibility, plus two of our own:
   - `chitchat` — off-topic conversational
   - `out_of_scope` — unrelated to our products
   Sensitivity tags applied on top:
   - `pricing`, `discount` → **sensitive** (wiki required)
   - `cancellation` → **sensitive** (wiki required; this is LumenX's
     refund/cancel bucket)
   - `billing` → **sensitive** (account-state changes)
   - others → standard
2. `agent/intent_router.py` — Haiku 4.5 with a JSON-mode prompt. Returns `{intent, confidence, requires_wiki, sensitivity}`.
3. Short-circuit logic: `greeting` and `chitchat` use a small templated drafter (no wiki, no Sonnet) — cheapest path.
4. Log every classification to the feedback log.

**Risks / decisions:**
- *Decision:* Haiku, not a local classifier. Cheap enough, and we benefit from improvements without retraining.
- *Risk:* JSON mode parsing failures. Mitigation: retry once, then fall back to `feature_question` with low confidence (forces human review).

**Success criteria:** Distribution of intents on the export matches the LumenX `intent_distribution` stat ±20%. Sensitive intents (`pricing`, `refund_or_cancel`) always set `requires_wiki=true`.

---

## Phase 3 — Context Builder (≈ 1 day)

**Goal:** Given a thread + intent, assemble the prompt context that the drafter will see. Token-budgeted, with clear sections.

**Deliverable:** A function `build_context(thread, intent) -> ContextWindow` and a CLI that prints the assembled context for any thread id.

**Tasks:**
1. `agent/context_builder.py` — composes a context with:
   - **System** — empathetic-but-professional persona, hallucination rules, citation format
   - **Product wiki snippets** — top-k retrieved by intent + last customer message
   - **Past Q&A from feedback log** — top-k similar resolved pairs
   - **This customer's history** — summarised if > N tokens (summary itself is cached to feedback log to avoid recomputation)
   - **All-customer summary stats** — short, e.g. "most common issue this week: billing login"
   - **Current thread** — full
2. Strict token budgets per section; truncation from the bottom of each section.
3. Cache the assembled context blob keyed by `(thread_id, intent, wiki_version)` for the dashboard's per-draft inspector.

**Risks / decisions:**
- *Decision:* the customer-history summary is generated by Haiku, not Sonnet — cost matters and the summary is non-sensitive.
- *Risk:* feedback log retrieval can pull in a previous *bad* reply if it was never corrected. Mitigation: only pull from feedback log entries marked `approved` or `human_final`.
- *Carried over from Phase 1 verification:* pure vector retrieval over the wiki misses cleanly on broad company-policy questions ("non-profit discount", "education eligibility"). The 20 product pricing chunks lexically dominate even after dropping the redundant "Annual discount" line from chunk text and adding company-side topic synonyms. The architectural fix is intent-aware retrieval: when intent ∈ {`discount`, `cancellation`, `pricing`}, also retrieve a top-k from a `where={"product_id": "_company"}` filter and merge. Phase 1's verbatim wiki and Chroma index already support this; Phase 3 just needs to call it.

**Success criteria:** For a known pricing question, the assembled context contains the matching wiki section verbatim and at least one similar past Q&A.

---

## Phase 4 — LLM Drafter (≈ 1 day) ✅ COMPLETE

**Goal:** Produce a candidate reply. Strict, citation-aware, never inventing pricing or refund detail.

**Deliverable:** A function `draft_reply(context) -> Draft` and an end-to-end CLI: pick a thread → draft → print.

**Tasks:**
1. `agent/drafter.py` — Sonnet 4.6 call. Returns structured JSON:
   ```json
   {
     "reply": "...",
     "cited_sources": ["product:emailpilot#refund_policy", "thread:abc123#msg42"],
     "uncertainty_flags": ["customer asked about X but wiki has no entry"]
   }
   ```
2. Post-generation guardrail: if intent is `pricing` or `refund_or_cancel` and `cited_sources` is empty → mark the draft as `low_confidence` and refuse to send without human review, regardless of MLP score.
3. Persist draft + context snapshot + all token/cost figures.

**Risks / decisions:**
- *Decision:* structured output via JSON schema, not free-text. Lets us run the guardrail mechanically.
- *Risk:* Sonnet may sometimes refuse JSON. Mitigation: one retry with a stricter prompt, then escalate to human review.

**Success criteria:** A draft for a refund question cites the wiki refund section. A draft for an unknown product asks for clarification instead of inventing.

**Verified (2026-05-27):**
- conv-055 (cancellation) → cites `calendarsync__cancellation` + `calendarsync__refund` inline; HIGH confidence; auto_sendable=True.
- conv-068 (discount, BillSplit) → cites `billsplit__pricing` + `_company__bundle`; uncertainty flag for volume discounts (not in wiki); LOW confidence; auto_sendable=False.
- conv-001 (integration, Airtable not in wiki) → uncertainty flag fires; auto_sendable=False.
- conv-048 (greeting) → no wiki, HIGH confidence, auto_sendable=True.
- Guardrail (sensitive intent + no wiki citations → blocked) is hard-coded in drafter logic.
- Bug fix: `_build_current_thread_section` now strips trailing assistant turns so coalesced messages always end with a user turn (Anthropic API requirement).
- Windows terminal: CLI uses UTF-8 stdout reconfigure for model replies containing Unicode arrows/bullets.
- Drafts persisted to `data/drafts/<uuid>.json` for Phase 5 pickup.
- Drafter cost per call: ~$0.004–$0.008 (Sonnet 4.6, 1,000–1,700 in tokens + 50–180 out).

---

## Phase 5 — Feedback Log + Persistence (≈ 1 day) ✅ COMPLETE

**Goal:** Every draft and every human action lands in a queryable store so we can (a) drive the dashboard, (b) feed retrieval, (c) build training labels.

**Deliverable:** SQLite DB with three tables, populated by Phase 4 drafts, plus a `record_human_action` API.

**Tasks:**
1. Schema:
   - `drafts(id, thread_id, intent, context_snapshot_json, draft_text, model, input_tokens, output_tokens, cost_usd, created_at)`
   - `human_actions(draft_id, action, final_text, edit_distance, reviewer, decided_at)` — action in `approved | edited | rejected | auto_sent`
   - `confidence_predictions(draft_id, features_json, score, threshold, would_auto_send)`
2. `agent/feedback_log/` — SQLAlchemy models + read/write helpers + similarity search over past `final_text` (uses the same Chroma instance).
3. `agent/api/feedback.py` — endpoints `POST /drafts/{id}/action` and `GET /drafts/{id}`.

**Risks / decisions:**
- *Decision:* SQLite for simplicity. Move to Postgres only if it becomes a bottleneck (won't, at this volume).
- *Risk:* PII in stored conversations. Mitigation: stored only on our server, never shipped externally, dashboard requires admin token.

**Success criteria:** Generating a draft creates a `drafts` row. Approving via API creates a `human_actions` row. Restarting the agent does not lose state.

**Verified (2026-05-27):**
- `data/feedback.db` (SQLite) created on first import; persists across restarts.
- `drafter.py` auto-records every draft to SQLite (best-effort, non-fatal). Existing JSON files preserved as backup.
- `scripts/migrate_drafts.py` successfully imported all 7 Phase 4 draft JSON files.
- API server: `uvicorn agent.main:app --port 8000`
- Tested endpoints:
  - `GET /health` → `{"status":"ok","drafts":8,"pending_review":6}`
  - `GET /api/drafts` → paginated list with intent/label/auto_sendable filters
  - `GET /api/drafts/pending` → 5 drafts awaiting review
  - `POST /api/drafts/{id}/action` → recorded "approved" (edit_distance=null) and "edited" (edit_distance=160) actions
  - `GET /api/drafts/{id}` → full record with human_actions and confidence_predictions
  - `GET /api/stats` → breakdowns by action, label, intent; cost totals
- New draft (conv-016 billing, annual discount question) auto-recorded to SQLite immediately upon generation.
- Swagger docs available at http://localhost:8000/docs

---

## Phase 6 — Bootstrap labels (≈ 1–2 days, mostly human time)

**Goal:** Get the first ~300 labelled examples so the Confidence Net has something to learn from.

**Deliverable:** ~300 rows in `human_actions` with action ∈ {approved, edited, rejected} covering a wide quality spread.

**Tasks:**
1. `scripts/bootstrap_labels.py` — iterate over the 100 seed threads. For each, generate drafts under *deliberately varied conditions* to create a quality spread:
   - normal Sonnet draft
   - Haiku draft (lower quality)
   - Sonnet draft with the wiki retrieval intentionally disabled (more likely to hallucinate)
   - Sonnet draft with a small temperature bump
2. A simple terminal review tool: prints draft + context + ground-truth historical reply side-by-side, asks user for `a / e / r` and a free-text correction if `e`.
3. Persist into `human_actions`.

**Risks / decisions:**
- *Decision:* hybrid (bootstrap + live) approach was chosen. This phase is the bootstrap half. The live half starts in Phase 8.
- *Risk:* user labelling fatigue. Mitigation: keep it under 300, batch in sessions of ~50.

**Success criteria:** ≥300 labelled drafts in the DB, with at least 25% in each of approved / edited / rejected.

---

## Phase 7 — Train the Confidence Net (≈ 1–2 days)

**Goal:** A small, calibrated MLP that takes a draft's features and outputs P(would_be_approved).

**Deliverable:** A saved `model.pt`, a calibration plot, and a CLI that scores any draft.

**Tasks:**
1. Define ~20 features in `agent/confidence_net/features.py`:
   - one-hot intent (9 dims)
   - draft length (chars / tokens)
   - draft length ratio vs. historical replies for this intent
   - top-k feedback-log similarity score
   - wiki coverage: fraction of draft sentences that align to a retrieved wiki chunk
   - sensitive-topic flag (pricing / refund) AND `cited_sources` empty
   - count of `uncertainty_flags` from the drafter
   - customer-thread length so far
   - is-this-a-returning-customer flag
   - avg Sonnet log-prob (if exposed; else omit)
2. Label = `approved` → 1, `edited` with low edit-distance → 0.5 (or drop), `rejected`/`edited`-heavy → 0.
3. Tiny MLP: 20 → 64 → 32 → 1, sigmoid output, binary cross-entropy. ~80/20 train/val.
4. Calibration: temperature scaling on the validation set.
5. Save checkpoint + features schema (so production inference can't drift).

**Risks / decisions:**
- *Decision:* drop ambiguous "lightly-edited" cases from the first training run rather than soft-label them. Simpler.
- *Decision:* features schema is versioned. Inference loads `features_v{n}.json` so model + schema can't desync.
- *Risk:* 300 examples may be on the thin side. Mitigation: aggressive regularisation (dropout, weight decay), and accept that auto-send threshold starts strict.

**Success criteria:** Val AUC ≥ 0.75. Calibration plot looks roughly diagonal. Model file < 1 MB.

---

## Phase 8 — Auto-send router (≈ 0.5 day)

**Goal:** Close the loop. If the MLP scores a draft above threshold AND no hard guardrail trips, send it via `POST /api/admin/threads/{id}/reply`.

**Deliverable:** End-to-end demo: a new customer message in LumenX → agent picks it up → drafts → MLP scores → sends if score ≥ threshold, else queues for human review.

**Tasks:**
1. `agent/router.py` — combines MLP score with hard rules:
   - hard veto if `pricing` or `refund_or_cancel` and no wiki citations
   - hard veto if `uncertainty_flags` non-empty
   - else gate on `score ≥ threshold`
2. Send path uses LumenX reply endpoint with `draft_source="agent"` and the `confidence` field, so the audit trail is visible in the LumenX admin UI too.
3. Default threshold: **0.90** (very strict). Configurable via env + dashboard.

**Risks / decisions:**
- *Decision:* start at 0.90, not 0.85, on the principle that the first auto-sends should be near-certain wins.
- *Risk:* the model overfits to bootstrap data and overestimates confidence on novel queries. Mitigation: dashboard charts auto-sent vs. later-corrected, so we can spot the failure mode.

**Success criteria:** A held-out greeting reply auto-sends; a held-out pricing question without wiki coverage does not.

---

## Phase 9 — Dashboard (≈ 2–3 days)

**Goal:** A real UI for reviewing drafts, watching costs, and inspecting any reply down to its full context window.

**Deliverable:** A Next.js app, deployable separately, with four pages:

1. **Inbox** — pending drafts. Shows confidence score, intent, and accept/edit/reject buttons. Editing inline writes back to the feedback log and uses the LumenX reply endpoint to send.
2. **History** — all sent replies, filterable by `auto | human`, by intent, by customer.
3. **Costs** — daily and per-reply spend; per-model breakdown; total spend this month.
4. **Draft detail** (`/drafts/[id]`) — expandable sections for:
   - Full context window (every section, with token counts)
   - Cited sources (links to wiki entries and past Q&A)
   - Feature vector + MLP score breakdown
   - Token/cost figures for that draft
   - Was-it-auto-sent? Did a human edit it later?

**Tasks:**
1. Scaffold `dashboard/` (Next.js 16 to match LumenX).
2. Wire to agent's FastAPI endpoints (`/api/drafts`, `/api/costs`, etc.).
3. Auth: simple admin token check, same pattern as LumenX.
4. Use the existing LumenX admin UI as visual reference for consistency.

**Risks / decisions:**
- *Decision:* dashboard is a separate Next.js app, not embedded in the agent service, because Python + JSX in one process is more pain than it's worth.
- *Risk:* dashboard becomes a time-sink. Mitigation: ship a v0 with the four pages even if visually plain, polish later.

**Success criteria:** A reviewer can sit on the Inbox page, approve five drafts in a row, and the cost page updates immediately.

---

## Phase 10 — Deploy to Railway (≈ 0.5–1 day)

**Goal:** Both services live, on the same Railway project as LumenX, polling and serving.

**Deliverable:** Public dashboard URL. Live polling. Logs visible in Railway.

**Tasks:**
1. `Dockerfile` for `agent/`. Includes background poller + FastAPI.
2. `Dockerfile` for `dashboard/`. Standard Next.js production build.
3. Railway project — two services, env vars set, volume mounted for `data/` (SQLite + Chroma + model checkpoint).
4. Healthchecks + log forwarding.
5. Polling interval: 5s in prod (still under LumenX's existing 2.5s UI polling cadence per the API notes).

**Risks / decisions:**
- *Decision:* Same Railway project, different services. Easy networking, separate scaling.
- *Risk:* SQLite on a volume is fine for one replica but doesn't scale to multi-replica. Acceptable for now; documented as a known limit.

**Success criteria:** A customer messages on the live LumenX site; within ~10s the agent either replies (high confidence) or surfaces a draft in the dashboard inbox.

---

## Phase 11 — Monitoring & weekly retrain (ongoing)

**Goal:** The system improves week-over-week without manual intervention beyond reviewing drafts.

**Tasks:**
1. Weekly cron (Railway scheduler) retrains the Confidence Net on all data since last train.
2. Weekly cost summary mailed/Slacked to the user.
3. Alert if cost/day exceeds threshold OR error rate > 1%.
4. A "promote checkpoint?" gate: a new model is staged but doesn't go live until the dashboard shows its val metrics and the user clicks promote.

**Success criteria:** Six weeks after launch, auto-send rate has moved (in either direction — both directions are signal) and the dashboard shows the trend.

---

## Open questions we don't yet need to resolve

- **Multilingual support.** Out of scope for now; intent router would need expansion.
- **Voice / audio replies.** Out of scope.
- **Customer-side identity verification.** LumenX handles this today; we trust it.
- **Multi-replica scaling.** Documented limit; address only if SQLite or polling becomes the bottleneck.

---

## Next action

User reads this file, agrees (or pushes back on) the phasing, and gives the go-ahead. **Phase 0 starts only after that.**
