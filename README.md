# LumenX Auto-Reply Agent

An LLM-powered customer-support agent for the [LumenX](https://lumenx-demo.up.railway.app) platform. Watches the LumenX admin inbox, drafts replies, decides — via a small confidence model — whether to send the reply automatically or hand it to a human reviewer, and learns from every edit.

> **Status:** Planning phase. No code yet. See `PLAN.md` for the phased build.

---

## What it does

```
   New customer message
            │
            ▼
   ┌──────────────────┐
   │  Intent Router   │  Haiku-4.5 — cheap classification
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  Pulls:
   │ Context Builder  │   • current thread
   │                  │   • this customer's past threads
   └────────┬─────────┘   • similar Q&A from feedback log
            │             • product wiki snippets
            ▼
   ┌──────────────────┐
   │   LLM Drafter    │  Sonnet-4.6 — produces candidate reply
   └────────┬─────────┘
            ▼
   ┌──────────────────┐  Tiny PyTorch MLP (~20-dim features)
   │  Confidence Net  │  outputs 0–1 score
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │      Router      │  score ≥ threshold → auto-send
   │                  │  score <  threshold → human review
   └──────────────────┘
```

Two source diagrams: `image copy.png` (pipeline) and `image.png` (confidence net).

## Why a confidence net?

A single LLM produces a reply but has no honest sense of when it should second-guess itself. A small model trained on **real human edit/accept signal** can. The MLP doesn't try to know the right answer — it only learns the question "does this draft look like one a human would have sent?" That keeps the model tiny, fast, and easy to retrain.

The threshold is yours to set. Start strict (high threshold → most drafts go to human review), relax as confidence in the model grows.

## Project goals

1. **Never hallucinate pricing or refund policy.** Sensitive topics are checked against the product wiki before send; if the wiki has no answer the agent admits ignorance.
2. **Be cost-aware.** Every LLM call records tokens and USD spend. Haiku does cheap work; Sonnet does drafts; Opus only when justified.
3. **Be observable.** Every reply has a full audit trail — context window, sources cited, confidence features, MLP score — viewable in the dashboard.
4. **Improve over time.** Every human edit becomes a future training label for the confidence net and a future retrieval source for the drafter.

## Stack

| Component | Tech |
|---|---|
| Agent service | Python 3.11+, FastAPI, background poller |
| LLM API | Anthropic — Haiku 4.5 (intent / utility), Sonnet 4.6 (drafts) |
| Vector retrieval | ChromaDB (local file index) |
| Feedback DB | SQLite via SQLAlchemy |
| Confidence model | PyTorch MLP, ~20-dim input, binary cross-entropy |
| Dashboard | Next.js (separate app) |
| Hosting | Railway (separate service from LumenX) |

## Talking to LumenX

The agent only uses public + admin HTTP endpoints. See `api_description.txt` for the full spec.

Hot paths:
- `GET /api/admin/inbox?since=ISO` — poll for new customer messages
- `GET /api/admin/threads/{id}` — read full thread
- `POST /api/admin/threads/{id}/reply` — send reply (with `draft_source` and `confidence` metadata)
- `GET /api/admin/products` — feeds the LLM Wiki
- `GET /api/admin/export` — full dump for offline analysis and bootstrap labelling

The LumenX repo and database are not touched.

## Phased build

See `PLAN.md` for full detail. Headlines:

0. Setup & API connectivity check
1. LLM Wiki from products endpoint
2. Intent router (Haiku)
3. Context builder
4. LLM drafter (Sonnet)
5. Feedback log + draft persistence
6. Bootstrap labels (hand-rate ~300 drafts)
7. Train the Confidence Net MLP
8. Auto-send router (threshold-gated)
9. Dashboard (inbox, history, costs, per-draft detail)
10. Deploy to Railway
11. Monitoring + weekly retrain

## Repository layout (target — does not yet exist)

```
agent/        # Python FastAPI service + ML
dashboard/    # Next.js admin UI
scripts/      # one-off data + bootstrap scripts
data/         # local: chroma index, sqlite, model checkpoints (gitignored)
```

## Secrets

- `ANTHROPIC_API_KEY` — currently in `Antrphic_API_Key.txt` (will move to `.env`, kept out of git).
- `LUMENX_ADMIN_TOKEN` — `lmx_GQlch0Q5NOwVuVSADXRuFNJvxIpzVGwI`, in `api_description.txt`. Will move to `.env`.

Neither is ever logged, committed, or shown in the dashboard.

## Deployment (Railway)

Two services in the same Railway project, both built from Dockerfiles in this repo.

### Agent service (this directory)

- Builder: `DOCKERFILE` (`Dockerfile`)
- Healthcheck path: `/health`
- Required env vars:
  - `ANTHROPIC_API_KEY`
  - `LUMENX_ADMIN_TOKEN`
  - `LUMENX_BASE_URL` (default: `https://lumenx-demo.up.railway.app`)
  - `DASHBOARD_URL` — public dashboard URL, added to the CORS allowlist
  - `AUTO_SEND_ENABLED` (default `false`)
  - `AUTO_SEND_THRESHOLD` (default `0.90`)
  - `AUTO_SEND_BLOCKED_INTENTS` (default `pricing,cancellation`)
- Persistent Volume: mount at `/app/data` so SQLite (`feedback.db`), the Chroma index, model checkpoints, and `poller_state.json` survive redeploys.

### Dashboard service (`dashboard/`)

- Builder: `DOCKERFILE` (`dashboard/Dockerfile`, multi-stage standalone build)
- Healthcheck path: `/`
- Build-time env vars (must be set as Railway build vars, not just runtime — they're baked into the JS bundle):
  - `NEXT_PUBLIC_AGENT_API_URL` — public URL of the agent service
  - `NEXT_PUBLIC_ADMIN_TOKEN` — same value as the agent's `LUMENX_ADMIN_TOKEN`

### Polling cadence

Production poll interval is 5s (configurable via `LUMENX_POLL_INTERVAL_SECONDS`). LumenX's own UI polls at ~2.5s, so we stay below it.

### Known limit

SQLite on a single mounted volume is fine for one replica. Multi-replica scaling would require moving the feedback store to Postgres.

## Working on this project with Claude Code

This repo ships with project-scoped skills under `.claude/skills/` that future Claude Code sessions will pick up automatically:

- `lumenx-api` — wraps every LumenX endpoint call (auth header, retries, polling)
- `confidence-net` — train/eval/serve the MLP
- `anthropic-cost-tracker` — wrap every Anthropic call so cost is always logged

Read `CLAUDE.md` for orientation rules.
