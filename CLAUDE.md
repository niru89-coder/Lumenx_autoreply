# CLAUDE.md — Auto-Reply Agent for LumenX

This file orients future Claude Code sessions working in this repository. Read it before doing anything else.

## What this project is

An **auto-reply LLM agent** that watches the customer-support inbox of an existing deployed product called **LumenX** and either (a) sends replies automatically when it is confident, or (b) drafts replies for a human reviewer when it is not. Confidence is decided by a small MLP trained on real human accept/edit signal.

LumenX itself is a separate, already-deployed Next.js app. We do **not** modify it. We talk to it over HTTP only.

- LumenX site: https://lumenx-demo.up.railway.app
- LumenX repo: https://github.com/VizuaraAI/lumenx
- LumenX API spec: see `api_description.txt` in this directory

## Architecture (one-line view)

```
inbox  →  Intent Router  →  Context Builder  →  LLM Draft  →  Confidence Net  →  Router
                                                                                  ├─ high → auto-send
                                                                                  └─ low  → human review
```

Two reference diagrams:
- `image copy.png` — full Auto-Reply Agent architecture
- `image.png` — Confidence Net training/inference detail

## Stack (locked in)

| Layer | Choice |
|---|---|
| Agent backend | Python 3.11+ / FastAPI |
| LLM (intent + cheap calls) | Anthropic `claude-haiku-4-5` |
| LLM (draft generation) | Anthropic `claude-sonnet-4-6` (escalate to `claude-opus-4-7` only when justified) |
| Confidence Net | PyTorch, tiny MLP (~2–4 hidden layers, ~64 units) |
| Vector store | ChromaDB (local, file-backed) |
| Feedback DB | SQLite via SQLAlchemy |
| Dashboard | Next.js (separate app, same domain pattern as LumenX admin UI) |
| Deployment | Railway, **separate service** from LumenX |
| Polling | Background worker polling `/api/admin/inbox?since=...` every 5s |

## Top-level constraints

1. **Never hallucinate pricing or refund details.** If the LLM Wiki does not contain the answer, the agent must say something like "Let me check that and get back to you" — never invent numbers or policy terms. This is enforced in the system prompt and verified by a post-generation check against retrieved wiki snippets.
2. **Minimize cost.** Haiku for classification and short utility calls. Sonnet for drafts. Opus only when explicitly justified. Every LLM call records input/output tokens and computed USD cost.
3. **Build in phases.** See `PLAN.md`. Do not jump ahead — each phase has a working deliverable that can be demoed before the next one starts.
4. **Every draft is observable.** Per-reply context window, token counts, MLP feature vector, and confidence score are all persisted and exposed in the dashboard.
5. **Human-in-the-loop by default.** Auto-send is off until the Confidence Net has been trained on real data AND the user has explicitly raised the threshold.

## Repository layout (target)

```
.
├── CLAUDE.md                  # this file
├── README.md                  # human-facing
├── PLAN.md                    # phased execution plan
├── api_description.txt        # LumenX API spec (do not edit)
├── instructions.txt           # original brief from user (do not edit)
├── image.png                  # confidence net diagram (do not edit)
├── image copy.png             # full architecture diagram (do not edit)
├── Antrphic_API_Key.txt       # secret — never commit, never read aloud
├── .env.example               # template for required env vars
├── .gitignore
├── pyproject.toml             # Python deps via uv or poetry
├── agent/                     # Python FastAPI service
│   ├── main.py                # FastAPI app + background poller
│   ├── config.py              # env, model IDs, thresholds
│   ├── lumenx_client.py       # thin HTTP client around LumenX admin API
│   ├── anthropic_client.py    # wrapped client with cost tracking
│   ├── intent_router.py       # Haiku-based intent classification
│   ├── context_builder.py     # assembles the prompt context
│   ├── llm_wiki/              # product wiki: builder + retriever
│   ├── feedback_log/          # SQLite models + retrieval over past Q&A
│   ├── confidence_net/        # MLP training + inference
│   ├── drafter.py             # produces candidate reply
│   ├── router.py              # auto-send vs human-review decision
│   ├── observability.py       # token + cost logging
│   └── api/                   # HTTP endpoints used by the dashboard
├── dashboard/                 # Next.js app
│   ├── app/inbox/             # pending drafts to review
│   ├── app/history/           # all sent replies
│   ├── app/costs/             # cost & token analytics
│   └── app/drafts/[id]/       # per-draft drill-down (context, features, score)
├── data/                      # local-only: chroma index, sqlite db, model checkpoints
└── scripts/                   # one-off scripts: pull export, bootstrap labels, etc.
```

## Working agreements

- **No code before plan approval.** Phase 0 is "user reads PLAN.md and approves" — do not start scaffolding until they say so.
- **Each phase ends with a demoable artifact.** A phase is not done until something the user can poke at exists.
- **Cost log starts on day one.** Every Anthropic call goes through `anthropic_client.py`, no exceptions.
- **Secrets stay in `.env`.** Never paste the Anthropic key or the LumenX admin token into code, logs, commits, or chat responses.
- **Avoid editing `instructions.txt`, `api_description.txt`, or the two PNGs.** They are the source-of-truth brief.

## Key files for orientation

| File | What it tells you |
|---|---|
| `instructions.txt` | The original user brief — read this first |
| `api_description.txt` | All LumenX endpoints, auth header, example curls |
| `PLAN.md` | What gets built, in what order, with what success criteria |
| `image copy.png` | The pipeline diagram |
| `image.png` | The MLP training + inference detail |

## Commands you'll often run (once built)

These don't exist yet — they're the target shape so future-Claude knows what to wire up.

```
# Local dev
uv run python -m agent.main                 # start agent + poller
cd dashboard && npm run dev                 # start dashboard

# Data ops
uv run python -m scripts.pull_export        # cache LumenX export locally
uv run python -m scripts.build_wiki         # (re)build LLM Wiki from products
uv run python -m scripts.bootstrap_labels   # generate candidate drafts for hand-rating

# ML ops
uv run python -m agent.confidence_net.train # train the MLP from feedback log
uv run python -m agent.confidence_net.eval  # eval current checkpoint

# Deploy
railway up                                  # deploys agent service
cd dashboard && railway up                  # deploys dashboard
```
