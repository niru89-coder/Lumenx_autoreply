---
name: lumenx-api
description: Talk to the LumenX deployed admin/public API. Use when reading the inbox, fetching threads/products, sending replies, or pulling the full export. Handles auth header, base URL, retries, and polling cadence.
---

# LumenX API skill

LumenX is the deployed Next.js customer-support platform we're building this agent for. Treat it as **read-mostly** plus one write endpoint (the reply).

- Base URL: `https://lumenx-demo.up.railway.app`
- Auth: every `/api/admin/*` call needs header `X-Admin-Token: <token>` (env: `LUMENX_ADMIN_TOKEN`)
- Reference: `api_description.txt` at repo root is the source of truth — re-read it if anything seems off.

## Endpoints we use, in order of frequency

| Verb | Path | Why |
|---|---|---|
| GET | `/api/admin/inbox?since=ISO` | The poll. `since` = last `server_time` we saw. |
| GET | `/api/admin/threads/{id}` | Full thread when we need to draft. |
| POST | `/api/admin/threads/{id}/reply` | Send the reply. Body `{text, draft_source, confidence}`. |
| GET | `/api/admin/products` | Feeds the LLM Wiki. Re-pull on each wiki rebuild. |
| GET | `/api/admin/export` | One-shot bulk dump. Use for bootstrap labelling. |
| GET | `/api/admin/stats` | Health/observability. |

## Conventions

- **Always go through `agent/lumenx_client.py`.** Never call `httpx` against LumenX directly from feature code. The client centralises auth, retries (5xx → exponential backoff, 3 attempts), timeouts (10s), and structured logging.
- **Polling:** 5s default. LumenX's own UI polls 2.5s; staying slower than it keeps us off the critical UX path.
- **`since` handling:** persist the last `server_time` to disk so we resume cleanly across restarts.
- **Reply metadata:** always set `draft_source` (`"agent"` for auto-sent, `"human"` for reviewer-edited) and `confidence` (the MLP score). LumenX's admin UI shows these.

## Hard rules

- Never log the admin token. The client masks it to `lmx_…` in any log line.
- Never assume the seeded threads stay the same — they reset on every container boot (per API notes). Live threads (`seeded=0`) are preserved.
- A failed reply POST does **not** retry automatically — duplicates would confuse the customer. Surface failures to the dashboard for manual re-send.

## Quick test

```bash
curl -H "X-Admin-Token: $LUMENX_ADMIN_TOKEN" \
     https://lumenx-demo.up.railway.app/api/admin/stats
```
