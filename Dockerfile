# ── LumenX Auto-Reply Agent — Production Dockerfile ──────────────────────────
#
# Data that must survive redeployments (SQLite, Chroma, model checkpoints)
# should be stored in a Railway Persistent Volume mounted at /app/data.
#
# Required env vars at runtime (set in Railway service settings):
#   ANTHROPIC_API_KEY
#   LUMENX_ADMIN_TOKEN
#   LUMENX_BASE_URL          (default: https://lumenx-demo.up.railway.app)
#   DASHBOARD_URL            (e.g. https://dashboard-xxxx.up.railway.app)
#   AUTO_SEND_ENABLED        (default: false)
#   AUTO_SEND_THRESHOLD      (default: 0.90)
#   AUTO_SEND_BLOCKED_INTENTS (default: pricing,cancellation)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Build deps for native extensions (chromadb, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Install CPU-only PyTorch first ───────────────────────────────────
# Prevents pip from pulling the ~2.5 GB CUDA wheel later.
# Do this before copying source so it's cached unless base image changes.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: Project metadata (cached unless pyproject.toml changes) ──────────
COPY pyproject.toml README.md ./

# ── Layer 3: Source code ───────────────────────────────────────────────────────
COPY agent/   agent/
COPY scripts/ scripts/

# ── Layer 4: Install remaining deps (torch already in cache) ─────────────────
RUN pip install --no-cache-dir -e .

# Create the data directory; in production this path should be a Railway volume.
RUN mkdir -p data

EXPOSE 8000

# Healthcheck — Railway uses this to decide when the container is ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn directly for clean signal handling (the startup event wires the poller).
CMD ["uvicorn", "agent.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
