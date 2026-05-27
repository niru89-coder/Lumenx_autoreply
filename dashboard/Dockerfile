# ── LumenX Dashboard — Production Dockerfile (multi-stage) ───────────────────
#
# NEXT_PUBLIC_* vars are baked into the JS bundle at build time.
# Set them as Railway build-time env vars (not just runtime env vars):
#   NEXT_PUBLIC_AGENT_API_URL   https://agent-xxxx.up.railway.app
#   NEXT_PUBLIC_ADMIN_TOKEN     <same value as LUMENX_ADMIN_TOKEN on the agent>
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Install npm deps ─────────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ── Stage 2: Build Next.js (standalone output) ────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* vars must be present at build time to be embedded in the bundle.
ARG NEXT_PUBLIC_AGENT_API_URL
ARG NEXT_PUBLIC_ADMIN_TOKEN
ENV NEXT_PUBLIC_AGENT_API_URL=$NEXT_PUBLIC_AGENT_API_URL
ENV NEXT_PUBLIC_ADMIN_TOKEN=$NEXT_PUBLIC_ADMIN_TOKEN

RUN npm run build

# ── Stage 3: Minimal runtime image ────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
# Disable Next.js telemetry in the container
ENV NEXT_TELEMETRY_DISABLED=1

# The standalone build already bundles its own node_modules subset.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static     ./.next/static
COPY --from=builder /app/public           ./public

EXPOSE 3000
ENV PORT=3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
