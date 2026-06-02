import type { Draft, FeedbackStats, HealthStatus, ModelsResponse } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_AGENT_API_URL ?? "http://localhost:8000";
const TOKEN = process.env.NEXT_PUBLIC_ADMIN_TOKEN ?? "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Token": TOKEN,
      ...((init?.headers as Record<string, string>) ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} → HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  /** Reviewer inbox — drafts with no human action yet */
  pendingDrafts: () =>
    req<Draft[]>("/api/drafts/pending?limit=200"),

  /** Paginated draft list with optional filters */
  listDrafts: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return req<Draft[]>(`/api/drafts${qs}`);
  },

  /** Full draft record including human_actions + confidence_predictions */
  getDraft: (id: string) => req<Draft>(`/api/drafts/${id}`),

  /** Saved ContextWindow snapshot (may be null) */
  getContext: (id: string) =>
    req<Record<string, unknown>>(`/api/drafts/${id}/context`),

  /** Record a human action on a draft */
  recordAction: (
    id: string,
    action: string,
    final_text?: string | null
  ) =>
    req<{ ok: boolean }>(`/api/drafts/${id}/action`, {
      method: "POST",
      body: JSON.stringify({ action, final_text: final_text ?? null }),
    }),

  /** Aggregate stats for the costs page */
  stats: () => req<FeedbackStats>("/api/stats"),

  /** Agent health + poller status */
  health: () => req<HealthStatus>("/health"),

  /** Confidence Net checkpoints + which is active (Phase 11) */
  listModels: () => req<ModelsResponse>("/api/models"),

  /** Promote a candidate checkpoint to active; hot-reloads the router scorer */
  promoteModel: (version: number, reviewer = "dashboard") =>
    req<{ promoted: ActiveRecordLike; loaded: unknown }>(
      `/api/models/${version}/promote?reviewer=${encodeURIComponent(reviewer)}`,
      { method: "POST" }
    ),
};

type ActiveRecordLike = {
  active_version: number;
  previous_version: number | null;
  promoted_at: string;
  promoted_by: string;
};
