"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Draft } from "@/lib/types";
import { ConfidenceBadge, IntentBadge, GuardrailBadge } from "@/components/StatusBadge";
import ActionPanel from "@/components/ActionPanel";

function DraftCard({ draft, onDone }: { draft: Draft; onDone: () => void }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article
      className="rounded-xl p-5 flex flex-col gap-3"
      style={{ background: "var(--card)", border: "1px solid var(--border)" }}
    >
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-2">
        <IntentBadge intent={draft.intent} />
        <ConfidenceBadge label={draft.confidence_label} />
        <GuardrailBadge triggered={draft.guardrail_triggered} />
        <span className="ml-auto text-xs" style={{ color: "var(--muted)" }}>
          {new Date(draft.created_at).toLocaleString()}
        </span>
      </div>

      {/* Thread + meta */}
      <div className="flex items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
        <span>Thread: <Link href={`/drafts/${draft.id}`}
          className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2">
          {draft.thread_id.slice(0, 12)}…
        </Link></span>
        <span>·</span>
        <span>{draft.model.split("-").slice(-2).join("-")}</span>
        <span>·</span>
        <span>{draft.input_tokens + draft.output_tokens} tok</span>
        <span>·</span>
        <span>${draft.cost_usd.toFixed(4)}</span>
      </div>

      {/* Draft text */}
      <div
        className={`text-sm leading-relaxed overflow-hidden ${expanded ? "" : "max-h-24"}`}
        style={{ color: "var(--foreground)" }}
      >
        <pre className="whitespace-pre-wrap font-sans">{draft.draft_text}</pre>
      </div>
      {draft.draft_text.length > 200 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs self-start"
          style={{ color: "var(--muted)" }}
        >
          {expanded ? "▲ Show less" : "▼ Show more"}
        </button>
      )}

      {/* Uncertainty flags */}
      {draft.uncertainty_flags.length > 0 && (
        <p className="text-xs text-yellow-400">
          ⚑ {draft.uncertainty_flags.join(" · ")}
        </p>
      )}

      {/* Actions */}
      <div className="pt-1 border-t" style={{ borderColor: "var(--border)" }}>
        <ActionPanel draftId={draft.id} draftText={draft.draft_text} onDone={onDone} />
      </div>
    </article>
  );
}

export default function InboxPage() {
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.pendingDrafts();
      setDrafts(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 15_000);
    return () => clearInterval(t);
  }, [load]);

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">Inbox</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>
            Drafts awaiting human review
          </p>
        </div>
        <button
          onClick={load}
          className="text-xs px-3 py-1.5 rounded-lg transition-colors"
          style={{ border: "1px solid var(--border)", color: "var(--muted)" }}
        >
          ↻ Refresh
        </button>
      </div>

      {loading && (
        <p className="text-sm text-center py-16" style={{ color: "var(--muted)" }}>
          Loading…
        </p>
      )}

      {error && (
        <div
          className="rounded-lg p-4 text-sm text-red-400 mb-4"
          style={{ background: "#1e0a0a", border: "1px solid #7f1d1d" }}
        >
          {error}
        </div>
      )}

      {!loading && !error && drafts.length === 0 && (
        <div
          className="rounded-xl p-12 text-center"
          style={{ background: "var(--card)", border: "1px solid var(--border)" }}
        >
          <p className="text-4xl mb-3">🎉</p>
          <p className="text-sm font-medium text-white">Inbox zero!</p>
          <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
            No drafts pending review right now.
          </p>
        </div>
      )}

      <div className="flex flex-col gap-4">
        {drafts.map((d) => (
          <DraftCard key={d.id} draft={d} onDone={load} />
        ))}
      </div>
    </div>
  );
}
