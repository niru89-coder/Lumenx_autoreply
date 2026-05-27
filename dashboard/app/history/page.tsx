"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Draft } from "@/lib/types";
import { ConfidenceBadge, IntentBadge, ActionBadge } from "@/components/StatusBadge";

const INTENT_OPTIONS = [
  "all", "pricing", "billing", "cancellation", "refund",
  "technical_support", "feature_request", "account", "shipping",
  "general_inquiry", "complaint", "partnership", "other",
];

const ACTION_OPTIONS = ["all", "approved", "edited", "rejected", "auto_sent"];

function fmt(usd: number) {
  return usd < 0.001 ? "<$0.001" : `$${usd.toFixed(4)}`;
}

export default function HistoryPage() {
  const [drafts, setDrafts]   = useState<Draft[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const [intent, setIntent]   = useState("all");
  const [action, setAction]   = useState("all");
  const [page, setPage]       = useState(0);
  const PAGE_SIZE = 25;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string> = {
        limit: String(PAGE_SIZE),
        offset: String(page * PAGE_SIZE),
      };
      if (intent !== "all") params.intent = intent;
      if (action !== "all") params.action_filter = action;
      const data = await api.listDrafts(params);
      setDrafts(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [intent, action, page]);

  useEffect(() => { load(); }, [load]);
  // Reset page when filters change
  useEffect(() => { setPage(0); }, [intent, action]);

  const selectClass =
    "text-xs rounded-lg px-2 py-1.5 outline-none cursor-pointer " +
    "bg-slate-800 text-slate-300 border border-slate-700 hover:border-indigo-500 transition-colors";

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold text-white">History</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>
            All drafts — acted and pending
          </p>
        </div>

        {/* Filters */}
        <div className="flex gap-2 flex-wrap">
          <select
            className={selectClass}
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
          >
            {INTENT_OPTIONS.map((o) => (
              <option key={o} value={o}>{o === "all" ? "All intents" : o}</option>
            ))}
          </select>
          <select
            className={selectClass}
            value={action}
            onChange={(e) => setAction(e.target.value)}
          >
            {ACTION_OPTIONS.map((o) => (
              <option key={o} value={o}>{o === "all" ? "All actions" : o}</option>
            ))}
          </select>
          <button
            onClick={load}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors"
            style={{ border: "1px solid var(--border)", color: "var(--muted)" }}
          >
            ↻
          </button>
        </div>
      </div>

      {error && (
        <div
          className="rounded-lg p-4 text-sm text-red-400 mb-4"
          style={{ background: "#1e0a0a", border: "1px solid #7f1d1d" }}
        >
          {error}
        </div>
      )}

      {/* Table */}
      <div
        className="rounded-xl overflow-hidden"
        style={{ border: "1px solid var(--border)" }}
      >
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr style={{ background: "#0f172a", color: "var(--muted)", fontSize: "0.7rem" }}>
              {["Time", "Thread", "Intent", "Confidence", "Action", "Tokens", "Cost", ""].map((h) => (
                <th key={h} className="px-4 py-3 text-left font-semibold tracking-wide uppercase">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={8} className="px-4 py-10 text-center text-xs" style={{ color: "var(--muted)" }}>
                  Loading…
                </td>
              </tr>
            )}
            {!loading && drafts.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-10 text-center text-xs" style={{ color: "var(--muted)" }}>
                  No drafts match the current filters.
                </td>
              </tr>
            )}
            {drafts.map((d, i) => {
              const lastAction = d.human_actions?.[d.human_actions.length - 1];
              return (
                <tr
                  key={d.id}
                  style={{
                    background: i % 2 === 0 ? "var(--card)" : "transparent",
                    borderTop: "1px solid var(--border)",
                  }}
                  className="hover:bg-slate-700/30 transition-colors"
                >
                  <td className="px-4 py-3 text-xs whitespace-nowrap" style={{ color: "var(--muted)" }}>
                    {new Date(d.created_at).toLocaleDateString()}{" "}
                    <span className="opacity-60">{new Date(d.created_at).toLocaleTimeString()}</span>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs" style={{ color: "var(--muted)" }}>
                    {d.thread_id.slice(0, 10)}…
                  </td>
                  <td className="px-4 py-3">
                    <IntentBadge intent={d.intent} />
                  </td>
                  <td className="px-4 py-3">
                    <ConfidenceBadge label={d.confidence_label} />
                  </td>
                  <td className="px-4 py-3">
                    {lastAction ? (
                      <ActionBadge action={lastAction.action} />
                    ) : (
                      <span className="intent-chip bg-slate-700/60 text-slate-400">pending</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs tabular-nums" style={{ color: "var(--muted)" }}>
                    {(d.input_tokens + d.output_tokens).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-xs tabular-nums" style={{ color: "var(--muted)" }}>
                    {fmt(d.cost_usd)}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/drafts/${d.id}`}
                      className="text-xs text-indigo-400 hover:text-indigo-300 whitespace-nowrap"
                    >
                      View →
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex justify-between items-center mt-4 text-xs" style={{ color: "var(--muted)" }}>
        <button
          disabled={page === 0}
          onClick={() => setPage((p) => p - 1)}
          className="px-3 py-1.5 rounded-lg disabled:opacity-30 hover:text-white transition-colors"
          style={{ border: "1px solid var(--border)" }}
        >
          ← Prev
        </button>
        <span>Page {page + 1}</span>
        <button
          disabled={drafts.length < PAGE_SIZE}
          onClick={() => setPage((p) => p + 1)}
          className="px-3 py-1.5 rounded-lg disabled:opacity-30 hover:text-white transition-colors"
          style={{ border: "1px solid var(--border)" }}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
