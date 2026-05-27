"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { FeedbackStats } from "@/lib/types";
import CostChart from "@/components/CostChart";

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div
      className="rounded-xl p-5 flex flex-col gap-1"
      style={{ background: "var(--card)", border: "1px solid var(--border)" }}
    >
      <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
        {label}
      </p>
      <p className="text-2xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs" style={{ color: "var(--muted)" }}>{sub}</p>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="text-sm font-semibold mb-3 text-white tracking-wide">{title}</h2>
      {children}
    </section>
  );
}

function recordToChartData(rec: Record<string, number>) {
  return Object.entries(rec)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }));
}

function fmtK(n: number) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

export default function CostsPage() {
  const [stats, setStats]     = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const load = async () => {
    try {
      setLoading(true);
      const s = await api.stats();
      setStats(s);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-sm" style={{ color: "var(--muted)" }}>
        Loading…
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div
        className="rounded-lg p-4 text-sm text-red-400 max-w-xl"
        style={{ background: "#1e0a0a", border: "1px solid #7f1d1d" }}
      >
        {error ?? "No stats available"}
      </div>
    );
  }

  const totalTok = stats.total_input_tokens + stats.total_output_tokens;
  const autoSentPct = stats.total_actions
    ? ((stats.action_breakdown["auto_sent"] ?? 0) / stats.total_actions * 100).toFixed(1)
    : "0";
  const avgCost = stats.total_drafts
    ? (stats.total_cost_usd / stats.total_drafts * 100).toFixed(3)
    : "0";

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">Costs &amp; Analytics</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>
            Aggregate spend and action breakdown
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

      {/* KPI row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total spend"
          value={`$${stats.total_cost_usd.toFixed(4)}`}
          sub={`${stats.total_drafts} drafts total`}
        />
        <StatCard
          label="Avg cost / draft"
          value={`¢${avgCost}`}
          sub="cents per reply"
        />
        <StatCard
          label="Total tokens"
          value={fmtK(totalTok)}
          sub={`in: ${fmtK(stats.total_input_tokens)} / out: ${fmtK(stats.total_output_tokens)}`}
        />
        <StatCard
          label="Auto-send rate"
          value={`${autoSentPct}%`}
          sub={`of ${stats.total_actions} actions`}
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
        {/* Action breakdown */}
        <Section title="Actions">
          <div
            className="rounded-xl p-4"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            <CostChart
              data={recordToChartData(stats.action_breakdown)}
              color="#6366f1"
            />
          </div>
        </Section>

        {/* Intent breakdown */}
        <Section title="Intents">
          <div
            className="rounded-xl p-4"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            <CostChart
              data={recordToChartData(stats.intent_breakdown)}
              color="#06b6d4"
            />
          </div>
        </Section>

        {/* Confidence breakdown */}
        <Section title="Confidence labels">
          <div
            className="rounded-xl p-4"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            <CostChart
              data={recordToChartData(stats.confidence_breakdown)}
              color="#f59e0b"
            />
          </div>
        </Section>

        {/* Extra KPIs */}
        <Section title="Other">
          <div className="flex flex-col gap-3">
            <div
              className="rounded-xl p-4 flex items-center justify-between"
              style={{ background: "var(--card)", border: "1px solid var(--border)" }}
            >
              <p className="text-sm" style={{ color: "var(--muted)" }}>Pending review</p>
              <p className="text-xl font-bold text-white">{stats.pending_review}</p>
            </div>
            <div
              className="rounded-xl p-4 flex items-center justify-between"
              style={{ background: "var(--card)", border: "1px solid var(--border)" }}
            >
              <p className="text-sm" style={{ color: "var(--muted)" }}>Guardrail hits</p>
              <p className="text-xl font-bold text-red-400">{stats.guardrail_hits}</p>
            </div>
            <div
              className="rounded-xl p-4 flex items-center justify-between"
              style={{ background: "var(--card)", border: "1px solid var(--border)" }}
            >
              <p className="text-sm" style={{ color: "var(--muted)" }}>Total actions logged</p>
              <p className="text-xl font-bold text-white">{stats.total_actions}</p>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
