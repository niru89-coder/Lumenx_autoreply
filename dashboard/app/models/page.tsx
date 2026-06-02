"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Checkpoint, ModelsResponse } from "@/lib/types";

function fmtAuc(v: number | null) {
  return v === null || v === undefined ? "—" : v.toFixed(3);
}

function fmtDate(s: string | null) {
  if (!s) return "—";
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
}

function CheckpointCard({
  ck,
  onPromote,
  promoting,
}: {
  ck: Checkpoint;
  onPromote: (v: number) => void;
  promoting: number | null;
}) {
  const aucOk = (ck.val_auc ?? 0) >= 0.75; // PLAN.md Phase 7 success bar
  return (
    <div
      className="rounded-xl p-5"
      style={{
        background: "var(--card)",
        border: ck.is_active
          ? "1px solid var(--accent)"
          : "1px solid var(--border)",
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-bold text-white">v{ck.version}</h3>
          {ck.is_active ? (
            <span
              className="text-xs font-bold px-2 py-0.5 rounded-full text-white"
              style={{ background: "var(--accent)" }}
            >
              ACTIVE
            </span>
          ) : (
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded-full"
              style={{ border: "1px solid var(--border)", color: "var(--muted)" }}
            >
              candidate
            </span>
          )}
        </div>
        {!ck.is_active && (
          <button
            onClick={() => onPromote(ck.version)}
            disabled={promoting !== null}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg text-white transition-colors disabled:opacity-50"
            style={{ background: "var(--accent)" }}
          >
            {promoting === ck.version ? "Promoting…" : "Promote"}
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <Metric
          label="Val AUC"
          value={fmtAuc(ck.val_auc)}
          good={aucOk}
          bad={ck.val_auc !== null && !aucOk}
        />
        <Metric label="Val BCE" value={fmtAuc(ck.val_bce)} />
        <Metric
          label="Train / Val"
          value={`${ck.n_train ?? "—"} / ${ck.n_val ?? "—"}`}
        />
        <Metric label="Temp (T)" value={fmtAuc(ck.temperature)} />
        <Metric label="Epochs" value={String(ck.n_epochs ?? "—")} />
        <Metric label="Feat. ver" value={String(ck.feature_version ?? "—")} />
        <Metric label="Seed" value={String(ck.train_seed ?? "—")} />
        <Metric label="Created" value={fmtDate(ck.created_at)} small />
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  good,
  bad,
  small,
}: {
  label: string;
  value: string;
  good?: boolean;
  bad?: boolean;
  small?: boolean;
}) {
  const color = good ? "#4ade80" : bad ? "#f87171" : "white";
  return (
    <div className="flex flex-col gap-0.5">
      <span
        className="text-xs uppercase tracking-wide"
        style={{ color: "var(--muted)" }}
      >
        {label}
      </span>
      <span
        className={small ? "text-xs font-medium" : "text-base font-semibold"}
        style={{ color }}
      >
        {value}
      </span>
    </div>
  );
}

export default function ModelsPage() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [promoting, setPromoting] = useState<number | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const load = async () => {
    try {
      setLoading(true);
      const d = await api.listModels();
      setData(d);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handlePromote = async (version: number) => {
    if (
      !window.confirm(
        `Promote v${version} to active? The router will hot-reload and ` +
          `start scoring with this checkpoint immediately.`
      )
    ) {
      return;
    }
    try {
      setPromoting(version);
      await api.promoteModel(version);
      setToast(`Promoted v${version} — router reloaded.`);
      await load();
    } catch (e) {
      setToast(e instanceof Error ? e.message : String(e));
    } finally {
      setPromoting(null);
      setTimeout(() => setToast(null), 5000);
    }
  };

  if (loading) {
    return (
      <div
        className="flex items-center justify-center h-64 text-sm"
        style={{ color: "var(--muted)" }}
      >
        Loading…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div
        className="rounded-lg p-4 text-sm text-red-400 max-w-xl"
        style={{ background: "#1e0a0a", border: "1px solid #7f1d1d" }}
      >
        {error ?? "No models available"}
      </div>
    );
  }

  const { checkpoints, active_version, active_record } = data;

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">Confidence Net Models</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--muted)" }}>
            Review candidate checkpoints and promote when metrics look good
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

      {toast && (
        <div
          className="rounded-lg p-3 text-sm mb-4"
          style={{ background: "var(--card)", border: "1px solid var(--accent)", color: "white" }}
        >
          {toast}
        </div>
      )}

      <div
        className="rounded-xl p-4 mb-6 flex items-center justify-between"
        style={{ background: "var(--card)", border: "1px solid var(--border)" }}
      >
        <div>
          <p className="text-xs uppercase tracking-wide" style={{ color: "var(--muted)" }}>
            Serving now
          </p>
          <p className="text-lg font-bold text-white">
            {active_version !== null ? `v${active_version}` : "latest (no pin set)"}
          </p>
        </div>
        {active_record && (
          <div className="text-right text-xs" style={{ color: "var(--muted)" }}>
            <p>promoted {fmtDate(active_record.promoted_at)}</p>
            <p>by {active_record.promoted_by}</p>
            {active_record.previous_version !== null && (
              <p>was v{active_record.previous_version}</p>
            )}
          </div>
        )}
      </div>

      <div className="flex flex-col gap-4">
        {checkpoints.length === 0 ? (
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            No checkpoints trained yet. Run{" "}
            <code>python -m agent.confidence_net.train</code> or the weekly
            retrain job.
          </p>
        ) : (
          checkpoints.map((ck) => (
            <CheckpointCard
              key={ck.version}
              ck={ck}
              onPromote={handlePromote}
              promoting={promoting}
            />
          ))
        )}
      </div>
    </div>
  );
}
