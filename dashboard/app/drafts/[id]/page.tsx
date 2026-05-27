import { api } from "@/lib/api";
import type { Draft } from "@/lib/types";
import {
  ConfidenceBadge,
  IntentBadge,
  ActionBadge,
  GuardrailBadge,
} from "@/components/StatusBadge";
import ActionPanel from "@/components/ActionPanel";

/* ── Helper sub-components ──────────────────────────────────────── */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section
      className="rounded-xl p-5"
      style={{ background: "var(--card)", border: "1px solid var(--border)" }}
    >
      <h2 className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "var(--muted)" }}>
        {title}
      </h2>
      {children}
    </section>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 text-sm py-1.5"
      style={{ borderBottom: "1px solid var(--border)" }}>
      <span style={{ color: "var(--muted)" }}>{k}</span>
      <span className="text-right text-white font-mono text-xs">{v}</span>
    </div>
  );
}

/* ── Server component — fetches draft on the server ─────────────── */
export default async function DraftDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  // Next.js 16: params is async
  const { id } = await params;

  let draft: Draft;
  let ctxError: string | null = null;
  let ctx: Record<string, unknown> | null = null;

  try {
    draft = await api.getDraft(id);
  } catch (e) {
    return (
      <div className="max-w-3xl mx-auto">
        <div
          className="rounded-lg p-4 text-sm text-red-400"
          style={{ background: "#1e0a0a", border: "1px solid #7f1d1d" }}
        >
          Failed to load draft {id}: {e instanceof Error ? e.message : String(e)}
        </div>
      </div>
    );
  }

  try {
    ctx = await api.getContext(id);
  } catch (e) {
    ctxError = e instanceof Error ? e.message : String(e);
  }

  const lastAction = draft.human_actions?.[draft.human_actions.length - 1];
  const prediction = draft.confidence_predictions?.[draft.confidence_predictions.length - 1];

  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-5">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
        <a href="/history" className="hover:text-white transition-colors">History</a>
        <span>/</span>
        <span className="font-mono">{id.slice(0, 16)}…</span>
      </div>

      {/* Title row */}
      <div className="flex flex-wrap items-center gap-3">
        <IntentBadge intent={draft.intent} />
        <ConfidenceBadge label={draft.confidence_label} />
        <GuardrailBadge triggered={draft.guardrail_triggered} />
        {lastAction && <ActionBadge action={lastAction.action} />}
        <span className="ml-auto text-xs" style={{ color: "var(--muted)" }}>
          {new Date(draft.created_at).toLocaleString()}
        </span>
      </div>

      {/* Draft text */}
      <Section title="Draft reply">
        <pre className="whitespace-pre-wrap text-sm font-sans leading-relaxed text-white">
          {draft.draft_text}
        </pre>
      </Section>

      {/* Action panel (only if no action taken yet) */}
      {!lastAction && (
        <Section title="Review">
          <ActionPanel draftId={draft.id} draftText={draft.draft_text} />
        </Section>
      )}

      {/* Metadata */}
      <Section title="Metadata">
        <KV k="Thread ID"       v={draft.thread_id} />
        <KV k="Sensitivity"     v={draft.sensitivity} />
        <KV k="Model"           v={draft.model} />
        <KV k="Input tokens"    v={draft.input_tokens.toLocaleString()} />
        <KV k="Output tokens"   v={draft.output_tokens.toLocaleString()} />
        <KV k="Cost (USD)"      v={`$${draft.cost_usd.toFixed(6)}`} />
        <KV k="Latency"         v={`${draft.latency_ms} ms`} />
        <KV k="Parse attempts"  v={String(draft.parse_attempts)} />
        <KV k="Auto-sendable"   v={draft.auto_sendable ? "yes" : "no"} />
        <KV k="Guardrail"       v={draft.guardrail_triggered ? "⚠ triggered" : "clear"} />
        <KV k="Cache key"       v={draft.context_cache_key || "—"} />
      </Section>

      {/* Cited sources */}
      {draft.cited_sources.length > 0 && (
        <Section title="Cited sources">
          <ul className="flex flex-col gap-1">
            {draft.cited_sources.map((s, i) => (
              <li key={i} className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                {s}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Uncertainty flags */}
      {draft.uncertainty_flags.length > 0 && (
        <Section title="Uncertainty flags">
          <ul className="flex flex-col gap-1">
            {draft.uncertainty_flags.map((f, i) => (
              <li key={i} className="text-xs text-yellow-400">{f}</li>
            ))}
          </ul>
        </Section>
      )}

      {/* Confidence prediction */}
      {prediction && (
        <Section title="Confidence net prediction">
          <KV k="Score"           v={prediction.score.toFixed(4)} />
          <KV k="Threshold"       v={prediction.threshold.toFixed(2)} />
          <KV k="Would auto-send" v={prediction.would_auto_send ? "yes" : "no"} />
          <KV k="Model version"   v={prediction.model_version ?? "—"} />
          <KV k="Evaluated at"    v={new Date(prediction.created_at).toLocaleString()} />
          <details className="mt-3">
            <summary className="text-xs cursor-pointer" style={{ color: "var(--muted)" }}>
              Feature vector ({Object.keys(prediction.features).length} dims)
            </summary>
            <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-0.5">
              {Object.entries(prediction.features).map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs py-0.5"
                  style={{ borderBottom: "1px solid var(--border)" }}>
                  <span style={{ color: "var(--muted)" }}>{k}</span>
                  <span className="font-mono text-white">{v.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </details>
        </Section>
      )}

      {/* Human actions history */}
      {draft.human_actions && draft.human_actions.length > 0 && (
        <Section title="Action history">
          <div className="flex flex-col gap-2">
            {draft.human_actions.map((a) => (
              <div
                key={a.id}
                className="flex items-start gap-3 text-xs"
                style={{ borderBottom: "1px solid var(--border)", paddingBottom: "8px" }}
              >
                <ActionBadge action={a.action} />
                <div className="flex flex-col gap-0.5 flex-1">
                  <span style={{ color: "var(--muted)" }}>
                    by {a.reviewer} · {new Date(a.decided_at).toLocaleString()}
                  </span>
                  {a.final_text && (
                    <pre className="whitespace-pre-wrap font-sans text-xs text-white mt-1 leading-relaxed">
                      {a.final_text}
                    </pre>
                  )}
                  {a.edit_distance !== null && (
                    <span style={{ color: "var(--muted)" }}>
                      Edit distance: {a.edit_distance}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Context window snapshot */}
      {ctx && !ctxError && (
        <Section title="Context window snapshot">
          <details>
            <summary className="text-xs cursor-pointer" style={{ color: "var(--muted)" }}>
              Expand full context
            </summary>
            <pre className="mt-3 text-xs font-mono whitespace-pre-wrap overflow-auto max-h-96"
              style={{ color: "var(--muted)" }}>
              {JSON.stringify(ctx, null, 2)}
            </pre>
          </details>
        </Section>
      )}
      {ctxError && (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          Context snapshot unavailable: {ctxError}
        </p>
      )}
    </div>
  );
}
