import type { Draft, HumanAction } from "@/lib/types";

/* ── Confidence label chip ─────────────────────────────────────── */
const CONFIDENCE_STYLE: Record<string, string> = {
  high:    "bg-green-900/60 text-green-300",
  low:     "bg-yellow-900/60 text-yellow-300",
  blocked: "bg-red-900/60 text-red-300",
};

export function ConfidenceBadge({ label }: { label: Draft["confidence_label"] }) {
  return (
    <span className={`intent-chip ${CONFIDENCE_STYLE[label] ?? "bg-slate-700 text-slate-300"}`}>
      {label}
    </span>
  );
}

/* ── Human action chip ──────────────────────────────────────────── */
const ACTION_STYLE: Record<string, string> = {
  approved:  "bg-green-900/60 text-green-300",
  edited:    "bg-blue-900/60 text-blue-300",
  rejected:  "bg-red-900/60 text-red-300",
  auto_sent: "bg-indigo-900/60 text-indigo-300",
};

export function ActionBadge({ action }: { action: HumanAction["action"] }) {
  return (
    <span className={`intent-chip ${ACTION_STYLE[action] ?? "bg-slate-700 text-slate-300"}`}>
      {action.replace("_", " ")}
    </span>
  );
}

/* ── Intent chip ────────────────────────────────────────────────── */
const INTENT_COLORS = [
  "bg-purple-900/60 text-purple-300",
  "bg-cyan-900/60 text-cyan-300",
  "bg-orange-900/60 text-orange-300",
  "bg-pink-900/60 text-pink-300",
  "bg-teal-900/60 text-teal-300",
  "bg-rose-900/60 text-rose-300",
  "bg-sky-900/60 text-sky-300",
  "bg-amber-900/60 text-amber-300",
];

function intentColorClass(intent: string) {
  let h = 0;
  for (let i = 0; i < intent.length; i++) h = (h * 31 + intent.charCodeAt(i)) >>> 0;
  return INTENT_COLORS[h % INTENT_COLORS.length];
}

export function IntentBadge({ intent }: { intent: string }) {
  return (
    <span className={`intent-chip ${intentColorClass(intent)}`}>
      {intent.replace(/_/g, " ")}
    </span>
  );
}

/* ── Guardrail indicator ────────────────────────────────────────── */
export function GuardrailBadge({ triggered }: { triggered: boolean }) {
  if (!triggered) return null;
  return (
    <span className="intent-chip bg-red-900/80 text-red-200" title="Guardrail triggered">
      ⚠ guardrail
    </span>
  );
}
