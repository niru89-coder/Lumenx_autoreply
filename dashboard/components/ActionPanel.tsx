"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

interface Props {
  draftId: string;
  draftText: string;
  /** Called after a successful action so the parent can refresh */
  onDone?: () => void;
}

export default function ActionPanel({ draftId, draftText, onDone }: Props) {
  const router = useRouter();
  const [mode, setMode] = useState<"idle" | "editing" | "loading">("idle");
  const [editText, setEditText] = useState(draftText);
  const [error, setError] = useState<string | null>(null);

  async function submit(action: string, finalText?: string) {
    setMode("loading");
    setError(null);
    try {
      await api.recordAction(draftId, action, finalText ?? null);
      onDone?.();
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMode("idle");
    }
  }

  if (mode === "loading") {
    return (
      <div className="flex items-center gap-2 text-sm" style={{ color: "var(--muted)" }}>
        <span className="animate-spin">⏳</span> Saving…
      </div>
    );
  }

  if (mode === "editing") {
    return (
      <div className="flex flex-col gap-3">
        <textarea
          className="w-full rounded-lg p-3 text-sm font-mono resize-y min-h-[140px]"
          style={{
            background: "#0f172a",
            border: "1px solid var(--border)",
            color: "var(--foreground)",
          }}
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
        />
        <div className="flex gap-2">
          <button
            onClick={() => submit("edited", editText)}
            className="px-4 py-2 rounded-lg text-sm font-semibold bg-blue-600 hover:bg-blue-500 text-white transition-colors"
          >
            Save edit
          </button>
          <button
            onClick={() => setMode("idle")}
            className="px-4 py-2 rounded-lg text-sm font-semibold text-slate-400 hover:text-white transition-colors"
            style={{ border: "1px solid var(--border)" }}
          >
            Cancel
          </button>
        </div>
        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-2 items-center">
      <button
        onClick={() => submit("approved")}
        className="px-4 py-2 rounded-lg text-sm font-semibold bg-green-700 hover:bg-green-600 text-white transition-colors"
      >
        ✓ Approve
      </button>
      <button
        onClick={() => { setEditText(draftText); setMode("editing"); }}
        className="px-4 py-2 rounded-lg text-sm font-semibold bg-blue-700 hover:bg-blue-600 text-white transition-colors"
      >
        ✎ Edit &amp; approve
      </button>
      <button
        onClick={() => submit("rejected")}
        className="px-4 py-2 rounded-lg text-sm font-semibold bg-red-700 hover:bg-red-600 text-white transition-colors"
      >
        ✗ Reject
      </button>
      {error && <p className="text-xs text-red-400 w-full mt-1">{error}</p>}
    </div>
  );
}
