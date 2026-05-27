"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";

const NAV = [
  { href: "/inbox",   label: "Inbox",   icon: "📥" },
  { href: "/history", label: "History", icon: "📋" },
  { href: "/costs",   label: "Costs",   icon: "💰" },
];

export default function Sidebar() {
  const path = usePathname();
  const [pending, setPending] = useState<number | null>(null);
  const [agentOk, setAgentOk] = useState<boolean | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const h = await api.health();
        setPending(h.pending_review);
        setAgentOk(h.status === "ok");
      } catch {
        setAgentOk(false);
      }
    };
    load();
    const t = setInterval(load, 10_000);
    return () => clearInterval(t);
  }, []);

  return (
    <aside
      className="flex flex-col w-56 shrink-0 h-screen sticky top-0 overflow-y-auto"
      style={{ background: "var(--sidebar)", borderRight: "1px solid var(--border)" }}
    >
      {/* Logo */}
      <div className="px-5 py-5 border-b" style={{ borderColor: "var(--border)" }}>
        <p className="text-xs font-semibold tracking-widest uppercase"
           style={{ color: "var(--muted)" }}>
          LumenX
        </p>
        <p className="text-base font-bold text-white mt-0.5">Agent Dashboard</p>
      </div>

      {/* Nav links */}
      <nav className="flex flex-col gap-1 px-3 py-4 flex-1">
        {NAV.map(({ href, label, icon }) => {
          const active = path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                active
                  ? "text-white"
                  : "text-slate-400 hover:text-white hover:bg-slate-700/50"
              }`}
              style={active ? { background: "var(--accent)" } : {}}
            >
              <span>{icon}</span>
              <span>{label}</span>
              {href === "/inbox" && pending !== null && pending > 0 && (
                <span className="ml-auto bg-red-500 text-white text-xs font-bold px-1.5 py-0.5 rounded-full">
                  {pending}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Agent status */}
      <div className="px-4 py-4 border-t" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              agentOk === null
                ? "bg-slate-500"
                : agentOk
                ? "bg-green-500"
                : "bg-red-500"
            }`}
          />
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {agentOk === null
              ? "Connecting…"
              : agentOk
              ? "Agent online"
              : "Agent offline"}
          </span>
        </div>
        <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
          :8000
        </p>
      </div>
    </aside>
  );
}
