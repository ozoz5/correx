"use client";

import { useMemory } from "@/lib/useMemory";

interface Policy {
  id: string;
  title: string;
  core: string;
  why: string;
  analogy: string;
  opposite: string;
  limits: string;
  maturity: string;
  evidence_count: number;
}

interface Law {
  law?: string;
  principle?: string;
  evidence_count?: number;
}

export default function DoctrinePage() {
  const { data, loading } = useMemory();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="w-2 h-2 rounded-full animate-ping" style={{ background: "var(--cyan)" }} />
      </div>
    );
  }

  const policies = ((data as any).policies || []) as Policy[];
  const universalLaws = ((data as any).universalLaws || []) as Law[];
  const positiveLaws = ((data as any).positiveLaws || []) as Law[];
  const activePolicies = policies.filter((p) => p.maturity === "active");

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-lg font-bold tracking-widest" style={{ color: "var(--text-primary)" }}>
          DOCTRINE
        </h1>
        <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          Policies, prohibition laws, and recommendation laws — the core judgment system
        </p>
      </div>

      {/* Active Policies */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full" style={{ background: "var(--color-emerald)" }} />
          <h2 className="text-sm font-bold tracking-widest" style={{ color: "var(--color-emerald)" }}>
            POLICIES ({activePolicies.length})
          </h2>
        </div>
        {activePolicies.length === 0 ? (
          <p className="text-xs pl-5" style={{ color: "var(--text-faint)" }}>No active policies yet</p>
        ) : (
          activePolicies.map((p) => (
            <div
              key={p.id}
              className="rounded-xl p-5 space-y-3"
              style={{ background: "var(--bg-subtle)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                  {p.title}
                </h3>
                <span
                  className="text-[9px] px-2 py-0.5 rounded-full font-medium"
                  style={{ background: "rgba(5,150,105,0.15)", color: "var(--color-emerald)" }}
                >
                  {p.evidence_count} corrections
                </span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                {p.core}
              </p>
              <div className="grid grid-cols-2 gap-3 pt-2" style={{ borderTop: "1px solid var(--border)" }}>
                {p.why && (
                  <div>
                    <p className="text-[9px] tracking-widest mb-1" style={{ color: "var(--text-label)" }}>WHY</p>
                    <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-muted)" }}>{p.why}</p>
                  </div>
                )}
                {p.analogy && (
                  <div>
                    <p className="text-[9px] tracking-widest mb-1" style={{ color: "var(--text-label)" }}>ANALOGY</p>
                    <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-muted)" }}>{p.analogy}</p>
                  </div>
                )}
                {p.opposite && (
                  <div>
                    <p className="text-[9px] tracking-widest mb-1" style={{ color: "var(--amber)" }}>OPPOSITE</p>
                    <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-muted)" }}>{p.opposite}</p>
                  </div>
                )}
                {p.limits && (
                  <div>
                    <p className="text-[9px] tracking-widest mb-1" style={{ color: "var(--text-label)" }}>LIMITS</p>
                    <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-muted)" }}>{p.limits}</p>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </section>

      {/* Prohibition Laws */}
      <section className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full" style={{ background: "var(--color-red, #ef4444)" }} />
          <h2 className="text-sm font-bold tracking-widest" style={{ color: "var(--color-red, #ef4444)" }}>
            PROHIBITIONS ({universalLaws.length})
          </h2>
        </div>
        <div className="space-y-1.5">
          {universalLaws.map((law, i) => {
            const text = law.law || law.principle || JSON.stringify(law);
            return (
              <div
                key={i}
                className="flex items-start gap-3 px-4 py-2.5 rounded-lg"
                style={{ background: "var(--bg-subtle)" }}
              >
                <span className="text-[10px] font-mono font-bold mt-0.5 shrink-0" style={{ color: "var(--color-red, #ef4444)" }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{text}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Recommendation Laws */}
      <section className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full" style={{ background: "var(--cyan)" }} />
          <h2 className="text-sm font-bold tracking-widest" style={{ color: "var(--cyan)" }}>
            RECOMMENDATIONS ({positiveLaws.length})
          </h2>
        </div>
        <div className="space-y-1.5">
          {positiveLaws.map((law, i) => {
            const text = law.law || law.principle || JSON.stringify(law);
            return (
              <div
                key={i}
                className="flex items-start gap-3 px-4 py-2.5 rounded-lg"
                style={{ background: "var(--bg-subtle)" }}
              >
                <span className="text-[10px] font-mono font-bold mt-0.5 shrink-0" style={{ color: "var(--cyan)" }}>
                  +{i + 1}
                </span>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{text}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Proposed Policies */}
      {policies.filter((p) => p.maturity === "proposed").length > 0 && (
        <section className="space-y-3 opacity-60">
          <h2 className="text-sm font-bold tracking-widest" style={{ color: "var(--text-label)" }}>
            PROPOSED ({policies.filter((p) => p.maturity === "proposed").length})
          </h2>
          {policies
            .filter((p) => p.maturity === "proposed")
            .map((p) => (
              <div
                key={p.id}
                className="rounded-lg p-4"
                style={{ background: "var(--bg-subtle)", border: "1px dashed var(--border)" }}
              >
                <p className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>{p.title}</p>
                <p className="text-[11px] mt-1" style={{ color: "var(--text-muted)" }}>{p.core}</p>
              </div>
            ))}
        </section>
      )}
    </div>
  );
}
