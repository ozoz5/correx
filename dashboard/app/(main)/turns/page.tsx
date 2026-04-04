"use client";

import { useMemo } from "react";
import { useMemory } from "@/lib/useMemory";
import { calculateGuidanceImpact } from "@/lib/maturity";

function scoreBar(score: number) {
  const color = score >= 0.8 ? "var(--color-emerald)" : score >= 0.5 ? "var(--color-amber)" : "#dc2626";
  const label = score >= 0.8 ? "Good" : score >= 0.5 ? "OK" : "Corrected";
  return { color, label };
}

export default function TurnsPage() {
  const { data, loading } = useMemory();
  const impact = useMemo(() => calculateGuidanceImpact(data.turns), [data.turns]);

  if (loading) return (
    <div className="flex items-center gap-2 p-8 text-xs" style={{ color: "var(--text-muted)" }}>
      <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />Loading…
    </div>
  );

  const turns = [...data.turns].reverse().map((t) => ({
    ...t,
    user_feedback: t.user_feedback && /^<[a-z-]+[>\s]/.test(t.user_feedback) ? "" : t.user_feedback,
    user_message: t.user_message && /^<[a-z-]+[>\s]/.test(t.user_message) ? "" : t.user_message,
  }));

  return (
    <div className="p-6 space-y-6">
      {/* Guidance Impact Panel */}
      {(impact.guidedCount > 0 || impact.unguidedCount > 0) && (
        <div className="card-neutral rounded-xl p-5">
          <p className="text-[9px] tracking-[0.2em] uppercase mb-3" style={{ color: "var(--text-label)" }}>GUIDANCE IMPACT</p>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg p-3" style={{ background: "var(--bg-subtle)" }}>
              <p className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Without rules ({impact.unguidedCount})</p>
              <p className="text-2xl font-mono" style={{ color: "var(--text-body)" }}>
                {impact.unguidedCount > 0 ? impact.unguidedAvg.toFixed(2) : "--"}
              </p>
            </div>
            <div className="rounded-lg p-3" style={{ background: "rgba(5,150,105,0.06)", border: "1px solid rgba(5,150,105,0.15)" }}>
              <p className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>With rules ({impact.guidedCount})</p>
              <p className="text-2xl font-mono text-emerald-600">
                {impact.guidedCount > 0 ? impact.guidedAvg.toFixed(2) : "--"}
              </p>
            </div>
          </div>
          {impact.guidedCount > 0 && impact.unguidedCount > 0 && (
            <div className="mt-3 flex items-center gap-2">
              <span className="text-[10px]" style={{ color: impact.delta >= 0 ? "var(--color-emerald)" : "#dc2626" }}>
                {impact.delta >= 0 ? "+" : ""}{impact.delta.toFixed(2)} delta
              </span>
              <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                {impact.delta > 0.05 ? "Rules are working" : impact.delta > -0.05 ? "Minimal difference" : "Needs improvement"}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Score legend */}
      <div className="flex gap-4 text-[10px] px-1">
        <span style={{ color: "var(--color-emerald)" }}>● ≥0.8 = Good</span>
        <span style={{ color: "var(--color-amber)" }}>● 0.5–0.8 = OK</span>
        <span style={{ color: "#dc2626" }}>● &lt;0.5 = Corrected</span>
      </div>

      <div className="space-y-3">
        {turns.map((turn) => {
          const score = turn.reaction_score;
          const bar = score !== null ? scoreBar(score) : null;
          return (
            <div key={turn.id} className="card-neutral rounded-xl p-4" style={{ borderLeft: `3px solid ${turn.guidance_applied ? "rgba(5,150,105,0.4)" : "rgba(217,119,6,0.2)"}` }}>
              <div className="flex items-center gap-3 mb-3">
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>{turn.recorded_at}</span>
                {turn.task_scope && (
                  <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: "var(--bg-subtle)", color: "var(--text-muted)" }}>
                    {turn.task_scope}
                  </span>
                )}
                {turn.guidance_applied && (
                  <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: "rgba(5,150,105,0.08)", color: "var(--color-emerald)" }}>
                    Rules applied
                  </span>
                )}
                {bar && (
                  <span className="ml-auto text-[10px]" style={{ color: bar.color }}>
                    {bar.label} ({score!.toFixed(2)})
                  </span>
                )}
              </div>

              {turn.user_feedback && (
                <p className="text-xs mb-3 leading-relaxed line-clamp-3" style={{ color: "var(--text-primary)" }}>{turn.user_feedback}</p>
              )}

              {turn.extracted_corrections.length > 0 && (
                <div>
                  <p className="text-[9px] mb-2 tracking-wide" style={{ color: "var(--text-muted)" }}>Extracted corrections</p>
                  <div className="flex flex-wrap gap-1.5">
                    {turn.extracted_corrections.map((c, i) => (
                      <span key={i} className="text-[10px] px-2 py-1 rounded" style={{ background: "rgba(217,119,6,0.06)", color: "var(--color-amber)", border: "1px solid rgba(217,119,6,0.15)" }}>
                        {c}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {turns.length === 0 && (
          <p className="text-xs py-8 text-center" style={{ color: "var(--text-label)" }}>No conversation turns yet</p>
        )}
      </div>
    </div>
  );
}
