"use client";

import { useState } from "react";
import { useMemory } from "@/lib/useMemory";
import type { GrowthRecord } from "@/lib/types";

function OutputComparison({ record }: { record: GrowthRecord }) {
  const baseline = record.baseline_run;
  const guided = record.guided_run;
  if (!baseline && !guided) return null;

  return (
    <div className="mt-4 pt-4 space-y-3" style={{ borderTop: "1px solid var(--border-divider)" }}>
      {guided?.guidance_text && (
        <div>
          <p className="text-[9px] tracking-[0.15em] uppercase mb-1" style={{ color: "var(--color-cyan)" }}>GUIDANCE APPLIED</p>
          <p className="text-[10px] p-2 rounded" style={{ background: "rgba(8,145,178,0.06)", color: "var(--text-body)", border: "1px solid rgba(8,145,178,0.15)" }}>
            {guided.guidance_text.slice(0, 300)}{guided.guidance_text.length > 300 ? "…" : ""}
          </p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-[9px] tracking-[0.15em] uppercase mb-1" style={{ color: "var(--text-label)" }}>BASELINE OUTPUT</p>
          <pre className="text-[10px] p-2 rounded overflow-auto whitespace-pre-wrap" style={{ background: "var(--bg-subtle)", color: "var(--text-muted)", maxHeight: "120px", border: "1px solid var(--border)" }}>
            {baseline?.output?.slice(0, 500) || "—"}
          </pre>
        </div>
        <div>
          <p className="text-[9px] tracking-[0.15em] uppercase mb-1" style={{ color: "var(--color-emerald)" }}>GUIDED OUTPUT</p>
          <pre className="text-[10px] p-2 rounded overflow-auto whitespace-pre-wrap" style={{ background: "rgba(5,150,105,0.04)", color: "var(--text-body)", maxHeight: "120px", border: "1px solid rgba(5,150,105,0.15)" }}>
            {guided?.output?.slice(0, 500) || "—"}
          </pre>
        </div>
      </div>
    </div>
  );
}

function GrowthCase({ caseId, records }: { caseId: string; records: GrowthRecord[] }) {
  const [expanded, setExpanded] = useState(false);
  const latest = records[records.length - 1];
  const delta = latest.delta ?? 0;
  const improving = records.length >= 2 && delta > (records[0].delta ?? 0);
  const deltaColor = delta > 0.05 ? "var(--color-emerald)" : delta > -0.05 ? "var(--color-amber)" : "#dc2626";
  
  const bScore = latest.baseline_score ?? 0;
  const gScore = latest.guided_score ?? 0;
  const maxScore = Math.max(0.1, bScore, gScore);
  const bWidth = Math.min(100, (bScore / maxScore) * 100);
  const gWidth = Math.min(100, (gScore / maxScore) * 100);

  return (
    <div className="card-neutral rounded-xl p-5 cursor-pointer" onClick={() => setExpanded(!expanded)}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-sm" style={{ color: "var(--text-primary)" }}>{latest.case_title || caseId || "Untitled"}</p>
            <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>{expanded ? "▲" : "▼"}</span>
          </div>
          <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>{records.length} measurements</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold font-mono" style={{ color: deltaColor }}>
            {delta >= 0 ? "+" : ""}{delta.toFixed(3)}
          </p>
          <p className="text-[9px]" style={{ color: "var(--text-muted)" }}>Improvement</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg p-3" style={{ background: "var(--bg-subtle)" }}>
          <p className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>Without rules (baseline)</p>
          <p className="text-lg font-mono" style={{ color: "var(--text-body)" }}>{(latest.baseline_score ?? 0).toFixed(2)}</p>
        </div>
        <div className="rounded-lg p-3" style={{ background: "rgba(5,150,105,0.06)", border: "1px solid rgba(5,150,105,0.15)" }}>
          <p className="text-[9px] mb-1" style={{ color: "var(--text-muted)" }}>With rules (guided)</p>
          <p className="text-lg font-mono text-emerald-600">{(latest.guided_score ?? 0).toFixed(2)}</p>
        </div>
      </div>

      {/* Comparison chart */}
      <div className="mb-6 space-y-2">
        <div className="flex items-center w-full">
           <div className="h-2 rounded-r-full bg-slate-400 dark:bg-slate-500 transition-all duration-700" style={{ width: `${bWidth}%` }} />
        </div>
        <div className="flex items-center w-full">
           <div className="h-2 rounded-r-full bg-emerald-500 dark:bg-emerald-400 transition-all duration-700" style={{ width: `${gWidth}%` }} />
        </div>
      </div>

      {records.length > 1 && (
        <div>
          <p className="text-[9px] mb-2" style={{ color: "var(--text-label)" }}>Improvement trend</p>
          <div className="flex gap-1 items-end h-20">
            {records.map((r, i) => {
              const d = r.delta ?? 0;
              const height = Math.max(4, Math.min(80, (d + 0.5) * 80));
              const bg = d > 0.05 ? "var(--color-emerald)" : d > -0.05 ? "var(--color-amber)" : "#dc2626";
              return (
                <div key={i} className="flex-1 rounded-sm opacity-60" style={{ height, background: bg }} title={`${d >= 0 ? "+" : ""}${d.toFixed(3)}`} />
              );
            })}
          </div>
          {improving && <p className="text-[9px] text-emerald-600 mt-2">↑ Improving</p>}
        </div>
      )}

      {/* Drilldown: measurement history with output comparison */}
      {expanded && (
        <div className="mt-4 space-y-3" onClick={(e) => e.stopPropagation()}>
          <p className="text-[9px] tracking-[0.15em] uppercase" style={{ color: "var(--text-label)" }}>MEASUREMENT HISTORY</p>
          {records.map((r, i) => {
            const d = r.delta ?? 0;
            const color = d > 0.05 ? "var(--color-emerald)" : d > -0.05 ? "var(--color-amber)" : "#dc2626";
            return (
              <div key={r.record_id || i} className="rounded-lg p-3" style={{ background: "var(--bg-subtle)", border: "1px solid var(--border)" }}>
                <div className="flex items-center justify-between">
                  <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                    #{i + 1} — {r.recorded_at?.slice(0, 16) || "—"}
                  </span>
                  <div className="flex items-center gap-3 text-[10px] font-mono">
                    <span style={{ color: "var(--text-label)" }}>{(r.baseline_score ?? 0).toFixed(2)}</span>
                    <span>→</span>
                    <span style={{ color: "var(--color-emerald)" }}>{(r.guided_score ?? 0).toFixed(2)}</span>
                    <span style={{ color }}>{d >= 0 ? "+" : ""}{d.toFixed(3)}</span>
                  </div>
                </div>
                <OutputComparison record={r} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function GrowthPage() {
  const { data, loading } = useMemory();

  if (loading) return (
    <div className="flex items-center gap-2 p-8 text-xs" style={{ color: "var(--text-muted)" }}>
      <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />Loading…
    </div>
  );

  const grouped: Record<string, GrowthRecord[]> = {};
  for (const r of data.growth) {
    (grouped[r.case_id] ||= []).push(r);
  }

  return (
    <div className="p-6 space-y-6">
      {/* Overall Trend Indicator */}
      {data.growth.length > 0 && (() => {
        const valid = data.growth.filter((g) => g.delta != null && !isNaN(g.delta));
        const avgD = valid.length > 0 ? valid.reduce((s, g) => s + g.delta, 0) / valid.length : 0;
        const trendIcon = avgD > 0.05 ? "\u2191" : avgD > -0.05 ? "\u2192" : "\u2193";
        const trendLabel = avgD > 0.05 ? "Growing" : avgD > -0.05 ? "Stable" : "Watch";
        const trendColor = avgD > 0.05 ? "var(--color-emerald)" : avgD > -0.05 ? "var(--color-amber)" : "#dc2626";
        return (
          <div className="card-neutral rounded-xl p-5 flex items-center gap-5">
            <span className="text-4xl font-bold" style={{ color: trendColor }}>{trendIcon}</span>
            <div>
              <p className="text-lg font-bold" style={{ color: trendColor }}>
                {avgD >= 0 ? "+" : ""}{Math.round(avgD * 100)}% {trendLabel}
              </p>
              <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{data.growth.length}{" measurements, avg improvement"}</p>
            </div>
          </div>
        );
      })()}

      {Object.keys(grouped).length === 0 ? (
        <p className="text-xs py-8 text-center" style={{ color: "var(--text-label)" }}>No growth records yet</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Object.entries(grouped).map(([caseId, records]) => (
            <GrowthCase key={caseId} caseId={caseId} records={records} />
          ))}
        </div>
      )}
    </div>
  );
}
