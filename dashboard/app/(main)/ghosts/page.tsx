"use client";

import { useMemo } from "react";
import { useMemory } from "@/lib/useMemory";
import type { Ghost, GhostTrajectory } from "@/lib/types";

const originConfig: Record<string, { label: string; color: string; weight: number }> = {
  scolded:   { label: "SCOLDED",   color: "var(--color-amber)",   weight: 2.0 },
  corrected: { label: "CORRECTED", color: "var(--color-emerald)", weight: 1.5 },
  rejected:  { label: "REJECTED",  color: "var(--text-label)",    weight: 1.0 },
};

function TrajectoryCard({ trajectory, ghosts }: { trajectory: GhostTrajectory; ghosts: Ghost[] }) {
  const fired = trajectory.fired;
  const relatedGhosts = ghosts.filter((g) => g.trajectory_id === trajectory.id);
  const pePercent = Math.min(100, ((trajectory.cumulative_pe ?? 0) / 1.5) * 100);

  return (
    <div
      className="card-neutral rounded-xl p-5 relative overflow-hidden"
      style={fired ? { borderLeft: "3px solid var(--color-amber)" } : undefined}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm">{fired ? "\u{1F525}" : "\u{1F47B}"}</span>
          <p className="text-[11px] font-medium" style={{ color: "var(--text-primary)" }}>
            {trajectory.theme || trajectory.id}
          </p>
        </div>
        <span
          className="text-[8px] px-2 py-0.5 rounded-full font-medium tracking-wider"
          style={
            fired
              ? { background: "rgba(245,158,11,0.15)", color: "var(--color-amber)" }
              : { background: "var(--bg-subtle)", color: "var(--text-label)" }
          }
        >
          {fired ? "FIRED" : "OPEN"}
        </span>
      </div>

      {/* Sublimated Principle */}
      {fired && trajectory.sublimated_principle && (
        <div className="mb-3 p-3 rounded-lg" style={{ background: "rgba(245,158,11,0.08)" }}>
          <p className="text-[9px] uppercase tracking-widest mb-1" style={{ color: "var(--color-amber)" }}>
            Sublimated Principle
          </p>
          <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-body)" }}>
            {trajectory.sublimated_principle}
          </p>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="text-center">
          <p className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
            {trajectory.source_ghost_count}
          </p>
          <p className="text-[8px]" style={{ color: "var(--text-label)" }}>Ghosts</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold" style={{ color: pePercent >= 100 ? "var(--color-amber)" : "var(--text-primary)" }}>
            {(trajectory.cumulative_pe ?? 0).toFixed(2)}
          </p>
          <p className="text-[8px]" style={{ color: "var(--text-label)" }}>Cumulative PE</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
            {trajectory.fired_at ? new Date(trajectory.fired_at).toLocaleDateString("ja-JP", { month: "short", day: "numeric" }) : "--"}
          </p>
          <p className="text-[8px]" style={{ color: "var(--text-label)" }}>Fired At</p>
        </div>
      </div>

      {/* PE Progress Bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <p className="text-[8px]" style={{ color: "var(--text-label)" }}>Prediction Error</p>
          <p className="text-[8px]" style={{ color: pePercent >= 100 ? "var(--color-amber)" : "var(--text-label)" }}>
            {pePercent.toFixed(0)}%
          </p>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-bar)" }}>
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${pePercent}%`,
              background: fired
                ? "linear-gradient(to right, var(--amber), #fbbf24)"
                : "linear-gradient(to right, var(--text-faint), var(--text-label))",
            }}
          />
        </div>
      </div>

      {/* Ghost list */}
      {relatedGhosts.length > 0 && (
        <div>
          <p className="text-[8px] uppercase tracking-widest mb-2" style={{ color: "var(--text-label)" }}>
            Rejected Proposals
          </p>
          {relatedGhosts.slice(0, 5).map((ghost) => {
            const cfg = originConfig[ghost.origin] || originConfig.rejected;
            return (
              <div
                key={ghost.id}
                className="py-2"
                style={{ borderBottom: "1px solid var(--border-divider)" }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="text-[7px] px-1.5 py-0.5 rounded-full font-medium"
                    style={{ background: "var(--bg-subtle)", color: cfg.color }}
                  >
                    {cfg.label} x{cfg.weight}
                  </span>
                  <span className="text-[8px]" style={{ color: "var(--text-faint)" }}>
                    PE: {ghost.prediction_error.toFixed(2)}
                  </span>
                  {ghost.task_scope && (
                    <span className="text-[8px]" style={{ color: "var(--text-label)" }}>
                      {ghost.task_scope}
                    </span>
                  )}
                </div>
                <p className="text-[10px] leading-relaxed" style={{ color: "var(--text-body)" }}>
                  {ghost.rejected_output.length > 120
                    ? ghost.rejected_output.slice(0, 120) + "..."
                    : ghost.rejected_output}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function GhostsPage() {
  const { data, loading } = useMemory();
  const ghosts: Ghost[] = data.ghosts ?? [];
  const trajectories: GhostTrajectory[] = data.ghostTrajectories ?? [];

  const { fired, open } = useMemo(() => {
    const fired = trajectories.filter((t) => t.fired);
    const open = trajectories.filter((t) => !t.fired);
    return { fired, open };
  }, [trajectories]);

  const originCounts = useMemo(() => {
    const counts: Record<string, number> = { scolded: 0, corrected: 0, rejected: 0 };
    for (const g of ghosts) counts[g.origin] = (counts[g.origin] || 0) + 1;
    return counts;
  }, [ghosts]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 gap-3">
        <div className="w-2 h-2 rounded-full animate-ping" style={{ background: "var(--amber)" }} />
        <span className="text-xs tracking-widest" style={{ color: "var(--text-label)" }}>LOADING...</span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="card-neutral rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-2xl">{"\u{1F47B}"}</span>
          <div>
            <h1 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>Ghost Engine</h1>
            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
              Rejected proposals accumulate into trajectories. When prediction error exceeds the threshold, principles are sublimated autonomously.
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          <div className="text-center py-3 rounded-lg" style={{ background: "var(--bg-subtle)" }}>
            <p className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>{ghosts.length}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Total Ghosts</p>
          </div>
          <div className="text-center py-3 rounded-lg" style={{ background: "var(--bg-subtle)" }}>
            <p className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>{trajectories.length}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Trajectories</p>
          </div>
          <div className="text-center py-3 rounded-lg" style={{ background: "rgba(245,158,11,0.08)" }}>
            <p className="text-2xl font-bold glow-amber">{fired.length}</p>
            <p className="text-[9px]" style={{ color: "var(--color-amber)" }}>Fired</p>
          </div>
          <div className="text-center py-3 rounded-lg" style={{ background: "var(--bg-subtle)" }}>
            <p className="text-2xl font-bold" style={{ color: "var(--color-amber)" }}>{originCounts.scolded}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Scolded (x2.0)</p>
          </div>
          <div className="text-center py-3 rounded-lg" style={{ background: "var(--bg-subtle)" }}>
            <p className="text-2xl font-bold" style={{ color: "var(--color-emerald)" }}>{originCounts.corrected}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Corrected (x1.5)</p>
          </div>
        </div>
      </div>

      {/* Empty state */}
      {ghosts.length === 0 && trajectories.length === 0 && (
        <div className="card-neutral rounded-xl p-8 text-center">
          <span className="text-4xl block mb-3">{"\u{1F47B}"}</span>
          <p className="text-[11px] font-medium mb-2" style={{ color: "var(--text-body)" }}>
            No ghosts yet
          </p>
          <p className="text-[10px] max-w-md mx-auto" style={{ color: "var(--text-muted)" }}>
            Ghosts appear when AI proposals are rejected or corrected. Use <code className="text-[9px] px-1 py-0.5 rounded" style={{ background: "var(--bg-subtle)" }}>ghost_option</code> in <code className="text-[9px] px-1 py-0.5 rounded" style={{ background: "var(--bg-subtle)" }}>save_conversation_turn</code> to record rejected proposals.
          </p>
          <div className="mt-4 p-3 rounded-lg text-left max-w-sm mx-auto" style={{ background: "var(--bg-subtle)" }}>
            <p className="text-[8px] uppercase tracking-widest mb-2" style={{ color: "var(--text-label)" }}>How it works</p>
            <div className="space-y-1.5">
              {[
                { icon: "\u{1F47B}", text: "Rejected proposal \u2192 Ghost" },
                { icon: "\u{1F300}", text: "Same-theme ghosts \u2192 Trajectory" },
                { icon: "\u26A1", text: "Cumulative PE \u2265 threshold \u2192 Firing" },
                { icon: "\u{1F525}", text: "Fired trajectory \u2192 Sublimated Principle" },
              ].map((step, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs">{step.icon}</span>
                  <span className="text-[9px]" style={{ color: "var(--text-body)" }}>{step.text}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Fired trajectories */}
      {fired.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm">{"\u{1F525}"}</span>
            <p className="text-[9px] tracking-[0.2em] uppercase" style={{ color: "var(--color-amber)" }}>
              FIRED TRAJECTORIES ({fired.length})
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {fired.map((t) => (
              <TrajectoryCard key={t.id} trajectory={t} ghosts={ghosts} />
            ))}
          </div>
        </div>
      )}

      {/* Open trajectories */}
      {open.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm">{"\u{1F47B}"}</span>
            <p className="text-[9px] tracking-[0.2em] uppercase" style={{ color: "var(--text-label)" }}>
              OPEN TRAJECTORIES ({open.length})
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {open.map((t) => (
              <TrajectoryCard key={t.id} trajectory={t} ghosts={ghosts} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
