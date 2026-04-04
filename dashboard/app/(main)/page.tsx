"use client";

import { useEffect, useRef, useMemo, useState } from "react";
import { useMemory } from "@/lib/useMemory";
import { relativeTime } from "@/lib/maturity";
import Tamagotchi, { type Personality } from "@/components/Tamagotchi";

type Color = "cyan" | "emerald" | "violet" | "amber";

const colorMap: Record<Color, { card: string; text: string }> = {
  cyan:    { card: "card-cyan",    text: "glow-cyan" },
  emerald: { card: "card-emerald", text: "glow-emerald" },
  violet:  { card: "card-violet",  text: "glow-violet" },
  amber:   { card: "card-amber",   text: "glow-amber" },
};

function extractPersonality(data: any): Personality | undefined {
  const p = data?.personality;
  if (!p) return undefined;
  return {
    metabolism: p.metabolism_rate ?? p.metabolism ?? 0.5,
    digestibility: p.digestibility ?? 0.5,
    curiosityLevel: p.curiosity_level ?? 0.5,
    rewardKeywords: p.reward_keywords ?? [],
    avoidanceCount: (p.avoidance_keywords ?? []).length,
    turnCount: p.sample_size ?? (data.turns?.length ?? 0),
    driftDetected: p.drift_detected ?? false,
  };
}

function Stat({ label, value, sub, color = "cyan" }: {
  label: string; value: string | number; sub?: string; color?: Color;
}) {
  const c = colorMap[color];
  return (
    <div className={`${c.card} rounded-xl p-4`}>
      <p className="text-[9px] tracking-[0.2em] uppercase mb-1" style={{ color: "var(--text-label)" }}>{label}</p>
      <p className={`text-2xl font-bold ${c.text}`}>{value}</p>
      {sub && <p className="text-[9px] mt-1" style={{ color: "var(--text-muted)" }}>{sub}</p>}
    </div>
  );
}

export default function OverviewPage() {
  const { data, loading, lastUpdated } = useMemory();

  const [dataVersion, setDataVersion] = useState(0);
  const prevTurnsRef = useRef(0);

  useEffect(() => {
    const newLen = (data.turns || []).length;
    if (newLen > prevTurnsRef.current && prevTurnsRef.current > 0) {
      setDataVersion((v) => v + 1);
    }
    prevTurnsRef.current = newLen;
  }, [data.turns]);

  const d = data as any;
  const trajectories = useMemo(() => (d.ghostTrajectories || []) as any[], [d.ghostTrajectories]);
  const policies = useMemo(() => (d.policies || []) as any[], [d.policies]);
  const universalLaws = useMemo(() => (d.universalLaws || []) as any[], [d.universalLaws]);
  const positiveLaws = useMemo(() => (d.positiveLaws || []) as any[], [d.positiveLaws]);

  const principles = useMemo(() => {
    const fired = trajectories.filter((t: any) => t.fired && t.sublimated_principle);
    const active = fired.filter((t: any) => !t.dormant);
    const dormant = fired.filter((t: any) => t.dormant);
    const awakened = fired.filter((t: any) => (t.awakened_count ?? 0) > 0 && !t.dormant);
    return { total: fired.length, active: active.length, dormant: dormant.length, awakened: awakened.length };
  }, [trajectories]);

  const growth = useMemo(() => {
    const records = (data.growth || []).filter((g: any) => g.delta != null && !isNaN(g.delta));
    if (records.length === 0) return null;
    const avg = records.reduce((a: number, g: any) => a + g.delta, 0) / records.length;
    return { avg, count: records.length, trend: avg > 0.05 ? "growing" : avg < -0.05 ? "degrading" : "stable" };
  }, [data.growth]);

  const rules = data.rules || [];
  const promoted = rules.filter((r: any) => r.status === "promoted");
  const turns = data.turns || [];
  const lastTurn = turns.length > 0 ? turns[0] : null;

  const recentScore = useMemo(() => {
    const recent = turns.slice(0, 5).filter((t: any) => t.reaction_score != null);
    if (recent.length === 0) return 0.5;
    return recent.reduce((a: number, t: any) => a + t.reaction_score, 0) / recent.length;
  }, [turns]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen gap-3">
        <div className="w-2 h-2 rounded-full animate-ping" style={{ background: "var(--cyan)" }} />
        <span className="text-xs tracking-widest" style={{ color: "var(--text-label)" }}>LOADING...</span>
      </div>
    );
  }

  const activePolicies = policies.filter((p: any) => p.maturity === "active");

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold tracking-widest" style={{ color: "var(--text-primary)" }}>
            SYSTEM OVERVIEW
          </h1>
          <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            Knowledge hierarchy health at a glance
          </p>
        </div>
        {lastUpdated && (
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--color-emerald)" }} />
            <span className="text-[9px] tracking-widest" style={{ color: "var(--text-label)" }}>
              {lastUpdated.toLocaleTimeString("en-US", { hour12: false })}
            </span>
          </div>
        )}
      </div>

      {/* Tamagotchi + Knowledge Hierarchy */}
      <div className="rounded-xl p-6 space-y-4" style={{ background: "var(--bg-subtle)", border: "1px solid var(--border)" }}>
        <Tamagotchi
          growthTrend={growth ? growth.trend as any : "none"}
          recentScore={recentScore}
          principleHealth={principles.total > 0 ? principles.active / principles.total : 0}
          policyCount={activePolicies.length}
          level={Math.min(6, activePolicies.length + Math.floor(principles.active / 10) + 1)}
          dataVersion={dataVersion}
          personality={extractPersonality(d)}
        />

        <p className="text-[9px] tracking-[0.2em] uppercase" style={{ color: "var(--text-label)" }}>KNOWLEDGE HIERARCHY</p>

        {/* Layer 1: Policies */}
        <HierarchyLayer
          label="POLICIES"
          count={activePolicies.length}
          color="var(--color-emerald)"
          description={activePolicies.length > 0 ? activePolicies[0].title : "No active policies"}
          width={100}
        />

        {/* Layer 2: Laws */}
        <HierarchyLayer
          label="LAWS"
          count={universalLaws.length + positiveLaws.length}
          color="var(--color-red, #ef4444)"
          description={`${universalLaws.length} prohibitions + ${positiveLaws.length} recommendations`}
          width={85}
        />

        {/* Layer 3: Principles (ghost-sublimated) */}
        <HierarchyLayer
          label="PRINCIPLES"
          count={principles.active}
          color="var(--cyan)"
          description={`${principles.active} active / ${principles.dormant} dormant / ${principles.awakened} awakened`}
          width={70}
          badge={principles.dormant > 0 ? `${principles.dormant} absorbed` : undefined}
        />

        {/* Layer 4: Rules */}
        <HierarchyLayer
          label="RULES"
          count={promoted.length}
          color="var(--violet)"
          description={`${promoted.length} promoted / ${rules.length} total`}
          width={55}
        />

        {/* Layer 5: Turns (raw data) */}
        <HierarchyLayer
          label="CORRECTIONS"
          count={turns.length}
          color="var(--text-muted)"
          description={lastTurn ? `Last: ${relativeTime(lastTurn.recorded_at)}` : "No corrections yet"}
          width={40}
        />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat
          label="GROWTH"
          value={growth ? `${growth.avg >= 0 ? "+" : ""}${Math.round(growth.avg * 100)}%` : "--"}
          sub={growth ? `${growth.count} measurements / ${growth.trend}` : "No data"}
          color={growth && growth.avg > 0 ? "emerald" : "amber"}
        />
        <Stat
          label="PRINCIPLE HEALTH"
          value={`${principles.active}/${principles.total}`}
          sub={principles.dormant > 0 ? `${principles.dormant} absorbed by laws` : "All active"}
          color="cyan"
        />
        <Stat
          label="RULE MATURITY"
          value={rules.length > 0 ? `${Math.round((promoted.length / rules.length) * 100)}%` : "0%"}
          sub={`${promoted.length} promoted of ${rules.length}`}
          color="violet"
        />
        <Stat
          label="CORRECTIONS"
          value={turns.length}
          sub={lastTurn ? `Last: ${relativeTime(lastTurn.recorded_at)}` : "None"}
          color="amber"
        />
      </div>

      {/* Growth trend bars */}
      {data.growth && data.growth.length > 0 && (
        <div className="rounded-xl p-5" style={{ background: "var(--bg-subtle)", border: "1px solid var(--border)" }}>
          <p className="text-[9px] tracking-[0.2em] uppercase mb-3" style={{ color: "var(--text-label)" }}>
            GROWTH TREND
            {growth && (
              <span className="ml-2 font-medium" style={{ color: growth.trend === "growing" ? "var(--color-emerald)" : "var(--text-muted)" }}>
                {growth.trend.toUpperCase()}
              </span>
            )}
          </p>
          <div className="flex items-end gap-1 h-12">
            {data.growth.filter((g: any) => g.delta != null && !isNaN(g.delta)).map((g: any, i: number) => {
              const h = Math.max(4, Math.abs(g.delta) * 80);
              const positive = g.delta >= 0;
              return (
                <div
                  key={i}
                  className="flex-1 rounded-t transition-all duration-500"
                  style={{
                    height: `${h}px`,
                    background: positive ? "var(--color-emerald)" : "var(--color-red, #ef4444)",
                    opacity: 0.3 + (i / data.growth.length) * 0.7,
                  }}
                  title={`${g.case_title}: ${g.delta >= 0 ? "+" : ""}${(g.delta * 100).toFixed(0)}%`}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Recent activity (compact) */}
      {turns.length > 0 && (
        <div className="rounded-xl p-5" style={{ background: "var(--bg-subtle)", border: "1px solid var(--border)" }}>
          <p className="text-[9px] tracking-[0.2em] uppercase mb-3" style={{ color: "var(--text-label)" }}>
            RECENT CORRECTIONS
          </p>
          <div className="space-y-0">
            {turns.slice(0, 5).map((t: any, i: number) => (
              <div key={i} className="flex items-start gap-3 py-2" style={{ borderBottom: "1px solid var(--border-divider)" }}>
                <span className="text-[9px] shrink-0 mt-0.5 px-1.5 py-0.5 rounded-full font-mono"
                  style={{
                    background: (t.reaction_score ?? 1) < 0.5 ? "rgba(239,68,68,0.1)" : "rgba(5,150,105,0.1)",
                    color: (t.reaction_score ?? 1) < 0.5 ? "var(--color-red, #ef4444)" : "var(--color-emerald)",
                  }}
                >
                  {((t.reaction_score ?? 0.5) * 100).toFixed(0)}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] truncate" style={{ color: "var(--text-body)" }}>
                    {t.user_feedback || t.user_message || "—"}
                  </p>
                  <p className="text-[9px]" style={{ color: "var(--text-faint)" }}>
                    {t.task_scope} &middot; {relativeTime(t.recorded_at)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function HierarchyLayer({ label, count, color, description, width, badge }: {
  label: string; count: number; color: string; description: string; width: number; badge?: string;
}) {
  return (
    <div className="flex items-center gap-4">
      <div
        className="rounded-lg py-2.5 px-4 flex items-center justify-between transition-all"
        style={{
          width: `${width}%`,
          background: `color-mix(in srgb, ${color} 8%, transparent)`,
          borderLeft: `3px solid ${color}`,
        }}
      >
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold" style={{ color }}>{count}</span>
          <div>
            <p className="text-[9px] tracking-widest font-bold" style={{ color }}>{label}</p>
            <p className="text-[9px]" style={{ color: "var(--text-muted)" }}>{description}</p>
          </div>
        </div>
        {badge && (
          <span className="text-[8px] px-1.5 py-0.5 rounded-full" style={{ background: "rgba(100,100,100,0.1)", color: "var(--text-label)" }}>
            {badge}
          </span>
        )}
      </div>
    </div>
  );
}
