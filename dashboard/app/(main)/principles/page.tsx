"use client";

import { useMemo } from "react";
import { useMemory } from "@/lib/useMemory";

interface Trajectory {
  id: string;
  fired: boolean;
  sublimated_principle: string;
  dormant?: boolean;
  dormant_reason?: string;
  awakened_count?: number;
  cumulative_pe: number;
  scopes: string[];
  source_ghost_count: number;
  ghost_ids: string[];
  fired_at?: string;
}

type Status = "active" | "dormant" | "awakened";

function getStatus(t: Trajectory): Status {
  if (t.dormant) return "dormant";
  if ((t.awakened_count ?? 0) > 0) return "awakened";
  return "active";
}

const statusConfig: Record<Status, { color: string; bg: string; label: string }> = {
  active:   { color: "var(--color-emerald)", bg: "rgba(5,150,105,0.1)",  label: "ACTIVE" },
  dormant:  { color: "var(--text-faint)",    bg: "rgba(100,100,100,0.05)", label: "INACTIVE" },
  awakened: { color: "var(--amber)",         bg: "rgba(245,158,11,0.1)",  label: "REACTIVATED" },
};

export default function PrinciplesPage() {
  const { data, loading } = useMemory();

  const trajectories = useMemo(() => {
    const raw = ((data as any).ghostTrajectories || []) as Trajectory[];
    return raw.filter((t) => t.fired && t.sublimated_principle);
  }, [data]);

  const grouped = useMemo(() => {
    const active: Trajectory[] = [];
    const dormant: Trajectory[] = [];
    const awakened: Trajectory[] = [];
    for (const t of trajectories) {
      const s = getStatus(t);
      if (s === "active") active.push(t);
      else if (s === "dormant") dormant.push(t);
      else awakened.push(t);
    }
    return { active, dormant, awakened };
  }, [trajectories]);

  // Open (unfired) trajectories progressing toward firing
  const openTrajectories = useMemo(() => {
    const raw = ((data as any).ghostTrajectories || []) as Trajectory[];
    return raw.filter((t) => !t.fired && t.ghost_ids?.length > 0);
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="w-2 h-2 rounded-full animate-ping" style={{ background: "var(--cyan)" }} />
      </div>
    );
  }

  const total = trajectories.length;

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-lg font-bold tracking-widest" style={{ color: "var(--text-primary)" }}>
          PRINCIPLES
        </h1>
        <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          Learned behavioral principles — active, inactive, and reactivated
        </p>
      </div>

      {/* Summary bar */}
      <div className="flex gap-4">
        {(["active", "dormant", "awakened"] as Status[]).map((s) => {
          const count = grouped[s].length;
          const cfg = statusConfig[s];
          const pct = total > 0 ? Math.round((count / total) * 100) : 0;
          return (
            <div
              key={s}
              className="flex-1 rounded-xl p-4"
              style={{ background: cfg.bg }}
            >
              <p className="text-[9px] tracking-widest" style={{ color: cfg.color }}>{cfg.label}</p>
              <p className="text-2xl font-bold mt-1" style={{ color: cfg.color }}>{count}</p>
              <p className="text-[9px] mt-1" style={{ color: "var(--text-faint)" }}>{pct}%</p>
            </div>
          );
        })}
      </div>

      {/* Lifecycle bar */}
      {total > 0 && (
        <div className="flex h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-bar)" }}>
          {grouped.active.length > 0 && (
            <div
              style={{
                width: `${(grouped.active.length / total) * 100}%`,
                background: "var(--color-emerald)",
              }}
            />
          )}
          {grouped.awakened.length > 0 && (
            <div
              style={{
                width: `${(grouped.awakened.length / total) * 100}%`,
                background: "var(--amber)",
              }}
            />
          )}
          {grouped.dormant.length > 0 && (
            <div
              style={{
                width: `${(grouped.dormant.length / total) * 100}%`,
                background: "var(--text-faint)",
                opacity: 0.3,
              }}
            />
          )}
        </div>
      )}

      {/* Active principles */}
      <Section
        title="ACTIVE"
        count={grouped.active.length}
        color="var(--color-emerald)"
        items={grouped.active}
      />

      {/* Awakened principles */}
      {grouped.awakened.length > 0 && (
        <Section
          title="AWAKENED"
          count={grouped.awakened.length}
          color="var(--amber)"
          items={grouped.awakened}
          showAwakened
        />
      )}

      {/* Dormant principles */}
      {grouped.dormant.length > 0 && (
        <Section
          title="DORMANT"
          count={grouped.dormant.length}
          color="var(--text-faint)"
          items={grouped.dormant}
          showReason
          dimmed
        />
      )}

      {/* Emerging trajectories */}
      {openTrajectories.length > 0 && (
        <section className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: "var(--violet)" }} />
            <h2 className="text-sm font-bold tracking-widest" style={{ color: "var(--violet)" }}>
              EMERGING ({openTrajectories.length})
            </h2>
          </div>
          <div className="space-y-1.5">
            {openTrajectories.slice(0, 10).map((t) => {
              const pct = Math.min(100, (t.cumulative_pe / 1.0) * 100);
              return (
                <div
                  key={t.id}
                  className="flex items-center gap-3 px-4 py-2 rounded-lg"
                  style={{ background: "var(--bg-subtle)" }}
                >
                  <div className="flex-1">
                    <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                      {t.ghost_ids.length} observations &middot; strength {(t.cumulative_pe * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="w-20 h-1.5 rounded-full" style={{ background: "var(--bg-bar)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, background: "var(--violet)" }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

function Section({
  title,
  count,
  color,
  items,
  showReason,
  showAwakened,
  dimmed,
}: {
  title: string;
  count: number;
  color: string;
  items: Trajectory[];
  showReason?: boolean;
  showAwakened?: boolean;
  dimmed?: boolean;
}) {
  return (
    <section className="space-y-3" style={dimmed ? { opacity: 0.5 } : undefined}>
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full" style={{ background: color }} />
        <h2 className="text-sm font-bold tracking-widest" style={{ color }}>
          {title} ({count})
        </h2>
      </div>
      <div className="space-y-1.5">
        {items.map((t) => (
          <div
            key={t.id}
            className="flex items-start gap-3 px-4 py-2.5 rounded-lg"
            style={{ background: "var(--bg-subtle)" }}
          >
            <div className="flex-1">
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                {t.sublimated_principle}
              </p>
              {showReason && t.dormant_reason && (
                <p className="text-[9px] mt-1" style={{ color: "var(--text-faint)" }}>
                  {t.dormant_reason}
                </p>
              )}
              {showAwakened && (t.awakened_count ?? 0) > 0 && (
                <p className="text-[9px] mt-1" style={{ color: "var(--amber)" }}>
                  awakened {t.awakened_count}x
                </p>
              )}
            </div>
            <div className="shrink-0 text-right">
              <p className="text-[9px]" style={{ color: "var(--text-faint)" }}>
                {t.scopes?.slice(0, 2).join(", ")}
              </p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
