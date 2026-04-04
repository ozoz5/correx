"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import type { MemoryData, PreferenceRule, ConversationTurn, GrowthRecord } from "@/lib/types";

const empty: MemoryData = { episodes: [], turns: [], rules: [], growth: [], transitions: [], dreamLog: [], meanings: [], principles: [], profiles: null, experiments: [], benchCases: [], transferEvaluations: [], skills: [], adaptiveRules: [], adaptiveCorrections: [], ghosts: [], ghostTrajectories: [], policies: [], universalLaws: [], positiveLaws: [], personality: null };

type FeedItem = {
  id: string;
  time: string;
  type: "correction" | "promoted" | "growth" | "episode";
  text: string;
  detail?: string;
};

function buildFeed(data: MemoryData): FeedItem[] {
  const items: FeedItem[] = [];

  // Turns → correction events
  for (const t of data.turns) {
    for (const c of t.extracted_corrections) {
      items.push({
        id: `${t.id}-${c}`,
        time: t.recorded_at,
        type: "correction",
        text: c,
        detail: t.task_scope,
      });
    }
  }

  // Promoted rules
  for (const r of data.rules.filter((r) => r.status === "promoted")) {
    items.push({
      id: r.id,
      time: r.last_recorded_at,
      type: "promoted",
      text: r.instruction,
      detail: `x${r.evidence_count} — ${r.applies_to_scope}`,
    });
  }

  // Growth records
  for (const g of data.growth) {
    const sign = g.delta >= 0 ? "+" : "";
    items.push({
      id: g.record_id,
      time: g.recorded_at?.slice(0, 16).replace("T", " ") || "",
      type: "growth",
      text: g.case_title,
      detail: `${sign}${g.delta.toFixed(2)} (${g.baseline_score.toFixed(2)} → ${g.guided_score.toFixed(2)})`,
    });
  }

  // Episodes
  for (const e of data.episodes) {
    items.push({
      id: e.id,
      time: e.timestamp,
      type: "episode",
      text: e.title,
      detail: `${e.task_type} — ${e.corrections.length} corrections`,
    });
  }

  // Sort newest first
  items.sort((a, b) => b.time.localeCompare(a.time));
  return items;
}

const typeConfig = {
  correction: { icon: "!", bg: "bg-yellow-500/10", border: "border-yellow-600/20", label: "FIX", labelColor: "text-yellow-600" },
  promoted: { icon: "^", bg: "bg-emerald-500/10", border: "border-emerald-600/20", label: "RULE", labelColor: "text-emerald-600" },
  growth: { icon: "+", bg: "bg-purple-500/10", border: "border-purple-600/20", label: "GROW", labelColor: "text-purple-600" },
  episode: { icon: "o", bg: "bg-blue-500/10", border: "border-blue-600/20", label: "SAVE", labelColor: "text-blue-600" },
};

function scoreColor(score: number): string {
  if (score >= 0.8) return "text-emerald-600";
  if (score >= 0.5) return "text-yellow-600";
  return "text-red-600";
}

export default function WidgetPage() {
  const [data, setData] = useState<MemoryData>(empty);
  const [flash, setFlash] = useState(false);

  const prevCountRef = useRef(0);

  const refresh = useCallback(() => {
    fetch("/api/memory")
      .then((r) => r.json())
      .then((d: MemoryData) => {
        const newCount = d.turns.length + d.rules.length + d.growth.length + d.episodes.length;
        if (newCount > prevCountRef.current && prevCountRef.current > 0) {
          setFlash(true);
          setTimeout(() => setFlash(false), 600);
        }
        prevCountRef.current = newCount;
        setData(d);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [refresh]);

  const feed = buildFeed(data);
  const promoted = data.rules.filter((r) => r.status === "promoted").length;
  const candidate = data.rules.length - promoted;
  const scores = data.turns.map((t) => t.reaction_score).filter((s): s is number => s !== null);
  const avgScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

  return (
    <div className={`min-h-screen bg-white text-slate-800 transition-colors duration-300 ${flash ? "bg-emerald-50/30" : ""}`}>
      {/* Header stats bar */}
      <div className="sticky top-0 z-10 bg-white/95 backdrop-blur border-b border-slate-200 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${flash ? "bg-emerald-500 animate-pulse" : "bg-emerald-600"}`} />
            <span className="text-xs font-bold text-emerald-600 tracking-widest">CORREX</span>
          </div>
          <span className="text-xs text-slate-400">LIVE</span>
        </div>
        <div className="flex gap-4 mt-2 text-xs">
          <div>
            <span className="text-slate-400">Rules </span>
            <span className="text-emerald-600 font-mono">{promoted}</span>
            <span className="text-slate-400">/{data.rules.length}</span>
          </div>
          <div>
            <span className="text-slate-400">Turns </span>
            <span className="font-mono">{data.turns.length}</span>
          </div>
          <div>
            <span className="text-slate-400">Score </span>
            <span className={`font-mono ${scoreColor(avgScore)}`}>{avgScore.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-slate-400">Growth </span>
            <span className="font-mono text-purple-600">{data.growth.length}</span>
          </div>
        </div>
      </div>

      {/* Live feed */}
      <div className="p-3 space-y-2">
        {feed.map((item) => {
          const cfg = typeConfig[item.type];
          return (
            <div
              key={item.id}
              className={`${cfg.bg} border ${cfg.border} rounded-lg px-3 py-2.5 transition-all duration-300`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[10px] font-bold tracking-wider ${cfg.labelColor}`}>
                  {cfg.label}
                </span>
                <span className="text-[10px] text-slate-400">{item.time}</span>
              </div>
              <p className="text-sm leading-snug">{item.text}</p>
              {item.detail && (
                <p className="text-xs text-slate-400 mt-0.5">{item.detail}</p>
              )}
            </div>
          );
        })}

        {feed.length === 0 && (
          <div className="text-center py-12 text-slate-400 text-sm">
            Waiting for corrections...
          </div>
        )}
      </div>
    </div>
  );
}
