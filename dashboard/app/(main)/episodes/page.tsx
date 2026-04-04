"use client";

import { useState } from "react";
import { useMemory } from "@/lib/useMemory";

export default function EpisodesPage() {
  const { data, loading } = useMemory();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  if (loading) return (
    <div className="flex items-center gap-2 p-8 text-xs" style={{ color: "var(--text-muted)" }}>
      <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />Loading…
    </div>
  );

  const episodes = [...data.episodes].reverse();

  return (
    <div className="p-6 space-y-6">
      {/* Task type breakdown */}
      {episodes.length > 0 && (() => {
        const typeCounts: Record<string, number> = {};
        for (const ep of episodes) {
          const t = ep.task_type || "other";
          typeCounts[t] = (typeCounts[t] || 0) + 1;
        }
        const total = episodes.length;
        const types = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
        const typeColors: Record<string, string> = {
          archived_turns: "#7c3aed",
          generic: "var(--color-cyan)",
          proposal: "var(--color-emerald)",
          other: "#64748b",
        };
        return (
          <div className="card-neutral rounded-xl p-5">
            <p className="text-[9px] tracking-[0.2em] uppercase mb-3" style={{ color: "var(--text-label)" }}>
              {total} episodes
            </p>
            <div className="flex h-2 rounded-full overflow-hidden gap-0.5">
              {types.map(([type, count]) => (
                <div
                  key={type}
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(count / total) * 100}%`,
                    background: typeColors[type] || "#64748b",
                    minWidth: "4px",
                  }}
                  title={`${type}: ${count}`}
                />
              ))}
            </div>
            <div className="flex gap-4 mt-2">
              {types.map(([type, count]) => (
                <span key={type} className="text-[9px] flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full inline-block" style={{ background: typeColors[type] || "#64748b" }} />
                  <span style={{ color: "var(--text-label)" }}>{type} ({count})</span>
                </span>
              ))}
            </div>
          </div>
        );
      })()}

      <div className="space-y-3">
        {episodes.map((ep) => {
          const hasTraining = ep.training_example?.accepted === true;
          const isArchive = ep.task_type === "archived_turns";
          const isExpanded = !!expanded[ep.id];

          return (
            <div 
              key={ep.id} 
              className={`rounded-xl p-4 transition-colors cursor-pointer ${isArchive ? "card-violet" : "card-neutral"} hover:bg-[var(--bg-card-hover)]`}
              onClick={() => toggle(ep.id)}
            >
              <div className="flex items-start gap-3 mb-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{ep.title}</p>
                    {hasTraining && (
                      <span className="text-[9px] px-2 py-0.5 rounded-full animate-pulse border" style={{ borderColor: 'rgba(8,145,178,0.4)', background: "rgba(8,145,178,0.08)", color: "var(--color-cyan)" }}>
                        Training data
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>{ep.timestamp?.slice(0, 16)}</span>
                    {isArchive ? (
                      <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: "rgba(124,58,237,0.08)", color: "#7c3aed" }}>
                        Archived
                      </span>
                    ) : (
                      <span className="text-[10px]" style={{ color: "var(--text-label)" }}>{ep.task_type}</span>
                    )}
                  </div>
                </div>
                {ep.corrections?.length > 0 && (
                  <span className="text-[10px] shrink-0" style={{ color: "var(--text-label)" }}>{ep.corrections.length} corrections</span>
                )}
                <span className="text-[10px] select-none" style={{ color: "var(--text-muted)" }}>
                  {isExpanded ? "▲" : "▼"}
                </span>
              </div>

              {/* 閉じていて、かつCorrectionsがあるときのプレビュー */}
              {!isExpanded && ep.corrections?.length > 0 && (
                <div className="mt-3 space-y-1.5">
                  {ep.corrections.slice(0, 3).map((c, i) => (
                    <div key={i} className="text-[10px] pl-3" style={{ color: "var(--text-label)", borderLeft: "2px solid var(--border)" }}>
                      {c.correction_note || c.decision_override || c.reuse_note || "—"}
                    </div>
                  ))}
                  {ep.corrections.length > 3 && (
                    <p className="text-[9px] pl-3" style={{ color: "var(--text-label)" }}>+{ep.corrections.length - 3} more…</p>
                  )}
                </div>
              )}

              {/* 展開時詳細ビュー */}
              {isExpanded && (
                <div className="mt-4 pt-4 space-y-4" style={{ borderTop: "1px dashed var(--border)" }} onClick={(e) => e.stopPropagation()}>
                  
                  {/* Corrections Detail */}
                  {ep.corrections?.length > 0 && (
                    <div>
                      <h4 className="text-[10px] uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Corrections</h4>
                      <div className="space-y-2">
                        {ep.corrections.map((c, i) => (
                          <div key={i} className="text-[10px] pl-3 pb-2" style={{ color: "var(--text-body)", borderLeft: "2px solid var(--color-cyan)" }}>
                            <p className="font-semibold mb-1">{c.correction_note || c.decision_override || c.reuse_note || "—"}</p>
                            {c.bad_output && <p className="opacity-70 mt-1 line-through">Bad: {c.bad_output}</p>}
                            {c.revised_output && <p className="opacity-90 mt-0.5">Revised: {c.revised_output}</p>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Output JSON */}
                  {ep.output && Object.keys(ep.output).length > 0 && (
                    <div>
                      <h4 className="text-[10px] uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Output</h4>
                      <pre className="text-[10px] p-2 overflow-auto rounded border" style={{ background: "var(--bg-subtle)", borderColor: "var(--border)", color: "var(--text-body)", maxHeight: "150px" }}>
                        {JSON.stringify(ep.output, null, 2)}
                      </pre>
                    </div>
                  )}

                  {/* Source Text Preview */}
                  {ep.source_text && (
                    <div>
                      <h4 className="text-[10px] uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Source Text</h4>
                      <div className="text-[10px] p-2 rounded border break-words whitespace-pre-wrap" style={{ background: "var(--bg-subtle)", borderColor: "var(--border)", color: "var(--text-muted)", maxHeight: "100px", overflow: "hidden", position: "relative" }}>
                        {ep.source_text.slice(0, 200)}
                        {ep.source_text.length > 200 && "..."}
                      </div>
                    </div>
                  )}

                </div>
              )}
            </div>
          );
        })}

        {episodes.length === 0 && (
          <p className="text-xs py-8 text-center" style={{ color: "var(--text-label)" }}>No episodes yet</p>
        )}
      </div>
    </div>
  );
}
