"use client";

import { useState } from "react";
import { useMemory } from "@/lib/useMemory";
import type { Meaning } from "@/lib/types";

function MeaningCard({ meaning, rules }: { meaning: Meaning; rules: { id: string; instruction: string; applies_to_scope: string }[] }) {
  const [expanded, setExpanded] = useState(false);
  const hasOverlap = (meaning.personal_settings_overlap?.length ?? 0) > 0;
  const borderColor = hasOverlap ? "var(--violet)" : meaning.cross_scope_count >= 2 ? "var(--cyan)" : "var(--border)";

  const sourceRules = rules.filter((r) => meaning.source_rule_ids.includes(r.id));

  return (
    <div
      className="card-neutral rounded-xl p-5 relative overflow-hidden cursor-pointer transition-all"
      style={{ borderLeft: `3px solid ${borderColor}` }}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Principle text */}
      <p className="text-sm font-medium leading-relaxed" style={{ color: "var(--text-primary)" }}>
        {meaning.principle}
      </p>

      {/* Badges */}
      <div className="flex flex-wrap items-center gap-2 mt-3">
        <span
          className="text-[9px] px-2 py-0.5 rounded-full font-medium"
          style={{ background: "rgba(5,150,105,0.1)", color: "var(--color-emerald)" }}
        >
          {meaning.strength} rules
        </span>
        <span
          className="text-[9px] px-2 py-0.5 rounded-full font-medium"
          style={
            meaning.cross_scope_count >= 2
              ? { background: "rgba(8,145,178,0.1)", color: "var(--cyan)" }
              : { background: "var(--bg-subtle)", color: "var(--text-label)" }
          }
        >
          {meaning.cross_scope_count} scopes
        </span>
        <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
          Confidence {Math.round(meaning.confidence * 100)}%
        </span>
        {hasOverlap && (
          <span
            className="text-[9px] px-2 py-0.5 rounded-full font-medium"
            style={{ background: "rgba(139,92,246,0.1)", color: "var(--violet)" }}
          >
            Settings match
          </span>
        )}
      </div>

      {/* Scopes */}
      <div className="flex flex-wrap gap-1 mt-2">
        {meaning.scopes.map((scope) => (
          <span
            key={scope}
            className="text-[8px] px-1.5 py-0.5 rounded"
            style={{ background: "var(--bg-subtle)", color: "var(--text-label)" }}
          >
            {scope}
          </span>
        ))}
      </div>

      {/* Expanded: source rules + CLAUDE.md overlap */}
      {expanded && (
        <div className="mt-4 space-y-3 pt-3" style={{ borderTop: "1px solid var(--border-divider)" }}>
          {/* Source rules */}
          <div>
            <p className="text-[9px] tracking-[0.15em] uppercase mb-2" style={{ color: "var(--text-label)" }}>
              SOURCE RULES ({sourceRules.length})
            </p>
            <div className="space-y-1.5">
              {sourceRules.map((r) => (
                <div key={r.id} className="flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full mt-1.5 shrink-0" style={{ background: "var(--color-emerald)" }} />
                  <p className="text-[10px] leading-relaxed" style={{ color: "var(--text-body)" }}>
                    {r.instruction}
                    {r.applies_to_scope && (
                      <span className="ml-1.5" style={{ color: "var(--text-faint)" }}>({r.applies_to_scope})</span>
                    )}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Settings overlap */}
          {hasOverlap && (
            <div>
              <p className="text-[9px] tracking-[0.15em] uppercase mb-2" style={{ color: "var(--violet)" }}>
                SETTINGS MATCH
              </p>
              {meaning.personal_settings_overlap.map((line, i) => (
                <p key={i} className="text-[10px] leading-relaxed pl-3" style={{ color: "var(--text-muted)", borderLeft: "2px solid rgba(139,92,246,0.3)" }}>
                  {line}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function MeaningsPage() {
  const { data, loading } = useMemory();

  if (loading) {
    return (
      <div className="flex items-center gap-2 p-8 text-xs" style={{ color: "var(--text-muted)" }}>
        <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />
        Loading…
      </div>
    );
  }

  const meanings = (data.meanings || []).filter((m) => m.status === "active");
  const principles = (data.principles || []).filter((p) => p.status === "active");
  const crossScope = meanings.filter((m) => m.cross_scope_count >= 2);
  const withOverlap = meanings.filter((m) => m.personal_settings_overlap?.length > 0);
  const totalRulesCovered = new Set(meanings.flatMap((m) => m.source_rule_ids)).size;

  return (
    <div className="p-6 space-y-6">
      {/* Principles — the deepest layer */}
      {principles.length > 0 && (
        <div className="rounded-xl p-6 relative overflow-hidden" style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.08), rgba(8,145,178,0.08))", border: "1px solid rgba(139,92,246,0.2)" }}>
          <p className="text-[9px] tracking-[0.25em] uppercase mb-1" style={{ color: "var(--violet)" }}>
            IDENTITY PRINCIPLES
          </p>
          <p className="text-[9px] mb-4" style={{ color: "var(--text-muted)" }}>
            Core identity values — declarations of who you are, discovered from your behavioral patterns.
          </p>
          <div className="space-y-3">
            {principles.map((p) => (
              <div key={p.id} className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full mt-1.5 shrink-0" style={{ background: "var(--violet)", boxShadow: "0 0 6px rgba(139,92,246,0.4)" }} />
                <div>
                  <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{p.declaration}</p>
                  <p className="text-[9px] mt-1" style={{ color: "var(--text-label)" }}>
                    {p.source_meaning_ids.length} meanings / {p.source_rule_count} rules
                    <span className="ml-2" style={{ color: "var(--text-faint)" }}>{p.scopes.join(", ")}</span>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Header stats */}
      <div className="card-neutral rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--violet)" }} />
          <p className="text-[9px] tracking-[0.2em] uppercase" style={{ color: "var(--text-label)" }}>
            MEANING LAYER
          </p>
        </div>
        <p className="text-[10px] mb-4" style={{ color: "var(--text-muted)" }}>
          Behavioral patterns discovered across different areas — values that exist in you but aren't written in any single rule.
        </p>
        <div className="grid grid-cols-4 gap-3">
          <div className="text-center">
            <p className="text-2xl font-bold glow-violet">{meanings.length}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Meanings found</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold glow-cyan">{crossScope.length}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Cross-scope</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold" style={{ color: "var(--color-emerald)" }}>{totalRulesCovered}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Rules covered</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold" style={{ color: "var(--color-amber)" }}>{withOverlap.length}</p>
            <p className="text-[9px]" style={{ color: "var(--text-label)" }}>Settings match</p>
          </div>
        </div>
      </div>

      {/* Meaning cards */}
      {meanings.length === 0 ? (
        <p className="text-xs py-8 text-center" style={{ color: "var(--text-label)" }}>
          No patterns discovered yet. Continue using the system to generate insights.
        </p>
      ) : (
        <div className="space-y-3">
          {meanings.map((meaning) => (
            <MeaningCard
              key={meaning.id}
              meaning={meaning}
              rules={data.rules}
            />
          ))}
        </div>
      )}
    </div>
  );
}
