"use client";

import { useState } from "react";
import { useMemory } from "@/lib/useMemory";
import type { PreferenceRule } from "@/lib/types";

// 定着に必要な証拠数（overview の RuleBar と合わせる）
const PROMOTE_THRESHOLD = 2;

// 汎用のアクションボタンコンポーネント
function RuleActions({ 
  rule, 
  onStateChange 
}: { 
  rule: PreferenceRule, 
  onStateChange: () => void 
}) {
  const [loading, setLoading] = useState(false);
  const isDemoted = rule.status === "demoted";

  async function handleAction(action: "demote" | "promote") {
    setLoading(true);
    await fetch("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: rule.id, action }),
    });
    setLoading(false);
    onStateChange();
  }

  return (
    <div className="flex flex-col gap-2 shrink-0">
      {isDemoted ? (
        <button
          onClick={() => handleAction("promote")}
          disabled={loading}
          className="text-[9px] px-2 py-1 rounded border transition-all tracking-wide"
          style={{ borderColor: "rgba(16,185,129,0.4)", color: "var(--color-emerald)", background: "rgba(16,185,129,0.06)" }}
        >
          {loading ? "…" : "Restore"}
        </button>
      ) : (
        <button
          onClick={() => handleAction("demote")}
          disabled={loading}
          className="text-[9px] px-2 py-1 rounded border transition-all tracking-wide"
          style={{ borderColor: "rgba(239,68,68,0.4)", color: "#ef4444", background: "rgba(239,68,68,0.06)" }}
        >
          {loading ? "…" : "Disable"}
        </button>
      )}
    </div>
  );
}

// 進捗バー付きカード（candidate 用）
function CandidateCard({ rule }: { rule: PreferenceRule }) {
  const pct = Math.min(100, (rule.evidence_count / PROMOTE_THRESHOLD) * 100);
  const almostThere = rule.evidence_count >= PROMOTE_THRESHOLD - 1 && rule.evidence_count < PROMOTE_THRESHOLD;
  const isDemoted = rule.status === "demoted";

  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(rule.instruction);
  const [saving, setSaving] = useState(false);
  const [updatedRule, setUpdatedRule] = useState(rule);

  async function saveEdit() {
    setSaving(true);
    await fetch("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: rule.id, action: "edit", instruction: draft }),
    });
    setUpdatedRule({ ...updatedRule, instruction: draft });
    setIsEditing(false);
    setSaving(false);
  }

  // 状態変更時はウィンドウリロードで全体更新する仕様（一時的UI状態だけでなくデータと同期）
  const handleStateChange = () => window.location.reload();

  return (
    <div
      className={`rounded-lg p-4 flex flex-col gap-3 transition-opacity ${almostThere ? "card-amber" : "card-neutral"} ${isDemoted ? "opacity-40 grayscale" : ""}`}
    >
      <div className="flex items-start gap-4">
        <div className="flex-1 min-w-0">
          {isEditing ? (
             <div className="flex flex-col gap-2">
               <textarea 
                 value={draft}
                 onChange={(e) => setDraft(e.target.value)}
                 className="w-full bg-transparent border rounded p-2 text-sm focus:outline-none"
                 style={{ borderColor: "var(--border)", color: "var(--text-body)" }}
                 rows={3}
               />
               <div className="flex gap-2">
                  <button onClick={saveEdit} disabled={saving} className="text-[10px] px-3 py-1 bg-cyan-700 text-white rounded"> {saving ? "Saving" : "Save"}</button>
                  <button onClick={() => { setIsEditing(false); setDraft(updatedRule.instruction); }} className="text-[10px] px-3 py-1 border rounded" style={{ borderColor: 'var(--border)'}}>Cancel</button>
               </div>
             </div>
          ) : (
            <div className="flex items-center gap-2">
              <p className="text-sm cursor-text hover:opacity-80" onClick={() => !isDemoted && setIsEditing(true)} style={{ color: almostThere ? "#d97706" : "var(--text-body)" }}>
                {updatedRule.instruction}
              </p>
              {!isDemoted && (
                <button onClick={() => setIsEditing(true)} className="text-[10px] px-2 py-0.5 rounded opacity-0 hover:opacity-100 transition-opacity" style={{ background: "var(--bg-subtle)" }}>✎ Edit</button>
              )}
            </div>
          )}
          
          <div className="flex items-center gap-3 mt-2">
            <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>{rule.applies_to_scope || "General"}</span>
            <span className="text-[9px]" style={{ color: "var(--text-label)" }}>Evidence ×{rule.evidence_count}</span>
            {rule.last_recorded_at && (
              <span className="text-[9px]" style={{ color: "var(--text-label)" }}>{rule.last_recorded_at.slice(0, 10)}</span>
            )}
            {almostThere && !isDemoted && (
              <span
                className="text-[9px] px-1.5 py-0.5 rounded tracking-wider"
                style={{ background: "rgba(217,119,6,0.1)", color: "#d97706" }}
              >
                1 more to promote
              </span>
            )}
          </div>
        </div>
        <RuleActions rule={rule} onStateChange={handleStateChange} />
      </div>
      {/* 進捗バー */}
      {!isDemoted && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[8px]" style={{ color: "var(--text-label)" }}>Progress</span>
            <span className="text-[8px]" style={{ color: "var(--text-muted)" }}>{rule.evidence_count} / {PROMOTE_THRESHOLD}</span>
          </div>
          <div className="h-0.5 rounded-full overflow-hidden" style={{ background: "var(--bg-bar)" }}>
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${pct}%`,
                background: almostThere
                  ? "linear-gradient(to right, #d97706, #fbbf24)"
                  : "linear-gradient(to right, #94a3b8, #cbd5e1)",
                boxShadow: almostThere ? "0 0 4px rgba(217,119,6,0.3)" : "none",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function RuleCard({ rule }: { rule: PreferenceRule }) {
  const promoted = rule.status === "promoted";
  const isDemoted = rule.status === "demoted";
  const conf = rule.confidence_score ?? 0;
  const gain = rule.expected_gain ?? 0;
  const strongSignals = rule.strong_signal_count ?? 0;

  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(rule.instruction);
  const [saving, setSaving] = useState(false);
  const [updatedRule, setUpdatedRule] = useState(rule);

  async function saveEdit() {
    setSaving(true);
    await fetch("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: rule.id, action: "edit", instruction: draft }),
    });
    setUpdatedRule({ ...updatedRule, instruction: draft });
    setIsEditing(false);
    setSaving(false);
  }

  const handleStateChange = () => window.location.reload();

  return (
    <div
      className={`rounded-lg p-4 flex items-start gap-4 transition-opacity ${promoted ? "card-emerald" : "card-neutral"} ${isDemoted ? "opacity-40 grayscale" : ""}`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 group">
          {isEditing ? (
             <div className="flex flex-col gap-2 w-full">
               <textarea 
                 value={draft}
                 onChange={(e) => setDraft(e.target.value)}
                 className="w-full bg-transparent border rounded p-2 text-sm focus:outline-none"
                 style={{ borderColor: "var(--border)", color: "var(--text-body)" }}
                 rows={3}
               />
               <div className="flex gap-2">
                  <button onClick={saveEdit} disabled={saving} className="text-[10px] px-3 py-1 bg-cyan-700 text-white rounded"> {saving ? "Saving" : "Save"}</button>
                  <button onClick={() => { setIsEditing(false); setDraft(updatedRule.instruction); }} className="text-[10px] px-3 py-1 border rounded" style={{ borderColor: 'var(--border)'}}>Cancel</button>
               </div>
             </div>
          ) : (
            <>
              <p className="text-sm cursor-text hover:opacity-80 flex-1" onClick={() => !isDemoted && setIsEditing(true)} style={{ color: promoted && !isDemoted ? "var(--text-primary)" : "var(--text-body)" }}>
                {updatedRule.instruction}
              </p>
              {!isDemoted && (
                <button onClick={() => setIsEditing(true)} className="text-[10px] px-2 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity shrink-0" style={{ background: "var(--bg-subtle)" }}>✎ Edit</button>
              )}
            </>
          )}
          {strongSignals >= 3 && !isEditing && (
            <span className="text-xs" title={`${strongSignals} strong signals`}>
              {"\uD83D\uDD25"}
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-3 mt-2 flex-wrap">
          <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>{rule.applies_to_scope || "General"}</span>
          <span className="text-[9px]" style={{ color: "var(--text-label)" }}>{"Evidence \u00D7"}{rule.evidence_count}</span>
          {gain > 0 && (
            <span className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: "rgba(5,150,105,0.08)", color: "var(--color-emerald)" }}>
              +{gain.toFixed(2)} {"gain"}
            </span>
          )}
          {rule.support_score != null && rule.support_score > 0 && (
            <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
              {"Support: "}{rule.support_score.toFixed(1)}
            </span>
          )}
          {rule.last_recorded_at && (
            <span className="text-[9px]" style={{ color: "var(--text-label)" }}>{rule.last_recorded_at.slice(0, 10)}</span>
          )}
        </div>
        
        {/* Confidence bar */}
        {conf > 0 && !isDemoted && (
          <div className="mt-2">
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-[8px]" style={{ color: "var(--text-label)" }}>{"Confidence"}</span>
              <span className="text-[8px]" style={{ color: "var(--text-muted)" }}>{(conf * 100).toFixed(0)}%</span>
            </div>
            <div className="h-0.5 rounded-full overflow-hidden" style={{ background: "var(--bg-bar)" }}>
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${conf * 100}%`,
                  background: "linear-gradient(to right, #0e7490, #0891b2)",
                  boxShadow: "0 0 2px rgba(8,145,178,0.2)",
                }}
              />
            </div>
          </div>
        )}
      </div>
      <RuleActions rule={rule} onStateChange={handleStateChange} />
    </div>
  );
}

export default function RulesPage() {
  const { data, loading } = useMemory();

  if (loading) {
    return (
      <div className="flex items-center gap-2 p-8 text-xs" style={{ color: "var(--text-muted)" }}>
        <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />
        Loading…
      </div>
    );
  }

  const promoted = data.rules.filter((r) => r.status === "promoted");
  const demoted = data.rules.filter((r) => r.status === "demoted");
  const candidates = data.rules
    .filter((r) => r.status !== "promoted" && r.status !== "demoted")
    .sort((a, b) => b.evidence_count - a.evidence_count);

  // あと1回で定着するルール
  const almostPromoted = candidates.filter(
    (r) => r.evidence_count >= PROMOTE_THRESHOLD - 1
  );
  const otherCandidates = candidates.filter(
    (r) => r.evidence_count < PROMOTE_THRESHOLD - 1
  );

  return (
    <div className="p-6 space-y-6">
      {/* NEAR PROMOTION */}
      {almostPromoted.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-3">
            <div
              className="w-1.5 h-1.5 rounded-full bg-amber-500"
              style={{ boxShadow: "0 0 4px rgba(217,119,6,0.3)" }}
            />
            <h2 className="text-[10px] tracking-[0.2em] text-amber-600">
              NEAR PROMOTION — {almostPromoted.length}
            </h2>
          </div>
          <p className="text-[9px] mb-3" style={{ color: "var(--text-label)" }}>
            One more correction will promote these candidates automatically.
          </p>
          <div className="space-y-2">
            {almostPromoted.map((rule) => (
              <CandidateCard key={rule.id} rule={rule} />
            ))}
          </div>
        </section>
      )}

      {/* Promoted */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" style={{ boxShadow: "0 0 4px rgba(5,150,105,0.3)" }} />
          <h2 className="text-[10px] tracking-[0.2em] text-emerald-600">PROMOTED — {promoted.length}</h2>
        </div>
        <div className="space-y-2">
          {promoted.map((rule) => (
            <RuleCard key={rule.id} rule={rule} />
          ))}
          {promoted.length === 0 && (
            <p className="text-[10px] py-4" style={{ color: "var(--text-label)" }}>None yet</p>
          )}
        </div>
      </section>

      {/* Other candidates */}
      {otherCandidates.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--text-label)" }} />
            <h2 className="text-[10px] tracking-[0.2em]" style={{ color: "var(--text-label)" }}>LEARNING — {otherCandidates.length}</h2>
          </div>
          <div className="space-y-2">
            {otherCandidates.map((rule) => (
              <CandidateCard key={rule.id} rule={rule} />
            ))}
          </div>
        </section>
      )}

      {/* Demoted Rules */}
      {demoted.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1.5 h-1.5 rounded-full bg-red-500" style={{ boxShadow: "0 0 4px rgba(239,68,68,0.3)" }} />
            <h2 className="text-[10px] tracking-[0.2em] text-red-600 opacity-80">DISABLED — {demoted.length}</h2>
          </div>
          <p className="text-[9px] mb-3" style={{ color: "var(--text-label)" }}>
            Intentionally disabled rules. Not included in AI prompts.
          </p>
          <div className="space-y-2">
            {demoted.map((rule) => (
              <RuleCard key={rule.id} rule={rule} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
