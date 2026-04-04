"""Dormancy engine — automatic retirement, awakening, and forgetting.

Like human memory: once a behavior is internalized (covered by a
higher-level law or policy), the explicit rule goes dormant. If the law
fails to prevent a mistake, the specific rule wakes up again. If it stays
dormant long enough, it's forgotten entirely — but its DNA lives on in
the policy that absorbed it.

Lifecycle:
  1. Ghost fires → sublimated_principle created → "active"
  2. Law/policy covers it → principle goes "dormant"
  3. User corrects despite law → principle "awakens"
  4. Dormant for too long without awakening → "forgotten" (deleted)
     The policy/law still carries the essence.

This module requires NO external dependencies beyond stdlib + re.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Similarity helpers (same char-bigram approach as cleanup_overfitting.py)
# ---------------------------------------------------------------------------

def _bigrams(text: str) -> set[str]:
    """Character 2-grams for Japanese text similarity."""
    t = re.sub(r"[\s、。をのがはでにと]", "", text)
    if len(t) < 2:
        return set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


# ---------------------------------------------------------------------------
# Keyword patterns mapped to law themes
# ---------------------------------------------------------------------------

# Each key is a regex pattern; value is a human-readable law label.
# A principle matching any pattern is considered "covered" by that law.
_LAW_PATTERNS: dict[str, str] = {
    # 禁止法理 coverage
    r"状況確認|現状把握|確認してから|把握してから|調査してから|理解してから"
    r"|前提条件|文脈確認|現状確認|前提知識":
        "状況確認→作業",
    r"問題.*調査|問題.*特定|原因.*特定|根本原因|問題.*修正.*順序":
        "調査→修正",
    r"順序.*保持|優先順位.*変え|指定.*順序|並び替え":
        "ユーザー順序維持",
    r"既存.*変更|仕様.*変更|構造.*破壊|表示形式.*破壊|構成.*把握"
    r"|仕様.*勝手":
        "既存変更するな",
    r"代替手段.*安易|権限不足":
        "安易に代替移行するな",
    r"範囲.*拡張|指定.*範囲":
        "作業範囲拡張するな",
    r"並行.*せず|一つずつ|並列.*前に|個別.*完了|集団.*前に":
        "並行より単体完了",
    r"独断.*開始|勝手に.*進め|指示なし|求めていない.*開始":
        "独断で作業するな",
    r"完全性.*確認|説明.*付け|使い方.*説明|提出.*前":
        "完全性確保して提出",
    # 推奨法理 coverage
    r"確認.*求め|意見.*確認|ユーザー.*確認":
        "ユーザー確認",
    r"段階的.*改善|完璧.*待たず|継続.*改善":
        "段階的改善",
    r"感情.*反応|読み取|感情.*読":
        "感情読み取り",
    r"表形式.*整理|変更内容.*明確":
        "表形式整理",
    r"技術的制約.*伝え|代替案.*提示":
        "制約伝達+代替提示",
    r"動作確認.*報告|検証.*報告|実装完了.*報告":
        "動作確認報告",
    r"曖昧.*要求.*選択肢|具体的.*選択肢.*変換":
        "選択肢変換",
    r"評価.*方向性|積極的.*意見.*求":
        "方向性意見",
    r"UI.*UX.*改善|使いやすさ.*追求":
        "UX改善",
    r"商業化.*実用性|価値.*評価.*フィードバック":
        "商業化視点",
    # Policy coverage (理解が行動に先行する)
    r"準備.*完了.*前|改善.*前.*現状|部分的.*確認.*全体"
    r"|実行.*約束.*前.*現状|専門知識.*確認":
        "理解→行動(policy)",
    # Session continuity (multiple laws cover this)
    r"継続.*状況把握|会話継続.*文脈.*推測|会話.*継続.*前提"
    r"|会話.*継続性.*仮定|会話.*継続性.*維持":
        "セッション継続(複数法理)",
    # Completion before next (禁止7 + policy)
    r"発見.*検証.*完了.*次|進行中.*完了.*次.*提案"
    r"|完全.*解決.*次.*提案|調査完了.*次工程":
        "完了→次へ",
    # Generic/vague (too abstract to be useful as standalone)
    r"品質確認.*実行手順|内容確認.*修正作業.*提出"
    r"|最適化.*問題.*探|包括的.*分析|全体.*包括的.*確認"
    r"|全体検証|作業完了.*宣言.*確認|検証完了.*関連処理":
        "汎用的すぎる",
}

# Compiled patterns (lazy init)
_COMPILED: list[tuple[re.Pattern, str]] | None = None


def _get_patterns() -> list[tuple[re.Pattern, str]]:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = [(re.compile(pat), label) for pat, label in _LAW_PATTERNS.items()]
    return _COMPILED


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def check_coverage(
    principle: str,
    laws: list[str] | None = None,
    policies: list[str] | None = None,
    *,
    keyword_only: bool = False,
) -> str | None:
    """Check if a principle is covered by existing laws/policies.

    Returns the covering law/policy label, or None if the principle is unique.

    Args:
        principle: The ghost principle text to check.
        laws: List of law texts (prohibitions + recommendations).
        policies: List of policy core texts.
        keyword_only: If True, skip bigram similarity (faster).
    """
    if not principle or len(principle) < 4:
        return None

    # 1. Keyword pattern matching
    for pat, label in _get_patterns():
        if pat.search(principle):
            return label

    if keyword_only:
        return None

    # 2. Bigram similarity against explicit law/policy texts
    all_refs = (laws or []) + (policies or [])
    for ref in all_refs:
        sim = _jaccard(principle, ref)
        if sim > 0.35:
            return f"類似({sim:.2f}): {ref[:30]}"

    return None


def scan_and_dormant(
    trajectories: list[dict],
    laws: list[str] | None = None,
    policies: list[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Scan all trajectories and mark covered principles as dormant.

    Modifies trajectories in-place. Returns (trajectories, dormant_count, active_count).

    Dormancy is stored as trajectory["dormant"] = True and
    trajectory["dormant_reason"] = "covered by: <law label>".
    The sublimated_principle is NOT cleared — it's preserved for awakening.
    """
    dormant_count = 0
    active_count = 0

    for t in trajectories:
        if not t.get("fired"):
            continue
        p = t.get("sublimated_principle", "").strip()
        if not p:
            continue

        # Already dormant — skip (don't re-check)
        if t.get("dormant"):
            dormant_count += 1
            continue

        covering = check_coverage(p, laws=laws, policies=policies)
        if covering:
            t["dormant"] = True
            t["dormant_reason"] = f"covered by: {covering}"
            dormant_count += 1
        else:
            active_count += 1

    return trajectories, dormant_count, active_count


def awaken_relevant(
    trajectories: list[dict],
    user_feedback: str,
    scope: str = "",
    *,
    similarity_threshold: float = 0.25,
) -> tuple[list[dict], list[str]]:
    """Awaken dormant principles relevant to a user correction.

    When a user corrects despite laws being active, check if any dormant
    principle would have prevented the mistake. If so, awaken it.

    Returns (trajectories, list_of_awakened_principles).
    """
    awakened: list[str] = []

    for t in trajectories:
        if not t.get("dormant") or not t.get("sublimated_principle"):
            continue

        p = t["sublimated_principle"]

        # Check relevance: does this dormant principle relate to the correction?
        # Scope match lowers the similarity bar; without scope match, need stronger signal
        scope_match = scope and any(s == scope for s in t.get("scopes", []))
        sim = _jaccard(user_feedback, p)
        threshold = similarity_threshold * 0.6 if scope_match else similarity_threshold

        if sim > threshold:
            t["dormant"] = False
            t["dormant_reason"] = ""
            t["awakened_count"] = t.get("awakened_count", 0) + 1
            awakened.append(p)

    return trajectories, awakened


def forget_stale(
    trajectories: list[dict],
    *,
    max_dormant_days: int = 30,
) -> tuple[list[dict], int]:
    """Permanently forget principles that have been dormant too long.

    Like human memory: individual episodes fade, but the lessons (policies)
    remain. A principle dormant for max_dormant_days without awakening is
    deleted — its sublimated_principle is cleared permanently.

    Rules that have been awakened at least once are never forgotten
    (they proved their value by waking up when needed).

    Returns (trajectories, forgotten_count).
    """
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_dormant_days)
    forgotten = 0

    for t in trajectories:
        if not t.get("dormant") or not t.get("sublimated_principle"):
            continue

        # Never forget principles that have been awakened before
        if (t.get("awakened_count") or 0) > 0:
            continue

        # Check when it was fired (became dormant around that time)
        fired_at = t.get("fired_at", "")
        if not fired_at:
            continue

        try:
            # Handle both ISO and slash formats
            if "T" in fired_at:
                fired_dt = datetime.fromisoformat(fired_at)
            else:
                fired_dt = datetime.strptime(fired_at, "%Y/%m/%d %H:%M")
                fired_dt = fired_dt.replace(tzinfo=timezone.utc)

            if fired_dt < cutoff:
                t["sublimated_principle"] = ""
                t["dormant"] = False
                t["dormant_reason"] = "forgotten"
                forgotten += 1
        except (ValueError, TypeError):
            continue

    return trajectories, forgotten


def forget_stale_rules(
    rules: list[dict],
    *,
    max_dormant_days: int = 30,
) -> tuple[list[dict], int]:
    """Permanently forget preference rules that have been dormant too long.

    Same principle as forget_stale for ghost principles.
    Dormant rules whose DNA is already in policies are safe to delete.

    Returns (remaining_rules, forgotten_count).
    """
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_dormant_days)
    remaining: list[dict] = []
    forgotten = 0

    for r in rules:
        if r.get("status") != "dormant":
            remaining.append(r)
            continue

        # Check last update time
        updated = r.get("updated_at") or r.get("created_at", "")
        if not updated:
            remaining.append(r)
            continue

        try:
            if "T" in updated:
                dt = datetime.fromisoformat(updated)
            else:
                dt = datetime.strptime(updated, "%Y/%m/%d %H:%M")
                dt = dt.replace(tzinfo=timezone.utc)

            if dt < cutoff:
                forgotten += 1  # don't append — truly deleted
            else:
                remaining.append(r)
        except (ValueError, TypeError):
            remaining.append(r)

    return remaining, forgotten
