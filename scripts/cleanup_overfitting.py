#!/usr/bin/env python3
"""Cleanup script for overfitting data in pseudo-intelligence.

Phase 1: Sanitize (remove markdown noise, prefixes, long explanations)
Phase 2: Semantic clustering (group near-identical principles)
Phase 3: Dedup (keep representative from each cluster)
Phase 4: Fix rules (self-ref, confidence cap)
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

BASE_DIR = Path.home() / ".correx"


# ── Sanitize ──────────────────────────────────────────────────────────

def sanitize_principle(text: str) -> str:
    if not text:
        return ""
    # Remove noise prefixes
    text = re.sub(r"^(汎用原則|固有原則|固有)\s*[:：]\s*", "", text)
    # Remove trailing explanations (broken sublimation output)
    text = re.sub(r"\s*(この原則は|解説[:：]|を汎用化すると|この場合|固有の「|この固有原則).*$", "", text, flags=re.DOTALL)
    # Remove broken fragments with closing brackets/quotes at start
    text = re.sub(r"^[^a-zA-Z\u3000-\u9FFF]*」", "", text)
    # Remove scope tags like [correx_development]
    text = re.sub(r"\[[\w_]+\]\s*(において)?", "", text)
    # Remove trailing arrows and partial mappings
    text = re.sub(r"」→\s*汎用[:：]?\s*「?", "", text)
    # Remove "csak" and other foreign garbage
    text = re.sub(r"\s*csak.*$", "", text)
    # Remove markdown
    text = re.sub(r"\|[^\n]*\|", "", text)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\([\d]+件の.*?\)", "", text)
    text = re.sub(r"[（(]\d+文字[）)]", "", text)
    # Remove parenthetical metadata
    text = re.sub(r"（[^）]*ゴーストから[^）]*）", "", text)
    text = re.sub(r"（自律抽出[^）]*）", "", text)
    text = text.strip("「」\"'（）() ")
    text = re.sub(r"\s+", " ", text).strip()
    # Cut at first sentence if too long
    if len(text) > 60:
        first = re.split(r"[。\n]", text)[0]
        if first and len(first) > 5:
            text = first
    # Reject if still garbage (too short or not Japanese)
    if len(text) < 4:
        return ""
    return text.strip()


# ── Similarity ────────────────────────────────────────────────────────

def char_bigrams(text: str) -> set[str]:
    """Character 2-grams — works well for Japanese without tokenization."""
    text = re.sub(r"[\s、。「」]", "", text)
    if len(text) < 2:
        return set()
    return {text[i:i+2] for i in range(len(text) - 1)}


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of character bigrams — robust for Japanese."""
    ba, bb = char_bigrams(a), char_bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def cluster_principles(principles: list[str], threshold: float = 0.20) -> list[list[int]]:
    """Group principles into clusters using transitive closure.

    If A≈B and B≈C, then A, B, C are all in the same cluster,
    even if A and C aren't directly similar.
    """
    n = len(principles)
    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if similarity(principles[i], principles[j]) >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


def pick_representative(cluster_texts: list[str]) -> str:
    """Pick the best representative: prefer shorter, cleaner text."""
    # Filter out ones with noise
    clean = [t for t in cluster_texts if not re.search(r"この原則は|解説|を汎用化すると", t)]
    candidates = clean or cluster_texts
    # Prefer moderate length (not too short, not too long)
    candidates.sort(key=lambda t: abs(len(t) - 25))
    return candidates[0]


# ── Confidence recalc ────────────────────────────────────────────────

def recalc_confidence(rule: dict) -> float:
    evidence = max(0, rule.get("evidence_count", 0))
    scopes = max(0, rule.get("distinct_scope_count", 0))
    tags = max(0, rule.get("distinct_tag_count", 0))
    strong = max(0, rule.get("strong_signal_count", 0))
    success = max(0, rule.get("success_count", 0))
    failure = max(0, rule.get("failure_count", 0))

    c = 0.1
    c += min(0.42, evidence * 0.11)
    c += min(0.18, scopes * 0.09)
    c += min(0.15, tags * 0.03)
    c += min(0.15, (success + failure) * 0.05)
    if strong > 0:
        c += 0.1
    cap = 1.0 if evidence >= 2 else 0.6
    return round(min(cap, c), 4)


# ── IO ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


def save_json(path: Path, data: list | dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("過学習クリーンアップ v2 (word+ngram combined)")
    print("=" * 60)

    # Backup
    backup_dir = BASE_DIR / "backup_pre_cleanup"
    if not backup_dir.exists():
        backup_dir.mkdir(exist_ok=True)
        for f in BASE_DIR.glob("*.json"):
            shutil.copy2(f, backup_dir / f.name)
        print(f"  バックアップ作成: {backup_dir}")
    else:
        print(f"  バックアップ既存: {backup_dir}")

    # ── 1. Ghost trajectories ─────────────────────────────────
    traj_file = BASE_DIR / "ghost_trajectories.json"
    traj_raw = load_json(traj_file)
    # Handle both formats: list or {"schema_version": ..., "items": [...]}
    traj_wrapper = None
    if isinstance(traj_raw, dict) and "items" in traj_raw:
        traj_wrapper = traj_raw
        trajectories = traj_raw["items"]
    elif isinstance(traj_raw, list):
        trajectories = traj_raw
    else:
        trajectories = []

    fired_indices = [
        i for i, t in enumerate(trajectories)
        if isinstance(t, dict) and t.get("fired") and t.get("sublimated_principle")
    ]
    original_count = len(fired_indices)
    print(f"\n[Ghost原則] 修正前: {original_count}件")

    # Sanitize
    for t in trajectories:
        if t.get("sublimated_principle"):
            t["sublimated_principle"] = sanitize_principle(t["sublimated_principle"])

    # Cluster and dedup
    fired_texts = [trajectories[i]["sublimated_principle"] for i in fired_indices]
    clusters = cluster_principles(fired_texts, threshold=0.45)

    # For each cluster, keep ONE representative, clear all others
    kept_count = 0
    cleared_count = 0
    for cluster in clusters:
        texts = [fired_texts[idx] for idx in cluster]
        rep = pick_representative(texts)
        kept_count += 1

        # Set the first occurrence to the representative, clear the rest
        first_set = False
        for idx in cluster:
            traj_idx = fired_indices[idx]
            if not first_set:
                trajectories[traj_idx]["sublimated_principle"] = rep
                first_set = True
            else:
                trajectories[traj_idx]["sublimated_principle"] = ""
                cleared_count += 1

    final_fired = [
        t["sublimated_principle"]
        for t in trajectories
        if t.get("fired") and t.get("sublimated_principle")
    ]
    print(f"  クラスタ数: {len(clusters)} (代表を保持)")
    print(f"  結果: {original_count} → {len(final_fired)} (除去: {cleared_count})")
    if traj_wrapper is not None:
        traj_wrapper["items"] = trajectories
        save_json(traj_file, traj_wrapper)
    else:
        save_json(traj_file, trajectories)

    # Print some cluster examples
    multi_clusters = [c for c in clusters if len(c) > 1]
    if multi_clusters:
        print(f"  統合クラスタ例 (上位5):")
        for cluster in sorted(multi_clusters, key=len, reverse=True)[:5]:
            texts = [fired_texts[idx] for idx in cluster]
            rep = pick_representative(texts)
            print(f"    [{len(cluster)}件→1] 代表: {rep[:50]}")

    # ── 2. Abstracted principles (list of dicts) ─────────────
    for law_file_name in [
        "ghost_abstracted_principles.json",
        "ghost_universal_laws.json",
        "ghost_positive_laws.json",
    ]:
        law_file = BASE_DIR / law_file_name
        laws = load_json(law_file)
        if not laws:
            continue
        original_count = len(laws)

        if laws and isinstance(laws[0], str):
            cleaned = [sanitize_principle(s) for s in laws if isinstance(s, str)]
            cleaned = [s for s in cleaned if s]
            # Cluster dedup
            clusters = cluster_principles(cleaned, threshold=0.45)
            deduped = []
            for cluster in clusters:
                texts = [cleaned[i] for i in cluster]
                deduped.append(pick_representative(texts))
            cleaned = deduped
        elif laws and isinstance(laws[0], dict):
            for law in laws:
                for key in ("principle", "statement", "specific", "universal"):
                    if key in law and isinstance(law[key], str):
                        law[key] = sanitize_principle(law[key])
            text_key = next(
                (k for k in ("principle", "statement", "universal") if k in laws[0]),
                None,
            )
            if text_key:
                texts = [law.get(text_key, "") for law in laws]
                clusters = cluster_principles(texts, threshold=0.45)
                cleaned = []
                for cluster in clusters:
                    # Keep the one with best text
                    cluster_laws = [laws[i] for i in cluster]
                    cluster_laws.sort(key=lambda l: len(l.get(text_key, "")))
                    cleaned.append(cluster_laws[0])
            else:
                cleaned = laws
        else:
            cleaned = laws

        print(f"  [{law_file_name}] {original_count} → {len(cleaned)}")
        save_json(law_file, cleaned)

    # ── 3. Preference rules ───────────────────────────────────
    rules_file = BASE_DIR / "preference_rules.json"
    rules_raw = load_json(rules_file)
    # Handle both formats: list or {"schema_version": ..., "items": [...]}
    rules_wrapper = None
    if isinstance(rules_raw, dict) and "items" in rules_raw:
        rules_wrapper = rules_raw
        rules = rules_raw["items"]
    elif isinstance(rules_raw, list):
        rules = rules_raw
    else:
        rules = []
    print(f"\n[選好ルール] 修正前: {len(rules)}件")

    selfref_fixes = 0
    confidence_changes = 0
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        neg = rule.get("negative_conditions", [])
        stmt = rule.get("statement", "")
        instr = rule.get("instruction", "")
        new_neg = [n for n in neg if n != stmt and n != instr]
        if len(new_neg) != len(neg):
            rule["negative_conditions"] = new_neg
            selfref_fixes += 1

        old_conf = rule.get("confidence_score", 0)
        new_conf = recalc_confidence(rule)
        if abs(old_conf - new_conf) > 0.001:
            rule["confidence_score"] = new_conf
            confidence_changes += 1

    print(f"  自己参照修正: {selfref_fixes}件")
    print(f"  confidence再計算: {confidence_changes}件")
    if rules_wrapper is not None:
        rules_wrapper["items"] = rules
        save_json(rules_file, rules_wrapper)
    else:
        save_json(rules_file, rules)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("完了。rebuild_preference_rules() でルール統合を実行してください。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
