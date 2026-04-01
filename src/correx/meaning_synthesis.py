"""Meaning Layer — extract emergent judgment principles from rule clusters.

Rules are individual corrections. Meanings are the unstated principles
that emerge when multiple rules from different scopes converge on the
same judgment. This module detects those convergences automatically
without LLM calls or embeddings.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .memory_manager import _char_ngrams, _ngram_similarity
from .schemas import Meaning, Principle, PreferenceRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tag_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _combined_similarity(ra: PreferenceRule, rb: PreferenceRule) -> float:
    text_a = ra.instruction or ra.statement or ""
    text_b = rb.instruction or rb.statement or ""
    text_sim = _ngram_similarity(text_a, text_b)

    # Use only the most meaningful tags (first 6 per rule, skip very short ones)
    tags_a = [t for t in ra.tags[:6] if len(t) > 3]
    tags_b = [t for t in rb.tags[:6] if len(t) > 3]
    tag_sim = _tag_jaccard(tags_a, tags_b)

    # Cross-scope bonus: rules from different domains converging = strong signal
    cross_scope = (
        ra.applies_to_scope
        and rb.applies_to_scope
        and ra.applies_to_scope != rb.applies_to_scope
    )
    scope_bonus = 0.20 if cross_scope else 0.0

    return text_sim * 0.50 + tag_sim * 0.25 + scope_bonus


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_rules(
    rules: list[PreferenceRule],
    threshold: float = 0.25,
    min_cluster_size: int = 3,
    max_cluster_size: int = 15,
) -> list[list[PreferenceRule]]:
    """Group rules into clusters by combined similarity."""
    n = len(rules)
    if n < min_cluster_size:
        return []

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _combined_similarity(rules[i], rules[j]) >= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    clusters: list[list[PreferenceRule]] = []
    for indices in groups.values():
        if len(indices) < min_cluster_size:
            continue
        cluster = [rules[i] for i in indices]
        if len(cluster) <= max_cluster_size:
            clusters.append(cluster)
        else:
            # Split oversized cluster by scope
            by_scope: dict[str, list[PreferenceRule]] = {}
            for r in cluster:
                key = r.applies_to_scope or "_ungrouped"
                by_scope.setdefault(key, []).append(r)
            for sub in by_scope.values():
                if len(sub) >= min_cluster_size:
                    clusters.append(sub[:max_cluster_size])
    return clusters


# ---------------------------------------------------------------------------
# Principle extraction
# ---------------------------------------------------------------------------

def _extract_principle(cluster: list[PreferenceRule]) -> tuple[str, str]:
    """Pick the best rule's instruction as the principle text.

    Returns (principle, summary).
    """
    best = max(
        cluster,
        key=lambda r: (r.expected_gain or 0.0) * (r.confidence_score or 0.5) + (r.evidence_count or 0) * 0.1,
    )
    principle = best.instruction or best.statement or ""
    scopes = sorted({r.applies_to_scope for r in cluster if r.applies_to_scope})
    summary = f"[{len(cluster)}ルール / {len(scopes)}スコープ] {principle[:60]}"
    return principle, summary


# ---------------------------------------------------------------------------
# CLAUDE.md cross-reference
# ---------------------------------------------------------------------------

def _load_settings_lines(paths: list[Path] | None) -> list[str]:
    """Load non-empty, non-comment lines from CLAUDE.md files."""
    if not paths:
        candidates = [
            Path.home() / ".claude" / "CLAUDE.md",
        ]
        # Also check common project CLAUDE.md locations
        cwd = Path.cwd()
        candidates.append(cwd / "CLAUDE.md")
    else:
        candidates = paths

    lines: list[str] = []
    for p in candidates:
        try:
            if p.exists():
                for raw in p.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if line and not line.startswith("#") and not line.startswith("```") and len(line) > 10:
                        lines.append(line)
        except OSError:
            pass
    return lines


def _cross_reference_settings(
    principle: str,
    settings_lines: list[str],
    threshold: float = 0.18,
) -> list[str]:
    """Find CLAUDE.md lines that resonate with a principle."""
    overlaps: list[str] = []
    for line in settings_lines:
        sim = _ngram_similarity(principle, line)
        if sim >= threshold:
            overlaps.append(line[:120])
    return overlaps[:5]


# ---------------------------------------------------------------------------
# Meaning construction
# ---------------------------------------------------------------------------

def _make_meaning_id(principle: str) -> str:
    slug = re.sub(r"[^a-z0-9\u3040-\u9fff]+", "-", principle.lower())[:40]
    h = hashlib.sha256(principle.encode()).hexdigest()[:8]
    return f"meaning-{slug}-{h}"


def _build_meaning(
    cluster: list[PreferenceRule],
    settings_lines: list[str],
    now: str,
) -> Meaning:
    principle, summary = _extract_principle(cluster)
    scopes = sorted({r.applies_to_scope for r in cluster if r.applies_to_scope})
    all_tags = sorted({t for r in cluster for t in r.tags[:6]})[:20]
    conf_scores = [r.confidence_score for r in cluster if r.confidence_score is not None]
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.5
    strength = len(cluster)

    return Meaning(
        id=_make_meaning_id(principle),
        principle=principle,
        normalized_principle=_normalize_text(principle),
        summary=summary,
        source_rule_ids=[r.id for r in cluster],
        scopes=scopes,
        tags=all_tags,
        strength=strength,
        cross_scope_count=len(scopes),
        confidence=round(avg_conf * min(1.0, strength / 5), 4),
        first_seen_at=now,
        last_seen_at=now,
        personal_settings_overlap=_cross_reference_settings(principle, settings_lines),
        status="active",
    )


# ---------------------------------------------------------------------------
# Merge with existing meanings
# ---------------------------------------------------------------------------

def _merge_with_existing(
    new_meanings: list[Meaning],
    existing: list[Meaning] | None,
    now: str,
) -> list[Meaning]:
    if not existing:
        return new_meanings

    existing_by_id: dict[str, Meaning] = {m.id: m for m in existing}
    result: list[Meaning] = []

    for nm in new_meanings:
        # Check for same ID (exact match)
        if nm.id in existing_by_id:
            old = existing_by_id.pop(nm.id)
            nm.first_seen_at = old.first_seen_at
            nm.last_seen_at = now
            result.append(nm)
            continue

        # Check for high-similarity match with any existing
        merged = False
        for oid, old in list(existing_by_id.items()):
            if _ngram_similarity(nm.principle, old.principle) >= 0.4:
                # Merge: keep the stronger one
                if nm.strength >= old.strength:
                    nm.first_seen_at = old.first_seen_at
                    nm.last_seen_at = now
                    # Absorb source rules
                    all_ids = set(nm.source_rule_ids) | set(old.source_rule_ids)
                    nm.source_rule_ids = sorted(all_ids)
                    nm.strength = max(nm.strength, len(all_ids))
                    result.append(nm)
                else:
                    old.last_seen_at = now
                    all_ids = set(nm.source_rule_ids) | set(old.source_rule_ids)
                    old.source_rule_ids = sorted(all_ids)
                    old.strength = max(old.strength, len(all_ids))
                    result.append(old)
                existing_by_id.pop(oid)
                merged = True
                break
        if not merged:
            result.append(nm)

    # Keep existing meanings that didn't match any new one (mark dormant if stale)
    for old in existing_by_id.values():
        old.status = "dormant"
        result.append(old)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_meanings(
    rules: list[PreferenceRule],
    existing_meanings: list[Meaning] | None = None,
    claude_md_paths: list[Path] | None = None,
    threshold: float = 0.25,
    min_cluster_size: int = 3,
) -> list[Meaning]:
    """Detect emergent judgment principles from cross-scope rule clusters.

    Pure function: rules in, meanings out. No LLM calls, no embeddings.
    Uses character bigram Jaccard + tag Jaccard + cross-scope bonus.
    """
    if len(rules) < min_cluster_size:
        return list(existing_meanings) if existing_meanings else []

    now = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")
    settings_lines = _load_settings_lines(claude_md_paths)
    clusters = _cluster_rules(rules, threshold, min_cluster_size)
    new_meanings = [_build_meaning(c, settings_lines, now) for c in clusters]

    # Sort by strength (strongest first), then by cross-scope count
    new_meanings.sort(key=lambda m: (-m.strength, -m.cross_scope_count))

    return _merge_with_existing(new_meanings, existing_meanings, now)


# ---------------------------------------------------------------------------
# Deferred Meanings Pool — hold candidates below depth threshold
# ---------------------------------------------------------------------------

def extract_deferred_meanings(
    rules: list[PreferenceRule],
    promoted_meanings: list[Meaning],
    threshold: float = 0.20,
    min_cluster_size: int = 2,
) -> list[Meaning]:
    """Extract meaning candidates that didn't meet the promotion threshold.

    These are clusters of 2 rules (below the default min_cluster_size=3)
    or clusters with lower similarity. Stored in deferred_meanings.json
    for later re-activation when context changes.

    Backed by neuroscience: memory reconsolidation (2025) — deferred
    insights become labile and reactivatable when prediction error occurs.
    """
    if len(rules) < min_cluster_size:
        return []

    now = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")
    settings_lines = _load_settings_lines(None)

    # Find clusters that are weaker than promoted meanings
    clusters = _cluster_rules(rules, threshold, min_cluster_size)

    # Filter out clusters already represented in promoted meanings
    promoted_rule_ids = set()
    for m in promoted_meanings:
        promoted_rule_ids.update(m.source_rule_ids)

    deferred: list[Meaning] = []
    for cluster in clusters:
        cluster_ids = {r.id for r in cluster}
        # Skip if most rules are already in a promoted meaning
        overlap = cluster_ids & promoted_rule_ids
        if len(overlap) >= len(cluster_ids) * 0.5:
            continue
        meaning = _build_meaning(cluster, settings_lines, now)
        meaning.status = "deferred"
        deferred.append(meaning)

    return deferred


def reactivate_deferred(
    deferred: list[Meaning],
    current_scope: str,
    current_tags: list[str],
    threshold: float = 0.15,
) -> list[Meaning]:
    """Check if any deferred meanings should reactivate given new context.

    A deferred meaning reactivates when the current task context
    overlaps with the meaning's scope/tags but wasn't relevant before.

    Backed by: reconsolidation boundary conditions (dopamine-mediated
    prediction error triggers lability).
    """
    reactivated: list[Meaning] = []
    current_tag_set = set(current_tags)

    for meaning in deferred:
        if meaning.status != "deferred":
            continue

        # Scope match
        scope_match = current_scope in meaning.scopes if current_scope else False

        # Tag overlap
        meaning_tags = set(meaning.tags)
        tag_overlap = len(current_tag_set & meaning_tags) / max(1, len(current_tag_set | meaning_tags))

        # Text similarity to current context
        if scope_match or tag_overlap >= threshold:
            meaning.status = "active"
            reactivated.append(meaning)

    return reactivated


# ---------------------------------------------------------------------------
# Creative Destruction — weaken rules subsumed by new meanings
# ---------------------------------------------------------------------------

def apply_creative_destruction(
    rules: list[PreferenceRule],
    new_meanings: list[Meaning],
    metabolism_rate: float = 0.5,
) -> tuple[list[PreferenceRule], list[dict[str, str]]]:
    """Weaken rules that are fully subsumed by a new meaning.

    When a meaning emerges from a cluster of rules, the individual rules
    become partially redundant. This function reduces their priority
    proportionally to the meaning's strength.

    This implements "generation and destruction as a single transaction"
    backed by the synaptic homeostasis hypothesis (Tononi):
    new learning (meaning) triggers proportional downscaling (rule weakening).

    Aggressive users (high metabolism) destroy more; conservative users preserve more.
    """
    log: list[dict[str, str]] = []
    rules_by_id = {r.id: r for r in rules}

    # Destruction strength scales with metabolism
    # Conservative (0.0): reduce priority by 0.5 at most
    # Aggressive (1.0): reduce priority by 2.0
    destruction_strength = 0.5 + metabolism_rate * 1.5

    for meaning in new_meanings:
        if meaning.status != "active":
            continue
        if meaning.strength < 3:
            continue  # don't destroy for weak meanings

        for rule_id in meaning.source_rule_ids:
            rule = rules_by_id.get(rule_id)
            if rule is None or rule.status != "promoted":
                continue

            # Don't destroy the strongest rule in the cluster (it IS the meaning)
            if rule.instruction == meaning.principle:
                continue

            # Reduce priority proportionally (ensure at least 1 reduction for promoted rules)
            max_reduction = max(0, rule.priority - 1)
            reduction = min(destruction_strength, max_reduction)
            if reduction >= 0.5:
                rule.priority = max(1, round(rule.priority - reduction))
                if "subsumed" not in rule.tags:
                    rule.tags.append("subsumed")
                log.append({
                    "action": "weaken",
                    "rule_id": rule.id,
                    "meaning_id": meaning.id,
                    "reduction": f"{reduction:.1f}",
                    "instruction": rule.instruction[:40],
                })

    return rules, log


# ---------------------------------------------------------------------------
# Meaning-based rule consolidation
# ---------------------------------------------------------------------------

def consolidate_rules_by_meaning(
    rules: list[PreferenceRule],
    meanings: list[Meaning],
) -> list[PreferenceRule]:
    """Boost the strongest rule in each meaning cluster with sibling evidence.

    When multiple rules say the same thing in different words, they belong
    to the same meaning cluster. This function finds the strongest rule in
    each cluster and adds the siblings' evidence to it, accelerating promotion.

    Rules are NOT deleted — only the best rule's evidence is boosted.
    Returns the modified rules list.
    """
    if not meanings:
        return rules

    rules_by_id = {r.id: r for r in rules}
    boosted_ids: set[str] = set()

    for meaning in meanings:
        if meaning.status != "active":
            continue

        # Find rules in this cluster that still exist
        cluster_rules = [rules_by_id[rid] for rid in meaning.source_rule_ids if rid in rules_by_id]
        if len(cluster_rules) < 2:
            continue

        # Find the strongest rule (highest evidence + confidence)
        best = max(
            cluster_rules,
            key=lambda r: r.evidence_count * 10 + (r.confidence_score or 0.5) * 5 + (r.support_score or 0),
        )

        if best.id in boosted_ids:
            continue

        # Count sibling evidence (rules other than the best)
        siblings = [r for r in cluster_rules if r.id != best.id]
        sibling_evidence = sum(max(1, r.evidence_count) for r in siblings)

        # Boost: add sibling evidence (but cap at cluster size to avoid runaway)
        boost = min(sibling_evidence, len(cluster_rules))
        if boost > 0:
            best.evidence_count += boost
            best.support_score = (best.support_score or 0.0) + boost * 0.5
            # Tag to track consolidation
            if "meaning_consolidated" not in best.tags:
                best.tags.append("meaning_consolidated")
            boosted_ids.add(best.id)

    return rules


# ---------------------------------------------------------------------------
# Principle Layer — meaning of meanings
# ---------------------------------------------------------------------------

def _meaning_similarity(a: Meaning, b: Meaning) -> float:
    """Similarity between two meanings based on principle text + scope overlap."""
    text_sim = _ngram_similarity(a.principle, b.principle)
    scope_sim = _tag_jaccard(a.scopes, b.scopes)
    # Different source rules = more interesting convergence
    rule_overlap = _tag_jaccard(a.source_rule_ids, b.source_rule_ids)
    independence_bonus = 0.15 if rule_overlap < 0.3 else 0.0
    return text_sim * 0.45 + scope_sim * 0.25 + independence_bonus


def synthesize_principles(
    meanings: list[Meaning],
    claude_md_paths: list[Path] | None = None,
    threshold: float = 0.22,
    min_cluster_size: int = 2,
) -> list[Principle]:
    """Extract higher-order principles from meaning clusters.

    Principles are the 'who you are' level — emergent from meanings
    the same way meanings emerge from rules. If meanings say 'why you
    should do this', principles say 'what kind of person you are'.
    """
    active = [m for m in meanings if m.status == "active"]
    if len(active) < min_cluster_size:
        return []

    now = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")
    settings_lines = _load_settings_lines(claude_md_paths)

    # Cluster meanings by similarity
    n = len(active)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _meaning_similarity(active[i], active[j]) >= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    principles: list[Principle] = []
    for indices in groups.values():
        if len(indices) < min_cluster_size:
            continue

        cluster = [active[i] for i in indices]
        # Pick the strongest meaning's principle as the declaration seed
        best = max(cluster, key=lambda m: m.strength * m.confidence)
        declaration = best.principle

        all_scopes = sorted({s for m in cluster for s in m.scopes})
        all_meaning_ids = [m.id for m in cluster]
        total_rules = len({rid for m in cluster for rid in m.source_rule_ids})
        avg_conf = sum(m.confidence for m in cluster) / len(cluster)
        overlaps = _cross_reference_settings(declaration, settings_lines)

        pid = _make_meaning_id(f"principle-{declaration}")
        principles.append(Principle(
            id=pid,
            declaration=declaration,
            normalized_declaration=_normalize_text(declaration),
            source_meaning_ids=all_meaning_ids,
            source_rule_count=total_rules,
            depth=3,
            scopes=all_scopes,
            confidence=round(avg_conf, 4),
            first_seen_at=now,
            last_seen_at=now,
            personal_settings_overlap=overlaps,
            status="active",
        ))

    principles.sort(key=lambda p: (-p.source_rule_count, -len(p.source_meaning_ids)))
    return principles
