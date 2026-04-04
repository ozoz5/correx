"""CuriosityEngine — Third learning layer: knowledge gap detection from questions.

The Surface Layer learns from explicit corrections.
The Ghost Layer learns from the gap between proposed and wanted.
The Curiosity Layer learns from questions — catching problems UPSTREAM
before they become corrections or anger.

Causal chain intercepted:
  question → repetition → resignation → anger → abandonment

Architecture (inverted):
  - Client LLM detects questions, classifies them (3 types + target)
  - Server (this module) stores signals, clusters them by keyword overlap,
    computes escalation scores, and builds the cognitive map
  - Client LLM reads the cognitive map and decides when to intervene

No LLM calls, no embeddings, no external API. Pure data pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import Any

from .schemas import CuriositySignal, KnowledgeGapCluster


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_QUESTION_TYPES = {"knowledge_gap", "judgment_uncertainty", "confirmation_seeking"}
VALID_TARGETS = {"self", "other"}
VALID_STATUSES = {"open", "resolved", "escalated"}

# Weights for escalation score computation
_REPEAT_WEIGHT = 0.25       # each repeat adds this much
_DENSITY_WEIGHT = 0.15      # high signal density boosts escalation
_TYPE_ESCALATION = {
    "knowledge_gap": 1.0,           # most urgent — user doesn't know
    "judgment_uncertainty": 0.7,    # medium — user can't decide
    "confirmation_seeking": 0.3,    # low — user just wants reassurance
}


# ---------------------------------------------------------------------------
# ID / timestamp helpers
# ---------------------------------------------------------------------------

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")


def _make_id(seed: str) -> str:
    h = hashlib.md5(
        f"{seed}-{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()
    return h[:16]


# ---------------------------------------------------------------------------
# Serialization (signal)
# ---------------------------------------------------------------------------

def signal_to_dict(signal: CuriositySignal) -> dict[str, Any]:
    return asdict(signal)


def signal_from_dict(d: dict[str, Any]) -> CuriositySignal:
    valid_fields = {f.name for f in fields(CuriositySignal)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    filtered.setdefault("id", "")
    filtered.setdefault("created_at", "")
    return CuriositySignal(**filtered)


# ---------------------------------------------------------------------------
# Serialization (cluster)
# ---------------------------------------------------------------------------

def cluster_to_dict(cluster: KnowledgeGapCluster) -> dict[str, Any]:
    return asdict(cluster)


def cluster_from_dict(d: dict[str, Any]) -> KnowledgeGapCluster:
    valid_fields = {f.name for f in fields(KnowledgeGapCluster)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    filtered.setdefault("id", "")
    filtered.setdefault("created_at", "")
    filtered.setdefault("updated_at", "")
    return KnowledgeGapCluster(**filtered)


# ---------------------------------------------------------------------------
# Signal creation
# ---------------------------------------------------------------------------

def create_signal(
    *,
    question_text: str,
    question_type: str,
    target: str = "self",
    task_scope: str = "",
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    confidence: float = 0.0,
    source_turn_id: str = "",
) -> CuriositySignal:
    """Create a new CuriositySignal from client LLM classification."""
    if question_type not in VALID_QUESTION_TYPES:
        question_type = "knowledge_gap"
    if target not in VALID_TARGETS:
        target = "self"

    now = _now_str()
    return CuriositySignal(
        id=_make_id(question_text),
        created_at=now,
        question_text=question_text.strip(),
        question_type=question_type,
        target=target,
        source_turn_id=source_turn_id,
        task_scope=task_scope,
        tags=list(tags or []),
        keywords=list(keywords or []),
        confidence=max(0.0, min(1.0, confidence)),
    )


# ---------------------------------------------------------------------------
# Clustering (keyword overlap — no LLM needed)
# ---------------------------------------------------------------------------

def _keyword_overlap(a: list[str], b: list[str]) -> float:
    """Jaccard similarity on keyword sets (lowered)."""
    sa = {k.lower() for k in a if k}
    sb = {k.lower() for k in b if k}
    if not sa or not sb:
        return 0.0
    intersection = sa & sb
    union = sa | sb
    return len(intersection) / len(union) if union else 0.0


def _char_bigram_similarity(a: str, b: str) -> float:
    """Char-bigram Jaccard for Japanese text."""
    if not a or not b:
        return 0.0
    sa = {a[i:i + 2] for i in range(len(a) - 1)} if len(a) > 1 else {a}
    sb = {b[i:i + 2] for i in range(len(b) - 1)} if len(b) > 1 else {b}
    intersection = sa & sb
    union = sa | sb
    return len(intersection) / len(union) if union else 0.0


def _cluster_similarity(signal: CuriositySignal, cluster: KnowledgeGapCluster) -> float:
    """Combined similarity score for signal-to-cluster matching."""
    # Keyword overlap: 50%
    kw_sim = _keyword_overlap(signal.keywords, cluster.theme_keywords)

    # Scope match: 30%
    scope_match = 1.0 if signal.task_scope and signal.task_scope == cluster.scope else 0.0

    # Question text vs theme keywords bigram: 20%
    theme_text = " ".join(cluster.theme_keywords)
    text_sim = _char_bigram_similarity(signal.question_text.lower(), theme_text.lower())

    return kw_sim * 0.5 + scope_match * 0.3 + text_sim * 0.2


def assign_signal_to_cluster(
    signal: CuriositySignal,
    clusters: list[KnowledgeGapCluster],
    threshold: float = 0.25,
) -> tuple[KnowledgeGapCluster, bool]:
    """Find the best matching cluster or create a new one.

    Returns (cluster, is_new).
    """
    best_cluster = None
    best_score = 0.0

    # Match against both open AND escalated clusters (not resolved)
    active_clusters = [c for c in clusters if c.status in ("open", "escalated")]
    for cluster in active_clusters:
        score = _cluster_similarity(signal, cluster)
        if score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is not None and best_score >= threshold:
        return best_cluster, False

    # Create new cluster
    now = _now_str()
    new_cluster = KnowledgeGapCluster(
        id=_make_id(signal.question_text + signal.task_scope),
        created_at=now,
        updated_at=now,
        scope=signal.task_scope,
        theme_keywords=list(signal.keywords),
        dominant_type=signal.question_type,
        signal_ids=[],
        signal_count=0,
        repeat_count=0,
        escalation_score=0.0,
        gap_strength=0.0,
        status="open",
        scopes=[signal.task_scope] if signal.task_scope else [],
    )
    return new_cluster, True


def add_signal_to_cluster(
    signal: CuriositySignal,
    cluster: KnowledgeGapCluster,
) -> KnowledgeGapCluster:
    """Add a signal to a cluster and update cluster metrics."""
    now = _now_str()

    # Link signal to cluster
    signal.cluster_id = cluster.id

    # Update cluster
    if signal.id not in cluster.signal_ids:
        cluster.signal_ids.append(signal.id)
    cluster.signal_count = len(cluster.signal_ids)
    cluster.updated_at = now

    # Merge keywords (keep unique, max 20)
    existing = set(cluster.theme_keywords)
    for kw in signal.keywords:
        if kw not in existing:
            cluster.theme_keywords.append(kw)
            existing.add(kw)
    cluster.theme_keywords = cluster.theme_keywords[:20]

    # Merge scopes
    if signal.task_scope and signal.task_scope not in cluster.scopes:
        cluster.scopes.append(signal.task_scope)

    # Update dominant type — track counts, pick majority
    type_counts: dict[str, int] = {}
    # Count from existing signals (approximate: we only store dominant_type + current signal)
    # Increment existing count
    if cluster.dominant_type:
        type_counts[cluster.dominant_type] = max(1, cluster.signal_count - 1)
    type_counts[signal.question_type] = type_counts.get(signal.question_type, 0) + 1
    cluster.dominant_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

    # Repeat count: signals beyond the first are repeats
    cluster.repeat_count = max(0, cluster.signal_count - 1)

    # Recompute escalation and gap strength
    cluster.escalation_score = compute_escalation_score(cluster)
    cluster.gap_strength = compute_gap_strength(cluster)

    # Auto-escalate: score threshold OR high repeat count (catches low-weight types)
    if cluster.status == "open" and (
        cluster.escalation_score >= 0.7 or cluster.repeat_count >= 4
    ):
        cluster.status = "escalated"

    return cluster


# ---------------------------------------------------------------------------
# Escalation & gap strength
# ---------------------------------------------------------------------------

def compute_escalation_score(cluster: KnowledgeGapCluster) -> float:
    """Compute escalation score (0.0=calm, 1.0=about to give up).

    Factors:
    - repeat_count: more repeats = higher frustration
    - signal density: rapid-fire questions = urgent
    - question type: knowledge_gap escalates faster than confirmation_seeking
    """
    repeat = cluster.repeat_count
    type_weight = _TYPE_ESCALATION.get(cluster.dominant_type, 0.5)

    # Base escalation from repeats (saturates around 5 repeats)
    repeat_factor = min(1.0, repeat * _REPEAT_WEIGHT)

    # Density bonus: many signals = rapid questioning
    density_factor = min(1.0, cluster.signal_count * _DENSITY_WEIGHT)

    # Combine with type weighting
    raw = (repeat_factor * 0.6 + density_factor * 0.4) * type_weight

    return min(1.0, max(0.0, raw))


def compute_gap_strength(cluster: KnowledgeGapCluster) -> float:
    """Compute gap strength — how significant this knowledge gap is.

    Higher for knowledge_gap type with many signals.
    Lower for confirmation_seeking with few signals.
    """
    type_weight = _TYPE_ESCALATION.get(cluster.dominant_type, 0.5)
    signal_factor = min(1.0, cluster.signal_count / 5.0)
    return min(1.0, type_weight * 0.6 + signal_factor * 0.4)


# ---------------------------------------------------------------------------
# Cluster resolution
# ---------------------------------------------------------------------------

def resolve_cluster(cluster: KnowledgeGapCluster) -> KnowledgeGapCluster:
    """Mark a cluster as resolved (user's question was answered satisfactorily)."""
    cluster.status = "resolved"
    cluster.resolved_at = _now_str()
    return cluster


# ---------------------------------------------------------------------------
# Cognitive map
# ---------------------------------------------------------------------------

def build_cognitive_map(clusters: list[KnowledgeGapCluster]) -> dict[str, Any]:
    """Build a scope-level knowledge gap map.

    Returns:
        {
            "scopes": {
                "scope_name": {
                    "gap_strength": float,
                    "dominant_type": str,
                    "open_clusters": int,
                    "escalated_clusters": int,
                    "total_questions": int,
                }
            },
            "total_open": int,
            "total_escalated": int,
            "hotspots": ["scope1", "scope2"],  # highest escalation
        }
    """
    scopes: dict[str, dict[str, Any]] = {}

    for cluster in clusters:
        if cluster.status == "resolved":
            continue

        scope = cluster.scope or "unknown"
        if scope not in scopes:
            scopes[scope] = {
                "gap_strength": 0.0,
                "dominant_type": "",
                "open_clusters": 0,
                "escalated_clusters": 0,
                "total_questions": 0,
            }

        entry = scopes[scope]
        entry["total_questions"] += cluster.signal_count
        if cluster.status == "open":
            entry["open_clusters"] += 1
        elif cluster.status == "escalated":
            entry["escalated_clusters"] += 1

        # Take max gap_strength across clusters in same scope
        if cluster.gap_strength > entry["gap_strength"]:
            entry["gap_strength"] = cluster.gap_strength
            entry["dominant_type"] = cluster.dominant_type

    total_open = sum(s["open_clusters"] for s in scopes.values())
    total_escalated = sum(s["escalated_clusters"] for s in scopes.values())

    # Hotspots: scopes with escalated clusters or high gap strength
    hotspots = sorted(
        [s for s, v in scopes.items() if v["escalated_clusters"] > 0 or v["gap_strength"] >= 0.5],
        key=lambda s: scopes[s]["gap_strength"],
        reverse=True,
    )

    return {
        "scopes": scopes,
        "total_open": total_open,
        "total_escalated": total_escalated,
        "hotspots": hotspots,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_curiosity_signal(
    signal: CuriositySignal,
    clusters: list[KnowledgeGapCluster],
) -> tuple[CuriositySignal, KnowledgeGapCluster, bool]:
    """Full pipeline: assign signal to cluster, update metrics.

    Returns (updated_signal, updated_cluster, is_new_cluster).
    """
    cluster, is_new = assign_signal_to_cluster(signal, clusters)
    cluster = add_signal_to_cluster(signal, cluster)
    return signal, cluster, is_new
