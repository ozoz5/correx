"""Memory management: smart forgetting, compression, and semantic search.

Implements a brain-like memory hierarchy:
- Short-term: recent conversation turns (working memory)
- Long-term: contextual rules, consolidated episodes
- Forgetting: score-based eviction instead of FIFO
- Compression: merge similar rules
- Semantic search: character n-gram similarity for retrieval
- Forgetting curve: Ebbinghaus-style priority decay
- Association: similarity-based rule linking
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .conversation_learning import extract_keywords, normalize_text
from .schemas import (
    ConversationTurn,
    CorrectionRecord,
    EpisodeRecord,
    LatentContext,
    LatentTransition,
    PreferenceRule,
    RuleContext,
)


# ---------------------------------------------------------------------------
# 1. Smart Forgetting — score-based eviction
# ---------------------------------------------------------------------------

def _turn_retention_score(turn: ConversationTurn) -> float:
    """Score a turn's value for retention. Higher = keep longer."""
    score = 0.0

    # Turns with corrections are more valuable
    score += len(turn.extracted_corrections) * 0.3

    # Turns that led to guidance are valuable
    if turn.guidance_applied:
        score += 0.5

    # High reaction scores (positive feedback) are valuable
    if turn.reaction_score is not None:
        if turn.reaction_score >= 0.8:
            score += 0.4  # strong positive
        elif turn.reaction_score <= 0.3:
            score += 0.2  # negative feedback has learning value

    # Turns with rich tags are more connected
    score += min(0.3, len(turn.tags) * 0.03)

    return score


def _episode_retention_score(episode: EpisodeRecord) -> float:
    """Score an episode's value for retention. Higher = keep longer."""
    score = 0.0

    # Episodes with corrections are much more valuable
    score += len(episode.corrections) * 0.5

    # Episodes with training examples are extremely valuable
    if episode.training_example and episode.training_example.accepted:
        score += 2.0

    # Episodes with rich metadata are more connected
    if episode.issuer:
        score += 0.2
    if episode.source_text:
        score += 0.1

    return score


def select_turns_for_eviction(
    turns: list[ConversationTurn],
    retention_limit: int = 200,
) -> list[str]:
    """Select low-value turns for eviction. Returns IDs to remove."""
    overflow = len(turns) - retention_limit
    if overflow <= 0:
        return []

    # Score each turn, evict lowest-scoring first
    scored = [(turn.id, _turn_retention_score(turn)) for turn in turns]
    scored.sort(key=lambda x: x[1])
    return [turn_id for turn_id, _ in scored[:overflow]]


def select_episodes_for_eviction(
    episodes: list[EpisodeRecord],
    retention_limit: int = 50,
) -> list[str]:
    """Select low-value episodes for eviction. Returns IDs to remove."""
    overflow = len(episodes) - retention_limit
    if overflow <= 0:
        return []

    scored = [(ep.id, _episode_retention_score(ep)) for ep in episodes]
    scored.sort(key=lambda x: x[1])
    return [ep_id for ep_id, _ in scored[:overflow]]


def evict_turns(
    turns: list[ConversationTurn],
    retention_limit: int = 200,
) -> list[ConversationTurn]:
    """Return turns with low-value items evicted."""
    evict_ids = set(select_turns_for_eviction(turns, retention_limit))
    if not evict_ids:
        return turns
    return [t for t in turns if t.id not in evict_ids]


def evict_episodes(
    episodes: list[EpisodeRecord],
    retention_limit: int = 50,
) -> list[EpisodeRecord]:
    """Return episodes with low-value items evicted."""
    evict_ids = set(select_episodes_for_eviction(episodes, retention_limit))
    if not evict_ids:
        return episodes
    return [e for e in episodes if e.id not in evict_ids]


def archive_turns_to_episode(
    turns: list[ConversationTurn],
) -> EpisodeRecord | None:
    """Archive a list of conversation turns into a single EpisodeRecord.

    Collects all extracted_corrections from the turns, wraps them as
    CorrectionRecord objects, and returns an EpisodeRecord that summarises
    the batch.  Returns None when the turn list is empty.
    """
    if not turns:
        return None

    # Collect all corrections from every turn
    corrections: list[CorrectionRecord] = []
    for turn in turns:
        for correction_text in turn.extracted_corrections:
            if correction_text.strip():
                corrections.append(
                    CorrectionRecord(
                        recorded_at=turn.recorded_at,
                        correction_note=correction_text.strip(),
                        scope=turn.task_scope,
                    )
                )

    # Determine date range for the title
    dates = [t.recorded_at for t in turns if t.recorded_at]
    earliest_date = min(dates) if dates else ""
    latest_date = max(dates) if dates else ""

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S%f")

    return EpisodeRecord(
        id=f"archive-{timestamp}",
        timestamp=now.strftime("%Y/%m/%d %H:%M"),
        title=f"アーカイブ {earliest_date} 〜 {latest_date} ({len(turns)}ターン)",
        task_type="archived_turns",
        source_text=f"{len(turns)}件の会話ターンを圧縮。修正{len(corrections)}件を保持。",
        corrections=corrections,
    )


# ---------------------------------------------------------------------------
# 2. Rule Compression — merge similar rules
# ---------------------------------------------------------------------------

from .text_similarity import ngram_jaccard as _ngram_similarity  # noqa: E402


def derive_context_mode(
    *,
    distinct_scope_count: int,
    distinct_tag_count: int,
    evidence_count: int,
) -> str:
    """Estimate whether a rule is local to one situation or broadly reusable."""
    if distinct_scope_count >= 2:
        return "general"
    if distinct_scope_count == 1 and evidence_count >= 3 and distinct_tag_count >= 5:
        return "mixed"
    if distinct_scope_count == 0 and distinct_tag_count >= 6:
        return "mixed"
    return "local"


def derive_latent_context_confidence_score(
    *,
    evidence_count: float,
    support_score: float = 0.0,
    posterior_mass: float = 0.0,
    strong_signal_count: float = 0.0,
    success_mass: float = 0.0,
    failure_mass: float = 0.0,
) -> float:
    """Estimate how trustworthy an inferred latent situation is."""
    confidence = 0.08
    confidence += min(0.34, max(0.0, evidence_count) * 0.12)
    confidence += min(0.2, max(0.0, support_score) * 0.04)
    confidence += min(0.14, max(0.0, posterior_mass) * 0.06)
    confidence += min(0.14, max(0.0, success_mass + failure_mass) * 0.07)
    if strong_signal_count > 0:
        confidence += 0.08
    return round(min(1.0, confidence), 4)


def derive_latent_context_prior_weight(
    *,
    evidence_count: float,
    support_score: float,
    expected_gain: float,
    confidence_score: float,
    posterior_mass: float = 0.0,
) -> float:
    """Estimate prior plausibility for selecting a latent context before observing the task."""
    prior = 0.05
    prior += min(0.28, max(0.0, evidence_count) * 0.08)
    prior += min(0.22, max(0.0, support_score) * 0.03)
    prior += min(0.2, max(0.0, expected_gain) * 0.05)
    prior += min(0.18, max(0.0, posterior_mass) * 0.04)
    prior += min(0.17, max(0.0, confidence_score) * 0.22)
    return round(min(1.0, prior), 4)


def derive_latent_context_expected_gain(
    *,
    support_score: float,
    confidence_score: float,
    strong_signal_count: float = 0.0,
    success_mass: float = 0.0,
    failure_mass: float = 0.0,
) -> float:
    """Estimate expected gain for a specific latent situation."""
    outcome_delta = max(0.0, success_mass) * 1.25 - max(0.0, failure_mass) * 0.75
    raw = support_score * 0.52 + max(0.0, strong_signal_count) * 0.3 + outcome_delta
    expected_gain = raw * (0.72 + confidence_score * 0.52)
    return round(max(0.0, expected_gain), 4)


def derive_transition_confidence_score(
    *,
    evidence_count: float,
    success_weight: float = 0.0,
    failure_weight: float = 0.0,
) -> float:
    """Estimate how much trust to place in a learned context transition."""
    confidence = 0.08
    confidence += min(0.36, max(0.0, evidence_count) * 0.12)
    confidence += min(0.24, max(0.0, success_weight) * 0.1)
    confidence -= min(0.18, max(0.0, failure_weight) * 0.07)
    return round(min(1.0, max(0.0, confidence)), 4)


def derive_transition_forecast_score(
    *,
    prediction_hit_count: float = 0.0,
    prediction_miss_count: float = 0.0,
) -> float:
    """Estimate how trustworthy a transition forecast is after observing its hits and misses."""
    total = max(0.0, prediction_hit_count) + max(0.0, prediction_miss_count)
    if total <= 0:
        return 0.0

    balance = (max(0.0, prediction_hit_count) - max(0.0, prediction_miss_count) * 0.92) / (total + 1.0)
    coverage = min(1.0, total / 6.0)
    score = balance * (0.45 + coverage * 0.8)
    return round(max(-1.0, min(1.0, score)), 4)


def build_context_signature(
    scope: str = "",
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
) -> str:
    """Build a stable signature for a latent situation across sessions."""
    normalized_scope = normalize_text(scope)
    normalized_tags = [
        normalize_text(tag)
        for tag in (tags or [])
        if normalize_text(tag)
    ]
    normalized_keywords = [
        normalize_text(keyword)
        for keyword in (keywords or [])
        if normalize_text(keyword)
    ]
    unique_tags = list(dict.fromkeys(normalized_tags))[:4]
    unique_keywords = list(dict.fromkeys(normalized_keywords))[:4]
    return "|".join(
        [
            normalized_scope or "_",
            "/".join(unique_tags) or "_",
            "/".join(unique_keywords) or "_",
        ]
    )


def derive_context_confidence_score(
    *,
    evidence_count: int,
    strong_signal_count: int = 0,
    success_count: int = 0,
    failure_count: int = 0,
) -> float:
    """Estimate how reliable a context signal is."""
    confidence = 0.12
    confidence += min(0.48, max(0, evidence_count) * 0.16)
    confidence += min(0.22, max(0, success_count + failure_count) * 0.08)
    if strong_signal_count > 0:
        confidence += 0.12
    return round(min(1.0, confidence), 4)


def derive_rule_confidence_score(
    *,
    evidence_count: int,
    distinct_scope_count: int,
    distinct_tag_count: int,
    strong_signal_count: int = 0,
    success_count: int = 0,
    failure_count: int = 0,
) -> float:
    """Estimate how trustworthy a rule's learned value is.

    Anti-overfitting: single-evidence rules are capped at 0.6 confidence
    to prevent premature high-confidence from tag/scope inflation alone.
    """
    confidence = 0.1
    confidence += min(0.42, max(0, evidence_count) * 0.11)
    confidence += min(0.18, max(0, distinct_scope_count) * 0.09)
    confidence += min(0.15, max(0, distinct_tag_count) * 0.03)
    confidence += min(0.15, max(0, success_count + failure_count) * 0.05)
    if strong_signal_count > 0:
        confidence += 0.1
    # Anti-overfitting: single-evidence rules cannot exceed 0.6
    cap = 1.0 if evidence_count >= 2 else 0.6
    return round(min(cap, confidence), 4)


def _aggregate_context_utility(contexts: list[RuleContext]) -> float:
    if not contexts:
        return 0.0
    ranked = sorted(
        (max(0.0, float(context.utility_score)) for context in contexts),
        reverse=True,
    )
    top = ranked[:3]
    return round(sum(top) / max(1, len(top)), 4)


def derive_rule_expected_gain(
    *,
    support_score: float,
    context_utility: float,
    confidence_score: float,
    strong_signal_count: int = 0,
    success_count: int = 0,
    failure_count: int = 0,
    context_mode: str = "local",
) -> float:
    """Estimate how much a rule is likely to help when the context matches."""
    mode_multiplier = {
        "local": 0.95,
        "mixed": 1.05,
        "general": 1.12,
    }.get(context_mode, 1.0)
    outcome_delta = max(0, success_count) * 1.4 - max(0, failure_count) * 0.7
    raw = support_score * 0.35 + context_utility * 0.55 + strong_signal_count * 0.35 + outcome_delta
    expected_gain = raw * (0.7 + confidence_score * 0.5) * mode_multiplier
    return round(max(0.0, expected_gain), 4)


def derive_rule_status(
    *,
    evidence_count: int,
    support_score: float,
    strong_signal_count: int,
    context_mode: str,
    expected_gain: float = 0.0,
    confidence_score: float = 0.0,
    min_promoted_evidence: int = 1,
) -> str:
    """Keep compatibility with candidate/promoted while using contextual confidence.

    Thresholds relaxed 2026-04-04: evidence >= 1 is enough to promote.
    The dormancy/awakening ecosystem prevents overfitting — policies absorb
    generic rules automatically, so the entry gate can be wide open.
    """
    if strong_signal_count >= 1 and context_mode == "local" and expected_gain >= 0.8:
        return "promoted"
    if confidence_score >= 0.5 and expected_gain >= 1.0 and evidence_count >= 1:
        return "promoted"
    if support_score >= 2.0 and expected_gain >= 1.0 and evidence_count >= 1:
        return "promoted"
    required_evidence = max(min_promoted_evidence, 2 if context_mode == "general" else 1)
    if evidence_count >= required_evidence and support_score >= 2.0:
        return "promoted"
    if evidence_count >= required_evidence and support_score >= 1.5 and expected_gain >= 0.8:
        return "promoted"
    return "candidate"


def merge_rule_contexts(contexts: list[RuleContext]) -> list[RuleContext]:
    """Merge duplicated context signals by kind/value."""
    merged: dict[tuple[str, str], RuleContext] = {}
    for context in contexts:
        key = (context.kind.strip(), context.value.strip())
        if not key[0] or not key[1]:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = RuleContext(
                kind=key[0],
                value=key[1],
                evidence_count=max(0, int(context.evidence_count)),
                reaction_min=context.reaction_min,
                reaction_max=context.reaction_max,
                last_seen_at=context.last_seen_at,
                utility_score=max(0.0, float(context.utility_score)),
                confidence_score=max(0.0, float(context.confidence_score)),
                strong_signal_count=max(0, int(context.strong_signal_count)),
                success_count=max(0, int(context.success_count)),
                failure_count=max(0, int(context.failure_count)),
            )
            continue
        existing.evidence_count += max(0, int(context.evidence_count))
        if context.reaction_min is not None:
            if existing.reaction_min is None or context.reaction_min < existing.reaction_min:
                existing.reaction_min = context.reaction_min
        if context.reaction_max is not None:
            if existing.reaction_max is None or context.reaction_max > existing.reaction_max:
                existing.reaction_max = context.reaction_max
        if context.last_seen_at > existing.last_seen_at:
            existing.last_seen_at = context.last_seen_at
        existing.utility_score += max(0.0, float(context.utility_score))
        existing.strong_signal_count += max(0, int(context.strong_signal_count))
        existing.success_count += max(0, int(context.success_count))
        existing.failure_count += max(0, int(context.failure_count))
    for context in merged.values():
        context.confidence_score = derive_context_confidence_score(
            evidence_count=context.evidence_count,
            strong_signal_count=context.strong_signal_count,
            success_count=context.success_count,
            failure_count=context.failure_count,
        )
    return sorted(
        merged.values(),
        key=lambda item: (-item.utility_score, item.kind, -item.evidence_count, item.value),
    )


def _latent_context_key(context: LatentContext) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (
        normalize_text(context.scope),
        tuple(sorted(normalize_text(tag) for tag in context.tags if normalize_text(tag))),
        tuple(sorted(normalize_text(keyword) for keyword in context.keywords if normalize_text(keyword)))[:4],
    )


def _latent_context_prototype_text(context: LatentContext) -> str:
    parts = [context.scope, " ".join(context.tags[:6]), " ".join(context.keywords[:6]), context.prototype_text]
    return " ".join(part.strip() for part in parts if str(part).strip()).strip()


def merge_latent_contexts(latent_contexts: list[LatentContext]) -> list[LatentContext]:
    """Merge duplicated latent situations by scope and cue set."""
    merged: dict[tuple[str, tuple[str, ...], tuple[str, ...]], LatentContext] = {}
    for context in latent_contexts:
        key = _latent_context_key(context)
        existing = merged.get(key)
        if existing is None:
            merged[key] = LatentContext(
                id=context.id or f"latent-{len(merged) + 1}",
                scope=context.scope,
                tags=list(dict.fromkeys(tag for tag in context.tags if tag)),
                keywords=list(dict.fromkeys(keyword for keyword in context.keywords if keyword))[:8],
                prototype_text=_latent_context_prototype_text(context),
                evidence_count=round(max(0.0, float(context.evidence_count)), 4),
                support_score=round(max(0.0, float(context.support_score)), 4),
                expected_gain=round(max(0.0, float(context.expected_gain)), 4),
                confidence_score=round(max(0.0, float(context.confidence_score)), 4),
                prior_weight=round(max(0.0, float(context.prior_weight)), 4),
                posterior_mass=round(max(0.0, float(context.posterior_mass)), 4),
                strong_signal_count=round(max(0.0, float(context.strong_signal_count)), 4),
                success_mass=round(max(0.0, float(context.success_mass)), 4),
                failure_mass=round(max(0.0, float(context.failure_mass)), 4),
                last_seen_at=context.last_seen_at,
            )
            continue
        existing.tags = list(dict.fromkeys(existing.tags + [tag for tag in context.tags if tag]))[:8]
        existing.keywords = list(
            dict.fromkeys(existing.keywords + [keyword for keyword in context.keywords if keyword])
        )[:8]
        existing.prototype_text = " ".join(
            part for part in [existing.prototype_text, _latent_context_prototype_text(context)] if part
        ).strip()
        existing.evidence_count = round(existing.evidence_count + max(0.0, float(context.evidence_count)), 4)
        existing.support_score = round(existing.support_score + max(0.0, float(context.support_score)), 4)
        existing.posterior_mass = round(existing.posterior_mass + max(0.0, float(context.posterior_mass)), 4)
        existing.strong_signal_count = round(
            existing.strong_signal_count + max(0.0, float(context.strong_signal_count)),
            4,
        )
        existing.success_mass = round(existing.success_mass + max(0.0, float(context.success_mass)), 4)
        existing.failure_mass = round(existing.failure_mass + max(0.0, float(context.failure_mass)), 4)
        if context.last_seen_at > existing.last_seen_at:
            existing.last_seen_at = context.last_seen_at
    for context in merged.values():
        context.confidence_score = derive_latent_context_confidence_score(
            evidence_count=context.evidence_count,
            support_score=context.support_score,
            posterior_mass=context.posterior_mass,
            strong_signal_count=context.strong_signal_count,
            success_mass=context.success_mass,
            failure_mass=context.failure_mass,
        )
        context.expected_gain = derive_latent_context_expected_gain(
            support_score=context.support_score,
            confidence_score=context.confidence_score,
            strong_signal_count=context.strong_signal_count,
            success_mass=context.success_mass,
            failure_mass=context.failure_mass,
        )
        context.prior_weight = derive_latent_context_prior_weight(
            evidence_count=context.evidence_count,
            support_score=context.support_score,
            expected_gain=context.expected_gain,
            confidence_score=context.confidence_score,
            posterior_mass=context.posterior_mass,
        )
    return sorted(
        merged.values(),
        key=lambda item: (-item.expected_gain, -item.prior_weight, -item.evidence_count, item.scope, item.id),
    )


def flatten_latent_contexts(latent_contexts: list[LatentContext]) -> list[RuleContext]:
    """Project latent situations into scope/tag signals for compatibility and indexing."""
    projected: list[RuleContext] = []
    for context in latent_contexts:
        evidence_count = max(1, round(context.evidence_count))
        strong_signal_count = max(0, round(context.strong_signal_count))
        success_count = max(0, round(context.success_mass))
        failure_count = max(0, round(context.failure_mass))
        if context.scope:
            projected.append(
                RuleContext(
                    kind="scope",
                    value=context.scope,
                    evidence_count=evidence_count,
                    last_seen_at=context.last_seen_at,
                    utility_score=round(max(0.0, context.expected_gain), 4),
                    confidence_score=round(max(0.0, context.confidence_score), 4),
                    strong_signal_count=strong_signal_count,
                    success_count=success_count,
                    failure_count=failure_count,
                )
            )
        for tag in context.tags[:6]:
            projected.append(
                RuleContext(
                    kind="tag",
                    value=tag,
                    evidence_count=max(1, round(max(1.0, context.posterior_mass))),
                    last_seen_at=context.last_seen_at,
                    utility_score=round(max(0.2, context.expected_gain * 0.72), 4),
                    confidence_score=round(max(0.0, context.confidence_score), 4),
                    strong_signal_count=strong_signal_count,
                    success_count=success_count,
                    failure_count=failure_count,
                )
            )
        for keyword in context.keywords[:4]:
            projected.append(
                RuleContext(
                    kind="tag",
                    value=keyword,
                    evidence_count=max(1, round(max(1.0, context.posterior_mass * 0.8))),
                    last_seen_at=context.last_seen_at,
                    utility_score=round(max(0.15, context.expected_gain * 0.55), 4),
                    confidence_score=round(max(0.0, context.confidence_score * 0.92), 4),
                    strong_signal_count=strong_signal_count,
                    success_count=success_count,
                    failure_count=failure_count,
                )
            )
    return merge_rule_contexts(projected)


def infer_latent_contexts_from_rule(rule: PreferenceRule) -> list[LatentContext]:
    """Reconstruct latent situations for older rules that only stored signal fragments."""
    if rule.latent_contexts:
        return merge_latent_contexts(rule.latent_contexts)

    scope_contexts = [context for context in rule.contexts if context.kind == "scope"]
    tag_values = list(dict.fromkeys(rule.applies_when_tags + rule.tags))[:6]
    reconstructed: list[LatentContext] = []
    if scope_contexts:
        for index, context in enumerate(scope_contexts, start=1):
            evidence_count = max(1.0, float(context.evidence_count))
            confidence_score = max(
                float(context.confidence_score),
                derive_latent_context_confidence_score(
                    evidence_count=evidence_count,
                    support_score=max(rule.support_score, evidence_count),
                    posterior_mass=evidence_count,
                    strong_signal_count=float(context.strong_signal_count),
                    success_mass=float(context.success_count),
                    failure_mass=float(context.failure_count),
                ),
            )
            expected_gain = max(
                float(context.utility_score),
                derive_latent_context_expected_gain(
                    support_score=max(rule.support_score, evidence_count),
                    confidence_score=confidence_score,
                    strong_signal_count=float(context.strong_signal_count),
                    success_mass=float(context.success_count),
                    failure_mass=float(context.failure_count),
                ),
            )
            reconstructed.append(
                LatentContext(
                    id=f"{rule.id}-latent-{index}",
                    scope=context.value,
                    tags=tag_values[:4],
                    keywords=tag_values[:4],
                    prototype_text=f"{context.value} {' '.join(tag_values[:4])}".strip(),
                    evidence_count=evidence_count,
                    support_score=max(rule.support_score, evidence_count),
                    expected_gain=expected_gain,
                    confidence_score=confidence_score,
                    prior_weight=derive_latent_context_prior_weight(
                        evidence_count=evidence_count,
                        support_score=max(rule.support_score, evidence_count),
                        expected_gain=expected_gain,
                        confidence_score=confidence_score,
                        posterior_mass=evidence_count,
                    ),
                    posterior_mass=evidence_count,
                    strong_signal_count=float(context.strong_signal_count),
                    success_mass=float(context.success_count),
                    failure_mass=float(context.failure_count),
                    last_seen_at=context.last_seen_at or rule.last_recorded_at,
                )
            )
    else:
        evidence_count = max(1.0, float(rule.evidence_count))
        confidence_score = max(
            float(rule.confidence_score),
            derive_latent_context_confidence_score(
                evidence_count=evidence_count,
                support_score=max(rule.support_score, evidence_count),
                posterior_mass=evidence_count,
                strong_signal_count=float(rule.strong_signal_count),
                success_mass=float(rule.success_count),
                failure_mass=float(rule.failure_count),
            ),
        )
        expected_gain = max(
            float(rule.expected_gain),
            derive_latent_context_expected_gain(
                support_score=max(rule.support_score, evidence_count),
                confidence_score=confidence_score,
                strong_signal_count=float(rule.strong_signal_count),
                success_mass=float(rule.success_count),
                failure_mass=float(rule.failure_count),
            ),
        )
        reconstructed.append(
            LatentContext(
                id=f"{rule.id}-latent-1",
                scope=rule.applies_to_scope,
                tags=tag_values[:4],
                keywords=tag_values[:4],
                prototype_text=" ".join(part for part in [rule.applies_to_scope, " ".join(tag_values[:4])] if part),
                evidence_count=evidence_count,
                support_score=max(rule.support_score, evidence_count),
                expected_gain=expected_gain,
                confidence_score=confidence_score,
                prior_weight=derive_latent_context_prior_weight(
                    evidence_count=evidence_count,
                    support_score=max(rule.support_score, evidence_count),
                    expected_gain=expected_gain,
                    confidence_score=confidence_score,
                    posterior_mass=evidence_count,
                ),
                posterior_mass=evidence_count,
                strong_signal_count=float(rule.strong_signal_count),
                success_mass=float(rule.success_count),
                failure_mass=float(rule.failure_count),
                last_seen_at=rule.last_recorded_at,
            )
        )

    return merge_latent_contexts(reconstructed)


def _aggregate_latent_context_gain(latent_contexts: list[LatentContext]) -> float:
    if not latent_contexts:
        return 0.0
    ranked = sorted(
        latent_contexts,
        key=lambda context: (
            context.expected_gain,
            context.prior_weight,
            context.confidence_score,
            context.posterior_mass,
        ),
        reverse=True,
    )[:3]
    total_weight = sum(max(0.12, context.prior_weight) for context in ranked)
    if total_weight <= 0:
        return 0.0
    weighted_gain = sum(context.expected_gain * max(0.12, context.prior_weight) for context in ranked)
    return round(weighted_gain / total_weight, 4)


@dataclass
class LatentContextMatch:
    context: LatentContext
    responsibility: float
    raw_score: float
    reason: str


def _normalize_context_node(node: dict | None) -> dict[str, object]:
    if not isinstance(node, dict):
        return {}
    scope = str(node.get("scope", "")).strip()
    tags = [
        normalize_text(str(tag).strip())
        for tag in node.get("tags", [])
        if normalize_text(str(tag).strip())
    ][:4]
    keywords = [
        normalize_text(str(keyword).strip())
        for keyword in node.get("keywords", [])
        if normalize_text(str(keyword).strip())
    ][:6]
    return {
        "context_id": str(node.get("context_id", "")).strip(),
        "scope": scope,
        "tags": tags,
        "keywords": keywords,
        "posterior": float(node.get("posterior", 0.0) or 0.0),
        "signature": str(node.get("signature", "")).strip()
        or build_context_signature(scope, tags, keywords),
    }


def _score_transition_prior(
    context: LatentContext,
    *,
    previous_context_nodes: list[dict] | None = None,
    transitions: list[LatentTransition] | None = None,
) -> tuple[float, str]:
    if not previous_context_nodes or not transitions:
        return 0.0, ""

    context_scope = normalize_text(context.scope)
    context_tags = {
        normalize_text(tag)
        for tag in context.tags
        if normalize_text(tag)
    }
    context_keywords = {
        normalize_text(keyword)
        for keyword in context.keywords
        if normalize_text(keyword)
    }
    best_bonus = 0.0
    best_reason = ""
    for raw_previous in previous_context_nodes:
        previous = _normalize_context_node(raw_previous)
        if not previous:
            continue
        previous_signature = str(previous.get("signature", "")).strip()
        if not previous_signature:
            continue
        previous_posterior = max(0.12, float(previous.get("posterior", 0.0) or 0.0))
        for transition in transitions:
            if transition.from_signature != previous_signature:
                continue
            transition_scope = normalize_text(transition.to_scope)
            transition_tags = {
                normalize_text(tag)
                for tag in transition.to_tags
                if normalize_text(tag)
            }
            transition_keywords = {
                normalize_text(keyword)
                for keyword in transition.to_keywords
                if normalize_text(keyword)
            }

            alignment = 0.0
            reasons: list[str] = []
            if transition_scope and context_scope:
                if transition_scope == context_scope:
                    alignment += 1.0
                    reasons.append(f"scope={transition.to_scope}")
                elif transition_scope in context_scope or context_scope in transition_scope:
                    alignment += 0.72
                    reasons.append(f"scope~={transition.to_scope}")
                else:
                    scope_similarity = semantic_similarity(transition_scope, context_scope)
                    if scope_similarity >= 0.2:
                        alignment += min(0.58, scope_similarity * 0.9)
                        reasons.append(f"scope_sem={scope_similarity:.2f}")

            tag_overlap = sorted(context_tags & transition_tags)
            if tag_overlap:
                alignment += min(0.54, len(tag_overlap) * 0.18)
                reasons.append("tags=" + "/".join(tag_overlap[:3]))

            keyword_overlap = sorted(context_keywords & transition_keywords)
            if keyword_overlap:
                alignment += min(0.42, len(keyword_overlap) * 0.12)
                reasons.append("kw=" + "/".join(keyword_overlap[:3]))

            if alignment <= 0:
                continue

            transition_strength = (
                max(0.0, float(transition.evidence_count)) * 0.18
                + max(0.0, float(transition.success_weight)) * 0.2
                - max(0.0, float(transition.failure_weight)) * 0.08
                + max(0.0, float(transition.confidence_score)) * 1.1
                + float(transition.forecast_score) * 0.95
                + max(0.0, float(transition.prediction_hit_count)) * 0.05
                - max(0.0, float(transition.prediction_miss_count)) * 0.04
            )
            bonus = alignment * max(0.0, transition_strength) * previous_posterior
            if bonus > best_bonus:
                best_bonus = bonus
                best_reason = (
                    f"transition:{previous.get('scope', '') or previous_signature}->{transition.to_scope or transition.to_signature}"
                    + (f" ({' | '.join(reasons)})" if reasons else "")
                )
    return round(min(2.4, best_bonus), 4), best_reason


def predict_next_contexts(
    *,
    previous_context_nodes: list[dict] | None = None,
    transitions: list[LatentTransition] | None = None,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Predict likely next latent situations from the current flow state."""
    if not previous_context_nodes or not transitions:
        return []

    aggregated: dict[str, dict[str, object]] = {}
    for raw_previous in previous_context_nodes:
        previous = _normalize_context_node(raw_previous)
        if not previous:
            continue
        previous_signature = str(previous.get("signature", "")).strip()
        if not previous_signature:
            continue
        previous_scope = str(previous.get("scope", "")).strip()
        previous_posterior = max(0.12, float(previous.get("posterior", 0.0) or 0.0))

        for transition in transitions:
            if transition.from_signature != previous_signature:
                continue
            score = previous_posterior * (
                max(0.0, float(transition.evidence_count)) * 0.16
                + max(0.0, float(transition.success_weight)) * 0.2
                - max(0.0, float(transition.failure_weight)) * 0.08
                + max(0.0, float(transition.confidence_score)) * 1.1
                + float(transition.forecast_score) * 0.95
                + max(0.0, float(transition.prediction_hit_count)) * 0.05
                - max(0.0, float(transition.prediction_miss_count)) * 0.04
            )
            if score <= 0:
                continue

            bucket = aggregated.setdefault(
                transition.to_signature,
                {
                    "to_signature": transition.to_signature,
                    "to_scope": transition.to_scope,
                    "to_tags": transition.to_tags[:4],
                    "to_keywords": transition.to_keywords[:6],
                    "score": 0.0,
                    "confidence_score": 0.0,
                    "evidence_count": 0.0,
                    "forecast_score": 0.0,
                    "prediction_hit_count": 0.0,
                    "prediction_miss_count": 0.0,
                    "supporting_flows": [],
                },
            )
            bucket["score"] = round(float(bucket["score"]) + score, 4)
            bucket["confidence_score"] = max(
                float(bucket["confidence_score"]),
                float(transition.confidence_score),
            )
            bucket["evidence_count"] = round(
                float(bucket["evidence_count"]) + float(transition.evidence_count),
                4,
            )
            bucket["forecast_score"] = max(
                float(bucket["forecast_score"]),
                float(transition.forecast_score),
            )
            bucket["prediction_hit_count"] = round(
                float(bucket["prediction_hit_count"]) + float(transition.prediction_hit_count),
                4,
            )
            bucket["prediction_miss_count"] = round(
                float(bucket["prediction_miss_count"]) + float(transition.prediction_miss_count),
                4,
            )
            supporting = {
                "from_signature": previous_signature,
                "from_scope": previous_scope,
                "weight": round(score, 4),
            }
            current_supporting = list(bucket["supporting_flows"])
            current_supporting.append(supporting)
            current_supporting.sort(key=lambda item: float(item.get("weight", 0.0)), reverse=True)
            bucket["supporting_flows"] = current_supporting[:3]

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            float(item["score"]),
            float(item["forecast_score"]),
            float(item["confidence_score"]),
            float(item["evidence_count"]),
        ),
        reverse=True,
    )
    return ranked[: max(1, min(int(limit), 20))]


def infer_latent_context_responsibilities(
    rule: PreferenceRule,
    *,
    task_scope: str = "",
    tags: list[str] | None = None,
    query_text: str = "",
    previous_context_nodes: list[dict] | None = None,
    transitions: list[LatentTransition] | None = None,
) -> tuple[list[LatentContextMatch], float]:
    """Infer which latent situations best explain the current task cues."""
    latent_contexts = infer_latent_contexts_from_rule(rule)
    if not latent_contexts:
        return [], 1.0

    task_scope.strip()
    normalized_scope = normalize_text(task_scope)
    normalized_tags = {
        normalize_text(tag)
        for tag in (tags or [])
        if normalize_text(tag)
    }
    query_keywords = extract_keywords(task_scope, query_text[:800])
    query_cue_text = " ".join(
        part for part in [task_scope, " ".join(sorted(normalized_tags)), " ".join(sorted(query_keywords)), query_text] if part
    ).strip()

    scored: list[tuple[LatentContext, float, str]] = []
    for context in latent_contexts:
        raw_score = 0.04 + max(0.02, float(context.prior_weight))
        reasons: list[str] = []

        context_scope = normalize_text(context.scope)
        if normalized_scope and context_scope:
            if normalized_scope == context_scope:
                raw_score += 2.6
                reasons.append(f"scope={context.scope}")
            elif normalized_scope in context_scope or context_scope in normalized_scope:
                raw_score += 1.9
                reasons.append(f"scope~={context.scope}")
            else:
                scope_similarity = semantic_similarity(normalized_scope, context_scope)
                if scope_similarity >= 0.18:
                    raw_score += min(1.4, scope_similarity * 3.8)
                    reasons.append(f"scope_sem={scope_similarity:.2f}")

        context_tags = {
            normalize_text(tag)
            for tag in context.tags
            if normalize_text(tag)
        }
        tag_overlap = sorted(normalized_tags & context_tags)
        if tag_overlap:
            raw_score += min(2.2, len(tag_overlap) * 0.85)
            reasons.append("tags=" + "/".join(tag_overlap[:3]))

        context_keywords = {
            normalize_text(keyword)
            for keyword in context.keywords
            if normalize_text(keyword)
        }
        keyword_overlap = sorted(query_keywords & context_keywords)
        if keyword_overlap:
            raw_score += min(1.6, len(keyword_overlap) * 0.42)
            reasons.append("kw=" + "/".join(keyword_overlap[:3]))

        if query_cue_text:
            prototype_similarity = semantic_similarity(query_cue_text, _latent_context_prototype_text(context))
            if prototype_similarity >= 0.12:
                raw_score += min(1.8, prototype_similarity * 3.2)
                reasons.append(f"proto={prototype_similarity:.2f}")

        transition_bonus, transition_reason = _score_transition_prior(
            context,
            previous_context_nodes=previous_context_nodes,
            transitions=transitions,
        )
        if transition_bonus > 0:
            raw_score += transition_bonus
            reasons.append(f"flow={transition_bonus:.2f}")
            if transition_reason:
                reasons.append(transition_reason)

        raw_score *= 0.72 + min(1.0, context.expected_gain / 4.5) * 0.38 + context.confidence_score * 0.28
        if raw_score >= 0.12:
            scored.append((context, raw_score, " | ".join(reasons)))

    if not scored:
        return [], 1.0

    novelty_base = 0.08
    max_raw = max(item[1] for item in scored)
    if max_raw < 1.15:
        novelty_base += 0.45
    elif max_raw < 1.8:
        novelty_base += 0.22
    total = sum(item[1] for item in scored) + novelty_base
    if total <= 0:
        return [], 1.0

    matches = [
        LatentContextMatch(
            context=context,
            responsibility=round(raw_score / total, 4),
            raw_score=round(raw_score, 4),
            reason=reason or "prior",
        )
        for context, raw_score, reason in scored
    ]
    matches.sort(key=lambda item: item.responsibility, reverse=True)
    return matches, round(novelty_base / total, 4)


def derive_trace_blame_weights(
    *,
    outcome_score: float,
    top_posterior: float,
    posterior_gap: float,
    novelty_probability: float,
    should_abstain: bool = False,
) -> tuple[float, float, str]:
    """Split blame between rule-value failure and context-inference failure.

    We explicitly avoid the convenient story that every bad result was a context mismatch.
    If the model was confident and decisive, the rule takes the hit.
    """
    if outcome_score >= 0.7:
        return 1.0, 0.0, "confirmed"
    if should_abstain and outcome_score <= 0.3:
        return 0.2, 0.8, "should_have_abstained"
    if top_posterior >= 0.68 and posterior_gap >= 0.18:
        return 0.88, 0.12, "confident_rule_failure"
    if top_posterior <= 0.42 or posterior_gap <= 0.06 or novelty_probability >= 0.45:
        return 0.34, 0.66, "uncertain_context_failure"
    return 0.58, 0.42, "mixed_failure"


@dataclass
class MergeResult:
    """Result of merging similar rules."""
    merged_rules: list[PreferenceRule]
    merge_count: int  # how many merges happened


def merge_similar_rules(
    rules: list[PreferenceRule],
    similarity_threshold: float = 0.25,
) -> MergeResult:
    """Merge rules with similar instructions into consolidated rules.

    Example: "余白を作れ" and "余白を増やせ" → merged into one rule
    with combined evidence.
    """
    if not rules:
        return MergeResult(merged_rules=[], merge_count=0)

    # Group rules by similarity
    merged: list[PreferenceRule] = []
    consumed: set[int] = set()
    merge_count = 0

    for i, rule_a in enumerate(rules):
        if i in consumed:
            continue

        # Find all rules similar to rule_a
        group = [rule_a]
        for j, rule_b in enumerate(rules):
            if j <= i or j in consumed:
                continue
            sim = _ngram_similarity(rule_a.instruction, rule_b.instruction)
            if sim >= similarity_threshold:
                group.append(rule_b)
                consumed.add(j)

        if len(group) == 1:
            merged.append(rule_a)
            continue

        # Merge the group: keep the one with highest evidence, absorb others
        merge_count += len(group) - 1
        group.sort(key=lambda r: (-r.expected_gain, -r.support_score, -r.evidence_count, -len(r.instruction)))
        primary = group[0]

        # Combine evidence
        total_evidence = sum(r.evidence_count for r in group)
        total_support_score = sum(float(r.support_score or r.evidence_count) for r in group)
        total_strong_signal_count = sum(max(0, int(r.strong_signal_count)) for r in group)
        total_success_count = sum(max(0, int(r.success_count)) for r in group)
        total_failure_count = sum(max(0, int(r.failure_count)) for r in group)
        all_tags = set()
        all_source_ids = []
        all_when_tags = set()
        all_latent_contexts: list[LatentContext] = []
        earliest = primary.first_recorded_at
        latest = primary.last_recorded_at

        for r in group:
            all_tags.update(r.tags)
            all_source_ids.extend(r.source_turn_ids)
            all_when_tags.update(r.applies_when_tags)
            all_latent_contexts.extend(infer_latent_contexts_from_rule(r))
            if r.first_recorded_at < earliest:
                earliest = r.first_recorded_at
            if r.last_recorded_at > latest:
                latest = r.last_recorded_at

        merged_latent_contexts = merge_latent_contexts(all_latent_contexts)
        merged_contexts = flatten_latent_contexts(merged_latent_contexts)
        distinct_scope_count = len([ctx for ctx in merged_contexts if ctx.kind == "scope"])
        distinct_tag_count = len([ctx for ctx in merged_contexts if ctx.kind == "tag"])
        context_mode = derive_context_mode(
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
            evidence_count=total_evidence,
        )
        confidence_score = derive_rule_confidence_score(
            evidence_count=total_evidence,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
            strong_signal_count=total_strong_signal_count,
            success_count=total_success_count,
            failure_count=total_failure_count,
        )
        expected_gain = derive_rule_expected_gain(
            support_score=total_support_score,
            context_utility=max(
                _aggregate_context_utility(merged_contexts),
                _aggregate_latent_context_gain(merged_latent_contexts),
            ),
            confidence_score=confidence_score,
            strong_signal_count=total_strong_signal_count,
            success_count=total_success_count,
            failure_count=total_failure_count,
            context_mode=context_mode,
        )
        status = derive_rule_status(
            evidence_count=total_evidence,
            support_score=total_support_score,
            strong_signal_count=total_strong_signal_count,
            context_mode=context_mode,
            expected_gain=expected_gain,
            confidence_score=confidence_score,
        )

        merged_rule = PreferenceRule(
            id=primary.id,
            statement=primary.statement,
            normalized_statement=primary.normalized_statement,
            instruction=primary.instruction,
            status=status,
            evidence_count=total_evidence,
            first_recorded_at=earliest,
            last_recorded_at=latest,
            applies_to_scope=primary.applies_to_scope,
            applies_when_tags=sorted(all_when_tags)[:12],
            negative_conditions=primary.negative_conditions,
            priority=max(1, min(5, round(max(total_support_score, expected_gain)))),
            version=primary.version + 1,
            tags=sorted(all_tags)[:20],
            source_turn_ids=list(dict.fromkeys(all_source_ids))[:30],
            contexts=merged_contexts[:14],
            latent_contexts=merged_latent_contexts[:8],
            context_mode=context_mode,
            support_score=round(total_support_score, 4),
            expected_gain=expected_gain,
            confidence_score=confidence_score,
            strong_signal_count=total_strong_signal_count,
            success_count=total_success_count,
            failure_count=total_failure_count,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
        )
        merged.append(merged_rule)

    return MergeResult(merged_rules=merged, merge_count=merge_count)


# ---------------------------------------------------------------------------
# 3. Semantic Search — n-gram based relevance scoring
# ---------------------------------------------------------------------------

def semantic_similarity(query: str, target: str) -> float:
    """Compute semantic similarity between query and target text.

    Uses character n-gram Jaccard similarity as a lightweight
    embedding-free approach that works for Japanese and English.
    """
    return _ngram_similarity(query, target)


def find_relevant_rules_semantic(
    rules: list[PreferenceRule],
    query: str,
    *,
    min_similarity: float = 0.15,
    limit: int = 5,
) -> list[tuple[PreferenceRule, float]]:
    """Find rules relevant to a query using semantic similarity.

    Checks against instruction, tags, and scope.
    Returns (rule, score) pairs sorted by relevance.
    """
    if not query.strip():
        return []

    results: list[tuple[PreferenceRule, float]] = []

    for rule in rules:
        # Score against multiple fields
        instruction_sim = semantic_similarity(query, rule.instruction)
        tag_text = " ".join(rule.tags + rule.applies_when_tags)
        tag_sim = semantic_similarity(query, tag_text) * 0.6
        scope_sim = semantic_similarity(query, rule.applies_to_scope) * 0.8
        latent_sim = 0.0
        latent_contexts = infer_latent_contexts_from_rule(rule)
        if latent_contexts:
            latent_sim = max(
                semantic_similarity(query, _latent_context_prototype_text(context)) * (0.45 + context.prior_weight * 0.35)
                for context in latent_contexts
            )

        # Take best match across fields
        best_sim = max(instruction_sim, tag_sim, scope_sim, latent_sim)

        # Bonus for rules with proven situational value
        best_sim += min(0.18, max(0.0, float(rule.expected_gain)) * 0.035)
        best_sim += min(0.08, max(0.0, float(rule.confidence_score)) * 0.08)

        # Bonus for high evidence
        best_sim += min(0.1, rule.evidence_count * 0.02)

        if best_sim >= min_similarity:
            results.append((rule, best_sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


# ---------------------------------------------------------------------------
# 4. Contradiction Detection — detect conflicting rules
# ---------------------------------------------------------------------------

_NEGATION_WORDS = re.compile(r"するな|やめろ|避けろ|[ぁ-ん]な$|[ぁ-ん]るな|[ぁ-ん]いな")


def detect_contradicting_rules(
    rules: list[PreferenceRule],
) -> list[tuple[PreferenceRule, PreferenceRule, str]]:
    """Detect rules that may contradict each other.

    Returns list of (rule_a, rule_b, reason) tuples where:
    - similarity >= 0.25 (same topic)
    - but one is a negative_condition of the other, OR
    - one contains "するな/やめろ/避けろ" while the other is affirmative on same topic

    reason: "negation_conflict" or "scope_conflict"
    """
    conflicts: list[tuple[PreferenceRule, PreferenceRule, str]] = []

    for i, rule_a in enumerate(rules):
        for rule_b in rules[i + 1 :]:
            sim = _ngram_similarity(rule_a.instruction, rule_b.instruction)
            if sim < 0.25:
                continue

            a_negated = bool(_NEGATION_WORDS.search(rule_a.instruction))
            b_negated = bool(_NEGATION_WORDS.search(rule_b.instruction))

            # One is negation of the other → negation_conflict
            if a_negated != b_negated:
                conflicts.append((rule_a, rule_b, "negation_conflict"))
                continue

            # Check if one appears in the other's negative_conditions → scope_conflict
            a_in_b_neg = any(
                _ngram_similarity(rule_a.instruction, nc) >= 0.25
                for nc in rule_b.negative_conditions
            )
            b_in_a_neg = any(
                _ngram_similarity(rule_b.instruction, nc) >= 0.25
                for nc in rule_a.negative_conditions
            )
            if a_in_b_neg or b_in_a_neg:
                conflicts.append((rule_a, rule_b, "scope_conflict"))

    return conflicts


# ---------------------------------------------------------------------------
# 4b. Conflict Resolution — auto-resolve detected conflicts
# ---------------------------------------------------------------------------


def resolve_contradicting_rules(
    rules: list[PreferenceRule],
    metabolism_rate: float = 0.5,
) -> tuple[list[PreferenceRule], list[dict[str, str]]]:
    """Detect and auto-resolve contradicting rules.

    Returns (updated_rules, resolution_log).
    WARNING: Mutates rule objects in-place. Pass copies if you need the originals.

    Resolution strategy:
    - negation_conflict: demote the rule with lower (confidence × evidence_count).
      Safety guard threshold scales with metabolism_rate.
    - scope_conflict: add each rule's instruction to the other's negative_conditions
      so they no longer overlap. Neither rule is demoted.
    """
    conflicts = detect_contradicting_rules(rules)
    if not conflicts:
        return rules, []

    demoted_ids: set[str] = set()
    log: list[dict[str, str]] = []

    for rule_a, rule_b, reason in conflicts:
        if rule_a.id in demoted_ids or rule_b.id in demoted_ids:
            continue  # already resolved

        if reason == "negation_conflict":
            strength_a = (getattr(rule_a, "confidence_score", 0.5) or 0.5) * rule_a.evidence_count
            strength_b = (getattr(rule_b, "confidence_score", 0.5) or 0.5) * rule_b.evidence_count

            # Safety guard: don't demote well-established rules
            # Aggressive users have a lower bar for demotion (3 instead of 5)
            guard_threshold = max(2, round(5 * (1.5 - metabolism_rate)))
            double_guard = guard_threshold * 2
            if rule_a.evidence_count >= guard_threshold and rule_b.evidence_count < double_guard:
                if strength_a < strength_b:
                    continue
            if rule_b.evidence_count >= guard_threshold and rule_a.evidence_count < double_guard:
                if strength_b < strength_a:
                    continue

            if strength_a >= strength_b:
                loser, winner = rule_b, rule_a
            else:
                loser, winner = rule_a, rule_b

            loser.status = "demoted"
            loser.priority = max(1, loser.priority - 2)
            if "conflict_demoted" not in loser.tags:
                loser.tags.append("conflict_demoted")
            demoted_ids.add(loser.id)
            log.append({
                "action": "demote",
                "reason": reason,
                "winner": winner.id,
                "loser": loser.id,
                "winner_instruction": winner.instruction[:60],
                "loser_instruction": loser.instruction[:60],
            })

        elif reason == "scope_conflict":
            # Scope separation: add to each other's negative_conditions
            if rule_a.instruction not in rule_b.negative_conditions:
                rule_b.negative_conditions.append(rule_a.instruction)
            if rule_b.instruction not in rule_a.negative_conditions:
                rule_a.negative_conditions.append(rule_b.instruction)
            log.append({
                "action": "scope_separate",
                "reason": reason,
                "rule_a": rule_a.id,
                "rule_b": rule_b.id,
            })

    return rules, log


# ---------------------------------------------------------------------------
# 4c. Self-Correction — auto-fix needs_revision rules
# ---------------------------------------------------------------------------


def auto_correct_flagged_rules(
    rules: list[PreferenceRule],
    now_str: str,
    *,
    max_revision_age_days: float = 14.0,
    min_confidence_for_keep: float = 0.3,
    metabolism_rate: float = 0.5,
) -> tuple[list[PreferenceRule], list[dict[str, str]]]:
    """Auto-correct rules flagged with needs_revision.

    Strategy:
    - needs_revision + evidence_count <= 1 → delete (not enough evidence)
    - needs_revision + confidence < min_confidence_for_keep + aged > max_revision_age_days → demote
    - needs_revision + confidence >= min_confidence_for_keep → narrow scope
      (add failure context to negative_conditions)

    Returns (updated_rules, correction_log).
    """
    # Aggressive users wait less before demoting flagged rules
    max_revision_age_days = max_revision_age_days * (1.5 - metabolism_rate)
    now = datetime.strptime(now_str, "%Y/%m/%d %H:%M")
    corrected: list[PreferenceRule] = []
    log: list[dict[str, str]] = []

    for rule in rules:
        if "needs_revision" not in (rule.tags or []):
            corrected.append(rule)
            continue

        confidence = getattr(rule, "confidence_score", 0.5) or 0.5

        # Low evidence → delete entirely
        if rule.evidence_count <= 1:
            log.append({
                "action": "delete",
                "rule_id": rule.id,
                "reason": "needs_revision with evidence_count <= 1",
                "instruction": rule.instruction[:60],
            })
            continue  # don't add to corrected

        # Check age of the revision flag
        try:
            last_dt = datetime.strptime(rule.last_recorded_at, "%Y/%m/%d %H:%M")
            age_days = (now - last_dt).total_seconds() / 86400
        except (ValueError, AttributeError):
            age_days = 0

        if confidence < min_confidence_for_keep and age_days > max_revision_age_days:
            # Demote: too long flagged, low confidence
            rule.status = "demoted"
            if "self_corrected" not in rule.tags:
                rule.tags.append("self_corrected")
            log.append({
                "action": "demote",
                "rule_id": rule.id,
                "reason": f"needs_revision for {age_days:.0f}d, confidence {confidence:.2f}",
                "instruction": rule.instruction[:60],
            })
            corrected.append(rule)
        else:
            # Narrow scope: add scope as negative_condition to limit where it applies
            scope = rule.applies_to_scope
            if scope and scope not in rule.negative_conditions:
                rule.negative_conditions.append(f"not-{scope}")
            log.append({
                "action": "narrow",
                "rule_id": rule.id,
                "reason": f"needs_revision, confidence {confidence:.2f}, narrowing scope",
                "instruction": rule.instruction[:60],
            })
            corrected.append(rule)

    return corrected, log


# ---------------------------------------------------------------------------
# 5. Forgetting Curve — Ebbinghaus-style priority decay
# ---------------------------------------------------------------------------

def apply_forgetting_curve(
    rules: list[PreferenceRule],
    now_str: str,
    half_life_days: float = 30.0,
    metabolism_rate: float = 0.5,
) -> list[PreferenceRule]:
    """Apply forgetting curve to rule priorities.

    Rules not reinforced recently decay in effective priority.
    Promoted rules decay slower than candidates.
    Returns rules with adjusted priority (does NOT change status).

    Args:
        rules: List of PreferenceRule objects.
        now_str: Current datetime string in "2026/03/29 00:00" format.
        half_life_days: Half-life in days for candidate rules (promoted uses 2x).
        metabolism_rate: 0.0=conservative (slow decay), 1.0=aggressive (fast decay).
            Modulates the effective half-life: aggressive users forget faster.
    """
    # Metabolism modulates base half-life: aggressive → shorter, conservative → longer
    # Range: 0.5x (aggressive) to 1.5x (conservative)
    metabolism_factor = 1.5 - metabolism_rate  # 0.0→1.5, 0.5→1.0, 1.0→0.5
    half_life_days = half_life_days * metabolism_factor
    now = datetime.strptime(now_str, "%Y/%m/%d %H:%M")
    result: list[PreferenceRule] = []

    for rule in rules:
        last_str = rule.last_recorded_at
        try:
            last_dt = datetime.strptime(last_str, "%Y/%m/%d %H:%M")
        except (ValueError, AttributeError):
            # If parsing fails, treat as no decay
            result.append(rule)
            continue

        days_elapsed = (now - last_dt).total_seconds() / 86400.0
        days_elapsed = max(0.0, days_elapsed)

        effective_half_life = half_life_days
        if rule.expected_gain > 0:
            effective_half_life += min(30.0, rule.expected_gain * 3.0)
        if rule.confidence_score > 0:
            effective_half_life *= 1.0 + min(0.9, rule.confidence_score)
        if rule.context_mode == "general":
            effective_half_life *= 1.25
        elif rule.context_mode == "mixed":
            effective_half_life *= 1.1
        elif rule.status == "promoted" and rule.expected_gain <= 0:
            effective_half_life *= 2.0

        decay = 0.5 ** (days_elapsed / effective_half_life)
        base_priority = float(rule.priority)
        effective_priority = base_priority * decay + rule.evidence_count * 0.1 + rule.expected_gain * 0.12

        # Round to nearest int, clamp to [1, 5]
        new_priority = max(1, min(5, round(effective_priority)))

        # Build a new rule with updated priority
        updated = PreferenceRule(
            id=rule.id,
            statement=rule.statement,
            normalized_statement=rule.normalized_statement,
            instruction=rule.instruction,
            status=rule.status,
            evidence_count=rule.evidence_count,
            first_recorded_at=rule.first_recorded_at,
            last_recorded_at=rule.last_recorded_at,
            applies_to_scope=rule.applies_to_scope,
            applies_when_tags=rule.applies_when_tags,
            negative_conditions=rule.negative_conditions,
            priority=new_priority,
            version=rule.version,
            expires_at=rule.expires_at,
            tags=rule.tags,
            source_turn_ids=rule.source_turn_ids,
            contexts=rule.contexts,
            latent_contexts=infer_latent_contexts_from_rule(rule),
            context_mode=rule.context_mode,
            support_score=rule.support_score,
            expected_gain=rule.expected_gain,
            confidence_score=rule.confidence_score,
            strong_signal_count=rule.strong_signal_count,
            success_count=rule.success_count,
            failure_count=rule.failure_count,
            distinct_scope_count=rule.distinct_scope_count,
            distinct_tag_count=rule.distinct_tag_count,
        )
        result.append(updated)

    return result


# ---------------------------------------------------------------------------
# 5. Rule Association — similarity-based linking
# ---------------------------------------------------------------------------

def build_rule_associations(
    rules: list[PreferenceRule],
    similarity_threshold: float = 0.2,
) -> dict[str, list[str]]:
    """Return {rule_id: [related_rule_ids]} for rules with similarity >= threshold.

    Uses _ngram_similarity on instruction text. A rule is never associated
    with itself.
    """
    associations: dict[str, list[str]] = {}

    for i, rule_a in enumerate(rules):
        related: list[str] = []
        for j, rule_b in enumerate(rules):
            if i == j:
                continue
            sim = _ngram_similarity(rule_a.instruction, rule_b.instruction)
            if sim >= similarity_threshold:
                related.append(rule_b.id)
        associations[rule_a.id] = related

    return associations


# ---------------------------------------------------------------------------
# 6. Reconsolidation — update rules based on how well they worked
# ---------------------------------------------------------------------------

def reconsolidate_rule(
    rule: PreferenceRule,
    applied: bool,
    outcome_score: float,
    *,
    task_scope: str = "",
    tags: list[str] | None = None,
    inference_trace: dict | None = None,
) -> PreferenceRule:
    """Update a rule based on how well it worked when applied.

    - applied=True, outcome_score >= 0.7 → evidence_count += 1, priority up
    - applied=True, outcome_score <= 0.3 → rule instruction may need revision
      (flag with tag "needs_revision")
    - applied=False → no change
    """
    if not applied:
        return rule

    raw_scope = task_scope.strip()
    normalized_scope = normalize_text(task_scope)
    normalized_tags = {
        normalize_text(tag)
        for tag in (tags or [])
        if normalize_text(tag)
    }
    query_text = " ".join(part for part in [task_scope, " ".join(sorted(normalized_tags))] if part).strip()
    latent_contexts = infer_latent_contexts_from_rule(rule)
    if isinstance(inference_trace, dict):
        trace_matches = inference_trace.get("latent_context_matches", [])
        responsibility_by_id = {
            str(item.get("context_id", "")).strip(): float(item.get("posterior", 0.0) or 0.0)
            for item in trace_matches
            if str(item.get("context_id", "")).strip()
        }
        novelty_probability = float(inference_trace.get("novelty_probability", 0.0) or 0.0)
        top_posterior = float(inference_trace.get("top_context_posterior", 0.0) or 0.0)
        posterior_gap = float(inference_trace.get("posterior_gap", 0.0) or 0.0)
        should_abstain = bool(inference_trace.get("should_abstain", False))
        matches = []
    else:
        matches, novelty_probability = infer_latent_context_responsibilities(
            rule,
            task_scope=task_scope,
            tags=tags,
            query_text=query_text,
        )
        responsibility_by_id = {
            match.context.id: float(match.responsibility)
            for match in matches
            if match.responsibility > 0
        }
        top_posterior = matches[0].responsibility if matches else 0.0
        second_posterior = matches[1].responsibility if len(matches) >= 2 else 0.0
        posterior_gap = max(0.0, top_posterior - second_posterior)
        should_abstain = False
    has_matching_scope_context = any(
        normalize_text(match.context.scope) == normalized_scope
        or (
            normalized_scope
            and normalize_text(match.context.scope)
            and (
                normalized_scope in normalize_text(match.context.scope)
                or normalize_text(match.context.scope) in normalized_scope
            )
        )
        for match in matches
    )
    if not matches and responsibility_by_id:
        has_matching_scope_context = any(
            normalized_scope
            and normalize_text(context.scope)
            and context.id in responsibility_by_id
            and (
                normalized_scope == normalize_text(context.scope)
                or normalized_scope in normalize_text(context.scope)
                or normalize_text(context.scope) in normalized_scope
            )
            for context in latent_contexts
        )
    updated_latent_contexts: list[LatentContext] = []
    now_str = datetime.now().strftime("%Y/%m/%d %H:%M")
    rule_blame_weight, inference_blame_weight, _ = derive_trace_blame_weights(
        outcome_score=outcome_score,
        top_posterior=top_posterior,
        posterior_gap=posterior_gap,
        novelty_probability=novelty_probability,
        should_abstain=should_abstain,
    )

    support_delta = 0.0
    if outcome_score >= 0.7:
        support_delta = 0.7 + outcome_score
    elif outcome_score <= 0.3:
        support_delta = 0.3 + (0.3 - outcome_score) * 0.8
    else:
        support_delta = 0.35 + outcome_score * 0.2

    for context in latent_contexts:
        responsibility = responsibility_by_id.get(context.id, 0.0)
        evidence_count = max(0.0, float(context.evidence_count))
        support_score = max(0.0, float(context.support_score))
        posterior_mass = max(0.0, float(context.posterior_mass))
        strong_signal_count = max(0.0, float(context.strong_signal_count))
        success_mass = max(0.0, float(context.success_mass))
        failure_mass = max(0.0, float(context.failure_mass))
        last_seen_at = context.last_seen_at

        if responsibility > 0:
            evidence_count = round(evidence_count + responsibility, 4)
            posterior_delta = responsibility
            if outcome_score <= 0.3:
                posterior_delta *= max(0.08, 1.0 - inference_blame_weight)
            posterior_mass = round(posterior_mass + posterior_delta, 4)
            last_seen_at = now_str
            if outcome_score >= 0.7:
                success_mass = round(success_mass + responsibility, 4)
                support_score = round(support_score + support_delta * responsibility, 4)
            elif outcome_score <= 0.3:
                failure_mass = round(failure_mass + responsibility * max(0.25, rule_blame_weight), 4)
                support_score = round(
                    max(0.0, support_score - support_delta * responsibility * (0.18 + rule_blame_weight * 0.82)),
                    4,
                )
            else:
                support_score = round(support_score + support_delta * responsibility * 0.35, 4)

        confidence_score = derive_latent_context_confidence_score(
            evidence_count=evidence_count,
            support_score=support_score,
            posterior_mass=posterior_mass,
            strong_signal_count=strong_signal_count,
            success_mass=success_mass,
            failure_mass=failure_mass,
        )
        expected_gain = derive_latent_context_expected_gain(
            support_score=support_score,
            confidence_score=confidence_score,
            strong_signal_count=strong_signal_count,
            success_mass=success_mass,
            failure_mass=failure_mass,
        )
        updated_latent_contexts.append(
            LatentContext(
                id=context.id,
                scope=context.scope,
                tags=context.tags,
                keywords=context.keywords,
                prototype_text=_latent_context_prototype_text(context),
                evidence_count=evidence_count,
                support_score=support_score,
                expected_gain=expected_gain,
                confidence_score=confidence_score,
                prior_weight=derive_latent_context_prior_weight(
                    evidence_count=evidence_count,
                    support_score=support_score,
                    expected_gain=expected_gain,
                    confidence_score=confidence_score,
                    posterior_mass=posterior_mass,
                ),
                posterior_mass=posterior_mass,
                strong_signal_count=strong_signal_count,
                success_mass=success_mass,
                failure_mass=failure_mass,
                last_seen_at=last_seen_at,
            )
        )

    should_spawn_context = (
        novelty_probability >= 0.34
        or not matches and not responsibility_by_id
        or (normalized_scope and not has_matching_scope_context)
        or (outcome_score <= 0.3 and inference_blame_weight >= 0.55)
        or should_abstain
    )
    if should_spawn_context and (normalized_scope or normalized_tags):
        cue_keywords = sorted(extract_keywords(task_scope, " ".join(sorted(normalized_tags))))[:6]
        base_support = max(0.6, support_delta)
        success_mass = 1.0 if outcome_score >= 0.7 else 0.0
        failure_mass = 1.0 if outcome_score <= 0.3 else 0.0
        confidence_score = derive_latent_context_confidence_score(
            evidence_count=1.0,
            support_score=base_support,
            posterior_mass=max(0.6, novelty_probability),
            success_mass=success_mass,
            failure_mass=failure_mass,
        )
        expected_gain = derive_latent_context_expected_gain(
            support_score=base_support,
            confidence_score=confidence_score,
            success_mass=success_mass,
            failure_mass=failure_mass,
        )
        updated_latent_contexts.append(
            LatentContext(
                id=f"{rule.id}-latent-{len(updated_latent_contexts) + 1}",
                scope=raw_scope,
                tags=list(sorted(normalized_tags))[:4],
                keywords=cue_keywords,
                prototype_text=" ".join(
                    part
                    for part in [raw_scope, " ".join(sorted(normalized_tags)[:4]), " ".join(cue_keywords[:4])]
                    if part
                ).strip(),
                evidence_count=1.0,
                support_score=base_support,
                expected_gain=expected_gain,
                confidence_score=confidence_score,
                prior_weight=derive_latent_context_prior_weight(
                    evidence_count=1.0,
                    support_score=base_support,
                    expected_gain=expected_gain,
                    confidence_score=confidence_score,
                    posterior_mass=max(0.6, novelty_probability),
                ),
                posterior_mass=max(0.6, novelty_probability),
                success_mass=success_mass,
                failure_mass=failure_mass,
                last_seen_at=now_str,
            )
        )

    merged_latent_contexts = merge_latent_contexts(updated_latent_contexts)
    merged_contexts = flatten_latent_contexts(merged_latent_contexts)
    distinct_scope_count = len([ctx for ctx in merged_contexts if ctx.kind == "scope"])
    distinct_tag_count = len([ctx for ctx in merged_contexts if ctx.kind == "tag"])
    next_context_mode = derive_context_mode(
        distinct_scope_count=distinct_scope_count,
        distinct_tag_count=distinct_tag_count,
        evidence_count=rule.evidence_count + (1 if outcome_score >= 0.7 else 0),
    )
    success_count = rule.success_count + (1 if outcome_score >= 0.7 else 0)
    failure_count = rule.failure_count + (1 if outcome_score <= 0.3 else 0)
    confidence_score = derive_rule_confidence_score(
        evidence_count=rule.evidence_count + (1 if outcome_score >= 0.7 else 0),
        distinct_scope_count=distinct_scope_count,
        distinct_tag_count=distinct_tag_count,
        strong_signal_count=rule.strong_signal_count,
        success_count=success_count,
        failure_count=failure_count,
    )
    support_score = max(rule.support_score, float(rule.evidence_count))
    if outcome_score >= 0.7:
        support_score += 0.6
    elif outcome_score <= 0.3:
        support_score = max(0.0, support_score - 0.2)
    else:
        support_score += 0.1
    expected_gain = derive_rule_expected_gain(
        support_score=support_score,
        context_utility=max(
            _aggregate_context_utility(merged_contexts),
            _aggregate_latent_context_gain(merged_latent_contexts),
        ),
        confidence_score=confidence_score,
        strong_signal_count=rule.strong_signal_count,
        success_count=success_count,
        failure_count=failure_count,
        context_mode=next_context_mode,
    )

    if outcome_score >= 0.7:
        # Good outcome: strengthen the rule
        new_evidence = rule.evidence_count + 1
        new_priority = max(1, min(5, rule.priority + 1))
        # Remove needs_revision tag if present (rule is working now)
        new_tags = [t for t in rule.tags if t != "needs_revision"]
        return PreferenceRule(
            id=rule.id,
            statement=rule.statement,
            normalized_statement=rule.normalized_statement,
            instruction=rule.instruction,
            status=derive_rule_status(
                evidence_count=new_evidence,
                support_score=max(support_score, float(new_evidence)),
                strong_signal_count=rule.strong_signal_count,
                context_mode=next_context_mode,
                expected_gain=expected_gain,
                confidence_score=confidence_score,
            ),
            evidence_count=new_evidence,
            first_recorded_at=rule.first_recorded_at,
            last_recorded_at=rule.last_recorded_at,
            applies_to_scope=rule.applies_to_scope,
            applies_when_tags=rule.applies_when_tags,
            negative_conditions=rule.negative_conditions,
            priority=new_priority,
            version=rule.version,
            expires_at=rule.expires_at,
            tags=new_tags,
            source_turn_ids=rule.source_turn_ids,
            contexts=merged_contexts,
            latent_contexts=merged_latent_contexts,
            context_mode=next_context_mode,
            support_score=max(support_score, float(new_evidence)),
            expected_gain=expected_gain,
            confidence_score=confidence_score,
            strong_signal_count=rule.strong_signal_count,
            success_count=success_count,
            failure_count=failure_count,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
        )

    elif outcome_score <= 0.3:
        # Bad outcome: flag the rule for revision
        new_tags = list(rule.tags)
        if "needs_revision" not in new_tags:
            new_tags.append("needs_revision")
        return PreferenceRule(
            id=rule.id,
            statement=rule.statement,
            normalized_statement=rule.normalized_statement,
            instruction=rule.instruction,
            status=derive_rule_status(
                evidence_count=rule.evidence_count,
                support_score=support_score,
                strong_signal_count=rule.strong_signal_count,
                context_mode=next_context_mode,
                expected_gain=expected_gain,
                confidence_score=confidence_score,
            ),
            evidence_count=rule.evidence_count,
            first_recorded_at=rule.first_recorded_at,
            last_recorded_at=rule.last_recorded_at,
            applies_to_scope=rule.applies_to_scope,
            applies_when_tags=rule.applies_when_tags,
            negative_conditions=rule.negative_conditions,
            priority=rule.priority,
            version=rule.version,
            expires_at=rule.expires_at,
            tags=new_tags,
            source_turn_ids=rule.source_turn_ids,
            contexts=merged_contexts,
            latent_contexts=merged_latent_contexts,
            context_mode=next_context_mode,
            support_score=support_score,
            expected_gain=expected_gain,
            confidence_score=confidence_score,
            strong_signal_count=rule.strong_signal_count,
            success_count=success_count,
            failure_count=failure_count,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
        )

    # Score in range (0.3, 0.7): no change
    return PreferenceRule(
        id=rule.id,
        statement=rule.statement,
        normalized_statement=rule.normalized_statement,
        instruction=rule.instruction,
        status=derive_rule_status(
            evidence_count=rule.evidence_count,
            support_score=support_score,
            strong_signal_count=rule.strong_signal_count,
            context_mode=next_context_mode,
            expected_gain=expected_gain,
            confidence_score=confidence_score,
        ),
        evidence_count=rule.evidence_count,
        first_recorded_at=rule.first_recorded_at,
        last_recorded_at=rule.last_recorded_at,
        applies_to_scope=rule.applies_to_scope,
        applies_when_tags=rule.applies_when_tags,
        negative_conditions=rule.negative_conditions,
        priority=rule.priority,
        version=rule.version,
        expires_at=rule.expires_at,
        tags=rule.tags,
        source_turn_ids=rule.source_turn_ids,
        contexts=merged_contexts,
        latent_contexts=merged_latent_contexts,
        context_mode=next_context_mode,
        support_score=support_score,
        expected_gain=expected_gain,
        confidence_score=confidence_score,
        strong_signal_count=rule.strong_signal_count,
        success_count=success_count,
        failure_count=failure_count,
        distinct_scope_count=distinct_scope_count,
        distinct_tag_count=distinct_tag_count,
    )


def reconsolidate_rules_from_turns(
    rules: list[PreferenceRule],
    turns: list[ConversationTurn],
) -> list[PreferenceRule]:
    """Apply reconsolidation to all rules based on recent turns.

    For each turn where guidance_applied=True:
    - Find which rules were likely applied (semantic similarity between
      turn scope/tags and rule scope/tags)
    - Call reconsolidate_rule with that turn's reaction_score
    """
    # Only process turns where guidance was applied and score is available
    guidance_turns = [
        t for t in turns
        if t.guidance_applied and t.reaction_score is not None
    ]
    if not guidance_turns or not rules:
        return rules

    updated_rules = list(rules)

    for turn in guidance_turns:
        inference_trace = turn.metadata.get("inference_trace") if isinstance(turn.metadata, dict) else None
        selected_rule_ids: list[str] = []
        selected_trace_by_rule_id: dict[str, dict] = {}
        if isinstance(inference_trace, dict):
            selected_rule_ids = [
                str(rule_id).strip()
                for rule_id in inference_trace.get("selected_rule_ids", [])
                if str(rule_id).strip()
            ]
            for item in inference_trace.get("selected_rules", []):
                if not isinstance(item, dict):
                    continue
                rule_id = str(item.get("rule_id", "")).strip()
                if rule_id:
                    selected_trace_by_rule_id[rule_id] = item
            if inference_trace.get("abstained_overall") and not selected_rule_ids:
                continue

        # Build a query from the turn's scope and tags
        query_parts = [turn.task_scope, turn.user_feedback] + list(turn.tags)
        query = " ".join(part for part in query_parts if part)
        if not query.strip():
            continue

        # Build a mapping of rule_id → rule for in-place updates
        rule_map = {r.id: r for r in updated_rules}

        if selected_rule_ids:
            for rule_id in selected_rule_ids:
                rule = rule_map.get(rule_id)
                if rule is None:
                    continue
                updated = reconsolidate_rule(
                    rule,
                    applied=True,
                    outcome_score=turn.reaction_score,
                    task_scope=turn.task_scope,
                    tags=turn.tags,
                    inference_trace=selected_trace_by_rule_id.get(rule_id),
                )
                rule_map[rule.id] = updated
        else:
            # Backward compatibility for older turns that have no inference trace.
            relevant = find_relevant_rules_semantic(
                updated_rules,
                query,
                min_similarity=0.1,
                limit=len(updated_rules),
            )

            for rule, similarity in relevant:
                if similarity >= 0.1:
                    updated = reconsolidate_rule(
                        rule,
                        applied=True,
                        outcome_score=turn.reaction_score,
                        task_scope=turn.task_scope,
                        tags=turn.tags,
                    )
                    rule_map[rule.id] = updated

        updated_rules = [rule_map.get(r.id, r) for r in updated_rules]

    return updated_rules


def find_relevant_turns_semantic(
    turns: list[ConversationTurn],
    query: str,
    *,
    min_similarity: float = 0.15,
    limit: int = 5,
) -> list[tuple[ConversationTurn, float]]:
    """Find turns relevant to a query using semantic similarity."""
    if not query.strip():
        return []

    results: list[tuple[ConversationTurn, float]] = []

    for turn in turns:
        # Score against feedback, corrections, and scope
        feedback_sim = semantic_similarity(query, turn.user_feedback)
        corrections_text = " ".join(turn.extracted_corrections)
        corr_sim = semantic_similarity(query, corrections_text)
        scope_sim = semantic_similarity(query, turn.task_scope) * 0.7

        best_sim = max(feedback_sim, corr_sim, scope_sim)

        # Bonus for turns with corrections
        if turn.extracted_corrections:
            best_sim += 0.05

        if best_sim >= min_similarity:
            results.append((turn, best_sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]
