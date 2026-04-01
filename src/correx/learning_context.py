from __future__ import annotations

import math

from .conversation_learning import extract_keywords, normalize_text
from .memory_manager import (
    build_context_signature,
    find_relevant_rules_semantic,
    find_relevant_turns_semantic,
    infer_latent_context_responsibilities,
    infer_latent_contexts_from_rule,
)
from .schemas import ConversationTurn, EpisodeRecord, LatentTransition, PreferenceRule


def _score_entry(
    entry: EpisodeRecord,
    *,
    current_domain: str,
    current_issuer: str,
    current_keywords: set[str],
) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []

    history_domain = normalize_text((entry.company_profile or {}).get("basic", {}).get("industry"))
    if current_domain and history_domain == current_domain:
        score += 5
        reasons.append("same_domain")

    history_issuer = normalize_text(entry.issuer)
    if current_issuer and history_issuer == current_issuer:
        score += 5
        reasons.append("same_issuer")

    latest = entry.corrections[0] if entry.corrections else None
    if latest:
        history_keywords = extract_keywords(entry.title, latest.correction_note, latest.reuse_note)
        overlap = sorted(current_keywords & history_keywords)
        if overlap:
            score += min(4, len(overlap))
            reasons.append("shared_terms:" + "/".join(overlap[:3]))

    return score, " | ".join(reasons)


def get_relevant_corrections(
    entries: list[EpisodeRecord],
    *,
    company_profile: dict | None = None,
    task_title: str = "",
    issuer: str = "",
    raw_text: str = "",
    limit: int = 3,
) -> list[dict]:
    current_domain = normalize_text((company_profile or {}).get("basic", {}).get("industry"))
    current_issuer = normalize_text(issuer)
    current_keywords = extract_keywords(task_title, raw_text[:1800])

    ranked: list[dict] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not entry.corrections:
            continue
        score, reason = _score_entry(
            entry,
            current_domain=current_domain,
            current_issuer=current_issuer,
            current_keywords=current_keywords,
        )
        if score <= 0:
            continue
        latest = entry.corrections[0]
        signature = (
            normalize_text(entry.title),
            normalize_text(latest.correction_note),
            normalize_text(latest.reuse_note),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ranked.append(
            {
                "title": entry.title,
                "issuer": entry.issuer,
                "task_type": entry.task_type,
                "reason": reason,
                "latest": {
                    "decision_override": latest.decision_override,
                    "correction_note": latest.correction_note,
                    "reuse_note": latest.reuse_note,
                },
                "score": score,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def _score_preference_rule(
    rule: PreferenceRule,
    *,
    task_scope: str,
    current_keywords: set[str],
    query_text: str = "",
    previous_context_nodes: list[dict] | None = None,
    transitions: list[LatentTransition] | None = None,
) -> tuple[int, str, dict]:
    score = 0
    reasons: list[str] = []
    scope_matched = False
    keyword_matched = False

    latent_contexts = infer_latent_contexts_from_rule(rule)
    latent_matches, novelty_probability = infer_latent_context_responsibilities(
        rule,
        task_scope=task_scope,
        tags=sorted(current_keywords),
        query_text=query_text,
        previous_context_nodes=previous_context_nodes,
        transitions=transitions,
    )
    if latent_matches:
        top_match = latent_matches[0]
        latent_context = top_match.context
        latent_scope = normalize_text(latent_context.scope)
        if latent_scope and task_scope and (
            latent_scope == task_scope
            or latent_scope in task_scope
            or task_scope in latent_scope
        ):
            scope_matched = True
            reasons.append(f"latent_scope:{latent_context.scope}")
        if set(latent_context.tags) & current_keywords or set(latent_context.keywords) & current_keywords:
            keyword_matched = True
            overlap = sorted((set(latent_context.tags) | set(latent_context.keywords)) & current_keywords)
            if overlap:
                reasons.append("latent_terms:" + "/".join(overlap[:4]))
        score += min(7, round(top_match.responsibility * 10))
        score += min(4, round(latent_context.expected_gain))
        score += min(3, round(latent_context.confidence_score * 3))
        reasons.append(f"latent_resp:{top_match.responsibility:.2f}")
        reasons.append(f"latent_gain:{latent_context.expected_gain:.1f}")
        reasons.append(f"latent_conf:{latent_context.confidence_score:.2f}")
    else:
        if task_scope and rule.context_mode == "local":
            score -= 2
            reasons.append("local_scope_mismatch")

    if not keyword_matched:
        overlap = sorted(current_keywords & set(rule.applies_when_tags or rule.tags))
        if overlap:
            keyword_matched = True
            score += min(3, len(overlap))
            reasons.append("shared_terms:" + "/".join(overlap[:3]))

    if rule.context_mode == "general":
        score += 2
        reasons.append("general")
    elif rule.context_mode == "mixed":
        score += 1
        reasons.append("mixed")

    if novelty_probability > 0 and novelty_probability < 0.45:
        score += 1
        reasons.append(f"novelty:{novelty_probability:.2f}")

    if rule.expected_gain > 0:
        score += min(5, round(rule.expected_gain / 1.2))
        reasons.append(f"gain:{rule.expected_gain:.1f}")

    if rule.confidence_score > 0:
        score += min(2, round(rule.confidence_score * 2))
        reasons.append(f"confidence:{rule.confidence_score:.2f}")

    if rule.support_score > 0:
        score += min(2, round(rule.support_score / 2))
        reasons.append(f"support:{rule.support_score:.1f}")

    if "needs_revision" in rule.tags:
        score -= 2
        reasons.append("needs_revision")

    # Semantic similarity bonus: catches cases like "窮屈" matching "余白を作れ"
    semantic_hit = False
    semantic_similarity_score = 0.0
    if query_text:
        from .memory_manager import semantic_similarity

        sim = semantic_similarity(query_text, rule.instruction)
        if latent_contexts:
            sim = max(
                sim,
                max(
                    semantic_similarity(query_text, context.prototype_text or " ".join(context.tags))
                    for context in latent_contexts
                ),
            )
        semantic_similarity_score = sim
        if sim >= 0.2:
            semantic_hit = True
            score += min(3, int(sim * 6))
            reasons.append(f"semantic:{sim:.2f}")

    probabilities = [match.responsibility for match in latent_matches if match.responsibility > 0]
    if novelty_probability > 0:
        probabilities.append(novelty_probability)
    posterior_entropy = 0.0
    if len(probabilities) >= 2:
        entropy_base = math.log(len(probabilities))
        if entropy_base > 0:
            posterior_entropy = -sum(prob * math.log(max(prob, 1e-9)) for prob in probabilities) / entropy_base

    top_match = latent_matches[0] if latent_matches else None
    second_match = latent_matches[1] if len(latent_matches) >= 2 else None
    top_posterior = top_match.responsibility if top_match is not None else 0.0
    second_posterior = second_match.responsibility if second_match is not None else 0.0
    posterior_gap = max(0.0, top_posterior - second_posterior)
    should_abstain = False
    abstain_reason = ""
    if top_match is None and rule.context_mode == "local":
        should_abstain = True
        abstain_reason = "local_without_context_match"
    elif top_posterior < 0.42 and novelty_probability >= 0.35:
        should_abstain = True
        abstain_reason = "uncertain_context"
    elif posterior_gap < 0.08 and posterior_entropy >= 0.72:
        should_abstain = True
        abstain_reason = "ambiguous_context"
    elif rule.expected_gain < 0.8 and rule.confidence_score < 0.45:
        should_abstain = True
        abstain_reason = "weak_rule_value"

    snapshot = {
        "rule_id": rule.id,
        "selected_for_guidance": False,
        "should_abstain": should_abstain,
        "abstain_reason": abstain_reason,
        "scope_matched": scope_matched,
        "keyword_matched": keyword_matched,
        "semantic_hit": semantic_hit,
        "semantic_similarity": round(semantic_similarity_score, 4),
        "novelty_probability": round(novelty_probability, 4),
        "posterior_entropy": round(posterior_entropy, 4),
        "top_context_id": top_match.context.id if top_match is not None else "",
        "top_context_scope": top_match.context.scope if top_match is not None else "",
        "top_context_tags": top_match.context.tags[:4] if top_match is not None else [],
        "top_context_keywords": top_match.context.keywords[:6] if top_match is not None else [],
        "top_context_signature": (
            build_context_signature(
                top_match.context.scope,
                top_match.context.tags,
                top_match.context.keywords,
            )
            if top_match is not None
            else ""
        ),
        "top_context_posterior": round(top_posterior, 4),
        "second_context_id": second_match.context.id if second_match is not None else "",
        "second_context_posterior": round(second_posterior, 4),
        "posterior_gap": round(posterior_gap, 4),
        "latent_context_matches": [
            {
                "context_id": match.context.id,
                "context_scope": match.context.scope,
                "posterior": round(match.responsibility, 4),
                "raw_score": round(match.raw_score, 4),
                "reason": match.reason,
            }
            for match in latent_matches[:4]
        ],
    }

    # --- Friston: surprise-weighted attention ---
    # Rules where confidence diverges from actual outcomes get attention boost.
    # High confidence + poor outcomes = prediction error = surprise = must attend.
    if hasattr(rule, 'confidence_score') and rule.confidence_score and rule.confidence_score > 0.5:
        # Check if recent evidence contradicts confidence
        actual_success_rate = (
            rule.success_count / max(1, rule.success_count + rule.failure_count)
            if hasattr(rule, 'success_count') and rule.success_count is not None
            else rule.confidence_score
        )
        prediction_error = abs(rule.confidence_score - actual_success_rate)
        if prediction_error > 0.2:
            surprise_bonus = min(3, round(prediction_error * 5))
            score += surprise_bonus
            reasons.append(f"surprise:{prediction_error:.2f}")

    if score <= 0:
        return 0, " | ".join(reasons), snapshot
    if rule.context_mode == "local" and not scope_matched and not keyword_matched and not semantic_hit:
        snapshot["should_abstain"] = True
        snapshot["abstain_reason"] = "local_without_match"
        return 0, "local_without_match", snapshot

    if not snapshot["should_abstain"]:
        snapshot["selected_for_guidance"] = True

    return score, " | ".join(reasons), snapshot


def _score_conversation_turn(
    turn: ConversationTurn,
    *,
    task_scope: str,
    current_keywords: set[str],
    query_text: str = "",
) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    scope_matched = False
    keyword_matched = False

    turn_scope = normalize_text(turn.task_scope)
    if task_scope and task_scope and task_scope in turn_scope:
        scope_matched = True
        score += 4
        reasons.append("same_scope")

    turn_keywords = set(turn.tags) | extract_keywords(
        turn.user_feedback,
        " ".join(turn.extracted_corrections),
    )
    overlap = sorted(current_keywords & turn_keywords)
    if overlap:
        keyword_matched = True
        score += min(4, len(overlap))
        reasons.append("shared_terms:" + "/".join(overlap[:3]))

    semantic_hit = False
    if query_text:
        from .memory_manager import semantic_similarity

        correction_text = " ".join(turn.extracted_corrections)
        sim = max(
            semantic_similarity(query_text, correction_text),
            semantic_similarity(query_text, turn.user_feedback),
        )
        if sim >= 0.2:
            semantic_hit = True
            score += min(3, int(sim * 6))
            reasons.append(f"semantic:{sim:.2f}")

    if turn.extracted_corrections and (scope_matched or keyword_matched or semantic_hit):
        score += 1
        reasons.append("has_correction")

    if not scope_matched and not keyword_matched and not semantic_hit:
        return 0, "no_context_match"

    return score, " | ".join(reasons)


def get_relevant_preference_rules(
    rules: list[PreferenceRule],
    *,
    task_scope: str = "",
    raw_text: str = "",
    limit: int = 5,
    previous_context_nodes: list[dict] | None = None,
    transitions: list[LatentTransition] | None = None,
) -> list[dict]:
    current_scope = normalize_text(task_scope)
    current_keywords = extract_keywords(task_scope, raw_text[:1800])
    ranked: list[dict] = []

    for rule in rules:
        if not rule.statement:
            continue
        score, reason, snapshot = _score_preference_rule(
            rule,
            task_scope=current_scope,
            current_keywords=current_keywords,
            query_text=raw_text,
            previous_context_nodes=previous_context_nodes,
            transitions=transitions,
        )
        if score <= 0:
            continue
        ranked.append(
            {
                "statement": rule.statement,
                "instruction": rule.instruction,
                "status": rule.status,
                "reason": reason,
                "evidence_count": rule.evidence_count,
                "applies_to_scope": rule.applies_to_scope,
                "applies_when_tags": rule.applies_when_tags,
                "negative_conditions": rule.negative_conditions,
                "priority": rule.priority,
                "context_mode": rule.context_mode,
                "support_score": rule.support_score,
                "expected_gain": rule.expected_gain,
                "confidence_score": rule.confidence_score,
                "strong_signal_count": rule.strong_signal_count,
                "latent_context_count": len(infer_latent_contexts_from_rule(rule)),
                "selected_for_guidance": bool(snapshot.get("selected_for_guidance", False)),
                "should_abstain": bool(snapshot.get("should_abstain", False)),
                "abstain_reason": snapshot.get("abstain_reason", ""),
                "top_context_id": snapshot.get("top_context_id", ""),
                "top_context_scope": snapshot.get("top_context_scope", ""),
                "top_context_tags": snapshot.get("top_context_tags", []),
                "top_context_keywords": snapshot.get("top_context_keywords", []),
                "top_context_signature": snapshot.get("top_context_signature", ""),
                "top_context_posterior": snapshot.get("top_context_posterior", 0.0),
                "second_context_id": snapshot.get("second_context_id", ""),
                "second_context_posterior": snapshot.get("second_context_posterior", 0.0),
                "posterior_gap": snapshot.get("posterior_gap", 0.0),
                "posterior_entropy": snapshot.get("posterior_entropy", 0.0),
                "novelty_probability": snapshot.get("novelty_probability", 0.0),
                "semantic_similarity": snapshot.get("semantic_similarity", 0.0),
                "latent_context_matches": snapshot.get("latent_context_matches", []),
                "score": score,
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["evidence_count"]), reverse=True)
    return ranked[:limit]


def get_relevant_conversation_corrections(
    turns: list[ConversationTurn],
    *,
    task_scope: str = "",
    raw_text: str = "",
    limit: int = 3,
) -> list[dict]:
    current_scope = normalize_text(task_scope)
    current_keywords = extract_keywords(task_scope, raw_text[:1800])
    ranked: list[dict] = []

    for turn in turns:
        if not turn.extracted_corrections:
            continue
        score, reason = _score_conversation_turn(
            turn,
            task_scope=current_scope,
            current_keywords=current_keywords,
            query_text=raw_text,
        )
        if score <= 0:
            continue
        ranked.append(
            {
                "task_scope": turn.task_scope,
                "user_feedback": turn.user_feedback,
                "extracted_corrections": turn.extracted_corrections,
                "reason": reason,
                "score": score,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def build_guidance_context(
    entries: list[EpisodeRecord],
    *,
    company_profile: dict | None = None,
    task_title: str = "",
    issuer: str = "",
    raw_text: str = "",
    limit: int = 3,
) -> str:
    relevant = get_relevant_corrections(
        entries,
        company_profile=company_profile,
        task_title=task_title,
        issuer=issuer,
        raw_text=raw_text,
        limit=limit,
    )
    if not relevant:
        return ""

    lines = [
        "# HUMAN CORRECTION MEMORY",
        "Use these notes only as strategic guidance. Do not treat them as facts about the current task.",
    ]
    for item in relevant:
        latest = item["latest"]
        lines.extend(
            [
                f"## Previous case: {item['title']} / {item['issuer']}",
                f"- Why relevant: {item['reason']}",
                f"- Actual override: {latest.get('decision_override') or 'No override recorded'}",
                f"- Human correction: {latest.get('correction_note') or 'No correction note'}",
                f"- Reuse guidance: {latest.get('reuse_note') or 'No reuse guidance'}",
            ]
        )
    return "\n".join(lines)


def build_conversation_guidance(
    turns: list[ConversationTurn],
    rules: list[PreferenceRule],
    *,
    task_scope: str = "",
    raw_text: str = "",
    rule_limit: int = 4,
    correction_limit: int = 3,
    previous_context_nodes: list[dict] | None = None,
    transitions: list[LatentTransition] | None = None,
    meanings: list | None = None,
) -> str:
    relevant_rules = get_relevant_preference_rules(
        rules,
        task_scope=task_scope,
        raw_text=raw_text,
        limit=max(rule_limit * 2, rule_limit),
        previous_context_nodes=previous_context_nodes,
        transitions=transitions,
    )
    selected_rules = [item for item in relevant_rules if item.get("selected_for_guidance", False)][:rule_limit]
    relevant_turns = get_relevant_conversation_corrections(
        turns,
        task_scope=task_scope,
        raw_text=raw_text,
        limit=correction_limit,
    )

    if not selected_rules and not relevant_turns:
        return ""

    lines = [
        "# USER PREFERENCE MEMORY",
        "Use these as stylistic and structural guidance. They are not facts about the current task.",
    ]

    if selected_rules:
        lines.append("## Contextual rules")
        for item in selected_rules:
            qualifiers: list[str] = []
            if item.get("applies_to_scope"):
                qualifiers.append(f"scope={item['applies_to_scope']}")
            if item.get("applies_when_tags"):
                qualifiers.append("tags=" + "/".join(item["applies_when_tags"][:4]))
            if item.get("negative_conditions"):
                qualifiers.append("avoid=" + " / ".join(item["negative_conditions"][:2]))
            if item.get("context_mode"):
                qualifiers.append(f"mode={item['context_mode']}")
            if item.get("latent_context_count"):
                qualifiers.append(f"latent={item['latent_context_count']}")
            if item.get("top_context_scope"):
                qualifiers.append(f"top={item['top_context_scope']}")
            lines.append(
                f"- {item['statement']} (reason: {item['reason']}; gain: {item['expected_gain']:.1f}; confidence: {item['confidence_score']:.2f}; evidence: {item['evidence_count']}; {'; '.join(qualifiers) if qualifiers else 'general'})"
            )

    if relevant_turns:
        lines.append("## Recent conversation corrections")
        for item in relevant_turns:
            lines.append(f"- Scope: {item['task_scope'] or 'generic'} ({item['reason']})")
            for correction in item["extracted_corrections"][:2]:
                lines.append(f"  - {correction}")

    # Inject meanings (emergent value principles)
    if meanings:
        top_meanings = sorted(
            [m for m in meanings if getattr(m, "status", "") == "active"],
            key=lambda m: (-getattr(m, "strength", 0), -getattr(m, "cross_scope_count", 0)),
        )[:3]
        if top_meanings:
            lines.append("## Value Principles (emergent from multiple rules)")
            for m in top_meanings:
                scopes_str = "/".join(getattr(m, "scopes", [])[:3])
                lines.append(
                    f"- {getattr(m, 'principle', '')} "
                    f"(strength: {getattr(m, 'strength', 0)}, scopes: {scopes_str})"
                )

    return "\n".join(lines)
