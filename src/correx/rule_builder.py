"""rule_builder.py — Pure rule construction logic.

Extracted from history_store.py to isolate the statistical rule-building
logic from I/O concerns. No file access, no locks, no side effects.
"""
from __future__ import annotations

from collections import Counter

from .conversation_learning import (
    extract_keywords,
    is_explicit_directive,
    normalize_correction_statement,
    normalize_text,
)
from .memory_manager import (
    derive_context_mode,
    derive_latent_context_confidence_score,
    derive_latent_context_expected_gain,
    derive_latent_context_prior_weight,
    derive_rule_confidence_score,
    derive_rule_expected_gain,
    derive_rule_status,
    flatten_latent_contexts,
    merge_rule_contexts,
    merge_latent_contexts,
)
from .schemas import (
    ConversationTurn,
    LatentContext,
    PreferenceRule,
    RuleContext,
)


def build_preference_rules(
    turns: list[ConversationTurn],
    min_promoted_evidence: int = 1,
) -> list[PreferenceRule]:
    """Build preference rules from conversation turns.

    Pure function — no I/O, no side effects.
    Groups corrections by normalized statement, computes confidence/gain
    scores, and derives promotion status for each rule candidate.
    """
    grouped: dict[str, dict] = {}
    for turn in reversed(turns):
        metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
        if metadata.get("exclude_from_preference_rules"):
            continue
        if str(metadata.get("auto_saved_by", "")).strip() == "stop_reminder":
            continue
        for correction in turn.extracted_corrections:
            normalized = normalize_correction_statement(correction)
            if not normalized:
                continue
            scope_value = turn.task_scope.strip()
            explicit_directive = is_explicit_directive(correction) or is_explicit_directive(turn.user_feedback)
            reaction_score = turn.reaction_score
            support_increment = 1.0
            if explicit_directive:
                support_increment += 0.9
            if reaction_score is not None:
                support_increment += max(0.0, 0.6 - min(0.6, reaction_score))
            bucket = grouped.setdefault(
                normalized,
                {
                    "statement": correction.strip(),
                    "first_recorded_at": turn.recorded_at,
                    "last_recorded_at": turn.recorded_at,
                    "tags": set(turn.tags),
                    "tag_counter": Counter(),
                    "scope_counter": Counter(),
                    "context_map": {},
                    "latent_context_map": {},
                    "source_turn_ids": [],
                    "evidence_count": 0,
                    "min_reaction_score": turn.reaction_score,
                    "strong_signal_count": 0,
                    "support_score": 0.0,
                    "success_count": 0,
                    "failure_count": 0,
                },
            )
            bucket["statement"] = correction.strip() or bucket["statement"]
            bucket["last_recorded_at"] = turn.recorded_at
            bucket["tags"].update(turn.tags)
            for tag in turn.tags:
                normalized_tag = str(tag).strip()
                if normalized_tag:
                    bucket["tag_counter"][normalized_tag] += 1
            if scope_value:
                bucket["scope_counter"][scope_value] += 1
            bucket["source_turn_ids"].append(turn.id)
            bucket["evidence_count"] += 1
            bucket["support_score"] += support_increment
            if explicit_directive or (reaction_score is not None and reaction_score <= 0.25):
                bucket["strong_signal_count"] += 1
            if reaction_score is not None and reaction_score >= 0.7:
                bucket["success_count"] += 1
            elif reaction_score is not None and reaction_score <= 0.3:
                bucket["failure_count"] += 1
            if turn.reaction_score is not None:
                prev = bucket["min_reaction_score"]
                if prev is None or turn.reaction_score < prev:
                    bucket["min_reaction_score"] = turn.reaction_score

            normalized_scope = normalize_text(scope_value)
            normalized_turn_tags = [
                normalize_text(str(tag).strip())
                for tag in turn.tags
                if normalize_text(str(tag).strip())
            ]
            turn_keywords = list(
                sorted(
                    extract_keywords(
                        turn.task_scope,
                        turn.user_message[:400],
                        turn.user_feedback[:400],
                    )
                )
            )[:6]
            latent_key = (
                normalized_scope,
                tuple(sorted(dict.fromkeys(normalized_turn_tags))[:4]),
            )
            if normalized_scope or normalized_turn_tags:
                latent_bucket = bucket["latent_context_map"].setdefault(
                    latent_key,
                    {
                        "id": f"latent-{len(bucket['latent_context_map']) + 1}",
                        "scope": scope_value,
                        "tags": list(sorted(dict.fromkeys(normalized_turn_tags)))[:4],
                        "keywords": turn_keywords,
                        "prototype_text": " ".join(
                            part
                            for part in [
                                scope_value,
                                " ".join(list(sorted(dict.fromkeys(normalized_turn_tags)))[:4]),
                                " ".join(turn_keywords[:4]),
                            ]
                            if part
                        ).strip(),
                        "evidence_count": 0.0,
                        "support_score": 0.0,
                        "posterior_mass": 0.0,
                        "strong_signal_count": 0.0,
                        "success_mass": 0.0,
                        "failure_mass": 0.0,
                        "last_seen_at": turn.recorded_at,
                    },
                )
                latent_bucket["scope"] = scope_value or latent_bucket["scope"]
                latent_bucket["tags"] = list(
                    dict.fromkeys(latent_bucket.get("tags", []) + list(sorted(dict.fromkeys(normalized_turn_tags)))[:4])
                )[:4]
                latent_bucket["keywords"] = list(
                    dict.fromkeys(latent_bucket.get("keywords", []) + turn_keywords)
                )[:6]
                latent_bucket["prototype_text"] = " ".join(
                    part
                    for part in [
                        latent_bucket.get("scope", ""),
                        " ".join(latent_bucket.get("tags", [])[:4]),
                        " ".join(latent_bucket.get("keywords", [])[:4]),
                    ]
                    if part
                ).strip()
                latent_bucket["evidence_count"] += 1.0
                latent_bucket["support_score"] += support_increment
                latent_bucket["posterior_mass"] += 1.0
                latent_bucket["last_seen_at"] = turn.recorded_at
                if explicit_directive or (reaction_score is not None and reaction_score <= 0.25):
                    latent_bucket["strong_signal_count"] += 1.0
                if reaction_score is not None and reaction_score >= 0.7:
                    latent_bucket["success_mass"] += 1.0
                elif reaction_score is not None and reaction_score <= 0.3:
                    latent_bucket["failure_mass"] += 1.0

            context_keys: list[tuple[str, str]] = []
            if scope_value:
                context_keys.append(("scope", scope_value))
            for tag in turn.tags:
                normalized_tag = str(tag).strip()
                if normalized_tag:
                    context_keys.append(("tag", normalized_tag))

            for kind, value in context_keys:
                context_bucket = bucket["context_map"].setdefault(
                    (kind, value),
                    {
                        "kind": kind,
                        "value": value,
                        "evidence_count": 0,
                        "reaction_min": reaction_score,
                        "reaction_max": reaction_score,
                        "last_seen_at": turn.recorded_at,
                        "utility_score": 0.0,
                        "strong_signal_count": 0,
                        "success_count": 0,
                        "failure_count": 0,
                    },
                )
                context_bucket["evidence_count"] += 1
                context_bucket["last_seen_at"] = turn.recorded_at
                context_bucket["utility_score"] += support_increment + 0.35
                if explicit_directive or (reaction_score is not None and reaction_score <= 0.25):
                    context_bucket["strong_signal_count"] += 1
                if reaction_score is not None and reaction_score >= 0.7:
                    context_bucket["success_count"] += 1
                elif reaction_score is not None and reaction_score <= 0.3:
                    context_bucket["failure_count"] += 1
                if reaction_score is not None:
                    previous_min = context_bucket["reaction_min"]
                    previous_max = context_bucket["reaction_max"]
                    if previous_min is None or reaction_score < previous_min:
                        context_bucket["reaction_min"] = reaction_score
                    if previous_max is None or reaction_score > previous_max:
                        context_bucket["reaction_max"] = reaction_score

    rules: list[PreferenceRule] = []
    for normalized, bucket in grouped.items():
        evidence_count = int(bucket["evidence_count"])
        statement = str(bucket["statement"])
        applies_to_scope = ""
        if bucket["scope_counter"]:
            applies_to_scope = bucket["scope_counter"].most_common(1)[0][0]
        tag_counter: Counter[str] = bucket["tag_counter"]
        applies_when_tags = [tag for tag, _ in tag_counter.most_common(12)]
        latent_contexts = merge_latent_contexts(
            [
                LatentContext(
                    id=str(context.get("id", "")),
                    scope=str(context.get("scope", "")).strip(),
                    tags=[
                        str(entry).strip()
                        for entry in context.get("tags", [])
                        if str(entry).strip()
                    ],
                    keywords=[
                        str(entry).strip()
                        for entry in context.get("keywords", [])
                        if str(entry).strip()
                    ],
                    prototype_text=str(context.get("prototype_text", "")).strip(),
                    evidence_count=float(context.get("evidence_count", 0.0) or 0.0),
                    support_score=float(context.get("support_score", 0.0) or 0.0),
                    expected_gain=derive_latent_context_expected_gain(
                        support_score=float(context.get("support_score", 0.0) or 0.0),
                        confidence_score=derive_latent_context_confidence_score(
                            evidence_count=float(context.get("evidence_count", 0.0) or 0.0),
                            support_score=float(context.get("support_score", 0.0) or 0.0),
                            posterior_mass=float(context.get("posterior_mass", 0.0) or 0.0),
                            strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                            success_mass=float(context.get("success_mass", 0.0) or 0.0),
                            failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                        ),
                        strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                        success_mass=float(context.get("success_mass", 0.0) or 0.0),
                        failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                    ),
                    confidence_score=derive_latent_context_confidence_score(
                        evidence_count=float(context.get("evidence_count", 0.0) or 0.0),
                        support_score=float(context.get("support_score", 0.0) or 0.0),
                        posterior_mass=float(context.get("posterior_mass", 0.0) or 0.0),
                        strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                        success_mass=float(context.get("success_mass", 0.0) or 0.0),
                        failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                    ),
                    prior_weight=derive_latent_context_prior_weight(
                        evidence_count=float(context.get("evidence_count", 0.0) or 0.0),
                        support_score=float(context.get("support_score", 0.0) or 0.0),
                        expected_gain=derive_latent_context_expected_gain(
                            support_score=float(context.get("support_score", 0.0) or 0.0),
                            confidence_score=derive_latent_context_confidence_score(
                                evidence_count=float(context.get("evidence_count", 0.0) or 0.0),
                                support_score=float(context.get("support_score", 0.0) or 0.0),
                                posterior_mass=float(context.get("posterior_mass", 0.0) or 0.0),
                                strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                                success_mass=float(context.get("success_mass", 0.0) or 0.0),
                                failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                            ),
                            strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                            success_mass=float(context.get("success_mass", 0.0) or 0.0),
                            failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                        ),
                        confidence_score=derive_latent_context_confidence_score(
                            evidence_count=float(context.get("evidence_count", 0.0) or 0.0),
                            support_score=float(context.get("support_score", 0.0) or 0.0),
                            posterior_mass=float(context.get("posterior_mass", 0.0) or 0.0),
                            strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                            success_mass=float(context.get("success_mass", 0.0) or 0.0),
                            failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                        ),
                        posterior_mass=float(context.get("posterior_mass", 0.0) or 0.0),
                    ),
                    posterior_mass=float(context.get("posterior_mass", 0.0) or 0.0),
                    strong_signal_count=float(context.get("strong_signal_count", 0.0) or 0.0),
                    success_mass=float(context.get("success_mass", 0.0) or 0.0),
                    failure_mass=float(context.get("failure_mass", 0.0) or 0.0),
                    last_seen_at=str(context.get("last_seen_at", "")),
                )
                for context in bucket["latent_context_map"].values()
            ]
        )
        contexts = merge_rule_contexts(
            [
                RuleContext(
                    kind=str(context["kind"]),
                    value=str(context["value"]),
                    evidence_count=int(context["evidence_count"]),
                    reaction_min=(
                        float(context["reaction_min"])
                        if isinstance(context.get("reaction_min"), (int, float))
                        else None
                    ),
                    reaction_max=(
                        float(context["reaction_max"])
                        if isinstance(context.get("reaction_max"), (int, float))
                        else None
                    ),
                    last_seen_at=str(context.get("last_seen_at", "")),
                    utility_score=float(context.get("utility_score", 0.0) or 0.0),
                    strong_signal_count=int(context.get("strong_signal_count", 0) or 0),
                    success_count=int(context.get("success_count", 0) or 0),
                    failure_count=int(context.get("failure_count", 0) or 0),
                )
                for context in bucket["context_map"].values()
            ]
        )
        if latent_contexts:
            contexts = flatten_latent_contexts(latent_contexts)
        distinct_scope_count = len([context for context in contexts if context.kind == "scope"])
        distinct_tag_count = len([context for context in contexts if context.kind == "tag"])
        context_mode = derive_context_mode(
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
            evidence_count=evidence_count,
        )
        support_score = round(float(bucket["support_score"]), 4)
        success_count = int(bucket["success_count"])
        failure_count = int(bucket["failure_count"])
        confidence_score = derive_rule_confidence_score(
            evidence_count=evidence_count,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
            strong_signal_count=int(bucket["strong_signal_count"]),
            success_count=success_count,
            failure_count=failure_count,
        )
        expected_gain = derive_rule_expected_gain(
            support_score=support_score,
            context_utility=max(
                sum(context.utility_score for context in contexts[:3]) / max(1, min(3, len(contexts))),
                sum(context.expected_gain for context in latent_contexts[:3]) / max(1, min(3, len(latent_contexts))),
            ),
            confidence_score=confidence_score,
            strong_signal_count=int(bucket["strong_signal_count"]),
            success_count=success_count,
            failure_count=failure_count,
            context_mode=context_mode,
        )
        status = derive_rule_status(
            evidence_count=evidence_count,
            support_score=support_score,
            strong_signal_count=int(bucket["strong_signal_count"]),
            context_mode=context_mode,
            expected_gain=expected_gain,
            confidence_score=confidence_score,
            min_promoted_evidence=min_promoted_evidence,
        )
        negative_conditions: list[str] = []
        # Note: do NOT add the rule's own statement as a negative condition.
        # That was a self-referential bug causing rules to conflict with themselves.
        priority = max(1, min(5, round(max(support_score, expected_gain))))
        rule = PreferenceRule(
            id=f"pref-{normalized[:48].replace(' ', '-')}",
            statement=statement,
            normalized_statement=normalized,
            instruction=statement,
            status=status,
            evidence_count=evidence_count,
            first_recorded_at=str(bucket["first_recorded_at"]),
            last_recorded_at=str(bucket["last_recorded_at"]),
            applies_to_scope=applies_to_scope,
            applies_when_tags=applies_when_tags,
            negative_conditions=negative_conditions[:6],
            priority=priority,
            version=1,
            tags=sorted(bucket["tags"])[:20],
            source_turn_ids=list(dict.fromkeys(bucket["source_turn_ids"]))[:20],
            contexts=contexts[:14],
            latent_contexts=latent_contexts[:8],
            context_mode=context_mode,
            support_score=support_score,
            expected_gain=expected_gain,
            confidence_score=confidence_score,
            strong_signal_count=int(bucket["strong_signal_count"]),
            success_count=success_count,
            failure_count=failure_count,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
        )
        rules.append(rule)

    rules.sort(
        key=lambda rule: (
            rule.expected_gain,
            rule.confidence_score,
            rule.support_score,
            rule.evidence_count,
            rule.last_recorded_at,
        ),
        reverse=True,
    )
    # Keep all promoted rules + top 200 candidates by score
    promoted = [r for r in rules if r.status == "promoted"]
    candidates = [r for r in rules if r.status != "promoted"][:200]
    return promoted + candidates


def synthesize_rules_from_turns(turns: list[ConversationTurn]) -> list[dict]:
    """Generate rule hypotheses from success/failure pattern differences.

    Pure function — no I/O. Derives candidate rules without human input,
    based on statistical patterns in past interactions.
    """
    success = [t for t in turns if (t.reaction_score or 0) >= 0.7]
    failure = [t for t in turns if (t.reaction_score or 0) < 0.4]
    if len(success) < 2 or len(failure) < 2:
        return []

    success_tags = Counter(tag for t in success for tag in (t.tags or []))
    failure_tags = Counter(tag for t in failure for tag in (t.tags or []))
    success_scopes = Counter(t.task_scope for t in success if t.task_scope)
    failure_scopes = Counter(t.task_scope for t in failure if t.task_scope)

    hypotheses = []

    for tag, count in success_tags.most_common(30):
        if len(tag) < 4:
            continue
        success_rate = count / len(success)
        failure_rate = failure_tags.get(tag, 0) / max(1, len(failure))
        if success_rate > 0.3 and failure_rate < 0.1:
            hypotheses.append({
                "instruction": f"タスクに「{tag}」が関連するとき、成功パターンを適用せよ (成功率{success_rate:.0%})",
                "source": "synthesized",
                "confidence": round(success_rate - failure_rate, 2),
                "tag": tag,
            })

    for scope, s_count in success_scopes.most_common(10):
        f_count = failure_scopes.get(scope, 0)
        if s_count > f_count * 2 and s_count >= 2:
            hypotheses.append({
                "instruction": f"スコープ「{scope}」では現在の方針が効いている。維持せよ",
                "source": "synthesized",
                "confidence": round(s_count / (s_count + f_count), 2),
                "tag": scope,
            })

    success_corrections = []
    for t in success:
        for c in (t.extracted_corrections or []):
            if len(c) > 10:
                success_corrections.append(c)
    correction_counts = Counter(success_corrections)
    for correction, count in correction_counts.most_common(5):
        if count >= 2:
            hypotheses.append({
                "instruction": correction,
                "source": "synthesized_from_success",
                "confidence": round(min(0.95, count * 0.2), 2),
                "tag": "success_pattern",
            })

    return hypotheses[:7]


def compute_self_overcome_proposals(rules: list[PreferenceRule]) -> list[dict]:
    """Identify weak, redundant, or contradicting rules and propose improvements.

    Pure function — no I/O. The system transcends itself.
    """
    promoted = [r for r in rules if r.status == "promoted"]
    if len(promoted) < 5:
        return []

    proposals = []

    for r in promoted:
        conf = getattr(r, "confidence_score", None) or 0
        ev = r.evidence_count
        if conf < 0.65 and ev <= 1:
            proposals.append({
                "action": "demote",
                "rule_id": r.id,
                "instruction": r.instruction,
                "reason": f"confidence={conf:.2f}, evidence={ev}: 根拠が薄い。降格して再検証すべき",
            })

    seen_pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(promoted):
        a_text = a.instruction.lower()
        a_bigrams = {a_text[k:k+2] for k in range(len(a_text)-1)}
        for j, b in enumerate(promoted):
            if j <= i:
                continue
            pair_key = (min(a.id, b.id), max(a.id, b.id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            b_text = b.instruction.lower()
            b_bigrams = {b_text[k:k+2] for k in range(len(b_text)-1)}
            if not a_bigrams or not b_bigrams:
                continue
            overlap = len(a_bigrams & b_bigrams) / min(len(a_bigrams), len(b_bigrams))
            if overlap > 0.4 and a.applies_to_scope == b.applies_to_scope:
                keep = a if (getattr(a, "confidence_score", 0) or 0) >= (getattr(b, "confidence_score", 0) or 0) else b
                drop = b if keep == a else a
                proposals.append({
                    "action": "merge",
                    "keep_id": keep.id,
                    "drop_id": drop.id,
                    "keep_instruction": keep.instruction,
                    "drop_instruction": drop.instruction,
                    "reason": f"overlap={overlap:.0%}: 類似ルールの統合。より強い方を残す",
                })

    scope_counts = Counter(r.applies_to_scope for r in promoted)
    for scope, count in scope_counts.items():
        if count >= 3:
            scope_rules = [r for r in promoted if r.applies_to_scope == scope]
            all_tokens: Counter[str] = Counter()
            for r in scope_rules:
                for token in r.instruction.lower().split():
                    if len(token) > 3:
                        all_tokens[token] += 1
            common = [tok for tok, c in all_tokens.most_common(5) if c >= 2]
            if common:
                proposals.append({
                    "action": "generalize",
                    "scope": scope,
                    "rule_count": count,
                    "common_themes": common[:5],
                    "reason": f"scope={scope}に{count}件のルール。共通テーマ: {', '.join(common[:3])}",
                })

    for i, a in enumerate(promoted):
        for j, b in enumerate(promoted):
            if j <= i:
                continue
            if a.applies_to_scope == b.applies_to_scope:
                a_neg = set(getattr(a, "negative_conditions", []) or [])
                if b.instruction in a_neg or a.instruction in (getattr(b, "negative_conditions", []) or []):
                    proposals.append({
                        "action": "resolve_conflict",
                        "rule_a_id": a.id,
                        "rule_b_id": b.id,
                        "rule_a": a.instruction,
                        "rule_b": b.instruction,
                        "reason": "直接矛盾。文脈限定で解決するか、一方を降格すべき",
                    })

    return proposals[:10]
