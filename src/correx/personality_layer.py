"""Personality Layer — per-user cognitive metabolism parameters.

Infers 5 personality dimensions from conversation history:
  1. metabolism_rate  — how fast the user discards/adopts rules (stable)
  2. reward_function  — what "success" means to this user (medium-speed)
  3. avoidance_function — what the user most dislikes (medium-speed)
  4. digestibility    — abstract vs concrete preference (stable)
  5. objective_drift  — real-time goal shift detection (fast)

Backed by:
  - PReF (MIT 2025): reward factorization into 5-15 basis functions
  - COMT polymorphism: individual dopamine metabolism differences
  - SHY (Tononi): synaptic homeostasis hypothesis
  - Cognitive Mirror (Frontiers 2025): mirror > oracle
  - Developer Field Study (2026): timing > content
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .schemas import ConversationTurn, PreferenceRule


# ── Personality Profile ─────────────────────────────────────────────────────

@dataclass(slots=True)
class PersonalityProfile:
    """The 6-parameter personality layer for a single user."""

    # 1. Metabolism: ratio of attack vs defense (0.0=very conservative, 1.0=very aggressive)
    metabolism_rate: float = 0.5
    metabolism_label: str = "balanced"

    # 2. Reward: what triggers highest positive reaction
    reward_keywords: list[str] = field(default_factory=list)
    reward_pattern: str = ""  # human-readable summary

    # 3. Avoidance: what triggers most negative reaction
    avoidance_keywords: list[str] = field(default_factory=list)
    avoidance_pattern: str = ""

    # 4. Digestibility: 0.0=needs very concrete, 1.0=handles abstract well
    digestibility: float = 0.5
    digestibility_label: str = "moderate"

    # 5. Objective drift detection
    current_objective: str = ""
    objective_confidence: float = 0.0
    drift_detected: bool = False
    drift_description: str = ""

    # 6. Curiosity: density of questions / exploratory behavior
    # 0.0=pragmatic (just wants results), 1.0=intellectually curious (asks why)
    curiosity_level: float = 0.5
    curiosity_label: str = "moderate"

    # Meta
    sample_size: int = 0
    computed_at: str = ""


# ── Intervention Signal ─────────────────────────────────────────────────────

@dataclass(slots=True)
class InterventionSignal:
    """A predicted cognitive trap the user is about to fall into."""

    pattern_type: str  # "premature_discard" | "stale_retention" | "repeated_failure" | "goal_drift"
    confidence: float  # 0.0-1.0
    evidence: str  # what data supports this
    mirror_prompt: str  # question to reflect back, not a command
    reward_frame: str  # why acting on this helps their current goal


# ── Metabolism Estimation ────────────────────────────────────────────────────

def _estimate_metabolism(
    rules: list[PreferenceRule],
    turns: list[ConversationTurn],
) -> tuple[float, str]:
    """Estimate attack/defense balance from rule lifecycle data.

    Attack signals: fast adoption (new rules promoted quickly), fast discarding (demotes)
    Defense signals: long-lived rules, resistance to change, re-promotions after demote
    """
    if not rules:
        return 0.5, "balanced"

    # Attack indicators
    demoted_count = sum(1 for r in rules if r.status == "demoted")
    conflict_demoted = sum(1 for r in rules if "conflict_demoted" in (r.tags or []))

    # Defense indicators
    promoted = [r for r in rules if r.status == "promoted"]
    high_evidence = sum(1 for r in promoted if r.evidence_count >= 4)
    restored = sum(1 for r in rules if "restored" in (r.tags or []))

    total = len(rules)
    attack_score = (demoted_count + conflict_demoted) / max(1, total)
    defense_score = (high_evidence + restored) / max(1, total)

    # Also check turn feedback patterns: frequent corrections = higher metabolism
    recent_turns = turns[:30] if turns else []
    correction_rate = sum(
        1 for t in recent_turns
        if t.reaction_score is not None and t.reaction_score < 0.5
    ) / max(1, len(recent_turns))

    raw = 0.5 + (attack_score - defense_score) * 2 + correction_rate * 0.3
    rate = max(0.0, min(1.0, raw))

    if rate > 0.65:
        label = "aggressive"
    elif rate < 0.35:
        label = "conservative"
    else:
        label = "balanced"

    return round(rate, 3), label


# ── Reward/Avoidance Extraction ──────────────────────────────────────────────

_STOPWORDS = frozenset({
    # Technical noise (file names, extensions, code terms)
    "py", "js", "ts", "tsx", "json", "md", "css", "html", "yaml", "toml",
    "src", "lib", "app", "api", "test", "tests", "config", "scripts",
    "schemas", "route", "page", "index", "main", "utils", "types",
    "import", "export", "class", "function", "return", "const", "let", "var",
    "auto_captured", "self", "none", "true", "false", "null",
    # Common filler words (JP + EN)
    "の", "は", "が", "を", "に", "で", "と", "も", "から", "まで", "より",
    "する", "した", "して", "される", "された", "ある", "ない", "いる",
    "これ", "それ", "あれ", "この", "その", "あの", "こと", "もの", "ため",
    "the", "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "that", "this", "it", "an",
})


def _extract_reward_avoidance(
    turns: list[ConversationTurn],
) -> tuple[list[str], str, list[str], str]:
    """Extract reward and avoidance patterns from high/low reaction scores.

    Filters out technical noise (file names, code terms) and common filler words.
    Only meaningful content words survive.
    """
    reward_words: dict[str, int] = {}
    avoid_words: dict[str, int] = {}

    for turn in turns:
        if turn.reaction_score is None:
            continue
        text = turn.user_feedback or ""
        words = [
            w for w in re.findall(r"[\w\u3040-\u9FFF]+", text.lower())
            if len(w) > 1 and w not in _STOPWORDS
        ]

        if turn.reaction_score >= 0.8:
            for w in words:
                reward_words[w] = reward_words.get(w, 0) + 1
        elif turn.reaction_score < 0.4:
            for w in words:
                avoid_words[w] = avoid_words.get(w, 0) + 1

    # Remove overlap (words appearing in both)
    overlap = set(reward_words) & set(avoid_words)
    for w in overlap:
        if reward_words[w] > avoid_words[w]:
            del avoid_words[w]
        else:
            del reward_words[w]

    top_reward = sorted(reward_words, key=reward_words.get, reverse=True)[:8]
    top_avoid = sorted(avoid_words, key=avoid_words.get, reverse=True)[:8]

    reward_pattern = ", ".join(top_reward[:4]) if top_reward else "not enough data"
    avoid_pattern = ", ".join(top_avoid[:4]) if top_avoid else "not enough data"

    return top_reward, reward_pattern, top_avoid, avoid_pattern


# ── Digestibility Estimation ─────────────────────────────────────────────────

def _estimate_digestibility(turns: list[ConversationTurn]) -> tuple[float, str]:
    """Estimate how well the user handles abstract vs concrete information.

    Signals (content-based, NOT length-based):
    - Concrete: asks for specifics, examples, how-to, numbers, steps
    - Abstract: asks for principles, meaning, reasons, concepts
    - Directive short commands ("やれ", "違う") are neutral, not abstract

    Length is not a reliable signal. "やれ" is short but concrete.
    "ひらめきのひらめきは終わりがない" is abstract regardless of length.
    """
    if not turns:
        return 0.5, "moderate"

    concrete_signals = 0
    abstract_signals = 0

    _CONCRETE_PATTERN = re.compile(
        r"(どうやって|具体的|例えば|how|example|具体|手順|ステップ"
        r"|いくら|何件|数字|コード|実装|直せ|作れ|やれ|動かせ"
        r"|step|number|code|implement|fix|build|show me)",
        re.IGNORECASE,
    )
    _ABSTRACT_PATTERN = re.compile(
        r"(なぜ|why|概念|本質|意味|つまり|原理|理論|仮説|哲学"
        r"|示唆|メタ|構造|抽象|パラダイム|フレームワーク"
        r"|principle|theory|concept|framework|paradigm|meta)",
        re.IGNORECASE,
    )

    for turn in turns[:50]:
        fb = turn.user_feedback or ""
        msg = turn.user_message or ""
        text = f"{fb} {msg}"

        if not text.strip():
            continue

        has_concrete = bool(_CONCRETE_PATTERN.search(text))
        has_abstract = bool(_ABSTRACT_PATTERN.search(text))

        if has_concrete:
            concrete_signals += 1
        if has_abstract:
            abstract_signals += 1
        # If neither matches, this turn doesn't contribute (neutral)

    total = concrete_signals + abstract_signals
    if total == 0:
        return 0.5, "moderate"

    score = abstract_signals / total
    score = max(0.0, min(1.0, score))

    if score > 0.65:
        label = "abstract-tolerant"
    elif score < 0.35:
        label = "concrete-preferring"
    else:
        label = "moderate"

    return round(score, 3), label


# ── Objective Drift Detection ────────────────────────────────────────────────

def _detect_objective_drift(
    turns: list[ConversationTurn],
    window: int = 5,
) -> tuple[str, float, bool, str]:
    """Detect if the user's goal has shifted in recent turns.

    Compares task_scope and keyword distribution between
    recent window and previous window.
    """
    if len(turns) < window:
        scope = turns[0].task_scope if turns else ""
        return scope, 0.5, False, ""

    recent = turns[:window]
    previous = turns[window:window * 2]

    recent_scopes = {t.task_scope for t in recent if t.task_scope}
    prev_scopes = {t.task_scope for t in previous if t.task_scope}

    current_objective = recent[0].task_scope if recent else ""

    if not prev_scopes:
        return current_objective, 0.5, False, ""

    # Scope overlap
    overlap = recent_scopes & prev_scopes
    total = recent_scopes | prev_scopes
    scope_similarity = len(overlap) / max(1, len(total))

    # Keyword overlap
    recent_kw = set()
    for t in recent:
        recent_kw.update(t.tags[:5] if t.tags else [])
    prev_kw = set()
    for t in previous:
        prev_kw.update(t.tags[:5] if t.tags else [])

    kw_overlap = len(recent_kw & prev_kw) / max(1, len(recent_kw | prev_kw))

    confidence = (scope_similarity + kw_overlap) / 2
    drift = confidence < 0.3

    description = ""
    if drift:
        description = f"Goal shifted: {', '.join(prev_scopes)} → {', '.join(recent_scopes)}"

    return current_objective, round(confidence, 3), drift, description


# ── Intervention Detection ───────────────────────────────────────────────────

def detect_interventions(
    rules: list[PreferenceRule],
    turns: list[ConversationTurn],
    profile: PersonalityProfile,
    escalated_clusters: list[dict] | None = None,
) -> list[InterventionSignal]:
    """Detect cognitive patterns that may cause failure.

    Returns mirror-style prompts, not commands.
    """
    signals: list[InterventionSignal] = []

    # Pattern 1: Premature discard — demoted then restored
    restored_rules = [r for r in rules if "restored" in (r.tags or [])]
    if restored_rules and profile.metabolism_rate > 0.5:
        signals.append(InterventionSignal(
            pattern_type="premature_discard",
            confidence=min(0.9, 0.5 + len(restored_rules) * 0.1),
            evidence=f"{len(restored_rules)} rules were discarded then brought back",
            mirror_prompt="You've restored rules you previously discarded. Want to keep this one in a holding pool instead of removing it?",
            reward_frame="Keeping it available could save rework later",
        ))

    # Pattern 2: Stale retention — promoted + repeated negative feedback
    for rule in rules:
        if rule.status != "promoted":
            continue
        failure_count = getattr(rule, "failure_count", 0) or 0
        success_count = getattr(rule, "success_count", 0) or 0
        if failure_count >= 3 and failure_count > success_count:
            signals.append(InterventionSignal(
                pattern_type="stale_retention",
                confidence=min(0.9, 0.4 + failure_count * 0.1),
                evidence=f"Rule '{rule.instruction[:40]}' failed {failure_count}x vs {success_count} successes",
                mirror_prompt=f"This rule has been failing more than succeeding recently. Is it still relevant to what you're doing now?",
                reward_frame="Retiring outdated rules makes the remaining ones more precise",
            ))

    # Pattern 3: Repeated failure in same scope
    scope_failures: dict[str, int] = {}
    for turn in turns[:20]:
        if turn.reaction_score is not None and turn.reaction_score < 0.4 and turn.task_scope:
            scope_failures[turn.task_scope] = scope_failures.get(turn.task_scope, 0) + 1
    for scope, count in scope_failures.items():
        if count >= 3:
            signals.append(InterventionSignal(
                pattern_type="repeated_failure",
                confidence=min(0.9, 0.3 + count * 0.15),
                evidence=f"{count} low-score turns in scope '{scope}'",
                mirror_prompt=f"The last {count} attempts in '{scope}' had low scores. Is there a different angle to try?",
                reward_frame=f"Breaking the pattern in '{scope}' unblocks progress",
            ))

    # Pattern 4: Goal drift
    if profile.drift_detected:
        signals.append(InterventionSignal(
            pattern_type="goal_drift",
            confidence=0.7,
            evidence=profile.drift_description,
            mirror_prompt="Your focus seems to have shifted. Is the new direction intentional?",
            reward_frame="Clarifying the current goal prevents scattered effort",
        ))

    # Pattern 5: Knowledge gap warning — escalated curiosity clusters
    if escalated_clusters:
        for cluster in escalated_clusters:
            scope = cluster.get("scope", "")
            signal_count = cluster.get("signal_count", 0)
            keywords = cluster.get("theme_keywords", [])[:3]
            kw_str = ", ".join(keywords) if keywords else scope
            if signal_count >= 2:
                signals.append(InterventionSignal(
                    pattern_type="knowledge_gap_warning",
                    confidence=min(0.9, 0.4 + signal_count * 0.1),
                    evidence=f"User asked about '{kw_str}' {signal_count} times in scope '{scope}'",
                    mirror_prompt=f"You've asked about '{kw_str}' multiple times. Would a foundational explanation help?",
                    reward_frame=f"Clarifying '{kw_str}' prevents repeated misunderstandings",
                ))

    return signals


# ── Main API ─────────────────────────────────────────────────────────────────

def _estimate_curiosity(
    curiosity_signals: list[dict] | None = None,
    turns: list[ConversationTurn] | None = None,
) -> tuple[float, str]:
    """Estimate curiosity level from signal density.

    Does NOT use regex on user messages — relies on the count of
    CuriositySignal records saved by the client LLM.
    Falls back to turn count if no signals available.
    """
    signal_count = len(curiosity_signals) if curiosity_signals else 0
    turn_count = len(turns) if turns else 1

    if signal_count == 0:
        return 0.5, "moderate"

    # Ratio of turns that had a curiosity signal
    density = signal_count / max(turn_count, 1)

    # Map to 0.0-1.0 (saturates around density=0.5)
    level = min(1.0, density * 2.0)

    if level >= 0.65:
        return level, "intellectually-curious"
    elif level <= 0.35:
        return level, "pragmatic"
    else:
        return level, "moderate"


def compute_personality_profile(
    turns: list[ConversationTurn],
    rules: list[PreferenceRule],
    curiosity_signals: list[dict] | None = None,
) -> PersonalityProfile:
    """Compute the full personality profile from accumulated data."""

    metabolism_rate, metabolism_label = _estimate_metabolism(rules, turns)
    reward_kw, reward_pattern, avoid_kw, avoid_pattern = _extract_reward_avoidance(turns)
    digestibility, digestibility_label = _estimate_digestibility(turns)
    objective, obj_conf, drift, drift_desc = _detect_objective_drift(turns)
    curiosity_level, curiosity_label = _estimate_curiosity(curiosity_signals, turns)

    return PersonalityProfile(
        metabolism_rate=metabolism_rate,
        metabolism_label=metabolism_label,
        reward_keywords=reward_kw,
        reward_pattern=reward_pattern,
        avoidance_keywords=avoid_kw,
        avoidance_pattern=avoid_pattern,
        digestibility=digestibility,
        digestibility_label=digestibility_label,
        current_objective=objective,
        objective_confidence=obj_conf,
        drift_detected=drift,
        drift_description=drift_desc,
        curiosity_level=curiosity_level,
        curiosity_label=curiosity_label,
        sample_size=len(turns),
        computed_at=datetime.now().strftime("%Y/%m/%d %H:%M"),
    )


def format_personality_guidance(
    profile: PersonalityProfile,
    interventions: list[InterventionSignal],
) -> str:
    """Format personality insights and interventions for injection into guidance.

    Uses the same markdown format as build_conversation_guidance() in learning_context.py.
    Mirror-style prompts, not commands.
    Adapts verbosity to digestibility level.

    Note: Interventions here are user-facing coaching (cognitive pattern awareness).
    They complement but do not duplicate rule-health checks in memory_manager
    (auto_correct_flagged_rules, resolve_contradicting_rules), which handle
    internal rule hygiene without user interaction.
    """
    parts: list[str] = []

    parts.append("## User cognitive profile")
    parts.append(f"- Metabolism: {profile.metabolism_label} ({profile.metabolism_rate:.2f})")
    parts.append(f"- Prefers: {profile.digestibility_label} information")
    if profile.reward_pattern and profile.reward_pattern != "not enough data":
        parts.append(f"- Responds well to: {profile.reward_pattern}")
    if profile.avoidance_pattern and profile.avoidance_pattern != "not enough data":
        parts.append(f"- Avoids: {profile.avoidance_pattern}")

    if profile.drift_detected:
        parts.append(f"- Goal drift detected: {profile.drift_description}")

    parts.append(f"- Curiosity: {profile.curiosity_label} ({profile.curiosity_level:.2f})")

    # Interventions — mirror-style coaching (not rule hygiene)
    actionable = [sig for sig in interventions if sig.confidence >= 0.4]
    if actionable:
        parts.append("## Cognitive pattern alerts")
        for sig in actionable:
            if profile.digestibility > 0.6:
                parts.append(f"- {sig.mirror_prompt} (confidence: {sig.confidence:.0%})")
            else:
                parts.append(f"- {sig.evidence}. {sig.mirror_prompt}")
                parts.append(f"  → {sig.reward_frame}")

    return "\n".join(parts)
