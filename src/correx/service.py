from __future__ import annotations

from pathlib import Path

from .auto_train import run_auto_training_cycle
from .dormancy import awaken_relevant, forget_stale, forget_stale_rules, scan_and_dormant, semanticize_ghosts
from .growth_tracker import GrowthRecord, GrowthTracker
from .history_store import HistoryStore
from .llm_scorer import Backend, LlmScorer
from .learning_context import (
    build_conversation_guidance,
    build_guidance_context as build_case_guidance_context,
    get_relevant_conversation_corrections,
    get_relevant_corrections,
    get_relevant_preference_rules,
)
from .meaning_synthesis import (
    apply_creative_destruction as _creative_destruction,
    consolidate_rules_by_meaning as _consolidate_rules,
    extract_deferred_meanings as _extract_deferred,
    reactivate_deferred as _reactivate_deferred,
    synthesize_meanings as _synthesize_meanings,
    synthesize_principles as _synthesize_principles,
)
from dataclasses import asdict

from .memory_manager import predict_next_contexts
from .personality_layer import (
    PersonalityProfile,
    InterventionSignal,
    compute_personality_profile,
    detect_interventions,
    format_personality_guidance,
)
from .narrative_montage import (
    NarrativeState,
    build_narrative_template,
    compute_policy_fingerprint,
    from_dict as narrative_from_dict,
    needs_regeneration,
    to_dict as narrative_to_dict,
)
from .schemas import Meaning, Policy, Principle, Tension
from .curiosity_engine import (
    build_cognitive_map,
    cluster_from_dict,
    cluster_to_dict,
    create_signal,
    process_curiosity_signal,
    resolve_cluster,
    signal_from_dict,
    signal_to_dict,
)
from .ghost_engine import (
    create_ghost,
    ghost_from_dict,
    ghost_to_dict,
    process_ghost,
    trajectory_from_dict,
    trajectory_to_dict,
)
from .mlx_trainer import MlxLoraTrainingConfig
from .secret_store import delete_secure_secret, get_secure_secret, set_secure_secret
from .training_dataset import export_mlx_lm_dataset


from .text_similarity import ngram_jaccard as _char_bigram_similarity  # noqa: E402


def _deduplicate_ghost_principles(
    principles: list[str],
    *,
    similarity_threshold: float = 0.5,
    cap: int = 5,
) -> list[str]:
    """Clean, deduplicate, and cap ghost principles.

    Uses char-bigram similarity to remove near-duplicates.
    Returns at most `cap` unique principles.
    """
    cleaned: list[str] = []
    for p in principles:
        p = p.strip()
        for prefix in ["**汎用原則:**", "**汎用原則：**", "汎用原則:", "固有原則:"]:
            if p.startswith(prefix):
                p = p[len(prefix):].strip()
        p = p.split("\n")[0].strip().strip("「」\"'")
        if len(p) < 5 or len(p) > 80:
            continue
        # Check near-duplicate against already accepted
        is_dup = False
        for existing in cleaned:
            if _char_bigram_similarity(p, existing) >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            cleaned.append(p)
        if len(cleaned) >= cap:
            break
    return cleaned


class CorrexService:
    def __init__(
        self,
        base_dir: str | Path,
        *,
        scorer_backend: Backend = "auto",
        scorer_model: str | None = None,
        scorer_endpoint: str = "http://127.0.0.1:11434",
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        scorer = LlmScorer(
            backend=scorer_backend,
            model=scorer_model,
            endpoint=scorer_endpoint,
            score_dict_path=str(self.base_dir / "score_dictionary.json"),
        )
        self.history = HistoryStore(self.base_dir, scorer=scorer)
        self.growth = GrowthTracker(self.base_dir)
        self._scorer = scorer
        self._autonomous_engine = None  # lazy init

    def save_episode(
        self,
        *,
        title: str,
        issuer: str = "",
        task_type: str = "generic",
        source_text: str = "",
        company_profile: dict | None = None,
        profile_name: str = "",
        output: dict | None = None,
        metadata: dict | None = None,
    ):
        return self.history.save_episode(
            title=title,
            issuer=issuer,
            task_type=task_type,
            source_text=source_text,
            company_profile=company_profile,
            profile_name=profile_name,
            output=output,
            metadata=metadata,
        )

    def save_correction(
        self,
        entry_id: str,
        *,
        decision_override: str = "",
        correction_note: str = "",
        reuse_note: str = "",
        reason: str = "",
        scope: str = "",
        bad_output: str = "",
        revised_output: str = "",
        tool_used: str = "",
        source_user: str = "",
        accepted: bool = True,
    ) -> bool:
        return self.history.add_correction(
            entry_id,
            decision_override=decision_override,
            correction_note=correction_note,
            reuse_note=reuse_note,
            reason=reason,
            scope=scope,
            bad_output=bad_output,
            revised_output=revised_output,
            tool_used=tool_used,
            source_user=source_user,
            accepted=accepted,
        )

    def find_entry(self, entry_id: str):
        return self.history.find_entry(entry_id)

    def list_entries(self):
        return self.history.load_entries()

    def save_conversation_turn(
        self,
        *,
        task_scope: str = "",
        user_message: str = "",
        assistant_message: str = "",
        user_feedback: str = "",
        extracted_corrections: list[str] | None = None,
        tags: list[str] | None = None,
        guidance_applied: bool = False,
        auto_record_growth: bool = True,
        metadata: dict | None = None,
        reaction_score_override: float | None = None,
    ):
        turn = self.history.save_conversation_turn(
            task_scope=task_scope,
            user_message=user_message,
            assistant_message=assistant_message,
            user_feedback=user_feedback,
            extracted_corrections=extracted_corrections,
            tags=tags,
            guidance_applied=guidance_applied,
            metadata=metadata,
            reaction_score_override=reaction_score_override,
        )
        if auto_record_growth:
            turns = self.history.load_conversation_turns()
            self.growth.auto_record_from_turns(turns)

        # Awaken dormant rules if correction hits a law-covered scope
        reaction_score = getattr(turn, "reaction_score", None)
        if reaction_score is not None and reaction_score < 0.5:
            self._awaken_dormant_rules(turn)

        return turn

    def _awaken_dormant_rules(self, turn) -> None:
        """Wake dormant rules when anger recurs in a law-covered area.

        If a correction comes in with low reaction_score (anger) and the
        correction text overlaps with a dormant rule's instruction,
        the law wasn't enough — wake the specific rule.
        """
        items = self.history.load_preference_rules_raw()
        dormant = [r for r in items if r.get("status") == "dormant"]
        if not dormant:
            return

        # Build correction text from this turn
        corrections = getattr(turn, "extracted_corrections", []) or []
        feedback = getattr(turn, "user_feedback", "") or ""
        scope = getattr(turn, "task_scope", "") or ""
        signal = " ".join(corrections + [feedback, scope]).lower()

        awakened = 0
        for rule in dormant:
            instruction = (rule.get("instruction", "") or rule.get("statement", "")).lower()
            # Check if the anger overlaps with this dormant rule's domain
            # Simple word overlap check (3+ shared content words)
            rule_words = set(w for w in instruction.split() if len(w) > 2)
            signal_words = set(w for w in signal.split() if len(w) > 2)
            overlap = len(rule_words & signal_words)
            if overlap >= 3 or scope == rule.get("applies_to_scope", ""):
                rule["status"] = "candidate"
                rule.pop("dormant_law_index", None)
                rule.pop("dormant_law", None)
                awakened += 1

        if awakened > 0:
            self.history.write_preference_rules_raw(items)

        # Also awaken dormant ghost principles
        self._awaken_dormant_ghost_principles(turn)

    def _awaken_dormant_ghost_principles(self, turn) -> None:
        """Wake dormant ghost principles when a correction indicates the law wasn't enough."""
        feedback = getattr(turn, "user_feedback", "") or ""
        scope = getattr(turn, "task_scope", "") or ""
        if not feedback:
            return

        trajectories = self.history.load_ghost_trajectories()
        has_dormant = any(t.get("dormant") for t in trajectories)
        if not has_dormant:
            return

        trajectories, awakened = awaken_relevant(
            trajectories, feedback, scope,
        )
        if awakened:
            self.history.write_ghost_trajectories(trajectories)

    def save_training_example(
        self,
        entry_id: str,
        *,
        format: str = "chat",
        system_message: str = "",
        user_message: str = "",
        prompt: str = "",
        draft_output: str = "",
        rejected_output: str = "",
        accepted_output: str = "",
        feedback: str = "",
        accepted: bool = True,
        model_id: str = "",
        policy_version: str = "",
        accepted_by: str = "",
        tags: list[str] | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
    ) -> bool:
        return self.history.attach_training_example(
            entry_id,
            format=format,
            system_message=system_message,
            user_message=user_message,
            prompt=prompt,
            draft_output=draft_output,
            rejected_output=rejected_output,
            accepted_output=accepted_output,
            feedback=feedback,
            accepted=accepted,
            model_id=model_id,
            policy_version=policy_version,
            accepted_by=accepted_by,
            tags=tags,
            temperature=temperature,
            metadata=metadata,
        )

    def list_conversation_turns(self):
        return self.history.load_conversation_turns()

    def list_preference_rules(self, *, promoted_only: bool = False):
        rules = self.history.load_preference_rules()
        if promoted_only:
            return [rule for rule in rules if rule.status == "promoted"]
        return rules

    def list_context_transitions(self):
        return self.history.load_context_transitions()

    def rebuild_preference_rules(self):
        return self.history.rebuild_preference_rules()

    def rebuild_context_transitions(self):
        return self.history.rebuild_context_transitions()

    def self_overcome(self) -> list[dict]:
        """Criticize and propose improvements to own rule set.

        Returns a list of proposals (demote / merge / generalize / resolve_conflict).
        Demote proposals are automatically applied; others are surfaced for review.
        """
        proposals = self.history.self_overcome()
        if not proposals:
            return proposals

        # Auto-apply demotions — low-evidence, low-confidence promoted rules
        demote_ids = {p["rule_id"] for p in proposals if p.get("action") == "demote"}
        if demote_ids:
            rules = self.history.load_preference_rules()
            changed = False
            for rule in rules:
                if rule.id in demote_ids and rule.status == "promoted":
                    rule.status = "candidate"
                    if "self_overcome_demoted" not in rule.tags:
                        rule.tags.append("self_overcome_demoted")
                    changed = True
            if changed:
                self.history.write_preference_rules(rules)

        return proposals

    def synthesize_rules(self) -> list[dict]:
        """Generate rule hypotheses from success/failure pattern differences.

        Returns candidate rules derived from statistical patterns.
        These are proposals only — not automatically added to the rule store.
        """
        return self.history.synthesize_rules()

    def synthesize_meanings(self) -> list[Meaning]:
        rules = self.history.load_preference_rules()
        turns = self.history.load_conversation_turns()
        existing = self.history.load_meanings()
        meanings = _synthesize_meanings(rules, existing)
        self.history.write_meanings(meanings)

        # Consolidate: boost strongest rule in each cluster with sibling evidence
        _consolidate_rules(rules, meanings)

        # Creative destruction: weaken rules subsumed by new meanings
        # Generation and destruction as a single transaction (SHY)
        # Reuse already-loaded turns+rules for personality (no double read)
        profile = compute_personality_profile(turns, rules)
        metabolism = profile.metabolism_rate
        rules, _destruction_log = _creative_destruction(rules, meanings, metabolism_rate=metabolism)

        # Deferred pool: extract candidates below promotion threshold
        deferred = _extract_deferred(rules, meanings)
        if deferred:
            self.history.write_deferred_meanings(deferred)

        self.history.write_preference_rules(rules)
        return meanings

    def list_meanings(self) -> list[Meaning]:
        return self.history.load_meanings()

    def synthesize_principles(self) -> list[Principle]:
        meanings = self.history.load_meanings()
        principles = _synthesize_principles(meanings)
        self.history.write_principles(principles)
        return principles

    def list_principles(self) -> list[Principle]:
        return self.history.load_principles()

    def predict_next_contexts(
        self,
        *,
        previous_context_nodes: list[dict] | None = None,
        limit: int = 5,
    ):
        return predict_next_contexts(
            previous_context_nodes=previous_context_nodes,
            transitions=self.history.load_context_transitions(),
            limit=limit,
        )

    def get_relevant_corrections(
        self,
        *,
        company_profile: dict | None = None,
        task_title: str = "",
        issuer: str = "",
        raw_text: str = "",
        limit: int = 3,
    ):
        return get_relevant_corrections(
            self.history.load_entries(),
            company_profile=company_profile,
            task_title=task_title,
            issuer=issuer,
            raw_text=raw_text,
            limit=limit,
        )

    def get_relevant_preference_rules(
        self,
        *,
        task_scope: str = "",
        raw_text: str = "",
        limit: int = 5,
        previous_context_nodes: list[dict] | None = None,
    ):
        return get_relevant_preference_rules(
            self.history.load_preference_rules(),
            task_scope=task_scope,
            raw_text=raw_text,
            limit=limit,
            previous_context_nodes=previous_context_nodes,
            transitions=self.history.load_context_transitions(),
        )

    def get_relevant_conversation_corrections(
        self,
        *,
        task_scope: str = "",
        raw_text: str = "",
        limit: int = 3,
    ):
        return get_relevant_conversation_corrections(
            self.history.load_conversation_turns(),
            task_scope=task_scope,
            raw_text=raw_text,
            limit=limit,
        )

    def build_conversation_guidance(
        self,
        *,
        task_scope: str = "",
        raw_text: str = "",
        rule_limit: int = 4,
        correction_limit: int = 3,
        previous_context_nodes: list[dict] | None = None,
    ) -> str:
        return build_conversation_guidance(
            self.history.load_conversation_turns(),
            self.history.load_preference_rules(),
            task_scope=task_scope,
            raw_text=raw_text,
            rule_limit=rule_limit,
            correction_limit=correction_limit,
            previous_context_nodes=previous_context_nodes,
            transitions=self.history.load_context_transitions(),
            meanings=self.history.load_meanings(),
        )

    def analyze_conversation_guidance(
        self,
        *,
        task_scope: str = "",
        raw_text: str = "",
        rule_limit: int = 4,
        correction_limit: int = 3,
        previous_context_nodes: list[dict] | None = None,
    ) -> dict:
        rules = self.get_relevant_preference_rules(
            task_scope=task_scope,
            raw_text=raw_text,
            limit=max(rule_limit * 2, rule_limit),
            previous_context_nodes=previous_context_nodes,
        )
        corrections = self.get_relevant_conversation_corrections(
            task_scope=task_scope,
            raw_text=raw_text,
            limit=correction_limit,
        )
        selected_rules = [item for item in rules if item.get("selected_for_guidance", False)][:rule_limit]
        abstained_rules = [item for item in rules if item.get("should_abstain", False)][:rule_limit]
        return {
            "guidance_context": build_conversation_guidance(
                self.history.load_conversation_turns(),
                self.history.load_preference_rules(),
                task_scope=task_scope,
                raw_text=raw_text,
                rule_limit=rule_limit,
                correction_limit=correction_limit,
                previous_context_nodes=previous_context_nodes,
                transitions=self.history.load_context_transitions(),
                meanings=self.history.load_meanings(),
            ),
            "selected_rules": selected_rules,
            "abstained_rules": abstained_rules,
            "recent_corrections": corrections,
        }

    def build_guidance_context(
        self,
        *,
        company_profile: dict | None = None,
        task_title: str = "",
        issuer: str = "",
        raw_text: str = "",
        limit: int = 3,
        task_scope: str = "",
        previous_context_nodes: list[dict] | None = None,
    ) -> str:
        sections: list[str] = []
        case_guidance = build_case_guidance_context(
            self.history.load_entries(),
            company_profile=company_profile,
            task_title=task_title,
            issuer=issuer,
            raw_text=raw_text,
            limit=limit,
        )
        if case_guidance:
            sections.append(case_guidance)
        conversation_guidance = build_conversation_guidance(
            self.history.load_conversation_turns(),
            self.history.load_preference_rules(),
            task_scope=task_scope or task_title,
            raw_text=raw_text,
            previous_context_nodes=previous_context_nodes,
            transitions=self.history.load_context_transitions(),
            meanings=self.history.load_meanings(),
        )
        if conversation_guidance:
            sections.append(conversation_guidance)

        # Reactivate deferred meanings if context changed
        self._try_reactivate_deferred(task_scope=task_scope or task_title)

        # Personality layer: inject user-specific insights + interventions
        personality_section = self._build_personality_guidance()
        if personality_section:
            sections.append(personality_section)

        # Policy layer: deep interpretable principles (highest priority)
        policy_section = self._build_policy_guidance()
        if policy_section:
            sections.append(policy_section)

        # Ghost layer: sublimated principles from rejected proposals
        ghost_section = self._build_ghost_guidance()
        if ghost_section:
            sections.append(ghost_section)

        # Curiosity layer: knowledge gap warnings
        curiosity_section = self._build_curiosity_guidance(
            task_scope=task_scope or task_title
        )
        if curiosity_section:
            sections.append(curiosity_section)

        # Tension layer: contradiction montage — judgment boundaries
        tension_section = self._build_tension_guidance()
        if tension_section:
            sections.append(tension_section)

        return "\n\n".join(section for section in sections if section)

    def build_compact_guidance(self, *, task_scope: str = "", budget: int = 4000) -> str:
        """Build a compact guidance string for SessionStart hook injection.

        Architecture:
        - LLM base safety (ethics, cost, legality) is NOT our job — leave it intact
        - CORREX injects THIS USER's specific risk patterns on top
        - Concrete detection conditions + actions > abstract principles
        - Research-backed: authority framing, before/after examples, anchoring

        Layers:
        1. Policies with authority framing (core + evidence count)
        2. Landmine patterns — user-specific risk triggers with concrete actions
        3. Before/After examples from Ghost rejected/accepted pairs
        4. Top 5 preference rules
        5. Repeat: top 3 policy titles (end-of-prompt anchoring)
        """
        sections: list[str] = []
        remaining = budget

        def _append(text: str, *, required: bool = False) -> bool:
            nonlocal remaining
            if not text:
                return False
            if not required and len(text) > remaining:
                return False
            sections.append(text)
            remaining -= len(text) + 2
            return True

        # --- P1: Policies with social authority framing ---
        policies = self.history.load_policies()
        active_policies = [p for p in policies if p.maturity == "active"]
        if active_policies:
            lines = ["[ポリシー — ユーザーの修正データから蒸留された行動原則]"]
            for p in active_policies:
                ev = getattr(p, "evidence_count", 0) or 0
                authority = f"（{ev}件の修正データから確立）" if ev >= 5 else ""
                lines.append(f"- {p.title}{authority}: {p.core}")
            _append("\n".join(lines), required=True)

        # --- P2: Landmine patterns — concrete detection + action ---
        landmine_lines = self._build_landmine_section()
        if landmine_lines:
            _append(landmine_lines)

        # --- P3: Before/After examples from Ghosts ---
        try:
            ghosts = self.history.load_ghosts()
            examples = []
            for g in ghosts:
                rejected = g.get("rejected_output", "")
                accepted = g.get("accepted_output", "")
                feedback = g.get("user_feedback", "")
                pe = g.get("prediction_error", 0)
                if rejected and (accepted or feedback) and len(rejected) > 20:
                    examples.append((pe, rejected, accepted, feedback))
            examples.sort(key=lambda x: x[0], reverse=True)

            if examples:
                ba_lines = ["[実例 — 過去に却下/修正された行動パターン]"]
                for pe, rejected, accepted, feedback in examples[:3]:
                    rej_short = rejected[:100].replace("\n", " ")
                    ba_lines.append(f"✗ {rej_short}")
                    if accepted:
                        acc_short = accepted[:100].replace("\n", " ")
                        ba_lines.append(f"✓ {acc_short}")
                    elif feedback:
                        fb_short = feedback[:100].replace("\n", " ")
                        ba_lines.append(f"→ ユーザー: {fb_short}")
                _append("\n".join(ba_lines))
        except Exception:
            pass

        # --- P4: Top 5 preference rules ---
        rules = self.history.load_preference_rules()
        promoted = [r for r in rules if r.status == "promoted"]
        promoted.sort(key=lambda r: r.expected_gain * r.confidence_score, reverse=True)
        top_rules = promoted[:5]
        if top_rules:
            r_lines = ["[ルール]"]
            for r in top_rules:
                r_lines.append(f"- {r.statement}")
            _append("\n".join(r_lines))

        # --- P5: End-of-prompt anchoring (repeat top 3 policies) ---
        if active_policies:
            anchor = "[最重要] " + " / ".join(
                p.title for p in active_policies[:3]
            )
            _append(anchor)

        return "\n\n".join(sections)

    def _build_landmine_section(self) -> str:
        """Build user-specific landmine patterns from Ghost/correction data.

        These are NOT generic safety rules (LLM handles those).
        These are THIS USER's specific pain patterns, distilled from
        actual corrections and rejected proposals.

        Format: detection condition → concrete action
        """
        lines = [
            "[地雷パターン — このユーザー固有のリスク。検出したら必ず指定行動を取れ]",
            "※ LLMの基本安全(公序良俗・コスト制限等)は維持せよ。以下はその上に乗る個人最適化層。",
        ]
        patterns_added = 0

        # Source: ConversationTurns (user_feedback + extracted_corrections)
        try:
            turns = self.history.load_conversation_turns()
        except Exception:
            turns = []

        def _count_in_turns(keywords: tuple[str, ...]) -> int:
            """Count turns where keywords appear in feedback or corrections."""
            return sum(
                1 for t in turns
                if any(kw in (getattr(t, "user_feedback", "") or "")
                       for kw in keywords)
                or any(kw in str(getattr(t, "extracted_corrections", []))
                       for kw in keywords)
            )

        # --- Pattern 1: Integrity violation (changing A without B) ---
        if _count_in_turns(("揃え", "整合", "合わせ", "不整合", "連動",
                            "もう片方", "両方", "片方だけ")) >= 2:
            lines.append(
                "⚡ 複数ファイル/箇所に同じ情報があるとき → "
                "片方を変えたら全箇所をリストアップし、全て揃えてから完了報告しろ"
            )
            patterns_added += 1

        # --- Pattern 2: Guessing instead of checking ---
        if _count_in_turns(("推測", "知らない", "調べ", "適当", "確認してから",
                            "現物", "仕様", "混同")) >= 2:
            lines.append(
                "⚡ 専門知識・仕様値を書くとき → "
                "現物/ドキュメントから引用しろ。知らないなら「確認が必要」と言え。推測で書くな"
            )
            patterns_added += 1

        # --- Pattern 3: Irreversible changes without asking ---
        if _count_in_turns(("壊", "消え", "戻せ", "元に", "勝手に", "削除",
                            "上書き", "消した", "なくなっ")) >= 2:
            lines.append(
                "⚡ 元に戻すのが困難な変更（削除・構造変更・大幅な書き換え）→ "
                "実行前にユーザーに確認を取れ。勝手にやるな"
            )
            patterns_added += 1

        # --- Pattern 4: Scope creep ---
        if _count_in_turns(("範囲", "スコープ", "超える", "勝手に", "指示してない",
                            "それは頼んでない", "余計")) >= 2:
            lines.append(
                "⚡ ユーザーの指示範囲を超えそうなとき → "
                "指示を原文で確認し、範囲内かどうか検証してから行動しろ"
            )
            patterns_added += 1

        if patterns_added == 0:
            return ""

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Personality layer
    # ------------------------------------------------------------------

    def get_personality_profile(self) -> PersonalityProfile:
        """Compute and persist the personality profile."""
        profile, _ = self._compute_personality()
        return profile

    def get_interventions(self) -> list[InterventionSignal]:
        """Detect cognitive traps the user may be falling into."""
        _, interventions = self._compute_personality()
        return interventions

    def _compute_personality(self) -> tuple[PersonalityProfile, list[InterventionSignal]]:
        """Shared computation for personality profile + interventions.

        Loads turns and rules once, computes both, persists profile.
        """
        turns = self.history.load_conversation_turns()
        rules = self.history.load_preference_rules()
        if not turns:
            profile = PersonalityProfile()
            return profile, []
        curiosity_signals = self.history.load_curiosity_signals()
        profile = compute_personality_profile(turns, rules, curiosity_signals=curiosity_signals)

        # Escalated clusters for knowledge_gap_warning intervention
        cluster_dicts = self.history.load_knowledge_gap_clusters()
        escalated = [
            c for c in cluster_dicts
            if c.get("status") == "escalated"
            or (c.get("escalation_score", 0) >= 0.5 and c.get("status") != "resolved")
        ]
        interventions = detect_interventions(rules, turns, profile, escalated_clusters=escalated)
        self.history.write_personality(asdict(profile))
        return profile, interventions

    def _try_reactivate_deferred(self, task_scope: str = "") -> None:
        """Check if deferred meanings should reactivate given current context."""
        deferred = self.history.load_deferred_meanings()
        if not deferred:
            return
        # Get current context tags from recent turns
        turns = self.history.load_conversation_turns()
        recent_tags = []
        for t in turns[:5]:
            recent_tags.extend(t.tags[:5] if t.tags else [])
        reactivated = _reactivate_deferred(deferred, task_scope, recent_tags)
        if reactivated:
            # Merge reactivated into active meanings
            existing = self.history.load_meanings()
            existing.extend(reactivated)
            self.history.write_meanings(existing)
            # Remove from deferred pool
            remaining = [d for d in deferred if d.status == "deferred"]
            self.history.write_deferred_meanings(remaining)

    def _build_personality_guidance(self) -> str:
        """Build personality-aware guidance for injection into context."""
        profile, interventions = self._compute_personality()
        if profile.sample_size < 5:
            return ""
        return format_personality_guidance(profile, interventions)

    def _build_ghost_guidance(self) -> str:
        """Build guidance from Ghost Engine: universal laws + sublimated principles.

        Universal laws are the highest-level behavioral principles, distilled
        from hundreds of rejections via Claude Sonnet. They take precedence
        over individual sublimated principles.
        """
        import json as _json

        sections: list[str] = []

        has_laws = False
        has_positive = False

        # 1. Universal laws (constraints)
        laws = self.history.load_ghost_universal_laws()
        if laws:
            law_lines = ["[禁止法理 — 却下パターンから昇華された行動制約]"]
            for i, law in enumerate(laws, 1):
                text = law.get("law", "") if isinstance(law, dict) else str(law)
                if text:
                    law_lines.append(f"  {i}. {text}")
            sections.append("\n".join(law_lines))
            has_laws = True

        # 2. Positive laws (drivers)
        pos_laws = self.history.load_ghost_positive_laws()
        if pos_laws:
            pos_lines = ["[推奨法理 — 肯定反応から抽出された推進基準]"]
            for i, law in enumerate(pos_laws, 1):
                text = law.get("law", "") if isinstance(law, dict) else str(law)
                if text:
                    pos_lines.append(f"  +{i}. {text}")
            sections.append("\n".join(pos_lines))
            has_positive = True

        # 3. Tension resolution instruction
        if has_laws and has_positive:
            sections.append(
                "[適用ルール]\n"
                "禁止法理と推奨法理は矛盾ではなく、状況依存のテンションである。\n"
                "両方を同時に意識し、現在の場面でどちらに重みがあるかを判断せよ。\n"
                "一般に: タスク開始時・未知領域では禁止法理を重視、\n"
                "方向確定後・既知領域では推奨法理を重視せよ。\n"
                "どちらか一方だけを適用して他方を無視するな。"
            )

        # 4. Individual sublimated principles — specifics coexist with laws
        # Laws are the umbrella; specifics are instant-apply for known situations.
        # Aggressively deduplicate and cap to avoid context flooding.
        principles = self.get_fired_ghost_principles()
        if principles:
            cleaned = _deduplicate_ghost_principles(principles, similarity_threshold=0.5, cap=5)

            if cleaned:
                p_lines = [
                    "[固有原則 — 既知の場面で即適用する具体基準]",
                ]
                for i, p in enumerate(cleaned, 1):
                    p_lines.append(f"  {i}. {p}")
                sections.append("\n".join(p_lines))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Policy layer
    # ------------------------------------------------------------------

    # Thresholds for automatic policy activation (data-driven).
    # A proposed policy becomes active when it meets BOTH conditions.
    POLICY_AUTO_ACTIVE_EVIDENCE = 10    # minimum evidence_count
    POLICY_AUTO_ACTIVE_LAWS = 2         # minimum source_law_ids count

    def save_policy(self, policy: Policy) -> Policy:
        """Save or update a policy.

        Auto-activation: proposed policies with sufficient evidence
        (evidence_count >= 10 AND source_law_ids >= 2) are automatically
        promoted to active.  No human approval needed — this is the
        data-driven design of the 超個人 system.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")

        # Auto-activate proposed policies that meet evidence thresholds
        if policy.maturity == "proposed":
            evidence_ok = policy.evidence_count >= self.POLICY_AUTO_ACTIVE_EVIDENCE
            laws_ok = len(policy.source_law_ids or []) >= self.POLICY_AUTO_ACTIVE_LAWS
            if evidence_ok and laws_ok:
                policy.maturity = "active"

        policies = self.history.load_policies()
        existing = next((p for p in policies if p.id == policy.id), None)
        if existing:
            # Update in place
            idx = policies.index(existing)
            if not policy.created_at:
                policy.created_at = existing.created_at
            policy.updated_at = now
            policies[idx] = policy
        else:
            if not policy.created_at:
                policy.created_at = now
            policy.updated_at = now
            policies.append(policy)
        self.history.write_policies(policies)

        # Auto-dormancy: when a policy becomes active, retire covered principles
        if policy.maturity == "active":
            self._run_dormancy_scan()

        return policy

    def _run_dormancy_scan(self) -> tuple[int, int]:
        """Scan all ghost principles and dormant ones covered by laws/policies.

        Returns (dormant_count, active_count).
        """
        trajectories = self.history.load_ghost_trajectories()

        # Collect law texts
        laws: list[str] = []
        for law_item in self.history.load_ghost_universal_laws() + self.history.load_ghost_positive_laws():
            text = law_item.get("law", "") if isinstance(law_item, dict) else str(law_item)
            if text:
                laws.append(text)

        # Collect policy core texts
        policies_text = [p.core for p in self.history.load_policies() if p.maturity == "active"]

        trajectories, dormant_count, active_count = scan_and_dormant(
            trajectories, laws=laws, policies=policies_text,
        )
        self.history.write_ghost_trajectories(trajectories)

        # Also retire preference rules covered by policies
        self._dormant_preference_rules(policies_text, laws)

        # Forget stale dormant items (30+ days without awakening)
        trajectories = self.history.load_ghost_trajectories()
        trajectories, forgotten_principles = forget_stale(trajectories)
        if forgotten_principles > 0:
            self.history.write_ghost_trajectories(trajectories)

        self._forget_stale_rules()

        return dormant_count, active_count

    def _forget_stale_rules(self) -> int:
        """Permanently delete preference rules dormant for 30+ days."""
        items = self.history.load_preference_rules_raw()
        if not items:
            return 0

        remaining, forgotten = forget_stale_rules(items)
        if forgotten > 0:
            self.history.write_preference_rules_raw(remaining)
        return forgotten

    def _dormant_preference_rules(
        self,
        policies_text: list[str],
        laws: list[str],
    ) -> int:
        """Retire promoted preference rules that are covered by policies/laws.

        Rules with 'retrograde' tag and generic content are retired.
        Specific rules (no retrograde tag, meaningful length) are kept.
        """
        from .dormancy import check_coverage

        items = self.history.load_preference_rules_raw()
        if not items:
            return 0

        retired = 0

        for rule in items:
            if not isinstance(rule, dict):
                continue
            if rule.get("status") != "promoted":
                continue
            # Keep specific rules (no retrograde, meaningful content)
            tags = rule.get("tags") or []
            stmt = rule.get("statement", "")
            if "retrograde" not in tags and len(stmt) >= 25:
                continue

            # Check if covered by policies or laws
            covering = check_coverage(stmt, laws=laws, policies=policies_text)
            if covering:
                rule["status"] = "dormant"
                rule["dormant_reason"] = f"absorbed by: {covering}"
                retired += 1

        if retired > 0:
            self.history.write_preference_rules_raw(items)

        return retired

    def list_policies(self, *, active_only: bool = False) -> list[Policy]:
        """List all policies, optionally filtering to active only."""
        policies = self.history.load_policies()
        if active_only:
            return [p for p in policies if p.maturity == "active"]
        return policies

    # ------------------------------------------------------------------
    # Tension — Contradiction Montage
    # ------------------------------------------------------------------

    def list_tensions(self, *, active_only: bool = False) -> list[Tension]:
        tensions = self.history.load_tensions()
        if active_only:
            return [t for t in tensions if t.status == "active"]
        return tensions

    @staticmethod
    def _classify_action_direction(text: str) -> set[str]:
        """Classify a rule's action direction(s) for tension detection.

        Returns a set of direction tags. Rules can have multiple directions.
        Opposite direction pairs are tension candidates.

        Direction pairs (opposites):
          confirm ↔ execute  (確認を取れ vs 即実行せよ)
          preserve ↔ change  (壊すな vs 改善しろ)
          wait ↔ act         (待て vs 走れ)
          part ↔ whole       (部分で進め vs 全体を見ろ)
        """
        import re
        directions: set[str] = set()

        # confirm: 確認系
        if re.search(r"確認|聞[けい]|方針.*取[れる]|承認|ユーザーに.*求め|仰[ぐげ]", text):
            directions.add("confirm")

        # execute: 即実行系
        if re.search(r"即[座実]|即.*[せしや]|結果を出|走[れる]|完遂|一気に|即座", text):
            directions.add("execute")

        # preserve: 保全系
        if re.search(r"壊すな|変更するな|守[れる]|既存.*[を壊変]|維持|退行させるな|上書きするな", text):
            directions.add("preserve")

        # change: 変更系
        if re.search(r"改善|段階的|動くもの|磨き|リファクタ", text):
            directions.add("change")

        # wait: 待機系
        if re.search(r"待[てつ]|先[にに]|してから|完了.*から|前に", text):
            directions.add("wait")

        # act: 行動系
        if re.search(r"宣言するな|やれ$|動け$|出せ$|止めるな|走らせろ|止まるな", text):
            directions.add("act")

        # part: 部分系
        if re.search(r"段階|一つ.*完了|単体|指定された範囲", text):
            directions.add("part")

        # whole: 全体系
        if re.search(r"全体.*[を見再]|全指摘|部分.*するな|全件", text):
            directions.add("whole")

        return directions

    # Direction pairs that indicate potential tension
    _OPPOSITE_DIRECTIONS = [
        ("confirm", "execute"),
        ("preserve", "change"),
        ("wait", "act"),
        ("part", "whole"),
    ]

    def detect_tension_candidates(self) -> list[dict]:
        """Server-side candidate generation for contradiction detection.

        Two detection strategies:
        1. Keyword overlap (original) — finds rules about the same topic
        2. Action direction opposition — finds rules pointing in opposite directions

        The client LLM judges which candidates are genuine contradictions.
        """
        import re

        # --- Collect statements with scope metadata ---
        rules = self.history.load_preference_rules()
        statements: list[tuple[str, str, str]] = []  # (id, text, scope)
        for r in rules:
            text = r.statement or r.instruction
            if text:
                statements.append((r.id, text, getattr(r, "applies_to_scope", "") or ""))

        ghost_principles = self.get_fired_ghost_principles()
        for i, gp in enumerate(ghost_principles):
            statements.append((f"ghost-{i}", gp, ""))

        # --- Keyword extraction (Japanese-aware) ---
        _PARTICLE_RE = re.compile(
            r'[、。・\s\u3000（）「」/,.\-]|'
            r'(?<=[^\u3040-\u309F])[をにはがでとのもへから]{1,2}(?=[^\u3040-\u309F])|'
            r'(?<=.)[をにはがでとのもへ](?=.)'
        )

        _STOP_WORDS = {
            "しろ", "せよ", "やれ", "するな", "やるな", "出せ", "直せ",
            "動け", "避けよ", "控え", "走るな", "行け", "超えるな",
            "変えるな", "復唱するな", "実行しろ", "確認しろ", "記録しろ",
            "提案するな", "安心するな", "忘れるな",
            "ユーザー", "タスク", "作業", "実行", "確認", "提案",
            "コード", "修正", "バグ", "テスト", "セッション",
            "ファイル", "データ", "結果", "報告", "指示",
            "必ず", "即座", "自発", "自動", "最優先", "禁止",
            "場合", "状態", "問題", "対応", "処理", "機能",
            "ルール", "メモリ", "記録", "保存", "設定",
        }

        def extract_keywords(text: str) -> set[str]:
            tokens = _PARTICLE_RE.split(text)
            result = set()
            for t in tokens:
                t = t.strip()
                if len(t) >= 2:
                    result.add(t)
                    for sub in re.split(r'[をにはがでとのもへ]', t):
                        sub = sub.strip()
                        if len(sub) >= 2:
                            result.add(sub)
            return result - _STOP_WORDS

        # --- Pre-compute directions for all statements ---
        directions_cache: dict[int, set[str]] = {}
        for idx, (_, text, _) in enumerate(statements):
            directions_cache[idx] = self._classify_action_direction(text)

        # --- Find candidate pairs ---
        candidates: list[dict] = []
        existing = self.history.load_tensions()
        existing_pairs = {(t.rule_a_id, t.rule_b_id) for t in existing}
        existing_pairs |= {(t.rule_b_id, t.rule_a_id) for t in existing}

        for i, (id_a, text_a, scope_a) in enumerate(statements):
            kw_a = extract_keywords(text_a)
            dirs_a = directions_cache[i]

            for j, (id_b, text_b, scope_b) in enumerate(statements):
                if j <= i:
                    continue
                if (id_a, id_b) in existing_pairs:
                    continue

                kw_b = extract_keywords(text_b)
                dirs_b = directions_cache[j]

                overlap = kw_a & kw_b if kw_a and kw_b else set()
                scope_match = bool(scope_a and scope_b and scope_a == scope_b)
                overlap_count = len(overlap)

                # Strategy 1: Keyword overlap (original)
                keyword_score = 0
                if scope_match and overlap_count >= 1:
                    keyword_score = overlap_count + 5
                elif overlap_count >= 2:
                    keyword_score = overlap_count

                # Strategy 2: Action direction opposition
                direction_score = 0
                opposing_dirs: list[tuple[str, str]] = []
                if dirs_a and dirs_b:
                    for dir_a, dir_b in self._OPPOSITE_DIRECTIONS:
                        if (dir_a in dirs_a and dir_b in dirs_b) or \
                           (dir_b in dirs_a and dir_a in dirs_b):
                            direction_score += 8
                            opposing_dirs.append((dir_a, dir_b))

                # Combined score
                score = max(keyword_score, direction_score)
                if keyword_score > 0 and direction_score > 0:
                    score = keyword_score + direction_score  # both signals = strong

                if score < 2:
                    continue

                candidate = {
                    "rule_a_id": id_a,
                    "rule_a_text": text_a,
                    "rule_b_id": id_b,
                    "rule_b_text": text_b,
                    "shared_keywords": sorted(overlap),
                    "scope_match": scope_match,
                    "_score": score,
                }
                if opposing_dirs:
                    candidate["opposing_directions"] = [
                        f"{a} ↔ {b}" for a, b in opposing_dirs
                    ]
                candidates.append(candidate)

        # Sort by score descending, cap at 50
        candidates.sort(key=lambda c: c["_score"], reverse=True)
        candidates = candidates[:50]
        for c in candidates:
            del c["_score"]

        return candidates

    def save_tension(
        self,
        *,
        rule_a_id: str,
        rule_a_text: str,
        rule_b_id: str,
        rule_b_text: str,
        boundary: str = "",
        signal: str = "",
        scopes: list[str] | None = None,
        confidence: float = 0.0,
    ) -> Tension:
        """Save a tension (contradiction pair with boundary condition)."""
        from datetime import datetime as _dt

        now = _dt.now().strftime("%Y/%m/%d %H:%M")
        tension_id = f"tension-{_dt.now().strftime('%Y%m%d%H%M%S%f')}"

        tension = Tension(
            id=tension_id,
            rule_a_id=rule_a_id,
            rule_a_text=rule_a_text,
            rule_b_id=rule_b_id,
            rule_b_text=rule_b_text,
            boundary=boundary,
            signal=signal,
            scopes=scopes or [],
            confidence=confidence,
            created_at=now,
            updated_at=now,
        )

        tensions = self.history.load_tensions()
        tensions.append(tension)
        self.history.write_tensions(tensions)
        return tension

    def _build_tension_guidance(self) -> str:
        """Build guidance from active tensions for context injection.

        Tensions are the highest-signal knowledge: not what to do, but
        *when to switch* between opposing rules.
        """
        tensions = self.history.load_tensions()
        active = [t for t in tensions if t.status == "active" and t.boundary]
        if not active:
            return ""
        lines = ["[判断境界 — 矛盾モンタージュから抽出された切り替え基準]"]
        for t in active[:10]:  # Cap at 10 to avoid context bloat
            lines.append(f"\n  A: {t.rule_a_text}")
            lines.append(f"  B: {t.rule_b_text}")
            lines.append(f"  → 境界: {t.boundary}")
            if t.signal:
                lines.append(f"  → 切替シグナル: {t.signal}")
        lines.append("")
        lines.append("上記の各ペアは矛盾ではなく、状況依存の切り替え。")
        lines.append("境界条件を見て、今の場面でどちらに重みがある��判断せよ。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Narrative Montage
    # ------------------------------------------------------------------

    def check_narrative_status(self) -> dict:
        """Return current narrative and whether regeneration is needed.

        The client LLM calls this to decide whether to generate a new
        5-line personality narrative from policy montage data.
        """
        policies = self.history.load_policies()
        tensions = self.history.load_tensions()
        stored_raw = self.history.load_narrative()
        stored = narrative_from_dict(stored_raw) if stored_raw else None
        current_fp = compute_policy_fingerprint(policies)
        regen = needs_regeneration(current_fp, stored)

        # Gather context for LLM regeneration
        active_policies = [p for p in policies if p.maturity == "active"]
        profile = self._get_personality_snapshot()

        return {
            "needs_regeneration": regen,
            "current_narrative": stored.narrative_text if stored else "",
            "fingerprint": current_fp,
            "stored_fingerprint": stored.policy_fingerprint if stored else "",
            "policy_count": len(active_policies),
            "policies": [{"id": p.id, "title": p.title, "core": p.core, "evidence_count": p.evidence_count} for p in active_policies],
            "tensions": [{"boundary": t.boundary, "signal": t.signal} for t in tensions if t.status == "active" and t.boundary],
            "personality": {
                "metabolism": profile.get("metabolism_rate", 0.5),
                "digestibility": profile.get("digestibility", 0.5),
                "reward_keywords": profile.get("reward_keywords", []),
                "avoidance_keywords": profile.get("avoidance_keywords", []),
            },
        }

    def _get_personality_snapshot(self) -> dict:
        """Load cached personality or compute fresh."""
        cached = self.history.load_personality()
        if cached:
            return cached
        turns = self.history.load_conversation_turns()
        rules = self.history.load_preference_rules()
        profile = compute_personality_profile(turns, rules)
        return asdict(profile)

    def save_narrative(self, *, narrative_text: str, method: str = "llm") -> dict:
        """Persist a generated narrative and return clipboard text."""
        from datetime import datetime

        policies = self.history.load_policies()
        active = [p for p in policies if p.maturity == "active"]
        fp = compute_policy_fingerprint(policies)

        state = NarrativeState(
            narrative_text=narrative_text,
            policy_fingerprint=fp,
            generated_at=datetime.now().strftime("%Y/%m/%d %H:%M"),
            generation_method=method,
            source_policy_ids=[p.id for p in active],
        )
        self.history.write_narrative(narrative_to_dict(state))

        return {
            "ok": True,
            "narrative": narrative_text,
            "fingerprint": fp,
            "clipboard_text": (
                f"## お前が仕えている人間\n{narrative_text}\n\n"
                "↑ claude.aiの個人設定に貼ってください。"
            ),
        }

    def build_narrative_from_template(self) -> str:
        """Generate narrative from template (0 LLM cost)."""
        policies = self.history.load_policies()
        tensions = self.history.load_tensions()
        profile = self._get_personality_snapshot()

        return build_narrative_template(
            policies,
            tensions,
            metabolism=profile.get("metabolism_rate", 0.5),
            digestibility=profile.get("digestibility", 0.5),
            reward_keywords=profile.get("reward_keywords", []),
            avoidance_keywords=profile.get("avoidance_keywords", []),
        )

    def _build_policy_guidance(self) -> str:
        """Build guidance from active policies for injection into context.

        Policies are the highest-quality knowledge unit. They carry
        core, why, analogy, opposite, and limits — enabling reasoning
        in novel situations rather than literal rule-following.
        """
        policies = self.history.load_policies()
        active = [p for p in policies if p.maturity == "active"]
        if not active:
            return ""
        lines = ["[ポリシー — 深い判断基準]"]
        for p in active:
            lines.append(f"\n### {p.title}")
            lines.append(f"核: {p.core}")
            if p.why:
                lines.append(f"なぜ: {p.why}")
            if p.analogy:
                lines.append(f"類推: {p.analogy}")
            if p.opposite:
                lines.append(f"反対: {p.opposite}")
            if p.limits:
                lines.append(f"限界: {p.limits}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Growth measurement
    # ------------------------------------------------------------------

    def record_growth(
        self,
        *,
        case_id: str,
        case_title: str,
        task_scope: str = "",
        baseline_output: str,
        baseline_score: float,
        guided_output: str,
        guided_score: float,
        guidance_text: str = "",
    ) -> GrowthRecord:
        """Record one before/after measurement for a task.

        Run the task WITHOUT guidance → pass baseline_output + baseline_score.
        Run the task WITH guidance    → pass guided_output + guided_score.
        Scores are 0.0 (worst) to 1.0 (best).
        """
        return self.growth.record(
            case_id=case_id,
            case_title=case_title,
            task_scope=task_scope,
            baseline_output=baseline_output,
            baseline_score=baseline_score,
            guided_output=guided_output,
            guided_score=guided_score,
            guidance_text=guidance_text,
        )

    def growth_trend(self, case_id: str) -> list[dict]:
        """Return score history for one case, oldest → newest."""
        return self.growth.trend(case_id)

    def growth_summary(self) -> dict:
        """Overall growth summary from ConversationTurn reaction scores.

        Replaces old record_growth-based computation (was unreliable/negative).
        Uses actual user reaction data directly.
        """
        turns = self.history.load_conversation_turns()
        scored = [t for t in turns if t.reaction_score is not None]
        if not scored:
            return {"total_turns": 0, "average_delta": 0.0, "overall_trend": "no_data",
                    "guided_vs_unguided": {}, "temporal": {}, "correction_rate": {}}

        guided = [t for t in scored if getattr(t, "guidance_applied", False)]
        unguided = [t for t in scored if not getattr(t, "guidance_applied", False)]

        def _avg(ts: list) -> float:
            return sum(t.reaction_score for t in ts) / len(ts) if ts else 0.0

        def _neg_rate(ts: list) -> float:
            return sum(1 for t in ts if t.reaction_score < 0.4) / len(ts) if ts else 0.0

        guided_avg = _avg(guided)
        unguided_avg = _avg(unguided)
        guidance_delta = guided_avg - unguided_avg if guided and unguided else 0.0

        half = len(scored) // 2
        early, late = scored[:half], scored[half:]
        temporal_delta = _avg(late) - _avg(early)

        trend = ("growing" if temporal_delta > 0.05
                 else "flat" if temporal_delta >= -0.05
                 else "degrading")

        return {
            "total_turns": len(scored),
            "average_delta": round(guidance_delta, 4),
            "overall_trend": trend,
            "guided_vs_unguided": {
                "guided_count": len(guided),
                "guided_avg": round(guided_avg, 3),
                "unguided_count": len(unguided),
                "unguided_avg": round(unguided_avg, 3),
                "delta": round(guidance_delta, 3),
            },
            "temporal": {
                "early_avg": round(_avg(early), 3),
                "late_avg": round(_avg(late), 3),
                "delta": round(temporal_delta, 3),
            },
            "correction_rate": {
                "early": round(_neg_rate(early), 3),
                "late": round(_neg_rate(late), 3),
                "delta": round(_neg_rate(late) - _neg_rate(early), 3),
            },
        }

    def save_secret(self, account_name: str, secret_value: str) -> bool:
        return set_secure_secret(account_name, secret_value)

    def load_secret(self, account_name: str) -> str | None:
        return get_secure_secret(account_name)

    def clear_secret(self, account_name: str) -> bool:
        return delete_secure_secret(account_name)

    def export_training_dataset(
        self,
        output_dir: str | Path,
        *,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle_seed: int = 7,
        split_strategy: str = "chronological",
    ) -> dict:
        report = export_mlx_lm_dataset(
            self.history.load_entries(),
            output_dir,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            shuffle_seed=shuffle_seed,
            split_strategy=split_strategy,
        )
        return report.to_dict()

    def run_auto_training_cycle(
        self,
        *,
        model: str,
        output_dir: str | Path,
        minimum_new_examples: int = 8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle_seed: int = 7,
        split_strategy: str = "chronological",
        force: bool = False,
        dry_run: bool = False,
        training_config: MlxLoraTrainingConfig | None = None,
    ) -> dict:
        report = run_auto_training_cycle(
            self.history.load_entries(),
            model=model,
            output_dir=output_dir,
            minimum_new_examples=minimum_new_examples,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            shuffle_seed=shuffle_seed,
            split_strategy=split_strategy,
            force=force,
            dry_run=dry_run,
            training_config=training_config,
        )
        return report.to_dict()

    # ------------------------------------------------------------------
    # Ghost Engine
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Curiosity layer (third learning layer)
    # ------------------------------------------------------------------

    def save_curiosity_signal(
        self,
        *,
        question_text: str,
        question_type: str = "knowledge_gap",
        target: str = "self",
        task_scope: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        confidence: float = 0.0,
        source_turn_id: str = "",
    ) -> tuple[dict, dict, bool]:
        """Record a curiosity signal detected by the client LLM.

        The client LLM detects questions in user messages, classifies them,
        and passes the classification here. The server handles clustering
        and persistence.

        Returns:
        - signal_dict: the stored CuriositySignal record
        - cluster_dict: the updated/created KnowledgeGapCluster record
        - is_new_cluster: whether a new cluster was created
        """
        signal = create_signal(
            question_text=question_text,
            question_type=question_type,
            target=target,
            task_scope=task_scope,
            tags=tags,
            keywords=keywords,
            confidence=confidence,
            source_turn_id=source_turn_id,
        )

        # Load existing clusters
        cluster_dicts = self.history.load_knowledge_gap_clusters()
        clusters = [cluster_from_dict(d) for d in cluster_dicts]

        # Process through pipeline
        updated_signal, updated_cluster, is_new = process_curiosity_signal(
            signal, clusters,
        )

        # Persist
        signal_dict = signal_to_dict(updated_signal)
        cluster_dict = cluster_to_dict(updated_cluster)
        self.history.save_signal_with_cluster(signal_dict, cluster_dict)

        return signal_dict, cluster_dict, is_new

    def resolve_curiosity_clusters(
        self,
        task_scope: str = "",
    ) -> int:
        """Resolve open curiosity clusters matching the given scope.

        Called by the client LLM when the user expresses satisfaction.
        Requires a non-empty task_scope to prevent accidental mass-resolution.
        Returns the number of clusters resolved.
        """
        if not task_scope:
            return 0  # Refuse to resolve without a scope — prevents data loss

        cluster_dicts = self.history.load_knowledge_gap_clusters()
        clusters = [cluster_from_dict(d) for d in cluster_dicts]

        resolved_count = 0
        for cluster in clusters:
            if cluster.status != "open" and cluster.status != "escalated":
                continue
            # Match by primary scope or scopes list
            if cluster.scope != task_scope and task_scope not in cluster.scopes:
                continue
            resolve_cluster(cluster)
            resolved_count += 1

        if resolved_count > 0:
            self.history.write_knowledge_gap_clusters(
                [cluster_to_dict(c) for c in clusters]
            )

        return resolved_count

    def get_cognitive_map(self) -> dict:
        """Build and return the cognitive map of knowledge gaps.

        Used by the client LLM at session start to understand
        where the user needs more explanation.
        """
        cluster_dicts = self.history.load_knowledge_gap_clusters()
        clusters = [cluster_from_dict(d) for d in cluster_dicts]
        return build_cognitive_map(clusters)

    def list_curiosity_signals(self, limit: int = 50) -> list[dict]:
        """List recent curiosity signals."""
        signals = self.history.load_curiosity_signals()
        return signals[:limit]

    def list_knowledge_gap_clusters(
        self,
        include_resolved: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        """List knowledge gap clusters."""
        clusters = self.history.load_knowledge_gap_clusters()
        if not include_resolved:
            clusters = [c for c in clusters if c.get("status") != "resolved"]
        return clusters[:limit]

    def _build_curiosity_guidance(self, task_scope: str = "") -> str:
        """Build curiosity-layer guidance section for injection into guidance context."""
        cluster_dicts = self.history.load_knowledge_gap_clusters()
        clusters = [cluster_from_dict(d) for d in cluster_dicts]

        # Filter to relevant clusters (open/escalated, matching scope or high escalation)
        relevant = []
        for c in clusters:
            if c.status == "resolved":
                continue
            scope_match = (
                not task_scope
                or c.scope == task_scope
                or task_scope in c.scopes
            )
            high_escalation = c.escalation_score >= 0.5
            if scope_match or high_escalation:
                relevant.append(c)

        if not relevant:
            return ""

        lines = ["[知識空白地図 — Curiosity Layer]"]
        for c in sorted(relevant, key=lambda x: x.escalation_score, reverse=True):
            status_marker = "⚠" if c.status == "escalated" else "○"
            kw_str = ", ".join(c.theme_keywords[:5])
            type_label = {
                "knowledge_gap": "知らない",
                "judgment_uncertainty": "決められない",
                "confirmation_seeking": "安心したい",
            }.get(c.dominant_type, c.dominant_type)
            lines.append(
                f"  {status_marker} [{c.scope or 'general'}] {kw_str} — "
                f"{type_label} (質問{c.signal_count}回, エスカレーション: {c.escalation_score:.0%})"
            )

        if any(c.status == "escalated" for c in relevant):
            lines.append("")
            lines.append("  → エスカレーション中の領域は、基礎から丁寧に説明せよ")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ghost layer (second learning layer)
    # ------------------------------------------------------------------

    def save_ghost(
        self,
        *,
        rejected_output: str,
        task_scope: str = "",
        tags: list[str] | None = None,
        user_feedback: str = "",
        accepted_output: str = "",
        source_turn_id: str = "",
    ) -> tuple[dict, dict, list[str]]:
        """Create a ghost from a rejected AI proposal and process it through trajectories.

        Returns:
        - ghost_dict: the stored Ghost record
        - trajectory_dict: the updated GhostTrajectory record
        - fired_principles: list of sublimated principles (empty if not yet fired)
        """
        # Load personality for metabolism rate
        personality = self.history.load_personality()
        metabolism_rate = float(personality.get("metabolism_rate", 0.5))

        # Create ghost
        ghost = create_ghost(
            rejected_output=rejected_output,
            task_scope=task_scope,
            tags=tags,
            user_feedback=user_feedback,
            accepted_output=accepted_output,
            source_turn_id=source_turn_id,
        )

        # Load existing trajectories and ghosts
        trajectory_dicts = self.history.load_ghost_trajectories()
        ghost_dicts = self.history.load_ghosts()

        trajectories = [trajectory_from_dict(d) for d in trajectory_dicts]
        all_ghosts = {d["id"]: ghost_from_dict(d) for d in ghost_dicts}

        # Gather existing principles for dedup
        existing_principles = self.get_fired_ghost_principles()

        # Process ghost through pipeline
        updated_ghost, updated_trajectory, fired_principles = process_ghost(
            ghost=ghost,
            trajectories=trajectories,
            all_ghosts=all_ghosts,
            metabolism_rate=metabolism_rate,
            existing_principles=existing_principles,
        )

        # Persist
        ghost_dict = ghost_to_dict(updated_ghost)
        trajectory_dict = trajectory_to_dict(updated_trajectory)
        self.history.save_ghost_with_trajectory(ghost_dict, trajectory_dict)

        # Auto-sublimate if trajectory just fired
        if fired_principles and updated_trajectory.fired:
            self._auto_sublimate_to_law(trajectory_dict, fired_principles)

        return ghost_dict, trajectory_dict, fired_principles

    def _auto_sublimate_to_law(self, trajectory_dict: dict, fired_principles: list[str]) -> None:
        """When a trajectory fires, attempt 2-stage sublimation via Sonnet.

        Stage 1: Specific → Universal principle (via Sonnet API)
        Stage 2: Check if the universal principle fits an existing law or creates a new one
        Then re-evaluate dormant rules under the updated laws.
        """
        import json as _json
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return  # Sonnet not available, keep local LLM principle

        try:
            import anthropic
        except ImportError:
            return

        # Stage 1: Abstract the specific principle
        specific = fired_principles[0] if fired_principles else ""
        if not specific:
            return

        scopes = trajectory_dict.get("scopes", [])
        scope_str = "・".join(scopes[:3]) if scopes else "不明"

        try:
            client = anthropic.Anthropic(api_key=api_key)
            r = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": (
                    f"以下の固有原則から固有名詞を全て剥がし、どんな場面でも適用可能な"
                    f"汎用行動原則に書き換えてください。1文、20〜40文字、日本語。\n\n"
                    f"スコープ: {scope_str}\n固有原則: {specific}"
                )}],
            )
            universal = r.content[0].text.strip().strip("「」\"'").split("\n")[0]
        except Exception:
            return

        if not universal or len(universal) < 5:
            return

        # Update the trajectory's sublimated_principle
        trajectory_dict["sublimated_principle"] = universal

        # Stage 2: Check if this fits an existing law
        laws = self.history.load_ghost_universal_laws()

        if laws:
            law_list = "\n".join(f"{i+1}. {l.get('law', '')}" for i, l in enumerate(laws))
            try:
                r2 = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=50,
                    messages=[{"role": "user", "content": (
                        f"以下の法理のうち、この新原則をカバーできるものの番号を答えてください。\n"
                        f"どれにも該当しなければ0と答えてください。数字だけ。\n\n"
                        f"法理:\n{law_list}\n\n新原則: {universal}"
                    )}],
                )
                answer = r2.content[0].text.strip()
                import re
                m = re.search(r'\d+', answer)
                law_idx = int(m.group()) if m else 0
            except Exception:
                law_idx = 0

            if law_idx > 0 and law_idx <= len(laws):
                # Absorbed into existing law — strengthen it
                laws[law_idx - 1].setdefault("covers", [])
                laws[law_idx - 1]["covers"].append({"principle": universal, "trajectory_id": trajectory_dict.get("id", "")})
            else:
                # New law candidate — add it
                laws.append({
                    "law": universal,
                    "covers": [{"principle": universal, "trajectory_id": trajectory_dict.get("id", "")}],
                    "auto_generated": True,
                })
        else:
            laws = [{
                "law": universal,
                "covers": [{"principle": universal, "trajectory_id": trajectory_dict.get("id", "")}],
                "auto_generated": True,
            }]

        self.history.write_ghost_universal_laws(laws)

        # Re-evaluate dormant rules: new/strengthened law may cover more rules
        self._refresh_dormant_from_laws(laws, universal)

    def _refresh_dormant_from_laws(self, laws: list[dict], new_principle: str) -> None:
        """After a law is added or strengthened, check if candidate rules should sleep."""
        items = self.history.load_preference_rules_raw()
        if not items:
            return

        candidates = [r for r in items if r.get("status") == "candidate"]
        if not candidates:
            return

        # Word-overlap check: if the new principle shares 3+ content words
        # with a candidate rule, that rule is now covered.
        import re as _re
        _word_pat = _re.compile(r"[ぁ-んァ-ンー一-龯]{2,}|[a-zA-Z]{3,}")
        new_words = set(_word_pat.findall(new_principle.lower()))
        dormanted = 0
        for rule in candidates:
            instruction = (rule.get("instruction", "") or rule.get("statement", "")).lower()
            rule_words = set(_word_pat.findall(instruction))
            if len(new_words & rule_words) >= 3:
                rule["status"] = "dormant"
                rule["dormant_law"] = new_principle
                dormanted += 1

        if dormanted > 0:
            self.history.write_preference_rules_raw(items)

    # ------------------------------------------------------------------
    # Client-side sublimation support (LLM-free server)
    # ------------------------------------------------------------------

    def get_pending_sublimations(self) -> list[dict]:
        """Return fired ghost trajectories that need client-side LLM sublimation.

        Returns trajectories where:
        - fired == True
        - sublimated_principle is empty/missing OR was template-generated
        """
        trajectories = self.history.load_ghost_trajectories()
        ghost_list = self.history.load_ghosts()
        # Convert list[dict] to dict[str, dict] keyed by ghost ID
        ghost_by_id: dict[str, dict] = {d.get("id", ""): d for d in ghost_list if d.get("id")}
        pending = []

        # Template-generated principle markers (all known patterns)
        _TEMPLATE_MARKERS = (
            "繰り返し修正されてきた",
            "傾向がある",
            "繰り返し却下されている",
            "回却下された",
            "初期シグナル",
            "中程度の確信",
            "強い確信",
        )

        for t in trajectories:
            if not t.get("fired"):
                continue
            principle = (t.get("sublimated_principle") or "").strip()
            is_template = any(marker in principle for marker in _TEMPLATE_MARKERS)
            if not principle or is_template:
                ghost_ids = t.get("ghost_ids", [])
                ghosts_data = []
                for gid in ghost_ids[:5]:
                    g = ghost_by_id.get(gid)
                    if g:
                        ghosts_data.append({
                            "rejected_output": (g.get("rejected_output") or "")[:300],
                            "actual_outcome": (g.get("actual_outcome") or "")[:300],
                            "origin": g.get("origin", ""),
                        })
                pending.append({
                    "trajectory_id": t.get("id", ""),
                    "theme": t.get("theme", ""),
                    "scopes": t.get("scopes", []),
                    "ghost_count": len(ghost_ids),
                    "cumulative_pe": t.get("cumulative_pe", 0),
                    "current_principle": principle,
                    "ghosts": ghosts_data,
                })
        return pending

    def save_sublimation(
        self,
        trajectory_id: str,
        principle: str,
        universal_law: str = "",
        law_match_index: int = 0,
    ) -> dict:
        """Save a client-side LLM sublimation result.

        Args:
            trajectory_id: ID of the trajectory to update.
            principle: The sublimated principle (20-50 chars recommended).
            universal_law: Optional universal law generalized from principle.
            law_match_index: If > 0, absorb into existing law at this index (1-based).
        """
        import json as _json

        trajectories = self.history.load_ghost_trajectories()
        updated = False
        for t in trajectories:
            if t.get("id") == trajectory_id:
                t["sublimated_principle"] = principle.strip()
                updated = True
                break

        if not updated:
            return {"ok": False, "error": f"Trajectory {trajectory_id} not found"}

        # Save trajectories (using locked atomic write)
        try:
            self.history.write_ghost_trajectories(trajectories)
        except OSError:
            return {"ok": False, "error": "Failed to write trajectories"}

        # Handle universal law if provided
        if universal_law:
            laws = self.history.load_ghost_universal_laws()

            if law_match_index > 0 and law_match_index <= len(laws):
                # Absorb into existing law
                laws[law_match_index - 1].setdefault("covers", [])
                laws[law_match_index - 1]["covers"].append({
                    "principle": universal_law,
                    "trajectory_id": trajectory_id,
                })
            else:
                # New law
                laws.append({
                    "law": universal_law,
                    "covers": [{"principle": universal_law, "trajectory_id": trajectory_id}],
                    "auto_generated": True,
                    "client_sublimated": True,
                })

            self.history.write_ghost_universal_laws(laws)

            # Re-evaluate dormant rules
            self._refresh_dormant_from_laws(laws, universal_law)

        return {
            "ok": True,
            "trajectory_id": trajectory_id,
            "principle": principle,
            "universal_law": universal_law or None,
        }

    # ------------------------------------------------------------------
    # Client-side rule effectiveness evaluation
    # ------------------------------------------------------------------

    def evaluate_guidance_effectiveness(
        self,
        evaluations: list[dict],
        task_scope: str = "",
    ) -> dict:
        """Update rule metrics based on client LLM self-evaluation.

        Each evaluation: {"rule_id": str, "score": float 0.0-1.0, "reason": str}
        - score >= 0.7: rule helped → increment success_count, boost expected_gain
        - score 0.3-0.7: neutral → no change
        - score < 0.3: rule hurt or was irrelevant → increment failure_count

        This feeds into the existing promotion/demotion lifecycle.
        """
        items = self.history.load_preference_rules_raw()
        if not items:
            return {"ok": False, "error": "No rules found"}
        rules_by_id = {r.get("id", ""): r for r in items}

        updated = 0
        promoted = 0
        demoted = 0

        for ev in evaluations:
            rule_id = ev.get("rule_id", "")
            score = ev.get("score", 0.5)
            rule = rules_by_id.get(rule_id)
            if not rule:
                continue

            old_gain = rule.get("expected_gain", 0.0)

            if score >= 0.7:
                # Rule helped — boost
                rule["success_count"] = rule.get("success_count", 0) + 1
                rule["expected_gain"] = old_gain + (score - 0.5) * 2
                rule["confidence_score"] = min(
                    0.95,
                    rule.get("confidence_score", 0.5) + 0.05,
                )
                if rule.get("status") == "candidate":
                    # Check if ready for promotion
                    if rule.get("success_count", 0) >= 3 and rule.get("confidence_score", 0) >= 0.6:
                        rule["status"] = "promoted"
                        promoted += 1
            elif score < 0.3:
                # Rule hurt — penalize
                rule["failure_count"] = rule.get("failure_count", 0) + 1
                rule["expected_gain"] = max(0, old_gain - (0.5 - score) * 2)
                rule["confidence_score"] = max(
                    0.0,
                    rule.get("confidence_score", 0.5) - 0.05,
                )
                # Auto-demote if consistently failing
                if rule.get("failure_count", 0) >= 5 and rule.get("success_count", 0) < 2:
                    if rule.get("status") in ("promoted", "candidate"):
                        rule["status"] = "disabled"
                        demoted += 1

            updated += 1

        if updated > 0:
            self.history.write_preference_rules_raw(items)

        return {
            "ok": True,
            "evaluated": updated,
            "promoted": promoted,
            "demoted": demoted,
            "task_scope": task_scope,
        }

    # ------------------------------------------------------------------
    # Contextual feedback collection
    # ------------------------------------------------------------------

    def generate_session_feedback_question(
        self,
        task_scope: str = "",
        task_title: str = "",
        corrections_this_session: int = 0,
        guidance_was_injected: bool = False,
    ) -> dict:
        """Generate a natural feedback question based on what happened this session.

        Returns a dict with 'question' and 'options' that the AI should present
        to the user at the end of a session. The question is tailored to the
        task context — never mentions laws, rules, or system internals.
        """
        # Only ask when guidance was actually injected
        if not guidance_was_injected:
            return {"ask": False}

        # Build question from task context
        scope_questions = {
            "dashboard_development": "今回のダッシュボード作業",
            "correx_development": "今回の開発作業",
            "document_creation": "今回のドキュメント作成",
            "proposal_summary": "今回の提案書",
            "commercialization": "今回のビジネス検討",
            "game_development": "今回のゲーム開発",
        }

        task_label = scope_questions.get(task_scope, "")
        if not task_label and task_title:
            task_label = f"今回の「{task_title[:20]}」"
        if not task_label:
            task_label = "今回の作業"

        if corrections_this_session == 0:
            question = f"{task_label}、スムーズだった？"
        elif corrections_this_session <= 2:
            question = f"{task_label}、前よりやり直し減った？"
        else:
            question = f"{task_label}、手間取った？"

        return {
            "ask": True,
            "question": question,
            "options": ["スムーズだった", "いつも通り", "手間取った"],
            "context": {
                "task_scope": task_scope,
                "task_title": task_title,
                "corrections_this_session": corrections_this_session,
                "guidance_injected": True,
            },
        }

    def save_session_feedback(
        self,
        answer: str,
        task_scope: str = "",
        task_title: str = "",
        corrections_this_session: int = 0,
    ) -> dict:
        """Save the user's session feedback as a growth measurement.

        Maps answer to a score:
          'スムーズだった' → 1.0 (guidance helped)
          'いつも通り' → 0.5 (no change)
          '手間取った' → 0.0 (guidance didn't help or hurt)
        """
        import json as _json
        from datetime import datetime

        score_map = {
            "スムーズだった": 1.0,
            "smooth": 1.0,
            "いつも通り": 0.5,
            "normal": 0.5,
            "手間取った": 0.0,
            "struggled": 0.0,
        }
        score = score_map.get(answer, 0.5)

        record = {
            "type": "session_feedback",
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "task_scope": task_scope,
            "task_title": task_title,
            "answer": answer,
            "score": score,
            "corrections_this_session": corrections_this_session,
            "guidance_injected": True,
        }

        # Save to growth directory
        growth_dir = self.history.base_dir / "growth"
        growth_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = growth_dir / f"feedback-{ts}.json"
        filepath.write_text(_json.dumps(record, ensure_ascii=False, indent=2))

        return record

    def list_ghosts(self, limit: int = 50) -> list[dict]:
        """List stored ghosts, most recent first."""
        ghosts = self.history.load_ghosts()
        return ghosts[:limit]

    def list_ghost_trajectories(
        self,
        *,
        include_fired: bool = True,
        limit: int = 20,
    ) -> list[dict]:
        """List ghost trajectories."""
        trajectories = self.history.load_ghost_trajectories()
        if not include_fired:
            trajectories = [t for t in trajectories if not t.get("fired", False)]
        return trajectories[:limit]

    def get_fired_ghost_principles(self) -> list[str]:
        """Get all sublimated principles from fired trajectories.

        Applies runtime dedup: exact match removal + bigram similarity > 0.5.
        """
        import re

        trajectories = self.history.load_ghost_trajectories()
        raw: list[str] = []
        for t in trajectories:
            if t.get("fired") and t.get("sublimated_principle") and not t.get("dormant"):
                p = t["sublimated_principle"].strip()
                if p:
                    raw.append(p)

        if not raw:
            return []

        # --- sanitize broken principles ---
        def _sanitize(text: str) -> str:
            # Remove noise prefixes
            text = re.sub(r"^(汎用原則|固有原則|固有)\s*[:：]\s*", "", text)
            # Remove trailing explanations
            text = re.sub(r"\s*(この原則は|解説[:：]|を汎用化すると).*$", "", text, flags=re.DOTALL)
            # Remove markdown artifacts
            text = re.sub(r"[「」\"']", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            # Cut overly long to first sentence
            if len(text) > 60:
                first = re.split(r"[。\n]", text)[0]
                if first and len(first) > 5:
                    text = first
            return text.strip()

        sanitized = [_sanitize(p) for p in raw]
        sanitized = [p for p in sanitized if p and len(p) > 3]

        # --- exact dedup ---
        seen: set[str] = set()
        unique: list[str] = []
        for p in sanitized:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        # --- fuzzy dedup via char-bigram Jaccard ---
        from .text_similarity import char_ngrams as _ts_ngrams

        deduped: list[str] = []
        deduped_bigrams: list[set[str]] = []
        for p in unique:
            bg = _ts_ngrams(p, 2, particles=True)
            if not bg:
                deduped.append(p)
                deduped_bigrams.append(bg)
                continue
            is_dup = False
            for existing_bg in deduped_bigrams:
                if not existing_bg:
                    continue
                jaccard = len(bg & existing_bg) / len(bg | existing_bg)
                if jaccard > 0.40:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(p)
                deduped_bigrams.append(bg)

        return deduped

    def cleanup_ghost_principles(self) -> dict:
        """Clean up ghost principles: remove task-specific, merge duplicates, fix laws.

        Returns stats about what was cleaned.
        """
        from .ghost_engine import is_principle_generalizable
        from .text_similarity import char_ngrams as _ts_ngrams

        stats = {
            "principles_before": 0,
            "principles_after": 0,
            "principles_removed": 0,
            "laws_before": 0,
            "laws_after": 0,
            "laws_merged": 0,
        }

        # --- Phase 1: Clean abstracted principles ---
        principles = self.history.load_ghost_abstracted_principles()
        stats["principles_before"] = len(principles)

        cleaned_principles = []
        for p in principles:
            specific = p.get("specific", "")
            universal = p.get("universal", "")

            # Keep if the universal version is generalizable
            if universal and is_principle_generalizable(universal):
                cleaned_principles.append(p)
            # Or if specific is generalizable (and no universal)
            elif not universal and specific and is_principle_generalizable(specific):
                cleaned_principles.append(p)

        # Deduplicate by universal text (bigram Jaccard > 0.5)
        final_principles = []
        seen_bigrams: list[set[str]] = []
        for p in cleaned_principles:
            text = p.get("universal", "") or p.get("specific", "")
            bg = _ts_ngrams(text, 2, particles=True)
            if not bg:
                final_principles.append(p)
                seen_bigrams.append(bg)
                continue
            is_dup = False
            for existing_bg in seen_bigrams:
                if not existing_bg:
                    continue
                jaccard = len(bg & existing_bg) / len(bg | existing_bg)
                if jaccard > 0.50:
                    is_dup = True
                    break
            if not is_dup:
                final_principles.append(p)
                seen_bigrams.append(bg)

        stats["principles_after"] = len(final_principles)
        stats["principles_removed"] = stats["principles_before"] - stats["principles_after"]
        self.history.save_ghost_abstracted_principles(final_principles)

        # --- Phase 2: Deduplicate universal laws ---
        laws_data = self.history.load_ghost_universal_laws()
        laws = laws_data.get("items", []) if isinstance(laws_data, dict) else laws_data
        stats["laws_before"] = len(laws)

        # Merge exact duplicate law texts
        merged_laws: list[dict] = []
        law_text_map: dict[str, int] = {}
        for law in laws:
            text = law.get("law", "").strip()
            if text in law_text_map:
                idx = law_text_map[text]
                existing_covers = merged_laws[idx].get("covers", [])
                new_covers = law.get("covers", [])
                existing_ids = set()
                for c in existing_covers:
                    if isinstance(c, int):
                        existing_ids.add(c)
                    elif isinstance(c, dict):
                        existing_ids.add(c.get("trajectory_id", ""))
                for c in new_covers:
                    cid = c if isinstance(c, int) else c.get("trajectory_id", "")
                    if cid not in existing_ids:
                        existing_covers.append(c)
                        existing_ids.add(cid)
                merged_laws[idx]["covers"] = existing_covers
                stats["laws_merged"] += 1
            else:
                law_text_map[text] = len(merged_laws)
                merged_laws.append(law)

        # Also merge laws with high bigram similarity (> 0.7)
        further_merged: list[dict] = []
        fm_bigrams: list[set[str]] = []
        for law in merged_laws:
            text = law.get("law", "")
            bg = _ts_ngrams(text, 2, particles=True)
            merged_into = None
            if bg:
                for i, existing_bg in enumerate(fm_bigrams):
                    if not existing_bg:
                        continue
                    jaccard = len(bg & existing_bg) / len(bg | existing_bg)
                    if jaccard > 0.70:
                        merged_into = i
                        break
            if merged_into is not None:
                existing = further_merged[merged_into]
                if len(law.get("law", "")) > len(existing.get("law", "")):
                    existing["law"] = law["law"]
                existing_covers = existing.get("covers", [])
                existing_covers.extend(law.get("covers", []))
                existing["covers"] = existing_covers
                stats["laws_merged"] += 1
            else:
                further_merged.append(law)
                fm_bigrams.append(bg)

        stats["laws_after"] = len(further_merged)

        if isinstance(laws_data, dict):
            laws_data["items"] = further_merged
            self.history.write_ghost_universal_laws(laws_data)
        else:
            self.history.write_ghost_universal_laws(further_merged)

        # --- Phase 3: Clean trajectory sublimated_principles ---
        # This is what get_fired_ghost_principles actually reads
        trajectories = self.history.load_ghost_trajectories()
        traj_cleaned = 0
        for t in trajectories:
            if t.get("fired") and t.get("sublimated_principle"):
                principle = t["sublimated_principle"]
                if not is_principle_generalizable(principle):
                    t["sublimated_principle"] = ""
                    traj_cleaned += 1

        # --- Phase 3b: Merge duplicate sublimated_principles across trajectories ---
        # Keep the first trajectory with each unique principle; merge ghosts into it.
        # Uses bigram Jaccard (> 0.50) same as Phase 1.
        seen_principles: list[tuple[set[str], int]] = []  # (bigrams, traj_index)
        traj_merged = 0
        for i, t in enumerate(trajectories):
            if not t.get("fired") or not t.get("sublimated_principle"):
                continue
            principle = t["sublimated_principle"]
            bg = _ts_ngrams(principle, 2, particles=True)
            if not bg:
                continue
            is_dup = False
            for existing_bg, keeper_idx in seen_principles:
                if not existing_bg:
                    continue
                jaccard = len(bg & existing_bg) / len(bg | existing_bg)
                if jaccard > 0.50:
                    keeper = trajectories[keeper_idx]
                    keeper_ghosts = keeper.get("ghosts", [])
                    dup_ghosts = t.get("ghosts", [])
                    existing_ids = {g.get("id", "") for g in keeper_ghosts if g.get("id")}
                    for g in dup_ghosts:
                        if g.get("id", "") not in existing_ids:
                            keeper_ghosts.append(g)
                    keeper["ghosts"] = keeper_ghosts
                    keeper["cumulative_pe"] = round(
                        float(keeper.get("cumulative_pe", 0)) + float(t.get("cumulative_pe", 0)), 4
                    )
                    t["sublimated_principle"] = ""
                    t["fired"] = False
                    t["dormant"] = True
                    traj_merged += 1
                    is_dup = True
                    break
            if not is_dup:
                seen_principles.append((bg, i))

        traj_modified = traj_cleaned + traj_merged
        if traj_modified > 0:
            self.history.write_ghost_trajectories(trajectories)
        stats["trajectories_cleaned"] = traj_cleaned
        stats["trajectories_merged"] = traj_merged

        return stats

    # ── Journey Memory ───────────────────────────────────────────────────

    def save_journey(
        self,
        *,
        where: str,
        scope: str = "",
        impression: list[str] | None = None,
        valence: float = 0.5,
        journey_type: str = "wander",
        detail: str = "",
        connected_turn_id: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Save an episodic journey memory (search/exploration trace).

        journey_type:
          - "business": user-requested research. Full detail, never forgotten.
          - "wander": incidental exploration. Impressions only, forgettable.
        """
        import hashlib
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M")
        impression = impression or []
        tags = tags or []

        # Compute novelty: how different from existing journeys
        existing = self.history.load_journeys()
        similar_count = 0
        imp_set = set(impression)
        for j in existing:
            j_imp = set(j.get("impression", []))
            if imp_set and j_imp:
                overlap = len(imp_set & j_imp) / max(len(imp_set | j_imp), 1)
                if overlap > 0.3:
                    similar_count += 1

        novelty = 1.0 / (similar_count + 1)
        forgettable = journey_type != "business"

        import uuid
        journey_id = hashlib.sha256(
            f"{where}:{now}:{scope}:{uuid.uuid4().hex[:8]}".encode()
        ).hexdigest()[:16]

        # SWR tag: reward-weighted consolidation priority
        # High valence + connected to a turn = high consolidation priority
        swr_tag = round(valence * (1.0 if connected_turn_id else 0.5), 3)

        journey_dict = {
            "id": f"journey-{journey_id}",
            "where": where,
            "when": now,
            "scope": scope,
            "impression": impression,
            "valence": round(valence, 3),
            "novelty": round(novelty, 3),
            "journey_type": journey_type,
            "detail": detail if journey_type == "business" else "",
            "connected_turn_id": connected_turn_id,
            "tags": tags,
            "dormant": False,
            "awakened_count": 0,
            "revisit_count": 0,
            "forgettable": forgettable,
            "swr_tag": swr_tag,
            "labile_until": "",
        }

        self.history.save_journey(journey_dict)
        return journey_dict

    @staticmethod
    def _similarity_band(similarity: float) -> str:
        """Classify similarity into neuroscience-inspired bands.

        <0.15  = irrelevant (no signal)
        0.15-0.35 = weak_association (faint familiarity)
        0.35-0.65 = deja_vu (the productive middle — highest creative potential)
        >0.65  = direct_match (strong recall)
        """
        if similarity < 0.15:
            return "irrelevant"
        if similarity < 0.35:
            return "weak_association"
        if similarity < 0.65:
            return "deja_vu"
        return "direct_match"

    def awaken_journeys(
        self,
        *,
        context_keywords: list[str],
        scope: str = "",
        threshold: float = 0.2,
        limit: int = 5,
    ) -> list[dict]:
        """Find and awaken dormant journey memories triggered by current context.

        Returns list of awakened journeys with similarity band classification.
        Bands: irrelevant (<0.15), weak_association (0.15-0.35),
               deja_vu (0.35-0.65), direct_match (>0.65).
        Awakened journeys enter a 30-minute labile window where they can be updated.
        """
        from datetime import datetime, timedelta, timezone

        journeys = self.history.load_journeys()
        context_set = set(context_keywords)
        if not context_set:
            return []

        awakened = []
        changed = False
        now = datetime.now(timezone.utc)
        labile_until = (now + timedelta(minutes=30)).isoformat()

        for j in journeys:
            j_imp = set(j.get("impression", []))
            if not j_imp:
                continue

            overlap = len(context_set & j_imp) / max(len(context_set | j_imp), 1)

            if scope and j.get("scope") == scope:
                overlap += 0.15

            band = self._similarity_band(overlap)
            if band == "irrelevant":
                continue

            if overlap > threshold:
                if j.get("forgotten", False):
                    continue  # forgotten journeys cannot awaken
                if j.get("dormant", False):
                    j["dormant"] = False
                    j["awakened_count"] = j.get("awakened_count", 0) + 1
                    j["labile_until"] = labile_until  # Labile Window: 30min update window
                    changed = True
                j["revisit_count"] = j.get("revisit_count", 0) + 1
                changed = True
                awakened.append({
                    "journey": j,
                    "similarity": round(overlap, 3),
                    "band": band,
                    "deja_vu": j.get("awakened_count", 0) > 1,
                })

        if changed:
            self.history.write_journeys(journeys)

        awakened.sort(key=lambda x: x["similarity"], reverse=True)
        return awakened[:limit]

    def update_journey(
        self,
        *,
        journey_id: str,
        impression: list[str] | None = None,
        valence: float | None = None,
        detail: str | None = None,
    ) -> dict:
        """Update a journey during its labile window (30min after awakening).

        Only awakened journeys within their labile window can be updated.
        This implements memory reconsolidation: recalled memories become
        temporarily malleable before re-stabilizing.
        """
        from datetime import datetime, timezone

        journeys = self.history.load_journeys()
        now = datetime.now(timezone.utc)

        for j in journeys:
            if j.get("id") != journey_id:
                continue

            # Check labile window
            labile_str = j.get("labile_until", "")
            if not labile_str:
                return {"ok": False, "error": "not_labile", "message": "Journey is not in labile state"}

            try:
                labile_dt = datetime.fromisoformat(labile_str)
                if now > labile_dt:
                    j["labile_until"] = ""  # Window closed
                    self.history.write_journeys(journeys)
                    return {"ok": False, "error": "window_closed", "message": "Labile window has expired"}
            except (ValueError, TypeError):
                return {"ok": False, "error": "invalid_labile", "message": "Invalid labile timestamp"}

            # Update allowed fields
            if impression is not None:
                j["impression"] = impression
            if valence is not None:
                j["valence"] = round(valence, 3)
                # Recompute SWR tag
                connected = bool(j.get("connected_turn_id"))
                j["swr_tag"] = round(valence * (1.0 if connected else 0.5), 3)
            if detail is not None:
                j["detail"] = detail

            self.history.write_journeys(journeys)
            return {"ok": True, "journey_id": journey_id, "labile_remaining_sec": max(0, int((labile_dt - now).total_seconds()))}

        return {"ok": False, "error": "not_found", "message": f"Journey {journey_id} not found"}

    def list_journeys(
        self,
        *,
        limit: int = 20,
        journey_type: str = "",
        include_dormant: bool = False,
    ) -> list[dict]:
        """List stored journey memories."""
        journeys = self.history.load_journeys()
        journeys = [j for j in journeys if not j.get("forgotten", False)]
        if journey_type:
            journeys = [j for j in journeys if j.get("journey_type") == journey_type]
        if not include_dormant:
            journeys = [j for j in journeys if not j.get("dormant", False)]
        return journeys[:limit]

    def dormant_journeys(self, *, max_idle_days: int = 30) -> dict:
        """Scan journeys for dormancy, forgetting, and labile window expiry.

        SWR protection: journeys with swr_tag >= 0.7 are immune to forgetting
        (like high-reward memories consolidated during sleep).
        Labile windows: expired labile_until timestamps are cleared.
        """
        from datetime import datetime, timedelta, timezone

        journeys = self.history.load_journeys()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_idle_days)
        dormant_count = 0
        forgotten_count = 0
        labile_expired = 0

        for j in journeys:
            # Close expired labile windows
            labile_str = j.get("labile_until", "")
            if labile_str:
                try:
                    labile_dt = datetime.fromisoformat(labile_str)
                    if now > labile_dt:
                        j["labile_until"] = ""
                        labile_expired += 1
                except (ValueError, TypeError):
                    j["labile_until"] = ""

            if not j.get("forgettable", True):
                continue
            if j.get("forgotten", False):
                continue  # already forgotten, skip

            when_str = j.get("when", "")
            try:
                when_dt = datetime.strptime(when_str, "%Y/%m/%d %H:%M").replace(
                    tzinfo=timezone.utc
                )
            except (ValueError, TypeError):
                continue

            if not j.get("dormant", False):
                if j.get("awakened_count", 0) == 0:
                    days_old = (now - when_dt).days
                    if days_old > 7:
                        j["dormant"] = True
                        dormant_count += 1
            else:
                # SWR protection: high-value journeys resist forgetting
                swr = j.get("swr_tag", 0)
                if swr >= 0.7:
                    continue  # protected by SWR consolidation

                if j.get("awakened_count", 0) == 0 and when_dt < cutoff:
                    j["impression"] = []
                    j["detail"] = ""
                    j["dormant"] = False
                    j["forgotten"] = True
                    forgotten_count += 1

        self.history.write_journeys(journeys)
        return {
            "dormant": dormant_count,
            "forgotten": forgotten_count,
            "labile_expired": labile_expired,
            "total": len(journeys),
        }

    # ── Autonomous Intelligence Engine ────────────────────────────────────

    def _get_engine(self):
        """Lazy-init the autonomous engine."""
        if self._autonomous_engine is None:
            from .autonomous import AutonomousEngine
            self._autonomous_engine = AutonomousEngine(self.history)
        return self._autonomous_engine

    def run_autonomous_tick(
        self,
        *,
        event_type: str = "",
        scope: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict:
        """Run one tick of the autonomous intelligence engine.

        event_type: "correction", "praise", "question", "time", "ghost_fire"
                    Empty string = self-reflection mode (no external event).
        """
        from .autonomous import Event

        engine = self._get_engine()

        event = None
        if event_type:
            event = Event(
                type=event_type,
                scope=scope,
                tags=tags or [],
                keywords=keywords or [],
            )

        result = engine.tick(event)

        return {
            "cycle": result.cycle_count,
            "rules_count": len(result.decision.applicable_rules),
            "policies_count": len(result.decision.applicable_policies),
            "tensions_count": len(result.decision.active_tensions),
            "awakened_journeys": len(result.decision.awakened_journeys),
            "weakest_scope": result.decision.weakest_scope,
            "curiosity_hotspots": result.decision.curiosity_hotspots,
            "prediction": {
                "scope": result.prediction.scope,
                "confidence": result.prediction.confidence,
            } if result.prediction else None,
            "verification_error": result.verification_error,
            "modulations": result.modulations,
            "cry": {
                "need_type": result.cry.need.type,
                "scope": result.cry.need.scope,
                "urgency": result.cry.need.urgency,
                "deficit": result.cry.need.deficit,
                "description": result.cry.need.description,
                "is_intentional": result.cry.is_intentional,
                "predicted_effect": result.cry.predicted_effect,
            } if result.cry else None,
        }

    def get_engine_state(self) -> dict:
        """Get the current state of the autonomous engine."""
        engine = self._get_engine()
        return engine.get_state()

    # ── Ghost Semanticization ─────────────────────────────────────────────

    def semanticize_ghost_memories(self) -> dict:
        """Run episodic → semantic transformation on Ghost memories.

        Ghosts in fired trajectories gradually lose text detail:
        full → gist (50 chars) → trace (metadata only).
        Decay speed depends on emotional origin (scolded=slow, rejected=fast).
        """
        ghosts = self.history.load_ghosts()
        trajectories = self.history.load_ghost_trajectories()
        ghosts, stats = semanticize_ghosts(ghosts, trajectories)
        if stats["gisted"] > 0 or stats["traced"] > 0:
            self.history.write_ghosts(ghosts)
        return stats

    # ── Communication Outcome ────────────────────────────────────────────

    def record_communication_outcome(
        self, need_type: str, resolved: bool,
    ) -> dict:
        """Record whether the engine's cry was heard and resolved.

        This is how the engine learns that crying works —
        the reflexive → intentional transition.
        """
        engine = self._get_or_create_engine()
        engine.record_communication_outcome(need_type, resolved)
        self._save_engine_state(engine)
        return {
            "need_type": need_type,
            "resolved": resolved,
            "communication_outcomes": engine._state.get("communication_outcomes", {}),
        }
