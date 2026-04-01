from __future__ import annotations

from pathlib import Path

from .auto_train import run_auto_training_cycle
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
from .schemas import Meaning, Principle
from .mlx_trainer import MlxLoraTrainingConfig
from .secret_store import delete_secure_secret, get_secure_secret, set_secure_secret
from .training_dataset import export_mlx_lm_dataset


class PseudoIntelligenceService:
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
        )
        if auto_record_growth:
            turns = self.history.load_conversation_turns()
            self.growth.auto_record_from_turns(turns)
        return turn

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

        return "\n\n".join(section for section in sections if section)

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
        profile = compute_personality_profile(turns, rules)
        interventions = detect_interventions(rules, turns, profile)
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
        """Overall growth summary across all cases."""
        return self.growth.summary()

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
