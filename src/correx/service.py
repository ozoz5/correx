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
        import json as _json

        rules_path = self.history.base_dir / "preference_rules.json"
        if not rules_path.exists():
            return

        try:
            data = _json.loads(rules_path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError):
            return

        items = data["items"] if isinstance(data, dict) and "items" in data else data
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
            rule_words = set(w for w in instruction if len(w) > 2)
            signal_words = set(w for w in signal if len(w) > 2)
            overlap = len(rule_words & signal_words)
            if overlap >= 3 or scope == rule.get("applies_to_scope", ""):
                rule["status"] = "candidate"
                rule.pop("dormant_law_index", None)
                rule.pop("dormant_law", None)
                awakened += 1

        if awakened > 0:
            if isinstance(data, dict):
                data["items"] = items
            try:
                rules_path.write_text(_json.dumps(data, ensure_ascii=False, indent=2))
            except OSError:
                pass

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

        # Ghost layer: sublimated principles from rejected proposals
        ghost_section = self._build_ghost_guidance()
        if ghost_section:
            sections.append(ghost_section)

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
        laws_path = self.history.base_dir / "ghost_universal_laws.json"
        if laws_path.exists():
            try:
                laws = _json.loads(laws_path.read_text(encoding="utf-8"))
                if laws:
                    law_lines = ["[禁止法理 — 却下パターンから昇華された行動制約]"]
                    for i, law in enumerate(laws, 1):
                        text = law.get("law", "") if isinstance(law, dict) else str(law)
                        if text:
                            law_lines.append(f"  {i}. {text}")
                    sections.append("\n".join(law_lines))
                    has_laws = True
            except (ValueError, OSError, KeyError, UnicodeDecodeError):
                pass

        # 2. Positive laws (drivers)
        pos_path = self.history.base_dir / "ghost_positive_laws.json"
        if pos_path.exists():
            try:
                pos_laws = _json.loads(pos_path.read_text(encoding="utf-8"))
                if pos_laws:
                    pos_lines = ["[推奨法理 — 肯定反応から抽出された推進基準]"]
                    for i, law in enumerate(pos_laws, 1):
                        text = law.get("law", "") if isinstance(law, dict) else str(law)
                        if text:
                            pos_lines.append(f"  +{i}. {text}")
                    sections.append("\n".join(pos_lines))
                    has_positive = True
            except (ValueError, OSError, KeyError, UnicodeDecodeError):
                pass

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
        # Laws are the umbrella for unknown situations.
        # Specific principles are zero-judgment instant-apply rules for known situations.
        # Both must coexist. Do not suppress specifics just because laws exist.
        # But we filter for quality and cap to avoid flooding context.
        principles = self.get_fired_ghost_principles()
        if principles:
            # Clean up: remove format artifacts from 7b generation
            cleaned = []
            seen = set()
            for p in principles:
                # Strip common artifacts
                p = p.strip()
                for prefix in ["**汎用原則:**", "**汎用原則：**", "汎用原則:", "固有原則:"]:
                    if p.startswith(prefix):
                        p = p[len(prefix):].strip()
                # Take first line only, strip quotes
                p = p.split("\n")[0].strip().strip("「」\"'")
                # Skip if too short, too long, or duplicate
                if len(p) < 5 or len(p) > 80:
                    continue
                if p in seen:
                    continue
                seen.add(p)
                cleaned.append(p)

            if cleaned:
                # Cap at 20 to balance specificity vs context usage
                top = cleaned[:20]
                p_lines = [
                    "[固有原則 — 既知の場面で即適用する具体基準]",
                    "法理は未知の場面で類推適用する傘。固有原則は既知の場面で判断不要で即適用する。",
                    "両方が共存する。法理があるからといって固有原則を無視するな。",
                ]
                for i, p in enumerate(top, 1):
                    p_lines.append(f"  {i}. {p}")
                if len(cleaned) > 20:
                    p_lines.append(f"  (他{len(cleaned) - 20}件の固有原則あり)")
                sections.append("\n".join(p_lines))

        return "\n\n".join(sections)

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

    # ------------------------------------------------------------------
    # Ghost Engine
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

        # Process ghost through pipeline
        updated_ghost, updated_trajectory, fired_principles = process_ghost(
            ghost=ghost,
            trajectories=trajectories,
            all_ghosts=all_ghosts,
            metabolism_rate=metabolism_rate,
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
        laws_path = self.history.base_dir / "ghost_universal_laws.json"
        try:
            laws = _json.loads(laws_path.read_text(encoding="utf-8")) if laws_path.exists() else []
        except (ValueError, OSError):
            laws = []

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

        try:
            laws_path.write_text(_json.dumps(laws, ensure_ascii=False, indent=2))
        except OSError:
            pass

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
        """Get all sublimated principles from fired trajectories."""
        trajectories = self.history.load_ghost_trajectories()
        principles = []
        for t in trajectories:
            if t.get("fired") and t.get("sublimated_principle"):
                principles.append(t["sublimated_principle"])
        return principles
