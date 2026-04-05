from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - unavailable on POSIX
    msvcrt = None

from .conversation_learning import (
    extract_correction_candidates,
    extract_keywords,
    is_explicit_directive,
    normalize_correction_statement,
    normalize_text,
)
from .memory_manager import (
    archive_turns_to_episode,
    build_context_signature,
    derive_context_confidence_score,
    derive_context_mode,
    derive_latent_context_confidence_score,
    derive_latent_context_expected_gain,
    derive_latent_context_prior_weight,
    derive_rule_confidence_score,
    derive_rule_expected_gain,
    derive_rule_status,
    derive_transition_confidence_score,
    derive_transition_forecast_score,
    evict_episodes,
    evict_turns,
    flatten_latent_contexts,
    infer_latent_contexts_from_rule,
    merge_rule_contexts,
    merge_latent_contexts,
    merge_similar_rules,
    reconsolidate_rules_from_turns,
    resolve_contradicting_rules,
    auto_correct_flagged_rules,
)
from .rule_builder import (
    build_preference_rules,
    synthesize_rules_from_turns,
    compute_self_overcome_proposals,
)
from .reaction_scorer import score_reaction
from .llm_scorer import LlmScorer
from .schemas import (
    ConversationTurn,
    CorrectionRecord,
    EpisodeRecord,
    LatentContext,
    LatentTransition,
    Meaning,
    Policy,
    PreferenceRule,
    Principle,
    RuleContext,
    Tension,
    TrainingExample,
)


class HistoryStore:
    def __init__(self, base_dir: str | Path, *, scorer: LlmScorer | None = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_dir / "history.json"
        self.conversation_file = self.base_dir / "conversation_history.json"
        self.preference_rules_file = self.base_dir / "preference_rules.json"
        self.context_transitions_file = self.base_dir / "context_transitions.json"
        self.meanings_file = self.base_dir / "meanings.json"
        self.principles_file = self.base_dir / "principles.json"
        self.personality_file = self.base_dir / "personality.json"
        self.deferred_meanings_file = self.base_dir / "deferred_meanings.json"
        self.policies_file = self.base_dir / "policies.json"
        self.ghosts_file = self.base_dir / "ghosts.json"
        self.ghost_trajectories_file = self.base_dir / "ghost_trajectories.json"
        self.curiosity_signals_file = self.base_dir / "curiosity_signals.json"
        self.tensions_file = self.base_dir / "tensions.json"
        self.narrative_file = self.base_dir / "narrative.json"
        self.knowledge_gap_clusters_file = self.base_dir / "knowledge_gap_clusters.json"
        self.ghost_universal_laws_file = self.base_dir / "ghost_universal_laws.json"
        self.ghost_positive_laws_file = self.base_dir / "ghost_positive_laws.json"
        self.ghost_abstracted_principles_file = self.base_dir / "ghost_abstracted_principles.json"
        self.lock_file = self.base_dir / ".store.lock"
        self._scorer = scorer  # None = rule-based only

    def _lock_handle(self) -> TextIO:
        handle = self.lock_file.open("a+", encoding="utf-8")
        handle.seek(0)
        if not handle.read(1):
            handle.write("0")
            handle.flush()
        handle.seek(0)

        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            return handle

        if msvcrt is not None:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return handle

        handle.close()
        raise RuntimeError("File locking is not supported on this platform")

    def _unlock_handle(self, handle: TextIO) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return

            if msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                return
        finally:
            handle.close()

    def _read_json_list(self, path: Path) -> list[dict]:
        if not path.exists():
            # Try backup if primary is missing
            backup = path.with_suffix(path.suffix + ".bak")
            if backup.exists():
                try:
                    payload = json.loads(backup.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        import sys
                        print(
                            f"[correx] WARNING: {path.name} missing, recovered from backup",
                            file=sys.stderr,
                        )
                        return [item for item in payload if isinstance(item, dict)]
                except Exception:
                    pass
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Primary file is corrupted — try backup
            import sys
            print(
                f"[correx] ERROR: {path.name} corrupted ({exc}), trying backup",
                file=sys.stderr,
            )
            backup = path.with_suffix(path.suffix + ".bak")
            if backup.exists():
                try:
                    payload = json.loads(backup.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        print(
                            f"[correx] RECOVERED: loaded {len(payload)} items from {backup.name}",
                            file=sys.stderr,
                        )
                        return [item for item in payload if isinstance(item, dict)]
                except Exception:
                    pass
            print(
                f"[correx] DATA LOSS: no usable backup for {path.name}",
                file=sys.stderr,
            )
            return []
        except Exception as exc:
            import sys
            print(
                f"[correx] ERROR: failed to read {path.name}: {exc}",
                file=sys.stderr,
            )
            return []
        # Support both legacy (bare array) and versioned ({"schema_version": ..., "items": [...]}) formats
        if isinstance(payload, dict) and "items" in payload:
            payload = payload["items"]
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    # Schema version embedded in every JSON file for forward/backward compat.
    SCHEMA_VERSION = "1.0"

    def _atomic_write_json(self, path: Path, payload: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create backup of existing file before overwriting
        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            try:
                import shutil
                shutil.copy2(path, backup)
            except OSError:
                pass  # Best-effort backup
        # Wrap payload with schema version for migration safety
        versioned = {"schema_version": self.SCHEMA_VERSION, "items": payload}
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(versioned, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)

    def _load_entries_unlocked(self) -> list[EpisodeRecord]:
        return [self._normalize_entry(item) for item in self._read_json_list(self.history_file)]

    def _write_entries_unlocked(self, entries: list[EpisodeRecord]) -> None:
        self._atomic_write_json(self.history_file, [asdict(entry) for entry in entries])

    def _load_conversation_turns_unlocked(self) -> list[ConversationTurn]:
        return [self._normalize_turn(item) for item in self._read_json_list(self.conversation_file)]

    def _write_conversation_turns_unlocked(self, turns: list[ConversationTurn]) -> None:
        self._atomic_write_json(self.conversation_file, [asdict(turn) for turn in turns])

    def _load_preference_rules_unlocked(self) -> list[PreferenceRule]:
        return [self._normalize_rule(item) for item in self._read_json_list(self.preference_rules_file)]

    def _write_preference_rules_unlocked(self, rules: list[PreferenceRule]) -> None:
        self._atomic_write_json(self.preference_rules_file, [asdict(rule) for rule in rules])

    def _load_context_transitions_unlocked(self) -> list[LatentTransition]:
        return [self._normalize_transition(item) for item in self._read_json_list(self.context_transitions_file)]

    def _write_context_transitions_unlocked(self, transitions: list[LatentTransition]) -> None:
        self._atomic_write_json(self.context_transitions_file, [asdict(transition) for transition in transitions])

    def _load_meanings_unlocked(self) -> list[Meaning]:
        return [self._normalize_meaning(item) for item in self._read_json_list(self.meanings_file)]

    def _write_meanings_unlocked(self, meanings: list[Meaning]) -> None:
        self._atomic_write_json(self.meanings_file, [asdict(m) for m in meanings])

    @staticmethod
    def _normalize_meaning(item: dict) -> Meaning:
        return Meaning(
            id=str(item.get("id", "")),
            principle=str(item.get("principle", "")),
            normalized_principle=str(item.get("normalized_principle", "")),
            summary=str(item.get("summary", "")),
            source_rule_ids=list(item.get("source_rule_ids", [])),
            scopes=list(item.get("scopes", [])),
            tags=list(item.get("tags", [])),
            strength=int(item.get("strength", 0)),
            cross_scope_count=int(item.get("cross_scope_count", 0)),
            confidence=float(item.get("confidence", 0.0)),
            first_seen_at=str(item.get("first_seen_at", "")),
            last_seen_at=str(item.get("last_seen_at", "")),
            personal_settings_overlap=list(item.get("personal_settings_overlap", [])),
            status=str(item.get("status", "active")),
        )

    def _load_principles_unlocked(self) -> list[Principle]:
        return [self._normalize_principle(item) for item in self._read_json_list(self.principles_file)]

    def _write_principles_unlocked(self, principles: list[Principle]) -> None:
        self._atomic_write_json(self.principles_file, [asdict(p) for p in principles])

    @staticmethod
    def _normalize_principle(item: dict) -> Principle:
        return Principle(
            id=str(item.get("id", "")),
            declaration=str(item.get("declaration", "")),
            normalized_declaration=str(item.get("normalized_declaration", "")),
            source_meaning_ids=list(item.get("source_meaning_ids", [])),
            source_rule_count=int(item.get("source_rule_count", 0)),
            depth=int(item.get("depth", 3)),
            scopes=list(item.get("scopes", [])),
            confidence=float(item.get("confidence", 0.0)),
            first_seen_at=str(item.get("first_seen_at", "")),
            last_seen_at=str(item.get("last_seen_at", "")),
            personal_settings_overlap=list(item.get("personal_settings_overlap", [])),
            status=str(item.get("status", "active")),
        )

    def _load_policies_unlocked(self) -> list[Policy]:
        return [self._normalize_policy(item) for item in self._read_json_list(self.policies_file)]

    def _write_policies_unlocked(self, policies: list[Policy]) -> None:
        self._atomic_write_json(self.policies_file, [asdict(p) for p in policies])

    @staticmethod
    def _normalize_policy(item: dict) -> Policy:
        return Policy(
            id=str(item.get("id", "")),
            title=str(item.get("title", "")),
            core=str(item.get("core", "")),
            why=str(item.get("why", "")),
            analogy=str(item.get("analogy", "")),
            opposite=str(item.get("opposite", "")),
            limits=str(item.get("limits", "")),
            source_rule_ids=list(item.get("source_rule_ids", [])),
            source_ghost_ids=list(item.get("source_ghost_ids", [])),
            source_law_ids=list(item.get("source_law_ids", [])),
            scopes=list(item.get("scopes", [])),
            tags=list(item.get("tags", [])),
            evidence_count=int(item.get("evidence_count", 0)),
            maturity=str(item.get("maturity", "proposed")),
            created_at=str(item.get("created_at", "")),
            updated_at=str(item.get("updated_at", "")),
            approved_by=str(item.get("approved_by", "")),
        )

    # ── Tension (contradiction montage) ─────────────────────────

    def _load_tensions_unlocked(self) -> list[Tension]:
        return [self._normalize_tension(item) for item in self._read_json_list(self.tensions_file)]

    def _write_tensions_unlocked(self, tensions: list[Tension]) -> None:
        self._atomic_write_json(self.tensions_file, [asdict(t) for t in tensions])

    @staticmethod
    def _normalize_tension(item: dict) -> Tension:
        return Tension(
            id=str(item.get("id", "")),
            rule_a_id=str(item.get("rule_a_id", "")),
            rule_a_text=str(item.get("rule_a_text", "")),
            rule_b_id=str(item.get("rule_b_id", "")),
            rule_b_text=str(item.get("rule_b_text", "")),
            boundary=str(item.get("boundary", "")),
            signal=str(item.get("signal", "")),
            evidence_a=list(item.get("evidence_a", [])),
            evidence_b=list(item.get("evidence_b", [])),
            scopes=list(item.get("scopes", [])),
            confidence=float(item.get("confidence", 0.0)),
            created_at=str(item.get("created_at", "")),
            updated_at=str(item.get("updated_at", "")),
            status=str(item.get("status", "active")),
        )

    def load_tensions(self) -> list[Tension]:
        handle = self._lock_handle()
        try:
            return self._load_tensions_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_tensions(self, tensions: list[Tension]) -> None:
        handle = self._lock_handle()
        try:
            self._write_tensions_unlocked(tensions)
        finally:
            self._unlock_handle(handle)

    # ── EpisodeRecord ─────────────────────────────────────────

    def _normalize_entry(self, item: dict) -> EpisodeRecord:
        corrections = [
            CorrectionRecord(
                recorded_at=str(entry.get("recorded_at", "")),
                decision_override=str(entry.get("decision_override", "")),
                correction_note=str(entry.get("correction_note", "")),
                reuse_note=str(entry.get("reuse_note", "")),
                reason=str(entry.get("reason", "")),
                scope=str(entry.get("scope", "")),
                bad_output=str(entry.get("bad_output", "")),
                revised_output=str(entry.get("revised_output", "")),
                tool_used=str(entry.get("tool_used", "")),
                source_user=str(entry.get("source_user", "")),
                accepted=bool(entry.get("accepted", True)),
            )
            for entry in item.get("corrections", [])
            if isinstance(entry, dict)
        ]
        training_example = self._normalize_training_example(item.get("training_example"))
        return EpisodeRecord(
            id=str(item.get("id", "")),
            timestamp=str(item.get("timestamp", "")),
            title=str(item.get("title", "No Title")),
            issuer=str(item.get("issuer", "")),
            task_type=str(item.get("task_type", "generic")),
            profile_name=str(item.get("profile_name", "")),
            source_text=str(item.get("source_text", "")),
            company_profile=item.get("company_profile") if isinstance(item.get("company_profile"), dict) else {},
            corrections=corrections,
            last_corrected_at=str(item.get("last_corrected_at", "")),
            output=item.get("output") if isinstance(item.get("output"), dict) else {},
            training_example=training_example,
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        )

    def _normalize_training_example(self, item: object) -> TrainingExample | None:
        if not isinstance(item, dict):
            return None

        format_name = str(item.get("format", "chat")).strip().lower() or "chat"
        if format_name not in {"chat", "completions"}:
            format_name = "chat"

        return TrainingExample(
            updated_at=str(item.get("updated_at", "")),
            format=format_name,
            system_message=str(item.get("system_message", "")),
            user_message=str(item.get("user_message", "")),
            prompt=str(item.get("prompt", "")),
            draft_output=str(item.get("draft_output", "")),
            rejected_output=str(item.get("rejected_output", "")),
            accepted_output=str(item.get("accepted_output", "")),
            feedback=str(item.get("feedback", "")),
            accepted=bool(item.get("accepted", False)),
            model_id=str(item.get("model_id", "")),
            policy_version=str(item.get("policy_version", "")),
            accepted_by=str(item.get("accepted_by", "")),
            tags=[str(entry).strip() for entry in item.get("tags", []) if str(entry).strip()],
            temperature=(
                float(item.get("temperature"))
                if isinstance(item.get("temperature"), (int, float))
                else None
            ),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        )

    def _serialize_text_payload(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False, indent=2).strip()

    def load_entries(self) -> list[EpisodeRecord]:
        handle = self._lock_handle()
        try:
            return self._load_entries_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_entries(self, entries: list[EpisodeRecord]) -> None:
        handle = self._lock_handle()
        try:
            self._write_entries_unlocked(entries)
        finally:
            self._unlock_handle(handle)

    def _normalize_turn(self, item: dict) -> ConversationTurn:
        corrections = [
            str(entry).strip()
            for entry in item.get("extracted_corrections", [])
            if str(entry).strip()
        ]
        tags = [str(entry).strip() for entry in item.get("tags", []) if str(entry).strip()]
        raw_score = item.get("reaction_score")
        reaction_score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        return ConversationTurn(
            id=str(item.get("id", "")),
            recorded_at=str(item.get("recorded_at", "")),
            task_scope=str(item.get("task_scope", "")),
            user_message=str(item.get("user_message", "")),
            assistant_message=str(item.get("assistant_message", "")),
            user_feedback=str(item.get("user_feedback", "")),
            extracted_corrections=corrections,
            tags=tags,
            reaction_score=reaction_score,
            guidance_applied=bool(item.get("guidance_applied", False)),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        )

    def load_conversation_turns(self) -> list[ConversationTurn]:
        handle = self._lock_handle()
        try:
            return self._load_conversation_turns_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_conversation_turns(self, turns: list[ConversationTurn]) -> None:
        handle = self._lock_handle()
        try:
            self._write_conversation_turns_unlocked(turns)
        finally:
            self._unlock_handle(handle)

    def _normalize_transition(self, item: dict) -> LatentTransition:
        return LatentTransition(
            id=str(item.get("id", "")).strip(),
            from_signature=str(item.get("from_signature", "")).strip(),
            to_signature=str(item.get("to_signature", "")).strip(),
            from_scope=str(item.get("from_scope", "")).strip(),
            to_scope=str(item.get("to_scope", "")).strip(),
            from_tags=[
                str(entry).strip()
                for entry in item.get("from_tags", [])
                if str(entry).strip()
            ][:4],
            to_tags=[
                str(entry).strip()
                for entry in item.get("to_tags", [])
                if str(entry).strip()
            ][:4],
            from_keywords=[
                str(entry).strip()
                for entry in item.get("from_keywords", [])
                if str(entry).strip()
            ][:6],
            to_keywords=[
                str(entry).strip()
                for entry in item.get("to_keywords", [])
                if str(entry).strip()
            ][:6],
            evidence_count=float(item.get("evidence_count", 0.0) or 0.0),
            success_weight=float(item.get("success_weight", 0.0) or 0.0),
            failure_weight=float(item.get("failure_weight", 0.0) or 0.0),
            confidence_score=float(item.get("confidence_score", 0.0) or 0.0),
            prediction_hit_count=float(item.get("prediction_hit_count", 0.0) or 0.0),
            prediction_miss_count=float(item.get("prediction_miss_count", 0.0) or 0.0),
            forecast_score=float(
                item.get(
                    "forecast_score",
                    derive_transition_forecast_score(
                        prediction_hit_count=float(item.get("prediction_hit_count", 0.0) or 0.0),
                        prediction_miss_count=float(item.get("prediction_miss_count", 0.0) or 0.0),
                    ),
                )
                or 0.0
            ),
            last_seen_at=str(item.get("last_seen_at", "")).strip(),
        )

    def _normalize_rule(self, item: dict) -> PreferenceRule:
        tags = [str(entry).strip() for entry in item.get("tags", []) if str(entry).strip()]
        applies_to_scope = str(item.get("applies_to_scope", "")).strip()
        applies_when_tags = [
            str(entry).strip()
            for entry in item.get("applies_when_tags", [])
            if str(entry).strip()
        ]
        source_turn_ids = [
            str(entry).strip()
            for entry in item.get("source_turn_ids", [])
            if str(entry).strip()
        ]
        contexts = []
        for raw_context in item.get("contexts", []):
            if not isinstance(raw_context, dict):
                continue
            raw_min = raw_context.get("reaction_min")
            raw_max = raw_context.get("reaction_max")
            evidence_count = int(raw_context.get("evidence_count", 0) or 0)
            strong_signal_count = int(raw_context.get("strong_signal_count", 0) or 0)
            success_count = int(raw_context.get("success_count", 0) or 0)
            failure_count = int(raw_context.get("failure_count", 0) or 0)
            utility_score = float(raw_context.get("utility_score", 0.0) or 0.0)
            confidence_score = float(raw_context.get("confidence_score", 0.0) or 0.0)
            contexts.append(
                RuleContext(
                    kind=str(raw_context.get("kind", "")).strip(),
                    value=str(raw_context.get("value", "")).strip(),
                    evidence_count=evidence_count,
                    reaction_min=float(raw_min) if isinstance(raw_min, (int, float)) else None,
                    reaction_max=float(raw_max) if isinstance(raw_max, (int, float)) else None,
                    last_seen_at=str(raw_context.get("last_seen_at", "")),
                    utility_score=utility_score if utility_score > 0 else max(0.5, float(evidence_count or 0)),
                    confidence_score=(
                        confidence_score
                        if confidence_score > 0
                        else derive_context_confidence_score(
                            evidence_count=evidence_count,
                            strong_signal_count=strong_signal_count,
                            success_count=success_count,
                            failure_count=failure_count,
                        )
                    ),
                    strong_signal_count=strong_signal_count,
                    success_count=success_count,
                    failure_count=failure_count,
                )
            )
        latent_contexts = []
        for raw_context in item.get("latent_contexts", []):
            if not isinstance(raw_context, dict):
                continue
            latent_contexts.append(
                LatentContext(
                    id=str(raw_context.get("id", "")).strip(),
                    scope=str(raw_context.get("scope", "")).strip(),
                    tags=[
                        str(entry).strip()
                        for entry in raw_context.get("tags", [])
                        if str(entry).strip()
                    ],
                    keywords=[
                        str(entry).strip()
                        for entry in raw_context.get("keywords", [])
                        if str(entry).strip()
                    ],
                    prototype_text=str(raw_context.get("prototype_text", "")).strip(),
                    evidence_count=float(raw_context.get("evidence_count", 0.0) or 0.0),
                    support_score=float(raw_context.get("support_score", 0.0) or 0.0),
                    expected_gain=float(raw_context.get("expected_gain", 0.0) or 0.0),
                    confidence_score=float(raw_context.get("confidence_score", 0.0) or 0.0),
                    prior_weight=float(raw_context.get("prior_weight", 0.0) or 0.0),
                    posterior_mass=float(raw_context.get("posterior_mass", 0.0) or 0.0),
                    strong_signal_count=float(raw_context.get("strong_signal_count", 0.0) or 0.0),
                    success_mass=float(raw_context.get("success_mass", 0.0) or 0.0),
                    failure_mass=float(raw_context.get("failure_mass", 0.0) or 0.0),
                    last_seen_at=str(raw_context.get("last_seen_at", "")),
                )
            )
        if not contexts:
            if applies_to_scope:
                contexts.append(
                    RuleContext(
                        kind="scope",
                        value=applies_to_scope,
                        evidence_count=max(1, int(item.get("evidence_count", 0) or 0)),
                        last_seen_at=str(item.get("last_recorded_at", "")),
                        utility_score=max(1.0, float(item.get("support_score", 0.0) or item.get("evidence_count", 0) or 1.0)),
                    )
                )
            for tag in applies_when_tags or tags:
                contexts.append(
                    RuleContext(
                        kind="tag",
                        value=tag,
                        evidence_count=1,
                        last_seen_at=str(item.get("last_recorded_at", "")),
                        utility_score=max(0.6, float(item.get("support_score", 0.0) or 1.0) * 0.6),
                    )
                )
        contexts = merge_rule_contexts(contexts)
        distinct_scope_count = int(item.get("distinct_scope_count", 0) or 0)
        if distinct_scope_count <= 0:
            distinct_scope_count = len([context for context in contexts if context.kind == "scope"])
        distinct_tag_count = int(item.get("distinct_tag_count", 0) or 0)
        if distinct_tag_count <= 0:
            distinct_tag_count = len([context for context in contexts if context.kind == "tag"])
        support_score = float(item.get("support_score", 0.0) or 0.0)
        if support_score <= 0:
            support_score = float(item.get("evidence_count", 0) or 0)
        success_count = int(item.get("success_count", 0) or 0)
        failure_count = int(item.get("failure_count", 0) or 0)
        context_mode = str(item.get("context_mode", "")).strip() or derive_context_mode(
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
            evidence_count=int(item.get("evidence_count", 0) or 0),
        )
        confidence_score = float(item.get("confidence_score", 0.0) or 0.0)
        if confidence_score <= 0:
            confidence_score = derive_rule_confidence_score(
                evidence_count=int(item.get("evidence_count", 0) or 0),
                distinct_scope_count=distinct_scope_count,
                distinct_tag_count=distinct_tag_count,
                strong_signal_count=int(item.get("strong_signal_count", 0) or 0),
                success_count=success_count,
                failure_count=failure_count,
            )
        expected_gain = float(item.get("expected_gain", 0.0) or 0.0)
        if expected_gain <= 0:
            expected_gain = derive_rule_expected_gain(
                support_score=support_score,
                context_utility=sum(context.utility_score for context in contexts[:3]) / max(1, min(3, len(contexts))),
                confidence_score=confidence_score,
                strong_signal_count=int(item.get("strong_signal_count", 0) or 0),
                success_count=success_count,
                failure_count=failure_count,
                context_mode=context_mode,
            )
        if latent_contexts:
            latent_contexts = merge_latent_contexts(latent_contexts)
        rule = PreferenceRule(
            id=str(item.get("id", "")),
            statement=str(item.get("statement", "")),
            normalized_statement=str(item.get("normalized_statement", "")),
            instruction=str(item.get("instruction", item.get("statement", ""))),
            status=str(item.get("status", "candidate")),
            evidence_count=int(item.get("evidence_count", 0) or 0),
            first_recorded_at=str(item.get("first_recorded_at", "")),
            last_recorded_at=str(item.get("last_recorded_at", "")),
            applies_to_scope=applies_to_scope,
            applies_when_tags=applies_when_tags,
            negative_conditions=[
                str(entry).strip()
                for entry in item.get("negative_conditions", [])
                if str(entry).strip()
            ],
            priority=int(item.get("priority", 1) or 1),
            version=int(item.get("version", 1) or 1),
            expires_at=str(item.get("expires_at", "")),
            tags=tags,
            source_turn_ids=source_turn_ids,
            contexts=flatten_latent_contexts(latent_contexts) if latent_contexts else contexts,
            latent_contexts=latent_contexts,
            context_mode=context_mode,
            support_score=support_score,
            expected_gain=expected_gain,
            confidence_score=confidence_score,
            strong_signal_count=int(item.get("strong_signal_count", 0) or 0),
            success_count=success_count,
            failure_count=failure_count,
            distinct_scope_count=distinct_scope_count,
            distinct_tag_count=distinct_tag_count,
        )
        if not rule.latent_contexts:
            rule.latent_contexts = infer_latent_contexts_from_rule(rule)
            if rule.latent_contexts:
                rule.contexts = flatten_latent_contexts(rule.latent_contexts)
                rule.distinct_scope_count = len([context for context in rule.contexts if context.kind == "scope"])
                rule.distinct_tag_count = len([context for context in rule.contexts if context.kind == "tag"])
        return rule

    def _get_active_profile(self) -> str:
        """Read active profile from profiles.json."""
        profiles_path = self.base_dir / "profiles.json"
        if profiles_path.exists():
            try:
                data = json.loads(profiles_path.read_text(encoding="utf-8"))
                return data.get("active", "personal")
            except Exception:
                pass
        return "personal"

    def _load_public_rules_unlocked(self) -> list[PreferenceRule]:
        public_file = self.base_dir / "profiles" / "public_rules.json"
        return [self._normalize_rule(item) for item in self._read_json_list(public_file)]

    def load_preference_rules(self) -> list[PreferenceRule]:
        handle = self._lock_handle()
        try:
            profile = self._get_active_profile()
            personal = self._load_preference_rules_unlocked()
            if profile == "public":
                return self._load_public_rules_unlocked()
            elif profile == "hybrid":
                return personal + self._load_public_rules_unlocked()
            return personal  # default: personal
        finally:
            self._unlock_handle(handle)

    def load_meanings(self) -> list[Meaning]:
        handle = self._lock_handle()
        try:
            return self._load_meanings_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_meanings(self, meanings: list[Meaning]) -> None:
        handle = self._lock_handle()
        try:
            self._write_meanings_unlocked(meanings)
        finally:
            self._unlock_handle(handle)

    def load_principles(self) -> list[Principle]:
        handle = self._lock_handle()
        try:
            return self._load_principles_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_principles(self, principles: list[Principle]) -> None:
        handle = self._lock_handle()
        try:
            self._write_principles_unlocked(principles)
        finally:
            self._unlock_handle(handle)

    def load_policies(self) -> list[Policy]:
        handle = self._lock_handle()
        try:
            return self._load_policies_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_policies(self, policies: list[Policy]) -> None:
        handle = self._lock_handle()
        try:
            self._write_policies_unlocked(policies)
        finally:
            self._unlock_handle(handle)

    def load_personality(self) -> dict:
        handle = self._lock_handle()
        try:
            data = self._read_json_list(self.personality_file)
            return data[0] if data else {}
        finally:
            self._unlock_handle(handle)

    def write_personality(self, profile_dict: dict) -> None:
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.personality_file, [profile_dict])
        finally:
            self._unlock_handle(handle)

    def load_narrative(self) -> dict:
        handle = self._lock_handle()
        try:
            data = self._read_json_list(self.narrative_file)
            return data[0] if data else {}
        finally:
            self._unlock_handle(handle)

    def write_narrative(self, state_dict: dict) -> None:
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.narrative_file, [state_dict])
        finally:
            self._unlock_handle(handle)

    def load_deferred_meanings(self) -> list[Meaning]:
        handle = self._lock_handle()
        try:
            return [self._normalize_meaning(item) for item in self._read_json_list(self.deferred_meanings_file)]
        finally:
            self._unlock_handle(handle)

    def write_deferred_meanings(self, meanings: list[Meaning]) -> None:
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.deferred_meanings_file, [asdict(m) for m in meanings])
        finally:
            self._unlock_handle(handle)

    def load_context_transitions(self) -> list[LatentTransition]:
        handle = self._lock_handle()
        try:
            return self._load_context_transitions_unlocked()
        finally:
            self._unlock_handle(handle)

    def write_preference_rules(self, rules: list[PreferenceRule]) -> None:
        handle = self._lock_handle()
        try:
            self._write_preference_rules_unlocked(rules)
        finally:
            self._unlock_handle(handle)

    # ------------------------------------------------------------------
    # Ghost Laws persistence
    # ------------------------------------------------------------------

    def load_ghost_universal_laws(self) -> list[dict]:
        """Load ghost universal (禁止) laws as raw dicts, with lock."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.ghost_universal_laws_file)
        finally:
            self._unlock_handle(handle)

    def write_ghost_universal_laws(self, laws: list[dict]) -> None:
        """Atomically write ghost universal laws, with lock."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.ghost_universal_laws_file, laws)
        finally:
            self._unlock_handle(handle)

    def load_ghost_abstracted_principles(self) -> list[dict]:
        """Load ghost abstracted principles as raw dicts, with lock."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.ghost_abstracted_principles_file)
        finally:
            self._unlock_handle(handle)

    def save_ghost_abstracted_principles(self, principles: list[dict]) -> None:
        """Atomically write ghost abstracted principles, with lock."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.ghost_abstracted_principles_file, principles)
        finally:
            self._unlock_handle(handle)

    def load_ghost_positive_laws(self) -> list[dict]:
        """Load ghost positive (推奨) laws as raw dicts, with lock."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.ghost_positive_laws_file)
        finally:
            self._unlock_handle(handle)

    def write_ghost_positive_laws(self, laws: list[dict]) -> None:
        """Atomically write ghost positive laws, with lock."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.ghost_positive_laws_file, laws)
        finally:
            self._unlock_handle(handle)

    # ------------------------------------------------------------------
    # Raw dict locked read/write (for service.py interop)
    # ------------------------------------------------------------------

    def load_preference_rules_raw(self) -> list[dict]:
        """Load preference rules as raw dicts (no dataclass conversion), with lock."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.preference_rules_file)
        finally:
            self._unlock_handle(handle)

    def write_preference_rules_raw(self, rules: list[dict]) -> None:
        """Atomically write preference rules from raw dicts, with lock."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.preference_rules_file, rules)
        finally:
            self._unlock_handle(handle)

    # ------------------------------------------------------------------
    # Ghost Engine persistence
    # ------------------------------------------------------------------

    def load_ghosts(self) -> list[dict]:
        """Load all stored Ghost records as raw dicts."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.ghosts_file)
        finally:
            self._unlock_handle(handle)

    def write_ghosts(self, ghosts: list[dict]) -> None:
        """Atomically write Ghost records."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.ghosts_file, ghosts)
        finally:
            self._unlock_handle(handle)

    def load_ghost_trajectories(self) -> list[dict]:
        """Load all stored GhostTrajectory records as raw dicts."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.ghost_trajectories_file)
        finally:
            self._unlock_handle(handle)

    def write_ghost_trajectories(self, trajectories: list[dict]) -> None:
        """Atomically write GhostTrajectory records."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.ghost_trajectories_file, trajectories)
        finally:
            self._unlock_handle(handle)

    def save_ghost_with_trajectory(
        self,
        ghost_dict: dict,
        trajectory_dict: dict,
    ) -> None:
        """Atomically save a ghost and its updated trajectory together."""
        handle = self._lock_handle()
        try:
            # Ghosts
            ghosts = self._read_json_list(self.ghosts_file)
            existing_ids = {g.get("id") for g in ghosts}
            if ghost_dict.get("id") not in existing_ids:
                ghosts.insert(0, ghost_dict)
            else:
                ghosts = [ghost_dict if g.get("id") == ghost_dict.get("id") else g for g in ghosts]
            self._atomic_write_json(self.ghosts_file, ghosts)

            # Trajectories
            trajectories = self._read_json_list(self.ghost_trajectories_file)
            traj_ids = {t.get("id") for t in trajectories}
            if trajectory_dict.get("id") not in traj_ids:
                trajectories.insert(0, trajectory_dict)
            else:
                trajectories = [
                    trajectory_dict if t.get("id") == trajectory_dict.get("id") else t
                    for t in trajectories
                ]
            self._atomic_write_json(self.ghost_trajectories_file, trajectories)
        finally:
            self._unlock_handle(handle)

    # ── Curiosity Layer persistence ────────────────────────────────────────

    def load_curiosity_signals(self) -> list[dict]:
        """Load all stored CuriositySignal records as raw dicts."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.curiosity_signals_file)
        finally:
            self._unlock_handle(handle)

    def write_curiosity_signals(self, signals: list[dict]) -> None:
        """Atomically write CuriositySignal records."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.curiosity_signals_file, signals)
        finally:
            self._unlock_handle(handle)

    def load_knowledge_gap_clusters(self) -> list[dict]:
        """Load all stored KnowledgeGapCluster records as raw dicts."""
        handle = self._lock_handle()
        try:
            return self._read_json_list(self.knowledge_gap_clusters_file)
        finally:
            self._unlock_handle(handle)

    def write_knowledge_gap_clusters(self, clusters: list[dict]) -> None:
        """Atomically write KnowledgeGapCluster records."""
        handle = self._lock_handle()
        try:
            self._atomic_write_json(self.knowledge_gap_clusters_file, clusters)
        finally:
            self._unlock_handle(handle)

    def save_signal_with_cluster(
        self,
        signal_dict: dict,
        cluster_dict: dict,
    ) -> None:
        """Atomically save a curiosity signal and its updated cluster together."""
        handle = self._lock_handle()
        try:
            # Signals
            signals = self._read_json_list(self.curiosity_signals_file)
            existing_ids = {s.get("id") for s in signals}
            if signal_dict.get("id") not in existing_ids:
                signals.insert(0, signal_dict)
            else:
                signals = [signal_dict if s.get("id") == signal_dict.get("id") else s for s in signals]
            self._atomic_write_json(self.curiosity_signals_file, signals)

            # Clusters
            clusters = self._read_json_list(self.knowledge_gap_clusters_file)
            cluster_ids = {c.get("id") for c in clusters}
            if cluster_dict.get("id") not in cluster_ids:
                clusters.insert(0, cluster_dict)
            else:
                clusters = [
                    cluster_dict if c.get("id") == cluster_dict.get("id") else c
                    for c in clusters
                ]
            self._atomic_write_json(self.knowledge_gap_clusters_file, clusters)
        finally:
            self._unlock_handle(handle)

    def _normalize_context_node_payload(self, item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        scope = str(item.get("scope", "")).strip()
        tags = [
            normalize_text(str(tag).strip())
            for tag in item.get("tags", [])
            if normalize_text(str(tag).strip())
        ][:4]
        keywords = [
            normalize_text(str(keyword).strip())
            for keyword in item.get("keywords", [])
            if normalize_text(str(keyword).strip())
        ][:6]
        signature = str(item.get("signature", "")).strip() or build_context_signature(scope, tags, keywords)
        if not signature:
            return None
        return {
            "context_id": str(item.get("context_id", "")).strip(),
            "scope": scope,
            "tags": tags,
            "keywords": keywords,
            "signature": signature,
            "posterior": float(item.get("posterior", 0.0) or 0.0),
        }

    def _infer_context_nodes_from_turn(self, turn: ConversationTurn) -> list[dict[str, object]]:
        scope = turn.task_scope.strip()
        tags = [
            normalize_text(str(tag).strip())
            for tag in turn.tags
            if normalize_text(str(tag).strip())
        ][:4]
        keywords = sorted(
            extract_keywords(
                turn.task_scope,
                turn.user_message[:400],
                turn.user_feedback[:400],
                " ".join(turn.extracted_corrections[:4]),
            )
        )[:6]
        if not scope and not tags and not keywords:
            return []

        posterior = 0.42
        if turn.extracted_corrections:
            posterior += 0.12
        if turn.guidance_applied:
            posterior += 0.08
        if len(tags) >= 2:
            posterior += 0.06
        if len(keywords) >= 3:
            posterior += 0.06
        if turn.reaction_score is not None and (turn.reaction_score >= 0.7 or turn.reaction_score <= 0.3):
            posterior += 0.08

        return [
            {
                "context_id": f"turn-{turn.id}",
                "scope": scope,
                "tags": tags,
                "keywords": keywords,
                "signature": build_context_signature(scope, tags, keywords),
                "posterior": round(min(0.92, posterior), 4),
                "source": "turn_fallback",
            }
        ]

    def _record_context_transitions_unlocked(
        self,
        transitions: list[LatentTransition],
        turn: ConversationTurn,
        fallback_previous_nodes: list[dict[str, object]] | None = None,
    ) -> tuple[list[LatentTransition], list[dict[str, object]]]:
        metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
        inference_trace = metadata.get("inference_trace") if isinstance(metadata.get("inference_trace"), dict) else {}
        transition_trace = (
            metadata.get("transition_trace")
            if isinstance(metadata.get("transition_trace"), dict)
            else inference_trace.get("transition_trace", {})
            if isinstance(inference_trace, dict)
            else {}
        )
        raw_previous = metadata.get("previous_context_nodes", [])
        raw_current = metadata.get("active_context_nodes", [])
        if not raw_previous and isinstance(inference_trace, dict):
            raw_previous = inference_trace.get("previous_context_nodes", [])
        if not raw_current and isinstance(inference_trace, dict):
            raw_current = inference_trace.get("active_context_nodes", [])
        if not raw_current:
            raw_current = self._infer_context_nodes_from_turn(turn)
        if not raw_previous and fallback_previous_nodes:
            raw_previous = fallback_previous_nodes

        previous_nodes = {}
        for item in raw_previous if isinstance(raw_previous, list) else []:
            normalized = self._normalize_context_node_payload(item)
            if normalized is None:
                continue
            previous_nodes[str(normalized["signature"])] = normalized

        current_nodes = {}
        for item in raw_current if isinstance(raw_current, list) else []:
            normalized = self._normalize_context_node_payload(item)
            if normalized is None:
                continue
            current_nodes[str(normalized["signature"])] = normalized

        current_node_list = list(current_nodes.values())
        if not previous_nodes or not current_nodes:
            return transitions, current_node_list

        by_key = {
            (transition.from_signature, transition.to_signature): transition
            for transition in transitions
        }
        for previous in previous_nodes.values():
            previous_signature = str(previous["signature"])
            for current in current_nodes.values():
                current_signature = str(current["signature"])
                key = (previous_signature, current_signature)
                transition = by_key.get(key)
                if transition is None:
                    transition = LatentTransition(
                        id=f"transition-{len(by_key) + 1}",
                        from_signature=previous_signature,
                        to_signature=current_signature,
                        from_scope=str(previous["scope"]),
                        to_scope=str(current["scope"]),
                        from_tags=list(previous["tags"])[:4],
                        to_tags=list(current["tags"])[:4],
                        from_keywords=list(previous["keywords"])[:6],
                        to_keywords=list(current["keywords"])[:6],
                    )
                    transitions.append(transition)
                    by_key[key] = transition

                weight = max(0.18, float(previous["posterior"] or 0.0)) * max(0.18, float(current["posterior"] or 0.0))
                transition.evidence_count = round(transition.evidence_count + weight, 4)
                if turn.reaction_score is not None and turn.reaction_score >= 0.7:
                    transition.success_weight = round(transition.success_weight + weight, 4)
                elif turn.reaction_score is not None and turn.reaction_score <= 0.3:
                    transition.failure_weight = round(transition.failure_weight + weight, 4)
                elif turn.reaction_score is not None:
                    transition.success_weight = round(transition.success_weight + weight * 0.2, 4)
                transition.confidence_score = derive_transition_confidence_score(
                    evidence_count=transition.evidence_count,
                    success_weight=transition.success_weight,
                    failure_weight=transition.failure_weight,
                )
                transition.forecast_score = derive_transition_forecast_score(
                    prediction_hit_count=transition.prediction_hit_count,
                    prediction_miss_count=transition.prediction_miss_count,
                )
                transition.last_seen_at = turn.recorded_at

        predicted_items = (
            transition_trace.get("predicted_next_contexts", [])
            if isinstance(transition_trace, dict)
            else []
        )
        predicted_matches = {
            str(entry).strip()
            for entry in (
                transition_trace.get("matched_prediction_signatures", [])
                if isinstance(transition_trace, dict)
                else []
            )
            if str(entry).strip()
        }
        matched_any_prediction = bool(predicted_matches)
        for raw_prediction in predicted_items if isinstance(predicted_items, list) else []:
            if not isinstance(raw_prediction, dict):
                continue
            to_signature = str(raw_prediction.get("to_signature", "")).strip()
            if not to_signature:
                continue
            predicted_hit = to_signature in current_nodes
            predicted_score = max(0.0, float(raw_prediction.get("score", 0.0) or 0.0))
            supporting_flows = raw_prediction.get("supporting_flows", [])
            if not isinstance(supporting_flows, list):
                continue
            for support in supporting_flows:
                if not isinstance(support, dict):
                    continue
                from_signature = str(support.get("from_signature", "")).strip()
                if not from_signature:
                    continue
                transition = by_key.get((from_signature, to_signature))
                if transition is None:
                    continue
                support_weight = max(0.0, float(support.get("weight", 0.0) or 0.0))
                calibration_weight = min(1.4, support_weight * 0.12 + predicted_score * 0.04)
                if calibration_weight <= 0:
                    continue
                if predicted_hit:
                    transition.prediction_hit_count = round(
                        transition.prediction_hit_count + calibration_weight,
                        4,
                    )
                else:
                    miss_multiplier = 0.18 if matched_any_prediction else 0.42
                    transition.prediction_miss_count = round(
                        transition.prediction_miss_count + calibration_weight * miss_multiplier,
                        4,
                    )
                transition.forecast_score = derive_transition_forecast_score(
                    prediction_hit_count=transition.prediction_hit_count,
                    prediction_miss_count=transition.prediction_miss_count,
                )
                transition.last_seen_at = turn.recorded_at

        transitions.sort(
            key=lambda item: (
                item.confidence_score,
                item.forecast_score,
                item.success_weight - item.failure_weight,
                item.evidence_count,
                item.last_seen_at,
            ),
            reverse=True,
        )
        return transitions[:400], current_node_list

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
    ) -> EpisodeRecord:
        now = datetime.now()
        entry = EpisodeRecord(
            id=now.strftime("%Y%m%d%H%M%S%f"),
            timestamp=now.strftime("%Y/%m/%d %H:%M"),
            title=title.strip() or "No Title",
            issuer=issuer.strip(),
            task_type=task_type.strip() or "generic",
            profile_name=profile_name.strip(),
            source_text=source_text.strip(),
            company_profile=company_profile if isinstance(company_profile, dict) else {},
            output=output if isinstance(output, dict) else {},
            metadata=metadata if isinstance(metadata, dict) else {},
        )

        handle = self._lock_handle()
        try:
            entries = self._load_entries_unlocked()
            entries.insert(0, entry)
            entries = evict_episodes(entries, retention_limit=50)
            self._write_entries_unlocked(entries)
        finally:
            self._unlock_handle(handle)
        return entry

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
        metadata: dict | None = None,
        reaction_score_override: float | None = None,
    ) -> ConversationTurn:
        now = datetime.now()
        metadata_payload = metadata if isinstance(metadata, dict) else {}
        authoritative_tags = bool(metadata_payload.get("authoritative_tags", False))
        auto_saved_by = str(metadata_payload.get("auto_saved_by", "")).strip()
        should_skip_inference = auto_saved_by == "stop_reminder"
        inferred_corrections = [
            str(entry).strip()
            for entry in (
                (extracted_corrections or [])
                if extracted_corrections is not None or should_skip_inference
                else extract_correction_candidates(user_feedback, user_message)
            )
            if str(entry).strip()
        ]
        if should_skip_inference:
            inferred_tags = sorted({str(entry).strip() for entry in (tags or []) if str(entry).strip()})
        else:
            inferred_tags = sorted(
                extract_keywords(
                    task_scope,
                    user_message[:600],
                    user_feedback[:600],
                    " ".join(inferred_corrections[:5]),
                )
            )
        if tags:
            provided_tags = [str(entry).strip() for entry in tags if str(entry).strip()]
            if authoritative_tags:
                inferred_tags = sorted(dict.fromkeys(provided_tags))
            else:
                inferred_tags = sorted({*inferred_tags, *provided_tags})

        # Score from reaction: client override > LLM scorer > rule-based
        if reaction_score_override is not None and 0.0 <= reaction_score_override <= 1.0:
            reaction_score = reaction_score_override
            # Feed override into dictionary cache so it keeps learning
            if self._scorer is not None:
                self._scorer.teach(user_feedback, inferred_corrections, reaction_score)
        elif self._scorer is not None:
            reaction_score = self._scorer.score(user_feedback, inferred_corrections)
        else:
            reaction_score = score_reaction(user_feedback, inferred_corrections)

        turn = ConversationTurn(
            id=now.strftime("%Y%m%d%H%M%S%f"),
            recorded_at=now.strftime("%Y/%m/%d %H:%M"),
            task_scope=task_scope.strip(),
            user_message=user_message.strip(),
            assistant_message=assistant_message.strip(),
            user_feedback=user_feedback.strip(),
            extracted_corrections=inferred_corrections[:8],
            tags=inferred_tags[:20],
            reaction_score=reaction_score,
            guidance_applied=guidance_applied,
            metadata=metadata_payload,
        )

        handle = self._lock_handle()
        try:
            turns = self._load_conversation_turns_unlocked()
            turns.insert(0, turn)

            # Identify turns to evict before removing them
            from .memory_manager import select_turns_for_eviction
            evict_ids = set(select_turns_for_eviction(turns, retention_limit=200))
            if evict_ids:
                evict_targets = [t for t in turns if t.id in evict_ids]
                archived_episode = archive_turns_to_episode(evict_targets)
                if archived_episode is not None:
                    entries = self._load_entries_unlocked()
                    entries.insert(0, archived_episode)
                    entries = evict_episodes(entries, retention_limit=50)
                    self._write_entries_unlocked(entries)
                turns = [t for t in turns if t.id not in evict_ids]

            rules = build_preference_rules(turns)
            merge_result = merge_similar_rules(rules)
            # Apply reconsolidation if the new turn had guidance applied
            final_rules = merge_result.merged_rules
            if guidance_applied:
                final_rules = reconsolidate_rules_from_turns(final_rules, [turn])
            transitions = self._load_context_transitions_unlocked()
            transitions, _ = self._record_context_transitions_unlocked(transitions, turn)
            self._write_conversation_turns_unlocked(turns)
            self._write_preference_rules_unlocked(final_rules)
            self._write_context_transitions_unlocked(transitions)
        finally:
            self._unlock_handle(handle)
        return turn

    def rebuild_preference_rules(self) -> list[PreferenceRule]:
        handle = self._lock_handle()
        try:
            turns = self._load_conversation_turns_unlocked()
            rules = build_preference_rules(turns)
            merge_result = merge_similar_rules(rules)
            final_rules = reconsolidate_rules_from_turns(
                merge_result.merged_rules,
                list(reversed(turns)),
            )
            # Load personality metabolism for adaptive thresholds (read directly, already locked)
            personality_data = self._read_json_list(self.personality_file)
            personality_dict = personality_data[0] if personality_data else {}
            metabolism = personality_dict.get("metabolism_rate", 0.5)
            # P0: Auto-resolve conflicting rules (metabolism modulates safety guard)
            final_rules, _conflict_log = resolve_contradicting_rules(final_rules, metabolism_rate=metabolism)
            # P0: Auto-correct needs_revision rules (metabolism modulates wait time)
            from datetime import datetime as _dt
            now_str = _dt.now().strftime("%Y/%m/%d %H:%M")
            final_rules, _correction_log = auto_correct_flagged_rules(final_rules, now_str, metabolism_rate=metabolism)
            self._write_preference_rules_unlocked(final_rules)
            return final_rules
        finally:
            self._unlock_handle(handle)

    def rebuild_context_transitions(self) -> list[LatentTransition]:
        handle = self._lock_handle()
        try:
            turns = list(reversed(self._load_conversation_turns_unlocked()))
            transitions: list[LatentTransition] = []
            previous_nodes: list[dict[str, object]] = []
            for turn in turns:
                transitions, current_nodes = self._record_context_transitions_unlocked(
                    transitions,
                    turn,
                    fallback_previous_nodes=previous_nodes,
                )
                if current_nodes:
                    previous_nodes = current_nodes
            self._write_context_transitions_unlocked(transitions)
            return transitions
        finally:
            self._unlock_handle(handle)

    def synthesize_rules(self) -> list[dict]:
        """Generate rule hypotheses from success/failure pattern differences.
        This is the system's autonomous reasoning — deriving new rules
        without human input, based on statistical patterns in past interactions."""
        handle = self._lock_handle()
        try:
            turns = self._load_conversation_turns_unlocked()
        finally:
            self._unlock_handle(handle)
        return synthesize_rules_from_turns(turns)

    def self_overcome(self) -> list[dict]:
        """Übermensch: self-overcome — criticize and rewrite own rules.
        Identifies weak, redundant, or overly narrow rules and proposes
        stronger replacements. The system transcends itself."""
        handle = self._lock_handle()
        try:
            rules = self._load_preference_rules_unlocked()
        finally:
            self._unlock_handle(handle)
        return compute_self_overcome_proposals(rules)

    def find_entry(self, entry_id: str) -> EpisodeRecord | None:
        handle = self._lock_handle()
        try:
            for entry in self._load_entries_unlocked():
                if entry.id == entry_id:
                    return entry
            return None
        finally:
            self._unlock_handle(handle)

    def add_correction(
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
        normalized = {
            "decision_override": decision_override.strip(),
            "correction_note": correction_note.strip(),
            "reuse_note": reuse_note.strip(),
            "reason": reason.strip(),
            "scope": scope.strip(),
            "bad_output": self._serialize_text_payload(bad_output),
            "revised_output": self._serialize_text_payload(revised_output),
            "tool_used": tool_used.strip(),
            "source_user": source_user.strip(),
        }
        if not any(normalized.values()):
            return False

        handle = self._lock_handle()
        try:
            entries = self._load_entries_unlocked()
            for entry in entries:
                if entry.id != entry_id:
                    continue
                timestamp = datetime.now().strftime("%Y/%m/%d %H:%M")
                entry.corrections.insert(
                    0,
                    CorrectionRecord(
                        recorded_at=timestamp,
                        decision_override=normalized["decision_override"],
                        correction_note=normalized["correction_note"],
                        reuse_note=normalized["reuse_note"],
                        reason=normalized["reason"],
                        scope=normalized["scope"],
                        bad_output=normalized["bad_output"],
                        revised_output=normalized["revised_output"],
                        tool_used=normalized["tool_used"],
                        source_user=normalized["source_user"],
                        accepted=bool(accepted),
                    ),
                )
                entry.corrections = entry.corrections[:20]
                entry.last_corrected_at = timestamp
                self._write_entries_unlocked(entries)
                return True
            return False
        finally:
            self._unlock_handle(handle)

    def attach_training_example(
        self,
        entry_id: str,
        *,
        format: str = "chat",
        system_message: object = "",
        user_message: object = "",
        prompt: object = "",
        draft_output: object = "",
        rejected_output: object = "",
        accepted_output: object = "",
        feedback: object = "",
        accepted: bool = True,
        model_id: str = "",
        policy_version: str = "",
        accepted_by: str = "",
        tags: list[str] | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
    ) -> bool:
        format_name = str(format or "chat").strip().lower() or "chat"
        if format_name not in {"chat", "completions"}:
            format_name = "chat"

        handle = self._lock_handle()
        try:
            entries = self._load_entries_unlocked()
            for entry in entries:
                if entry.id != entry_id:
                    continue

                timestamp = datetime.now().strftime("%Y/%m/%d %H:%M")
                accepted_payload = self._serialize_text_payload(accepted_output)
                if not accepted_payload and entry.output:
                    accepted_payload = self._serialize_text_payload(entry.output)
                entry.training_example = TrainingExample(
                    updated_at=timestamp,
                    format=format_name,
                    system_message=self._serialize_text_payload(system_message),
                    user_message=self._serialize_text_payload(user_message),
                    prompt=self._serialize_text_payload(prompt),
                    draft_output=self._serialize_text_payload(draft_output),
                    rejected_output=self._serialize_text_payload(rejected_output),
                    accepted_output=accepted_payload,
                    feedback=self._serialize_text_payload(feedback),
                    accepted=bool(accepted),
                    model_id=model_id.strip(),
                    policy_version=policy_version.strip(),
                    accepted_by=accepted_by.strip(),
                    tags=[str(entry).strip() for entry in (tags or []) if str(entry).strip()][:20],
                    temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
                self._write_entries_unlocked(entries)
                return True
            return False
        finally:
            self._unlock_handle(handle)

