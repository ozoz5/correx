from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .chat_adapter import ChatLoopAdapter
from .mlx_trainer import MlxLoraTrainingConfig
from .service import PseudoIntelligenceService

SERVER_NAME = "correx"
SERVER_INSTRUCTIONS = (
    "Use this server when you need persistent human correction memory, "
    "conversation-derived preference rules, training dataset export, or automatic "
    "LoRA training orchestration for the Claude Pseudo Intelligence Core."
)


def _require_fastmcp():
    """Import FastMCP with version checking and clear error messages."""
    try:
        from mcp.server.fastmcp import Context, FastMCP
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "MCP support requires the official Python SDK. Install it with "
            "`python3 -m pip install -e '.[mcp]'` or `pip install \"mcp[cli]>=1.9.4,<2\"`."
        ) from error
    # Version guard: detect MCP SDK version for forward-compat warnings
    try:
        import mcp
        mcp_version = getattr(mcp, "__version__", "unknown")
        major = int(mcp_version.split(".")[0]) if mcp_version != "unknown" else 1
        if major >= 2:
            import sys
            print(
                f"[correx] WARNING: MCP SDK v{mcp_version} detected. "
                f"This server was built for MCP <2. Some tools may not work correctly.",
                file=sys.stderr,
            )
    except (ValueError, AttributeError):
        pass
    return Context, FastMCP


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    return value


def _truncate_text(value: str, limit: int = 280) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _summarize_entry(entry: Any) -> dict[str, Any]:
    payload = _to_plain_data(entry)
    training_example = payload.get("training_example") or {}
    corrections = payload.get("corrections") or []
    return {
        "id": payload.get("id", ""),
        "timestamp": payload.get("timestamp", ""),
        "title": payload.get("title", ""),
        "issuer": payload.get("issuer", ""),
        "task_type": payload.get("task_type", ""),
        "correction_count": len(corrections),
        "has_training_example": bool(training_example),
        "accepted_training_example": bool(training_example.get("accepted", False)),
        "source_text_preview": _truncate_text(payload.get("source_text", "")),
    }


def _summarize_turn(turn: Any) -> dict[str, Any]:
    payload = _to_plain_data(turn)
    return {
        "id": payload.get("id", ""),
        "recorded_at": payload.get("recorded_at", ""),
        "task_scope": payload.get("task_scope", ""),
        "user_feedback_preview": _truncate_text(payload.get("user_feedback", "")),
        "extracted_corrections": payload.get("extracted_corrections", [])[:4],
        "tags": payload.get("tags", [])[:8],
    }


def _summarize_rule(rule: Any) -> dict[str, Any]:
    payload = _to_plain_data(rule)
    return {
        "id": payload.get("id", ""),
        "statement": payload.get("statement", ""),
        "instruction": payload.get("instruction", ""),
        "status": payload.get("status", ""),
        "evidence_count": payload.get("evidence_count", 0),
        "applies_to_scope": payload.get("applies_to_scope", ""),
        "applies_when_tags": payload.get("applies_when_tags", [])[:6],
        "negative_conditions": payload.get("negative_conditions", [])[:3],
        "priority": payload.get("priority", 1),
        "tags": payload.get("tags", [])[:8],
        "context_mode": payload.get("context_mode", "local"),
        "latent_context_count": len(payload.get("latent_contexts", []) or []),
        "support_score": payload.get("support_score", 0.0),
        "expected_gain": payload.get("expected_gain", 0.0),
        "confidence_score": payload.get("confidence_score", 0.0),
        "strong_signal_count": payload.get("strong_signal_count", 0),
    }


def _summarize_transition(transition: Any) -> dict[str, Any]:
    payload = _to_plain_data(transition)
    return {
        "id": payload.get("id", ""),
        "from_signature": payload.get("from_signature", ""),
        "to_signature": payload.get("to_signature", ""),
        "from_scope": payload.get("from_scope", ""),
        "to_scope": payload.get("to_scope", ""),
        "to_tags": payload.get("to_tags", [])[:4],
        "to_keywords": payload.get("to_keywords", [])[:6],
        "evidence_count": payload.get("evidence_count", 0.0),
        "success_weight": payload.get("success_weight", 0.0),
        "failure_weight": payload.get("failure_weight", 0.0),
        "confidence_score": payload.get("confidence_score", 0.0),
        "prediction_hit_count": payload.get("prediction_hit_count", 0.0),
        "prediction_miss_count": payload.get("prediction_miss_count", 0.0),
        "forecast_score": payload.get("forecast_score", 0.0),
        "last_seen_at": payload.get("last_seen_at", ""),
    }


def _memory_summary(service: PseudoIntelligenceService) -> dict[str, Any]:
    entries = service.list_entries()
    turns = service.list_conversation_turns()
    rules = service.list_preference_rules()
    transitions = service.list_context_transitions()
    stable_rules = [rule for rule in rules if getattr(rule, "status", "") == "promoted"]
    high_value_rules = [rule for rule in rules if getattr(rule, "expected_gain", 0.0) >= 0.8]
    general_rules = [rule for rule in rules if getattr(rule, "context_mode", "") == "general"]
    mixed_rules = [rule for rule in rules if getattr(rule, "context_mode", "") == "mixed"]
    local_rules = [rule for rule in rules if getattr(rule, "context_mode", "") == "local"]
    latent_context_count = sum(len(getattr(rule, "latent_contexts", []) or []) for rule in rules)
    meanings = service.list_meanings()
    trainable_entries = [
        entry
        for entry in entries
        if getattr(getattr(entry, "training_example", None), "accepted", False)
    ]
    return {
        "memory_dir": str(service.base_dir),
        "entry_count": len(entries),
        "conversation_turn_count": len(turns),
        "preference_rule_count": len(rules),
        "stable_rule_count": len(stable_rules),
        "promoted_rule_count": len([r for r in rules if getattr(r, "status", "") == "promoted"]),
        "high_value_rule_count": len(high_value_rules),
        "general_rule_count": len(general_rules),
        "mixed_rule_count": len(mixed_rules),
        "local_rule_count": len(local_rules),
        "latent_context_count": latent_context_count,
        "context_transition_count": len(transitions),
        "meaning_count": len(meanings),
        "cross_scope_meanings": len([m for m in meanings if m.cross_scope_count >= 2]),
        "accepted_training_example_count": len(trainable_entries),
        "personality_metabolism": service.history.load_personality().get("metabolism_label", "unknown"),
        "personality_digestibility": service.history.load_personality().get("digestibility_label", "unknown"),
        "latest_entries": [_summarize_entry(entry) for entry in entries[:5]],
        "latest_preference_rules": [_summarize_rule(rule) for rule in rules[:5]],
        "latest_context_transitions": [_summarize_transition(item) for item in transitions[:5]],
    }


def create_mcp_server(
    memory_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
) -> Any:
    Context, FastMCP = _require_fastmcp()
    globals()["Context"] = Context
    service = PseudoIntelligenceService(memory_dir)
    chat_adapter = ChatLoopAdapter(memory_dir)

    mcp = FastMCP(
        SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
        json_response=True,
        stateless_http=True,
    )
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.streamable_http_path = path

    @mcp.resource("memory://summary")
    def memory_summary() -> str:
        """Read a compact summary of stored episodes, corrections, and training state."""
        return json.dumps(_memory_summary(service), ensure_ascii=False, indent=2)

    @mcp.resource("memory://entries/{limit}")
    def recent_entries(limit: int = 10) -> str:
        """Read recent episode summaries for audit or debugging."""
        normalized_limit = max(1, min(int(limit), 50))
        entries = service.list_entries()[:normalized_limit]
        payload = {
            "items": [_summarize_entry(entry) for entry in entries],
            "count": len(entries),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @mcp.resource("memory://guidance/{task_scope}")
    def procedural_guidance(task_scope: str) -> str:
        """Read contextual guidance for a stable task scope."""
        guidance = service.build_conversation_guidance(task_scope=task_scope, raw_text="")
        payload = {
            "task_scope": task_scope,
            "guidance_context": guidance,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @mcp.tool()
    def build_guidance_context(
        task_title: str,
        raw_text: str,
        issuer: str = "",
        task_scope: str = "",
        company_profile: dict | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Use this when you want reusable human correction memory for a new task prompt."""
        guidance = service.build_guidance_context(
            company_profile=company_profile,
            task_title=task_title,
            issuer=issuer,
            raw_text=raw_text,
            limit=limit,
            task_scope=task_scope,
        )
        return {
            "guidance_context": guidance,
            "task_title": task_title,
            "issuer": issuer,
            "task_scope": task_scope,
        }

    @mcp.tool()
    def prepare_chat_session(
        task_title: str = "",
        raw_text: str = "",
        issuer: str = "",
        task_scope: str = "",
        company_profile: dict | None = None,
        system_message: str = "",
        user_message: str = "",
        prompt: str = "",
        metadata: dict | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Create a chat session, persist task context, and precompute reusable guidance."""
        return chat_adapter.prepare(
            session_id=session_id,
            task_scope=task_scope,
            task_title=task_title,
            issuer=issuer,
            raw_text=raw_text,
            company_profile=company_profile,
            system_message=system_message,
            user_message=user_message,
            prompt=prompt,
            metadata=metadata,
        )

    @mcp.tool()
    def save_chat_feedback(
        session_id: str,
        assistant_message: str,
        user_feedback: str,
        user_message: str = "",
        extracted_corrections: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Persist feedback for a prepared chat session using the stored guidance state."""
        return chat_adapter.save_feedback(
            session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            user_feedback=user_feedback,
            extracted_corrections=extracted_corrections,
            tags=tags,
            metadata=metadata,
        )

    @mcp.tool()
    def accept_chat_response(
        session_id: str,
        title: str = "",
        task_type: str = "generic",
        user_message: str = "",
        assistant_message: str = "",
        accepted_output: str = "",
        feedback: str = "",
        output: dict | None = None,
        create_training_example: bool = True,
        model_id: str = "",
        policy_version: str = "",
        accepted_by: str = "human",
        tags: list[str] | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
        close_session: bool = True,
    ) -> dict[str, Any]:
        """Persist the accepted response for a chat session and optionally attach training data."""
        return chat_adapter.accept_response(
            session_id,
            title=title,
            task_type=task_type,
            user_message=user_message,
            assistant_message=assistant_message,
            accepted_output=accepted_output,
            feedback=feedback,
            output=output,
            create_training_example=create_training_example,
            model_id=model_id,
            policy_version=policy_version,
            accepted_by=accepted_by,
            tags=tags,
            temperature=temperature,
            metadata=metadata,
            close_session=close_session,
        )

    @mcp.tool()
    def get_chat_session(session_id: str) -> dict[str, Any]:
        """Inspect the stored state of a prepared chat session."""
        return chat_adapter.session_summary(session_id)

    @mcp.tool()
    def list_entries(limit: int = 10, with_training_only: bool = False) -> dict[str, Any]:
        """Use this when you need recent episode history or want to inspect training coverage."""
        normalized_limit = max(1, min(int(limit), 50))
        entries = service.list_entries()
        if with_training_only:
            entries = [entry for entry in entries if getattr(entry, "training_example", None)]
        items = [_summarize_entry(entry) for entry in entries[:normalized_limit]]
        return {
            "items": items,
            "count": len(items),
            "with_training_only": with_training_only,
        }

    @mcp.tool()
    def list_conversation_turns(limit: int = 10) -> dict[str, Any]:
        """Use this when you need to inspect recent user corrections captured from conversation."""
        normalized_limit = max(1, min(int(limit), 50))
        turns = service.list_conversation_turns()[:normalized_limit]
        return {
            "items": [_summarize_turn(turn) for turn in turns],
            "count": len(turns),
        }

    @mcp.tool()
    def list_preference_rules(promoted_only: bool = False, limit: int = 10) -> dict[str, Any]:
        """Use this when you want the learned contextual rules that may shape future outputs."""
        normalized_limit = max(1, min(int(limit), 50))
        rules = service.list_preference_rules(promoted_only=promoted_only)[:normalized_limit]
        return {
            "items": [_summarize_rule(rule) for rule in rules],
            "count": len(rules),
            "promoted_only": promoted_only,
        }

    @mcp.tool()
    def list_context_transitions(limit: int = 10) -> dict[str, Any]:
        """Use this when you want to inspect learned flow between latent situations."""
        normalized_limit = max(1, min(int(limit), 50))
        transitions = service.list_context_transitions()[:normalized_limit]
        return {
            "items": [_summarize_transition(item) for item in transitions],
            "count": len(transitions),
        }

    @mcp.tool()
    def rebuild_context_transitions() -> dict[str, Any]:
        """Rebuild learned latent-context transitions from stored conversation turns."""
        transitions = service.rebuild_context_transitions()
        return {
            "ok": True,
            "count": len(transitions),
            "items": [_summarize_transition(item) for item in transitions[:10]],
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    def synthesize_meanings() -> dict[str, Any]:
        """Detect emergent value principles from cross-scope rule clusters.

        Analyzes all preference rules to find clusters of rules from different
        scopes that point to the same unstated principle. These meanings represent
        the user's implicit value system -- judgment principles that exist in no
        individual rule but emerge from their intersection.

        Run this after accumulating new rules, or during a sleep/reflection phase.
        """
        meanings = service.synthesize_meanings()
        new_ones = [m for m in meanings if m.first_seen_at == m.last_seen_at]
        return {
            "ok": True,
            "count": len(meanings),
            "new_count": len(new_ones),
            "items": [
                {
                    "id": m.id,
                    "principle": m.principle,
                    "summary": m.summary,
                    "strength": m.strength,
                    "cross_scope_count": m.cross_scope_count,
                    "scopes": m.scopes,
                    "confidence": m.confidence,
                    "source_rule_count": len(m.source_rule_ids),
                    "personal_settings_overlap": m.personal_settings_overlap,
                    "status": m.status,
                }
                for m in meanings[:15]
            ],
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    def list_meanings(limit: int = 10) -> dict[str, Any]:
        """View the user's emergent value principles synthesized from rule clusters."""
        meanings = service.list_meanings()[:max(1, min(limit, 50))]
        return {
            "items": [
                {
                    "id": m.id,
                    "principle": m.principle,
                    "summary": m.summary,
                    "strength": m.strength,
                    "cross_scope_count": m.cross_scope_count,
                    "scopes": m.scopes,
                    "confidence": m.confidence,
                    "source_rule_count": len(m.source_rule_ids),
                    "personal_settings_overlap": m.personal_settings_overlap,
                    "status": m.status,
                }
                for m in meanings
            ],
            "count": len(meanings),
        }

    @mcp.tool()
    def synthesize_principles() -> dict[str, Any]:
        """Extract higher-order identity principles from meaning clusters.

        Principles are the 'who you are' level. If rules say 'do this' and
        meanings say 'why', principles say 'what kind of person you are'.
        Run after synthesize_meanings to extract the deepest layer.
        """
        principles = service.synthesize_principles()
        return {
            "ok": True,
            "count": len(principles),
            "items": [
                {
                    "id": p.id,
                    "declaration": p.declaration,
                    "source_meaning_count": len(p.source_meaning_ids),
                    "source_rule_count": p.source_rule_count,
                    "depth": p.depth,
                    "scopes": p.scopes,
                    "confidence": p.confidence,
                    "personal_settings_overlap": p.personal_settings_overlap,
                    "status": p.status,
                }
                for p in principles
            ],
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    def list_principles(limit: int = 10) -> dict[str, Any]:
        """View identity-level principles — the deepest layer of the user's value system."""
        principles = service.list_principles()[:max(1, min(limit, 50))]
        return {
            "items": [
                {
                    "id": p.id,
                    "declaration": p.declaration,
                    "source_meaning_count": len(p.source_meaning_ids),
                    "source_rule_count": p.source_rule_count,
                    "depth": p.depth,
                    "scopes": p.scopes,
                    "confidence": p.confidence,
                    "personal_settings_overlap": p.personal_settings_overlap,
                    "status": p.status,
                }
                for p in principles
            ],
            "count": len(principles),
        }

    @mcp.tool()
    def predict_next_contexts(
        previous_context_nodes: list[dict] | None = None,
        session_id: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Predict likely next latent situations from active context nodes or a saved session."""
        context_nodes = previous_context_nodes if isinstance(previous_context_nodes, list) else []
        if session_id.strip():
            session = chat_adapter.session_summary(session_id.strip())
            metadata = session.get("metadata") if isinstance(session, dict) else {}
            if isinstance(metadata, dict):
                active_nodes = metadata.get("active_context_nodes", [])
                if isinstance(active_nodes, list) and active_nodes:
                    context_nodes = active_nodes
        predictions = service.predict_next_contexts(
            previous_context_nodes=context_nodes,
            limit=limit,
        )
        return {
            "items": predictions,
            "count": len(predictions),
            "used_context_nodes": context_nodes,
        }

    @mcp.tool()
    def get_personality_profile() -> dict[str, Any]:
        """Compute the user's personality profile from conversation history.

        Returns 5 dimensions inferred from behavior data:
        - metabolism_rate: how aggressively the user discards/adopts rules (0=conservative, 1=aggressive)
        - reward_function: what triggers positive reactions (keywords + pattern)
        - avoidance_function: what triggers negative reactions (keywords + pattern)
        - digestibility: abstract vs concrete preference (0=concrete, 1=abstract)
        - objective_drift: whether the user's goal has shifted recently

        Also returns any detected intervention signals (cognitive traps).
        """
        profile, interventions = service._compute_personality()
        return {
            "profile": {
                "metabolism_rate": profile.metabolism_rate,
                "metabolism_label": profile.metabolism_label,
                "reward_keywords": profile.reward_keywords,
                "reward_pattern": profile.reward_pattern,
                "avoidance_keywords": profile.avoidance_keywords,
                "avoidance_pattern": profile.avoidance_pattern,
                "digestibility": profile.digestibility,
                "digestibility_label": profile.digestibility_label,
                "current_objective": profile.current_objective,
                "objective_confidence": profile.objective_confidence,
                "drift_detected": profile.drift_detected,
                "drift_description": profile.drift_description,
                "sample_size": profile.sample_size,
                "computed_at": profile.computed_at,
            },
            "interventions": [
                {
                    "pattern_type": sig.pattern_type,
                    "confidence": sig.confidence,
                    "evidence": sig.evidence,
                    "mirror_prompt": sig.mirror_prompt,
                    "reward_frame": sig.reward_frame,
                }
                for sig in interventions
            ],
            "intervention_count": len(interventions),
            "memory_summary": _memory_summary(service),
            "self_overcome_proposals": service.self_overcome(),
        }

    @mcp.tool()
    def synthesize_rules() -> dict[str, Any]:
        """Generate rule hypotheses from success/failure pattern differences.

        Derives candidate rules without human input, based on statistical patterns.
        Returns proposed rules with confidence scores. These are NOT automatically
        added to the rule store — use them as guidance or review manually.
        """
        hypotheses = service.synthesize_rules()
        return {
            "hypotheses": hypotheses,
            "count": len(hypotheses),
        }

    @mcp.tool()
    async def save_episode(
        title: str,
        source_text: str = "",
        issuer: str = "",
        task_type: str = "generic",
        company_profile: dict | None = None,
        profile_name: str = "",
        output: dict | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Use this when you want to persist a completed task outcome as a reusable episode."""
        entry = service.save_episode(
            title=title,
            issuer=issuer,
            task_type=task_type,
            source_text=source_text,
            company_profile=company_profile,
            profile_name=profile_name,
            output=output,
            metadata=metadata,
        )
        if ctx is not None:
            await ctx.info(f"Saved episode {entry.id}: {entry.title}")
        return {
            "ok": True,
            "entry": _to_plain_data(entry),
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    async def save_correction(
        entry_id: str,
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
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Use this when a human corrected the output and you want that correction to persist."""
        saved = service.save_correction(
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
        if ctx is not None:
            await ctx.info(f"Correction {'saved' if saved else 'failed'} for episode {entry_id}")
        return {
            "ok": saved,
            "entry_id": entry_id,
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    async def save_conversation_turn(
        task_scope: str = "",
        user_message: str = "",
        assistant_message: str = "",
        user_feedback: str = "",
        extracted_corrections: list[str] | None = None,
        tags: list[str] | None = None,
        guidance_applied: bool = False,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Use this when conversation feedback should become reusable preference memory.

        Set guidance_applied=True when guidance from build_guidance_context was used
        before generating assistant_message. This enables automatic growth measurement:
        turns without guidance become the baseline; turns with guidance show improvement.
        Reaction score is inferred automatically — no explicit scoring needed.
        """
        turn = service.save_conversation_turn(
            task_scope=task_scope,
            user_message=user_message,
            assistant_message=assistant_message,
            user_feedback=user_feedback,
            extracted_corrections=extracted_corrections,
            tags=tags,
            guidance_applied=guidance_applied,
            auto_record_growth=True,
            metadata=metadata,
        )
        if ctx is not None:
            await ctx.info(f"Saved conversation turn {turn.id} in scope {turn.task_scope or 'generic'} | reaction_score={turn.reaction_score}")
        return {
            "ok": True,
            "turn": _to_plain_data(turn),
            "reaction_score": turn.reaction_score,
            "guidance_applied": turn.guidance_applied,
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    async def save_training_example(
        entry_id: str,
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
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Use this when you want to attach an accepted final output as supervised training data."""
        saved = service.save_training_example(
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
        if ctx is not None:
            await ctx.info(f"Training example {'saved' if saved else 'failed'} for episode {entry_id}")
        return {
            "ok": saved,
            "entry_id": entry_id,
            "memory_summary": _memory_summary(service),
        }

    @mcp.tool()
    def record_growth(
        case_id: str,
        case_title: str,
        baseline_output: str,
        baseline_score: float,
        guided_output: str,
        guided_score: float,
        task_scope: str = "",
        guidance_text: str = "",
    ) -> dict[str, Any]:
        """Use this to record a before/after growth measurement.

        Run the same task twice:
        1. WITHOUT guidance → pass baseline_output and baseline_score
        2. WITH guidance    → pass guided_output and guided_score
        Scores are 0.0 (worst) to 1.0 (best).
        delta > 0 means guidance helped. Repeated records build a growth curve.
        """
        record = service.record_growth(
            case_id=case_id,
            case_title=case_title,
            task_scope=task_scope,
            baseline_output=baseline_output,
            baseline_score=baseline_score,
            guided_output=guided_output,
            guided_score=guided_score,
            guidance_text=guidance_text,
        )
        return {
            "ok": True,
            "record_id": record.record_id,
            "case_id": record.case_id,
            "case_title": record.case_title,
            "baseline_score": record.baseline_score,
            "guided_score": record.guided_score,
            "delta": record.delta,
            "recorded_at": record.recorded_at,
        }

    @mcp.tool()
    def get_growth_summary() -> dict[str, Any]:
        """Use this to see whether the AI is growing overall.

        Returns average delta and per-case trends.
        trend label: 'growing' / 'flat' / 'degrading'
        """
        return service.growth_summary()

    @mcp.tool()
    def get_growth_trend(case_id: str) -> dict[str, Any]:
        """Use this to see the score history for one specific task over time.

        Returns a list of {recorded_at, baseline, guided, delta} entries,
        oldest first. Growing AI shows delta increasing over time.
        """
        trend = service.growth_trend(case_id)
        return {
            "case_id": case_id,
            "runs": len(trend),
            "trend": trend,
        }

    @mcp.tool()
    def export_training_dataset(
        output_dir: str,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle_seed: int = 7,
        split_strategy: str = "chronological",
    ) -> dict[str, Any]:
        """Use this when you need MLX-LM training files from accepted examples."""
        report = service.export_training_dataset(
            output_dir,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            shuffle_seed=shuffle_seed,
            split_strategy=split_strategy,
        )
        return {
            "ok": True,
            "report": report,
        }

    @mcp.tool()
    async def run_auto_training_cycle(
        model: str,
        output_dir: str,
        minimum_new_examples: int = 8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle_seed: int = 7,
        split_strategy: str = "chronological",
        force: bool = False,
        dry_run: bool = False,
        iters: int = 600,
        batch_size: int = 1,
        grad_accumulation_steps: int = 1,
        fine_tune_type: str = "lora",
        num_layers: int | None = None,
        learning_rate: float | None = None,
        mask_prompt: bool = True,
        grad_checkpoint: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Use this when you want to export data and trigger an MLX LoRA training cycle."""
        config = MlxLoraTrainingConfig(
            model=model,
            data_dir=Path(output_dir) / "dataset",
            adapter_path=Path(output_dir) / "adapters" / "pending",
            iters=iters,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            fine_tune_type=fine_tune_type,
            learning_rate=learning_rate,
            num_layers=num_layers,
            mask_prompt=mask_prompt,
            grad_checkpoint=grad_checkpoint,
        )
        if ctx is not None:
            await ctx.info(f"Starting auto-training cycle for model {model}")
        report = service.run_auto_training_cycle(
            model=model,
            output_dir=output_dir,
            minimum_new_examples=minimum_new_examples,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            shuffle_seed=shuffle_seed,
            split_strategy=split_strategy,
            force=force,
            dry_run=dry_run,
            training_config=config,
        )
        if ctx is not None:
            await ctx.info(f"Auto-training cycle finished with status {report['status']}")
        return {
            "ok": True,
            "report": report,
        }

    return mcp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Claude Pseudo Intelligence Core MCP server.")
    parser.add_argument("--memory-dir", default=str(Path.cwd() / ".local-memory"))
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mcp = create_mcp_server(
        args.memory_dir,
        host=args.host,
        port=args.port,
        path=args.path,
    )
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown path
        return


if __name__ == "__main__":
    main()
