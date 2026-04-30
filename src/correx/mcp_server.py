from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .chat_adapter import ChatLoopAdapter
from .mlx_trainer import MlxLoraTrainingConfig
from .service import CorrexService

SERVER_NAME = "correx"
SERVER_INSTRUCTIONS = (
    "Use this server when you need persistent human correction memory, "
    "conversation-derived preference rules, training dataset export, or automatic "
    "LoRA training orchestration for the Correx AI Correction OS."
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


def _memory_summary(service: CorrexService, *, compact: bool = True) -> dict[str, Any]:
    """Return memory statistics.

    When *compact=True* (default for inline tool responses), return only
    counts — no lists.  This keeps tool output short and avoids flooding
    the user's screen with JSON.

    When *compact=False* (used by the ``memory://summary`` resource),
    include the detailed latest_entries / latest_preference_rules /
    latest_context_transitions arrays.
    """
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
    personality = service.history.load_personality()
    result: dict[str, Any] = {
        "entries": len(entries),
        "turns": len(turns),
        "rules": len(rules),
        "promoted": len(stable_rules),
        "meanings": len(meanings),
        "transitions": len(transitions),
    }
    if not compact:
        # Full version — used by memory://summary resource only
        result.update({
            "memory_dir": str(service.base_dir),
            "high_value_rule_count": len(high_value_rules),
            "general_rule_count": len(general_rules),
            "mixed_rule_count": len(mixed_rules),
            "local_rule_count": len(local_rules),
            "latent_context_count": latent_context_count,
            "cross_scope_meanings": len([m for m in meanings if m.cross_scope_count >= 2]),
            "accepted_training_example_count": len(trainable_entries),
            "personality_metabolism": personality.get("metabolism_label", "unknown"),
            "personality_digestibility": personality.get("digestibility_label", "unknown"),
            "latest_entries": [_summarize_entry(entry) for entry in entries[:5]],
            "latest_preference_rules": [_summarize_rule(rule) for rule in rules[:5]],
            "latest_context_transitions": [_summarize_transition(item) for item in transitions[:5]],
        })
    return result


def create_mcp_server(
    memory_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
) -> Any:
    Context, FastMCP = _require_fastmcp()
    globals()["Context"] = Context
    service = CorrexService(memory_dir)
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
        return json.dumps(_memory_summary(service, compact=False), ensure_ascii=False, indent=2)

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
        return_trace: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Use this when you want reusable human correction memory for a new task prompt.

        Set ``return_trace=True`` to also receive ``inference_trace`` and
        ``guidance_id``. The trace lists the rule/context IDs that produced the
        guidance text so that, after the turn completes, hit/miss can be
        measured. Store ``guidance_id`` in
        ``save_conversation_turn(metadata={"inference_trace": {...}})`` to
        close the Organic Loop. Default ``False`` preserves the legacy
        two-field response.

        ``verbose`` controls payload size (default ``False`` = Medium diet):
        drops policy ``analogy``, caps prohibition laws at top 15 by coverage,
        and slims rule entries inside ``inference_trace`` (clears ``reason``,
        caps tag/keyword lists at 3, collapses ``latent_context_matches`` to
        a ``{count, max_posterior}`` summary). Set ``verbose=True`` when you
        need the full payload for dashboard inspection or deep debugging.
        """
        result = service.build_guidance_context(
            company_profile=company_profile,
            task_title=task_title,
            issuer=issuer,
            raw_text=raw_text,
            limit=limit,
            task_scope=task_scope,
            return_trace=return_trace,
            verbose=verbose,
        )
        if return_trace and isinstance(result, dict):
            guidance = result.get("guidance_context", "")
            inference_trace = result.get("inference_trace")
            guidance_id = result.get("guidance_id")
        else:
            guidance = result
            inference_trace = None
            guidance_id = None
        ghost_principles = service.get_fired_ghost_principles()
        payload: dict[str, Any] = {
            "guidance_context": guidance,
            "ghost_principles_count": len(ghost_principles),
        }
        if inference_trace is not None:
            payload["inference_trace"] = inference_trace
        if guidance_id is not None:
            payload["guidance_id"] = guidance_id
        return payload

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
        }

    @mcp.tool()
    def rebuild_preference_rules() -> dict[str, Any]:
        """Rebuild preference rules from all stored conversation turns.

        Re-scans the full correction history and promotes patterns that have
        appeared consistently across sessions into preference rules. Run this
        after importing corrections in bulk, or to repair a degraded rule set.
        """
        rules = service.rebuild_preference_rules()
        promoted = [r for r in rules if getattr(r, "status", "") == "promoted"]
        return {
            "ok": True,
            "total": len(rules),
            "promoted": len(promoted),
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
    def save_policy(
        id: str,
        title: str,
        core: str,
        why: str,
        analogy: str = "",
        opposite: str = "",
        limits: str = "",
        source_rule_ids: list[str] | None = None,
        source_ghost_ids: list[str] | None = None,
        source_law_ids: list[str] | None = None,
        scopes: list[str] | None = None,
        tags: list[str] | None = None,
        evidence_count: int = 0,
        maturity: str = "proposed",
        approved_by: str = "",
    ) -> dict[str, Any]:
        """Save a deep, interpretable policy derived from accumulated rules/ghosts.

        A policy carries: core (essence), why (reasoning), analogy (extension),
        opposite (when NOT to apply), limits (boundary conditions).
        Unlike literal rules, policies enable reasoning in novel situations.

        Maturity: "proposed" (needs user approval) | "active" (injected into guidance) | "superseded"
        """
        from .schemas import Policy as _Policy
        policy = _Policy(
            id=id,
            title=title,
            core=core,
            why=why,
            analogy=analogy,
            opposite=opposite,
            limits=limits,
            source_rule_ids=source_rule_ids or [],
            source_ghost_ids=source_ghost_ids or [],
            source_law_ids=source_law_ids or [],
            scopes=scopes or [],
            tags=tags or [],
            evidence_count=evidence_count,
            maturity=maturity,
            approved_by=approved_by,
        )
        saved = service.save_policy(policy)
        return {"ok": True, "policy": asdict(saved)}

    @mcp.tool()
    def list_policies(active_only: bool = False) -> dict[str, Any]:
        """List all policies. Active policies are injected into guidance context.

        Policies are the highest-quality knowledge unit — deep, interpretable
        principles with core/why/analogy/opposite/limits structure.
        """
        policies = service.list_policies(active_only=active_only)
        return {
            "items": [
                {
                    "id": p.id,
                    "title": p.title,
                    "core": p.core,
                    "why": p.why,
                    "analogy": p.analogy,
                    "opposite": p.opposite,
                    "limits": p.limits,
                    "source_rule_ids": p.source_rule_ids,
                    "source_ghost_ids": p.source_ghost_ids,
                    "source_law_ids": p.source_law_ids,
                    "evidence_count": p.evidence_count,
                    "maturity": p.maturity,
                    "scopes": p.scopes,
                    "tags": p.tags,
                    "approved_by": p.approved_by,
                    "created_at": p.created_at,
                    "updated_at": p.updated_at,
                }
                for p in policies
            ],
            "count": len(policies),
        }

    # ── Tension — Contradiction Montage ─────────────��──────────

    @mcp.tool()
    def detect_tension_candidates() -> dict[str, Any]:
        """Detect contradiction candidates for the client LLM to judge.

        Server-side: generates candidate pairs using keyword overlap and scope
        matching.  No opposition judgment — that's the client LLM's job.

        The client LLM should:
        1. Read each pair's full text in context
        2. Judge: would applying both rules in the SAME situation produce
           opposite actions?  If yes → genuine contradiction
        3. If different topics/scopes → skip (false positive)
        4. For genuine contradictions, extract:
           - boundary: "when A applies, when B applies"
           - signal: what observable cue triggers the switch
        5. Call save_tension() with the boundary and signal

        This is the Contradiction Montage pipeline — extracting *judgment*
        (when to switch between rules) rather than just rules themselves.
        """
        candidates = service.detect_tension_candidates()
        return {
            "ok": True,
            "count": len(candidates),
            "candidates": candidates,
            "instructions": (
                "各ペアを文脈込みで読み、以下を判定せよ:\n"
                "1. 同一状況で両方適用すると逆の行動になるか？ → 真の矛盾\n"
                "2. 別の話題・スコープなら → 除外（偽陽性）\n"
                "真の矛盾の場合、境界条件（いつAで、いつBか）と"
                "切替シグナル（何を見てA/Bを判断するか）を抽出して "
                "save_tension() を呼べ。"
            ),
        }

    @mcp.tool()
    def list_tensions(active_only: bool = False) -> dict[str, Any]:
        """List all detected tensions (contradiction pairs with boundary conditions).

        Tensions represent the highest form of learned knowledge: not WHAT to do,
        but WHEN to switch between opposing rules. Each tension has:
        - rule_a / rule_b: the contradicting pair
        - boundary: the decision function ("when A, when B")
        - signal: what observable cue triggers the switch
        """
        tensions = service.list_tensions(active_only=active_only)
        return {
            "items": [_to_plain_data(t) for t in tensions],
            "count": len(tensions),
        }

    @mcp.tool()
    def save_tension(
        rule_a_id: str,
        rule_a_text: str,
        rule_b_id: str,
        rule_b_text: str,
        boundary: str = "",
        signal: str = "",
        scopes: list[str] | None = None,
        confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Save a tension — a contradiction pair with its boundary condition.

        Call this after reviewing candidates from detect_tension_candidates().
        The boundary should describe WHEN each rule applies:
        e.g., "方針が不明なら確認。方針が明確なら即実行"

        The signal should identify the observable cue that triggers the switch:
        e.g., "ユーザーの指示の具体性レベル"
        """
        tension = service.save_tension(
            rule_a_id=rule_a_id,
            rule_a_text=rule_a_text,
            rule_b_id=rule_b_id,
            rule_b_text=rule_b_text,
            boundary=boundary,
            signal=signal,
            scopes=scopes,
            confidence=confidence,
        )
        return {
            "ok": True,
            "tension": _to_plain_data(tension),
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
            "self_overcome_proposals": service.self_overcome(),
        }

    @mcp.tool()
    def check_narrative_status() -> dict[str, Any]:
        """Check if the user personality narrative needs regeneration.

        Returns the current narrative, whether regeneration is needed,
        and raw policy/personality data for the client LLM to generate
        a new 5-line narrative if needed.

        Call this at session start or after policy changes. If
        needs_regeneration is True, generate a 5-line narrative from
        the returned data and call save_narrative() with the result.
        """
        return service.check_narrative_status()

    @mcp.tool()
    def save_narrative(
        narrative_text: str,
        method: str = "llm",
    ) -> dict[str, Any]:
        """Save a generated personality narrative.

        Called after the client LLM generates a 5-line personality
        narrative from check_narrative_status() data.  The narrative
        describes WHO the user is — their decision style, values,
        contradictions, growth edges, and working preferences.

        The narrative is persisted and auto-written to CLAUDE.md at
        session start.  Returns clipboard_text for manual paste into
        claude.ai personal settings.
        """
        return service.save_narrative(
            narrative_text=narrative_text,
            method=method,
        )

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
            "entry_id": entry.id,
            "title": entry.title,
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
        ghost_option: str = "",
        metadata: dict | None = None,
        reaction_score_override: float | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Use this when conversation feedback should become reusable preference memory.

        Set guidance_applied=True when guidance from build_guidance_context was used
        before generating assistant_message. This enables automatic growth measurement:
        turns without guidance become the baseline; turns with guidance show improvement.
        Reaction score is inferred automatically by default, or can be overridden
        by the client LLM for higher accuracy.

        reaction_score_override: Optional float 0.0-1.0. When provided, the client
        LLM's own judgment of the user's reaction is used instead of the rule-based
        scorer. The client LLM is IN the conversation and understands context better
        than pattern matching. Use this when you're confident about the user's sentiment.
        Scale: 0.0=strong rejection, 0.5=neutral, 0.75=acceptance, 0.9=strong praise.

        ghost_option: Optional. If the AI proposed something that was rejected or
        corrected, pass the rejected proposal text here. It will be stored as a
        ghost memory for the autonomous learning layer (GhostEngine). Over time,
        related ghosts cluster into trajectories that fire and sublimate into
        autonomous principles — without further human correction.
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
            reaction_score_override=reaction_score_override,
        )
        if ctx is not None:
            await ctx.info(f"Saved conversation turn {turn.id} in scope {turn.task_scope or 'generic'} | reaction_score={turn.reaction_score}")

        result: dict[str, Any] = {
            "ok": True,
            "turn_id": turn.id,
            "task_scope": turn.task_scope or "",
            "reaction_score": turn.reaction_score,
            "guidance_applied": turn.guidance_applied,
        }

        # Optionally record the rejected proposal as a ghost
        if ghost_option:
            ghost_dict, trajectory_dict, fired_principles = service.save_ghost(
                rejected_output=ghost_option,
                task_scope=task_scope,
                tags=tags,
                user_feedback=user_feedback,
                accepted_output=assistant_message,
                source_turn_id=turn.id,
            )
            result["ghost"] = {
                "ghost_id": ghost_dict["id"],
                "origin": ghost_dict["origin"],
                "prediction_error": ghost_dict["prediction_error"],
                "trajectory_id": trajectory_dict["id"],
                "trajectory_cumulative_pe": trajectory_dict["cumulative_pe"],
                "trajectory_ghost_count": trajectory_dict["source_ghost_count"],
                "fired": bool(fired_principles),
                "fired_principles": fired_principles,
            }
            if ctx is not None and fired_principles:
                await ctx.info(f"Ghost trajectory fired! Autonomous principle extracted: {fired_principles[0][:80]}")

        return result

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

    # ------------------------------------------------------------------
    # Ghost Engine tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def save_ghost(
        rejected_output: str,
        task_scope: str = "",
        tags: list[str] | None = None,
        user_feedback: str = "",
        accepted_output: str = "",
        source_turn_id: str = "",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Record a rejected AI proposal as a ghost memory for autonomous learning.

        Call this when the AI proposed something that was rejected or corrected.
        The ghost captures the rejected output, computes prediction error against
        the actual user response, and assigns it to an interference trajectory.

        When a trajectory accumulates enough prediction error (ghosts > 2 and
        cumulative PE > threshold), it fires automatically and sublimates into
        an autonomous principle — without requiring further human correction.

        Parameters:
          rejected_output: The AI's proposal that was rejected or corrected.
          task_scope: The task scope (same as save_conversation_turn).
          tags: Relevant tags for clustering.
          user_feedback: What the user actually said (the correction/rejection).
          accepted_output: The output that was actually accepted (if known).
          source_turn_id: The ConversationTurn ID that created this ghost.
        """
        ghost_dict, trajectory_dict, fired_principles = service.save_ghost(
            rejected_output=rejected_output,
            task_scope=task_scope,
            tags=tags,
            user_feedback=user_feedback,
            accepted_output=accepted_output,
            source_turn_id=source_turn_id,
        )
        fired = bool(fired_principles)
        if ctx is not None:
            status = f"fired! principles extracted: {len(fired_principles)}" if fired else "queued in trajectory"
            await ctx.info(f"Ghost {ghost_dict['id'][:8]} saved — trajectory {trajectory_dict['id'][:8]} — {status}")
        return {
            "ok": True,
            "ghost_id": ghost_dict["id"],
            "origin": ghost_dict["origin"],
            "prediction_error": ghost_dict["prediction_error"],
            "interference": ghost_dict["interference"],
            "trajectory_id": trajectory_dict["id"],
            "trajectory_cumulative_pe": trajectory_dict["cumulative_pe"],
            "trajectory_ghost_count": trajectory_dict["source_ghost_count"],
            "fired": fired,
            "fired_principles": fired_principles,
        }

    @mcp.tool()
    def list_ghost_trajectories(
        include_fired: bool = True,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List ghost trajectories — interference patterns detected from rejected proposals.

        Each trajectory is a cluster of related rejections converging on the same
        interference theme. When cumulative prediction error crosses the threshold,
        the trajectory fires and sublimates into an autonomous principle.

        Parameters:
          include_fired: Whether to include already-fired trajectories.
          limit: Maximum number of trajectories to return.
        """
        trajectories = service.list_ghost_trajectories(
            include_fired=include_fired,
            limit=limit,
        )
        fired = [t for t in trajectories if t.get("fired")]
        open_t = [t for t in trajectories if not t.get("fired")]
        return {
            "ok": True,
            "total": len(trajectories),
            "fired_count": len(fired),
            "open_count": len(open_t),
            "trajectories": trajectories,
        }

    @mcp.tool()
    def get_ghost_principles() -> dict[str, Any]:
        """Get all autonomous principles sublimated from fired ghost trajectories.

        These principles were extracted without human correction — purely from
        the accumulated prediction errors of rejected AI proposals. They represent
        the deepest layer of the Engram engine: patterns the AI kept getting wrong
        until the ghost trajectory fired and distilled the lesson autonomously.

        Include these alongside guidance from build_guidance_context for maximum
        behavioral alignment.
        """
        principles = service.get_fired_ghost_principles()
        return {
            "ok": True,
            "count": len(principles),
            "ghost_principles": principles,
            "note": (
                "These principles were extracted autonomously from fired ghost trajectories. "
                "No human correction was needed to generate them."
            ) if principles else (
                "No ghost trajectories have fired yet. "
                "Ghost principles emerge after enough rejected proposals accumulate "
                "in the same interference trajectory."
            ),
        }

    # ── Curiosity Layer (third learning layer) ────────────────────────────

    @mcp.tool()
    def save_curiosity_signal(
        question_text: str,
        question_type: str = "knowledge_gap",
        target: str = "self",
        task_scope: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        confidence: float = 0.0,
        source_turn_id: str = "",
    ) -> dict[str, Any]:
        """Record a user question detected by the client LLM.

        The client LLM detects questions in user messages, classifies them,
        and passes the result here. The server clusters related questions
        and tracks knowledge gaps.

        question_type:
          - "knowledge_gap": user doesn't know (「って何？」「教えて」)
          - "judgment_uncertainty": user can't decide (「どう思う？」「どっちがいい」)
          - "confirmation_seeking": user wants reassurance (「合ってる？」「大丈夫？」)

        target:
          - "self": user is asking for themselves (「わかりやすく教えて」)
          - "other": user needs to explain to someone else (「わかりやすくまとめて」)

        keywords: core keywords of the question for clustering.
        """
        signal_dict, cluster_dict, is_new = service.save_curiosity_signal(
            question_text=question_text,
            question_type=question_type,
            target=target,
            task_scope=task_scope,
            tags=tags,
            keywords=keywords,
            confidence=confidence,
            source_turn_id=source_turn_id,
        )
        return {
            "ok": True,
            "signal_id": signal_dict["id"],
            "question_type": signal_dict["question_type"],
            "cluster_id": cluster_dict["id"],
            "cluster_is_new": is_new,
            "cluster_signal_count": cluster_dict["signal_count"],
            "cluster_escalation_score": cluster_dict["escalation_score"],
            "cluster_status": cluster_dict["status"],
        }

    @mcp.tool()
    def resolve_curiosity_clusters(
        task_scope: str = "",
    ) -> dict[str, Any]:
        """Resolve open knowledge gap clusters when user is satisfied.

        Call this when the user expresses satisfaction or the topic is resolved.
        Resolves all open/escalated clusters matching the given scope.
        """
        resolved_count = service.resolve_curiosity_clusters(task_scope=task_scope)
        return {
            "ok": True,
            "resolved_count": resolved_count,
        }

    @mcp.tool()
    def get_cognitive_map() -> dict[str, Any]:
        """Get the cognitive map — a scope-level view of knowledge gaps.

        Use at session start to understand where the user needs more explanation.
        Returns gap strengths per scope, hotspots (escalated areas),
        and counts of open/escalated clusters.

        High gap_strength + escalated = user has been asking about this repeatedly.
        Provide foundational explanations in those areas.
        """
        cognitive_map = service.get_cognitive_map()
        return {
            "ok": True,
            **cognitive_map,
        }

    @mcp.tool()
    def list_knowledge_gap_clusters(
        include_resolved: bool = False,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List knowledge gap clusters for detailed analysis.

        Each cluster represents a group of related questions in the same scope.
        Shows escalation scores, question types, and signal counts.
        """
        clusters = service.list_knowledge_gap_clusters(
            include_resolved=include_resolved,
            limit=limit,
        )
        open_c = [c for c in clusters if c.get("status") == "open"]
        escalated_c = [c for c in clusters if c.get("status") == "escalated"]
        return {
            "ok": True,
            "total": len(clusters),
            "open_count": len(open_c),
            "escalated_count": len(escalated_c),
            "clusters": clusters,
        }

    # ── Session ingestion ──────────────────────────────────────────────────

    @mcp.tool()
    def ingest_claude_sessions(
        projects_dir: str = "",
        include_positive: bool = True,
    ) -> dict[str, Any]:
        """Ingest past Claude Code sessions into the correction/ghost pipeline.

        Scans ~/.claude/projects/ for all session JSONL files, extracts
        user→assistant pairs where corrections or praise occurred, and
        feeds them into conversation_history (for rule promotion) and
        ghost engine (for trajectory analysis).

        Call this once to bootstrap the system with historical data.
        Set include_positive=True to also capture praise patterns.

        Args:
            projects_dir: Override path to Claude projects dir. Default: ~/.claude/projects/
            include_positive: Also extract positive feedback ("いいね", "完璧" etc.)
        """
        import re
        import hashlib
        from datetime import datetime

        base = Path(projects_dir) if projects_dir else Path.home() / ".claude" / "projects"
        if not base.exists():
            return {"ok": False, "error": f"Directory not found: {base}"}

        neg_words = re.compile(
            r"違う|そうじゃない|やり直|ダメ|だめ|雑|抽象|具体|ちがう|余計|不要|いらん|やめ|捨て"
            r"|するな|じゃない|直せ|バグ|嘘|ひどい|おかしい|冗談|忘れるな|使うな|待つな|逃げるな"
            r"|なんで|できてない|動かない|壊れ|間違|やめろ|変えろ|効かない|聞いてない"
        )
        pos_words = re.compile(
            r"いいね|いいよ|完璧|素晴らしい|最高|ありがとう|感動|正解|それでいい|それだ"
            r"|good|great|perfect|exactly|nice|awesome|love it|well done"
        )

        def _infer_scope(name: str) -> str:
            """Infer task scope from Claude project directory name."""
            n = name.lower()
            if "document" in n: return "document_creation"
            return "general"

        # Load existing to avoid duplicates
        existing_turns = service.history.load_conversation_turns()
        existing_ids = {getattr(t, "id", "") for t in existing_turns}
        # Also track by original_id in metadata to prevent duplicates on re-run
        existing_ids.update(
            getattr(t, "metadata", {}).get("original_id", "")
            for t in existing_turns
            if getattr(t, "metadata", {}).get("source") == "ingest_claude_sessions"
        )

        new_corrections = 0
        new_positives = 0
        new_ghosts = 0
        projects_scanned = 0

        for proj in base.iterdir():
            if not proj.is_dir():
                continue
            projects_scanned += 1
            scope = _infer_scope(proj.name)

            for sf in sorted(proj.glob("*.jsonl")):
                messages: list[dict] = []
                _MAX_MESSAGES_PER_FILE = 5000
                for line in sf.open():
                    try:
                        d = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    msg = d.get("message", {})
                    role = msg.get("role", "")
                    if not role:
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        texts = [c.get("text", "") for c in content
                                 if c.get("type") == "text" and c.get("text")]
                        text = "\n".join(texts)
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""
                    ts = d.get("timestamp", "")
                    if text.strip():
                        messages.append({"role": role, "text": text.strip(), "ts": ts})
                        if len(messages) >= _MAX_MESSAGES_PER_FILE:
                            break

                for i in range(len(messages) - 1):
                    if messages[i]["role"] != "user":
                        continue
                    user_text = messages[i]["text"]
                    if len(user_text) < 6 or user_text.startswith("{"):
                        continue

                    is_negative = bool(neg_words.search(user_text))
                    is_positive = bool(pos_words.search(user_text)) if include_positive else False

                    if not is_negative and not is_positive:
                        continue

                    # Find next assistant response
                    asst_text = ""
                    for j in range(i + 1, min(i + 3, len(messages))):
                        if messages[j]["role"] == "assistant" and messages[j]["text"]:
                            asst_text = messages[j]["text"]
                            break
                    if not asst_text:
                        continue

                    turn_id = hashlib.sha256(
                        f"ingest-{messages[i]['ts']}-{user_text[:50]}".encode()
                    ).hexdigest()[:20]
                    if turn_id in existing_ids:
                        continue
                    existing_ids.add(turn_id)

                    score = 0.25 if is_negative else 0.85
                    tags = [scope, "ingested"]
                    if is_positive:
                        tags.append("positive")

                    # Save as conversation turn via service
                    service.save_conversation_turn(
                        task_scope=scope,
                        user_message=user_text[:400],
                        assistant_message=asst_text[:300],
                        user_feedback=user_text[:400],
                        extracted_corrections=[],
                        tags=tags,
                        guidance_applied=False,
                        metadata={"source": "ingest_claude_sessions", "original_id": turn_id},
                    )

                    if is_negative:
                        new_corrections += 1
                        # Also create ghost
                        try:
                            service.save_ghost(
                                rejected_output=asst_text[:500],
                                task_scope=scope,
                                tags=tags[:3],
                                user_feedback=user_text[:500],
                                source_turn_id=turn_id,
                            )
                            new_ghosts += 1
                        except Exception as e:
                            print(f"[correx] ghost save failed: {e}", file=sys.stderr)
                    else:
                        new_positives += 1

        return {
            "ok": True,
            "projects_scanned": projects_scanned,
            "new_corrections": new_corrections,
            "new_positives": new_positives,
            "new_ghosts": new_ghosts,
            "total_turns_now": len(existing_ids),
        }

    @mcp.tool()
    def get_unprocessed_turns(
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get conversation turns that have empty extracted_corrections.

        These turns were imported (e.g. via ingest_claude_sessions) but their
        correction rules haven't been extracted yet. The client LLM should:
        1. Read each turn's user_message + assistant_message + user_feedback
        2. Extract generalized rules (「次回以降も使える一般化されたルール」)
        3. Call update_turn_corrections() to save the extracted rules

        Args:
            limit: Max turns to return per call. Default 20.
            offset: Skip this many unprocessed turns.
        """
        turns = service.history.load_conversation_turns()
        unprocessed = [
            t for t in turns
            if not getattr(t, "extracted_corrections", None)
            and (getattr(t, "user_feedback", "") or getattr(t, "user_message", ""))
        ]
        total = len(unprocessed)
        page = unprocessed[offset:offset + limit]

        return {
            "ok": True,
            "total_unprocessed": total,
            "returned": len(page),
            "offset": offset,
            "turns": [
                {
                    "id": getattr(t, "id", ""),
                    "task_scope": getattr(t, "task_scope", ""),
                    "user_message": (getattr(t, "user_message", "") or "")[:500],
                    "assistant_message": (getattr(t, "assistant_message", "") or "")[:500],
                    "user_feedback": (getattr(t, "user_feedback", "") or "")[:500],
                    "reaction_score": getattr(t, "reaction_score", None),
                    "tags": getattr(t, "tags", []),
                }
                for t in page
            ],
        }

    @mcp.tool()
    def update_turn_corrections(
        turn_id: str,
        extracted_corrections: list[str],
    ) -> dict[str, Any]:
        """Update a conversation turn with extracted correction rules.

        Called by the client LLM after reading unprocessed turns from
        get_unprocessed_turns(). The corrections should be generalized rules
        that apply beyond this specific turn.

        Args:
            turn_id: The turn ID to update.
            extracted_corrections: List of generalized rules extracted by the LLM.
        """
        if not turn_id:
            return {"ok": False, "error": "turn_id is required"}
        if not extracted_corrections:
            return {"ok": False, "error": "extracted_corrections cannot be empty"}

        turns = service.history.load_conversation_turns()
        found = False
        for t in turns:
            if getattr(t, "id", "") == turn_id:
                t.extracted_corrections = extracted_corrections
                found = True
                break

        if not found:
            return {"ok": False, "error": f"Turn {turn_id} not found"}

        service.history.write_conversation_turns(turns)

        # Trigger rule promotion for the new corrections
        try:
            service.history.rebuild_preference_rules()
        except Exception as e:
            print(f"[correx] rule rebuild failed: {e}", file=sys.stderr)

        return {
            "ok": True,
            "turn_id": turn_id,
            "corrections_count": len(extracted_corrections),
        }

    @mcp.tool()
    def process_ingested_data() -> dict[str, Any]:
        """Run post-processing pipeline on ingested data.

        Call this after ingest_claude_sessions + update_turn_corrections
        to rebuild rules, synthesize patterns, and prepare for sublimation.
        No LLM needed for this step — it's pure data processing.

        Pipeline:
        1. rebuild_preference_rules (promote patterns from corrections)
        2. synthesize_rules (derive rule hypotheses from patterns)
        3. Return stats + count of pending sublimations for client LLM
        """
        # 1. Rebuild rules
        rules_before = len(service.history.load_preference_rules())
        try:
            service.history.rebuild_preference_rules()
        except Exception as e:
            print(f"[correx] rule rebuild failed: {e}", file=sys.stderr)
        rules_after = len(service.history.load_preference_rules())

        # 2. Count pending sublimations
        trajectories = service.list_ghost_trajectories(include_fired=True)
        pending_sublimations = sum(
            1 for t in trajectories
            if t.get("fired") and not t.get("sublimated_principle")
        )

        return {
            "ok": True,
            "rules_before": rules_before,
            "rules_after": rules_after,
            "new_rules": rules_after - rules_before,
            "pending_sublimations": pending_sublimations,
            "next_step": "Call get_pending_sublimations() and sublimate each trajectory"
            if pending_sublimations > 0
            else "No pending sublimations. Pipeline complete.",
        }

    @mcp.tool()
    def cleanup_ghost_principles() -> dict[str, Any]:
        """Clean up ghost principles: remove task-specific noise, merge duplicates, fix law duplicates.

        Run this when ghost principles have accumulated too many task-specific
        or duplicate entries. Removes non-generalizable principles and merges
        duplicate universal laws.

        Returns cleanup statistics.
        """
        return service.cleanup_ghost_principles()

    @mcp.tool()
    def get_pending_sublimations() -> dict[str, Any]:
        """Get ghost trajectories that need client-side LLM sublimation.

        Returns fired trajectories where the principle is missing or was
        template-generated. Each entry includes the ghost rejection records
        so you (the client LLM) can extract a behavioral principle.

        For each pending trajectory:
        1. Read the ghosts (rejected AI outputs + user reactions)
        2. Extract ONE behavioral principle (20-50 chars, Japanese)
        3. Optionally generalize into a universal law
        4. Call save_sublimation() with the result

        Format: "〜するな" or "〜してから〜せよ" — concrete, actionable.
        """
        pending = service.get_pending_sublimations()
        return {
            "ok": True,
            "count": len(pending),
            "pending": pending,
            "instructions": (
                "For each trajectory, read the ghost rejection patterns and write "
                "a behavioral principle in 20-50 chars Japanese. "
                "Then call save_sublimation(trajectory_id, principle, universal_law)."
            ) if pending else "No pending sublimations.",
        }

    @mcp.tool()
    def save_sublimation(
        trajectory_id: str,
        principle: str,
        universal_law: str = "",
        law_match_index: int = 0,
    ) -> dict[str, Any]:
        """Save a sublimated principle extracted by the client LLM.

        Called after get_pending_sublimations(). The client LLM reads
        the ghost rejection patterns and writes a behavioral principle.

        Args:
            trajectory_id: The trajectory ID from get_pending_sublimations().
            principle: Behavioral principle (20-50 chars Japanese recommended).
                       Format: "〜するな" or "〜してから〜せよ"
            universal_law: Optional. A generalized version that removes
                          scope-specific nouns (applicable to any context).
            law_match_index: If universal_law matches an existing law,
                           pass its 1-based index. Pass 0 for new law.
        """
        return service.save_sublimation(
            trajectory_id=trajectory_id,
            principle=principle,
            universal_law=universal_law,
            law_match_index=law_match_index,
        )

    @mcp.tool()
    def evaluate_guidance_effectiveness(
        evaluations: list[dict],
        task_scope: str = "",
    ) -> dict[str, Any]:
        """Self-evaluate which guidance rules actually helped your output.

        Call this after completing a meaningful task where build_guidance_context
        was used. You (the client LLM) are the best judge of whether the
        injected rules improved your output — you were IN the conversation.

        For each rule from build_guidance_context, score its effectiveness:
        - 0.9: This rule directly improved my output quality
        - 0.7: This rule was somewhat helpful
        - 0.5: Neutral, didn't affect my output
        - 0.3: This rule was irrelevant to this task
        - 0.1: This rule actively made my output worse

        Args:
            evaluations: List of {rule_id: str, score: float 0.0-1.0, reason: str}.
                        rule_id is the preference rule ID from build_guidance_context.
            task_scope: The scope of the task that was completed.

        Rules scoring >= 0.7 get boosted (promoted faster).
        Rules scoring < 0.3 get penalized (demoted if consistently failing).
        This creates a self-optimizing feedback loop with zero user intervention.
        """
        return service.evaluate_guidance_effectiveness(
            evaluations=evaluations,
            task_scope=task_scope,
        )

    @mcp.tool()
    def generate_session_feedback(
        task_scope: str = "",
        task_title: str = "",
        corrections_this_session: int = 0,
        guidance_was_injected: bool = False,
    ) -> dict[str, Any]:
        """Generate a natural feedback question for the end of a session.

        Call this at session end when meaningful work was completed.
        The question is tailored to the task context and never mentions
        internal system concepts (laws, rules, ghosts).

        Present the returned question and options to the user, then call
        save_session_feedback with their answer.
        """
        return service.generate_session_feedback_question(
            task_scope=task_scope,
            task_title=task_title,
            corrections_this_session=corrections_this_session,
            guidance_was_injected=guidance_was_injected,
        )

    @mcp.tool()
    def save_session_feedback(
        answer: str,
        task_scope: str = "",
        task_title: str = "",
        corrections_this_session: int = 0,
    ) -> dict[str, Any]:
        """Save the user's session feedback response.

        Call after the user answers the feedback question from
        generate_session_feedback. Maps their response to a growth score
        and persists it for quantitative analysis.

        answer should be one of: 'スムーズだった', 'いつも通り', '手間取った'
        """
        return service.save_session_feedback(
            answer=answer,
            task_scope=task_scope,
            task_title=task_title,
            corrections_this_session=corrections_this_session,
        )

    # ── Journey Memory (episodic memory from searches/exploration) ────────

    @mcp.tool()
    def save_journey(
        where: str,
        scope: str = "",
        impression: list[str] | None = None,
        valence: float = 0.5,
        journey_type: str = "wander",
        detail: str = "",
        connected_turn_id: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Save an episodic journey memory — a trace of search or exploration.

        Call this when the AI visits a URL, reads a file, or explores a codebase.
        journey_type: "business" (user-requested, never forgotten) or "wander" (incidental, forgettable).
        impression: keyword traces of what was found (not full content).
        valence: 0.0 (useless) to 1.0 (treasure).
        """
        j = service.save_journey(
            where=where,
            scope=scope,
            impression=impression,
            valence=valence,
            journey_type=journey_type,
            detail=detail,
            connected_turn_id=connected_turn_id,
            tags=tags,
        )
        return {
            "ok": True,
            "journey_id": j["id"],
            "novelty": j["novelty"],
            "journey_type": j["journey_type"],
        }

    @mcp.tool()
    def awaken_journeys(
        context_keywords: list[str],
        scope: str = "",
        threshold: float = 0.2,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Find journey memories triggered by current context (déjà vu).

        Pass keywords from the current task. Returns journeys whose impressions
        overlap — places the AI has been before that may be relevant now.
        Dormant journeys that match are awakened automatically.
        """
        results = service.awaken_journeys(
            context_keywords=context_keywords,
            scope=scope,
            threshold=threshold,
            limit=limit,
        )
        compact = [
            {
                "id": r["journey"]["id"],
                "where": r["journey"].get("where", ""),
                "scope": r["journey"].get("scope", ""),
                "impression": r["journey"].get("impression", []),
                "similarity": r["similarity"],
                "band": r.get("band", ""),
                "deja_vu": r["deja_vu"],
            }
            for r in results
        ]
        return {
            "ok": True,
            "awakened_count": len(compact),
            "journeys": compact,
        }

    @mcp.tool()
    def update_journey(
        journey_id: str,
        impression: list[str] | None = None,
        valence: float | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Update a journey during its labile window (30min after awakening).

        Only recently awakened journeys can be updated. This implements memory
        reconsolidation: recalled memories become temporarily malleable.
        Use this to enrich a journey with new impressions discovered during revisit.
        """
        return service.update_journey(
            journey_id=journey_id,
            impression=impression,
            valence=valence,
            detail=detail,
        )

    @mcp.tool()
    def list_journeys(
        limit: int = 20,
        journey_type: str = "",
        include_dormant: bool = False,
    ) -> dict[str, Any]:
        """List stored journey memories (episodic traces of searches and exploration)."""
        journeys = service.list_journeys(
            limit=limit,
            journey_type=journey_type,
            include_dormant=include_dormant,
        )
        compact = [
            {
                "id": j["id"],
                "where": j.get("where", ""),
                "scope": j.get("scope", ""),
                "journey_type": j.get("journey_type", ""),
                "impression": j.get("impression", []),
                "valence": j.get("valence", 0),
                "awakened_count": j.get("awakened_count", 0),
            }
            for j in journeys
        ]
        return {
            "ok": True,
            "total": len(compact),
            "journeys": compact,
        }

    @mcp.tool()
    def scan_journey_dormancy(
        max_idle_days: int = 30,
    ) -> dict[str, Any]:
        """Scan journeys for dormancy and forgetting. Wander journeys go dormant after 7 idle days, forgotten after max_idle_days."""
        return service.dormant_journeys(max_idle_days=max_idle_days)

    # ── Autonomous Intelligence Engine ────────────────────────────────────

    @mcp.tool()
    def run_autonomous_tick(
        event_type: str = "",
        scope: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run one tick of the autonomous intelligence engine (LLM-free).

        The engine processes all layers simultaneously: rules, ghosts, journeys,
        policies, tensions, and transitions. Returns applicable rules, predictions,
        and cross-layer modulations.

        event_type: "correction", "praise", "question", "time", "ghost_fire".
        Empty = self-reflection mode (finds weakest knowledge area).
        """
        return service.run_autonomous_tick(
            event_type=event_type,
            scope=scope,
            tags=tags,
            keywords=keywords,
        )

    @mcp.tool()
    def get_engine_state() -> dict[str, Any]:
        """Get the current state of the autonomous intelligence engine.

        Returns cycle count, last prediction, scope coverage map,
        and prediction error history.
        """
        return service.get_engine_state()

    @mcp.tool()
    def record_communication_outcome(
        need_type: str,
        resolved: bool,
    ) -> dict[str, Any]:
        """Record whether the engine's cry was heard and resolved.

        Call this when the user responds to an engine cry (need).
        This teaches the engine that expressing needs works,
        driving the reflexive → intentional transition.

        need_type: The type of need that was expressed (e.g., "knowledge_gap").
        resolved: True if the user addressed the need, False if ignored.
        """
        return service.record_communication_outcome(need_type, resolved)

    @mcp.tool()
    def semanticize_ghost_memories() -> dict[str, Any]:
        """Run episodic-to-semantic transformation on Ghost memories.

        Like the brain: specific episodes ("I said X and got scolded")
        gradually fade to gist (truncated) then trace (metadata only).
        The lesson (sublimated principle) survives; the episode fades.
        Decay speed: scolded=slow (180d), corrected=medium (90d), rejected=fast (60d).
        Only affects ghosts in fired trajectories (principle already extracted).
        """
        return service.semanticize_ghost_memories()

    return mcp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Correx MCP server.")
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
