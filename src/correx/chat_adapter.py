from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO
from uuid import uuid4

from .service import CorrexService

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - unavailable on POSIX
    msvcrt = None


def _serialize_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(payload, temp_file, ensure_ascii=False, indent=2)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def _read_text_payload(raw_text: str = "", raw_text_file: str = "") -> str:
    if raw_text_file:
        return Path(raw_text_file).read_text(encoding="utf-8").strip()
    return str(raw_text).strip()


def _extract_active_context_nodes(scored_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes_by_signature: dict[str, dict[str, Any]] = {}
    for item in scored_rules:
        signature = str(item.get("top_context_signature", "")).strip()
        if not signature:
            continue
        node = {
            "context_id": str(item.get("top_context_id", "")).strip(),
            "scope": str(item.get("top_context_scope", "")).strip(),
            "tags": [str(entry).strip() for entry in item.get("top_context_tags", []) if str(entry).strip()][:4],
            "keywords": [str(entry).strip() for entry in item.get("top_context_keywords", []) if str(entry).strip()][:6],
            "posterior": float(item.get("top_context_posterior", 0.0) or 0.0),
            "signature": signature,
        }
        existing = nodes_by_signature.get(signature)
        if existing is None or float(node["posterior"]) > float(existing.get("posterior", 0.0) or 0.0):
            nodes_by_signature[signature] = node
    return sorted(
        nodes_by_signature.values(),
        key=lambda item: float(item.get("posterior", 0.0) or 0.0),
        reverse=True,
    )[:3]


def _evaluate_transition_forecast(
    predicted_next_contexts: list[dict[str, Any]] | None,
    realized_context_nodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    predictions = [
        item
        for item in (predicted_next_contexts or [])
        if isinstance(item, dict) and str(item.get("to_signature", "")).strip()
    ]
    realized_signatures = {
        str(item.get("signature", "")).strip()
        for item in (realized_context_nodes or [])
        if isinstance(item, dict) and str(item.get("signature", "")).strip()
    }
    matched_prediction_signatures = [
        str(item.get("to_signature", "")).strip()
        for item in predictions
        if str(item.get("to_signature", "")).strip() in realized_signatures
    ]
    missed_prediction_signatures = [
        str(item.get("to_signature", "")).strip()
        for item in predictions
        if str(item.get("to_signature", "")).strip() not in realized_signatures
    ]
    total_score_mass = round(
        sum(max(0.0, float(item.get("score", 0.0) or 0.0)) for item in predictions),
        4,
    )
    matched_score_mass = round(
        sum(
            max(0.0, float(item.get("score", 0.0) or 0.0))
            for item in predictions
            if str(item.get("to_signature", "")).strip() in realized_signatures
        ),
        4,
    )
    top_prediction_signature = str(predictions[0].get("to_signature", "")).strip() if predictions else ""
    top_prediction_hit = bool(top_prediction_signature and top_prediction_signature in realized_signatures)
    return {
        "prediction_available": bool(predictions),
        "top_prediction_signature": top_prediction_signature,
        "top_prediction_hit": top_prediction_hit,
        "matched_prediction_signatures": matched_prediction_signatures[:4],
        "missed_prediction_signatures": missed_prediction_signatures[:4],
        "matched_score_mass": matched_score_mass,
        "total_score_mass": total_score_mass,
        "coverage_ratio": round(matched_score_mass / total_score_mass, 4) if total_score_mass > 0 else 0.0,
        "predicted_next_contexts": predictions[:4],
        "realized_context_signatures": sorted(realized_signatures)[:4],
    }


@dataclass(slots=True)
class ChatSessionState:
    session_id: str
    created_at: str
    updated_at: str
    status: str
    task_scope: str = ""
    task_title: str = ""
    issuer: str = ""
    raw_text: str = ""
    company_profile: dict = field(default_factory=dict)
    system_message: str = ""
    user_message: str = ""
    prompt: str = ""
    metadata: dict = field(default_factory=dict)
    guidance_context: str = ""
    guidance_applied: bool = False
    latest_assistant_message: str = ""
    latest_feedback: str = ""
    linked_entry_id: str = ""


class ChatLoopAdapter:
    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.service = CorrexService(self.memory_dir)
        self.session_dir = self.memory_dir / "chat_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.memory_dir / ".chat-session.lock"

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

    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"

    def _load_session(self, session_id: str) -> ChatSessionState:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ChatSessionState(
            session_id=str(payload.get("session_id", session_id)),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            status=str(payload.get("status", "open")),
            task_scope=str(payload.get("task_scope", "")),
            task_title=str(payload.get("task_title", "")),
            issuer=str(payload.get("issuer", "")),
            raw_text=str(payload.get("raw_text", "")),
            company_profile=payload.get("company_profile") if isinstance(payload.get("company_profile"), dict) else {},
            system_message=str(payload.get("system_message", "")),
            user_message=str(payload.get("user_message", "")),
            prompt=str(payload.get("prompt", "")),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            guidance_context=str(payload.get("guidance_context", "")),
            guidance_applied=bool(payload.get("guidance_applied", False)),
            latest_assistant_message=str(payload.get("latest_assistant_message", "")),
            latest_feedback=str(payload.get("latest_feedback", "")),
            linked_entry_id=str(payload.get("linked_entry_id", "")),
        )

    def _save_session(self, state: ChatSessionState) -> None:
        state.updated_at = datetime.now().strftime("%Y/%m/%d %H:%M")
        _serialize_json(self._session_path(state.session_id), asdict(state))

    def _load_session_locked(self, session_id: str) -> tuple[TextIO, ChatSessionState]:
        handle = self._lock_handle()
        try:
            return handle, self._load_session(session_id)
        except Exception:
            self._unlock_handle(handle)
            raise

    def prepare(
        self,
        *,
        task_scope: str = "",
        task_title: str = "",
        issuer: str = "",
        raw_text: str = "",
        raw_text_file: str = "",
        company_profile: dict | None = None,
        system_message: str = "",
        user_message: str = "",
        prompt: str = "",
        metadata: dict | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        now = datetime.now().strftime("%Y/%m/%d %H:%M")
        normalized_session_id = session_id.strip() or f"chat-{uuid4().hex[:12]}"
        session_path = self._session_path(normalized_session_id)
        if session_id.strip() and session_path.exists():
            state = self._load_session(normalized_session_id)
            state.status = "open"
            if task_scope.strip():
                state.task_scope = task_scope.strip()
            if task_title.strip():
                state.task_title = task_title.strip()
            if issuer.strip():
                state.issuer = issuer.strip()
            if raw_text.strip() or raw_text_file.strip():
                state.raw_text = _read_text_payload(raw_text, raw_text_file)
            if isinstance(company_profile, dict):
                state.company_profile = company_profile
            if system_message.strip():
                state.system_message = system_message.strip()
            if user_message.strip():
                state.user_message = user_message.strip()
            if prompt.strip():
                state.prompt = prompt.strip()
            if isinstance(metadata, dict):
                state.metadata = {
                    **state.metadata,
                    **metadata,
                }
        else:
            state = ChatSessionState(
                session_id=normalized_session_id,
                created_at=now,
                updated_at=now,
                status="open",
                task_scope=task_scope.strip(),
                task_title=task_title.strip(),
                issuer=issuer.strip(),
                raw_text=_read_text_payload(raw_text, raw_text_file),
                company_profile=company_profile if isinstance(company_profile, dict) else {},
                system_message=system_message.strip(),
                user_message=user_message.strip(),
                prompt=prompt.strip(),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        self._save_session(state)
        previous_context_nodes = (
            state.metadata.get("active_context_nodes", [])
            if isinstance(state.metadata, dict)
            else []
        )
        previous_predicted_next_contexts = (
            state.metadata.get("predicted_next_contexts", [])
            if isinstance(state.metadata, dict)
            else []
        )
        conversation_analysis = self.service.analyze_conversation_guidance(
            task_scope=state.task_scope or state.task_title,
            raw_text=state.raw_text,
            previous_context_nodes=previous_context_nodes if isinstance(previous_context_nodes, list) else None,
        )
        guidance_context = self.service.build_guidance_context(
            company_profile=state.company_profile,
            task_title=state.task_title,
            issuer=state.issuer,
            raw_text=state.raw_text,
            task_scope=state.task_scope or state.task_title,
            previous_context_nodes=previous_context_nodes if isinstance(previous_context_nodes, list) else None,
        )
        active_context_nodes = _extract_active_context_nodes(conversation_analysis["selected_rules"])
        transition_trace = _evaluate_transition_forecast(
            previous_predicted_next_contexts if isinstance(previous_predicted_next_contexts, list) else None,
            active_context_nodes,
        )
        predicted_next_contexts = self.service.predict_next_contexts(
            previous_context_nodes=active_context_nodes,
            limit=4,
        )
        inference_trace = {
            "task_scope": state.task_scope or state.task_title,
            "raw_text_preview": state.raw_text[:240],
            "selected_rule_ids": [item.get("rule_id", "") for item in conversation_analysis["selected_rules"] if item.get("rule_id")],
            "abstained_rule_ids": [item.get("rule_id", "") for item in conversation_analysis["abstained_rules"] if item.get("rule_id")],
            "selected_rules": conversation_analysis["selected_rules"],
            "abstained_rules": conversation_analysis["abstained_rules"],
            "recent_corrections": conversation_analysis["recent_corrections"],
            "previous_context_nodes": previous_context_nodes if isinstance(previous_context_nodes, list) else [],
            "active_context_nodes": active_context_nodes,
            "transition_trace": transition_trace,
            "predicted_next_contexts": predicted_next_contexts,
            "abstained_overall": not bool(conversation_analysis["selected_rules"]),
            "selection_policy": "latent-trace-v2",
        }
        state.guidance_context = guidance_context
        state.guidance_applied = bool(guidance_context.strip())
        state.metadata = {
            **state.metadata,
            "previous_context_nodes": previous_context_nodes if isinstance(previous_context_nodes, list) else [],
            "active_context_nodes": active_context_nodes,
            "transition_trace": transition_trace,
            "predicted_next_contexts": predicted_next_contexts,
            "inference_trace": inference_trace,
        }
        self._save_session(state)
        return {
            "session_id": state.session_id,
            "guidance_context": guidance_context,
            "guidance_applied": state.guidance_applied,
            "task_scope": state.task_scope,
            "task_title": state.task_title,
            "issuer": state.issuer,
            "inference_trace": inference_trace,
        }

    def save_feedback(
        self,
        session_id: str,
        *,
        user_message: str = "",
        assistant_message: str,
        user_feedback: str,
        extracted_corrections: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        handle, state = self._load_session_locked(session_id)
        try:
            current_user_message = user_message.strip() or state.user_message
            combined_metadata = dict(state.metadata)
            if isinstance(metadata, dict):
                combined_metadata.update(metadata)
            turn = self.service.save_conversation_turn(
                task_scope=state.task_scope or state.task_title,
                user_message=current_user_message,
                assistant_message=assistant_message,
                user_feedback=user_feedback,
                extracted_corrections=extracted_corrections,
                tags=tags,
                guidance_applied=state.guidance_applied,
                metadata=combined_metadata,
            )
            state.user_message = current_user_message
            state.latest_assistant_message = assistant_message.strip()
            state.latest_feedback = user_feedback.strip()
            state.metadata = combined_metadata
            self._save_session(state)
            return {
                "session_id": state.session_id,
                "turn_id": turn.id,
                "extracted_corrections": turn.extracted_corrections,
                "inference_trace": combined_metadata.get("inference_trace", {}),
            }
        finally:
            self._unlock_handle(handle)

    def accept_response(
        self,
        session_id: str,
        *,
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
        handle, state = self._load_session_locked(session_id)
        try:
            current_user_message = user_message.strip() or state.user_message
            draft_output = assistant_message.strip() or state.latest_assistant_message
            accepted_text = accepted_output.strip() or draft_output
            entry = self.service.save_episode(
                title=title.strip() or state.task_title or "Untitled task",
                issuer=state.issuer,
                task_type=task_type.strip() or "generic",
                source_text=state.raw_text,
                company_profile=state.company_profile,
                output=output if isinstance(output, dict) else {"accepted_output": accepted_text},
                metadata={
                    **state.metadata,
                    **(metadata if isinstance(metadata, dict) else {}),
                    "session_id": state.session_id,
                },
            )
            if create_training_example:
                self.service.save_training_example(
                    entry.id,
                    format="chat",
                    system_message=state.system_message,
                    user_message=current_user_message,
                    prompt=state.prompt or current_user_message,
                    draft_output=draft_output,
                    rejected_output=draft_output if draft_output and draft_output != accepted_text else "",
                    accepted_output=accepted_text,
                    feedback=feedback.strip() or state.latest_feedback,
                    accepted=True,
                    model_id=model_id,
                    policy_version=policy_version,
                    accepted_by=accepted_by,
                    tags=tags,
                    temperature=temperature,
                )

            state.user_message = current_user_message
            state.linked_entry_id = entry.id
            state.latest_assistant_message = draft_output
            state.status = "closed" if close_session else "open"
            self._save_session(state)
            return {
                "session_id": state.session_id,
                "entry_id": entry.id,
                "status": state.status,
                "accepted_output": accepted_text,
            }
        finally:
            self._unlock_handle(handle)

    def session_summary(self, session_id: str) -> dict[str, Any]:
        state = self._load_session(session_id)
        return asdict(state)
