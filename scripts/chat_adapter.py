from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from claude_pseudo_intelligence import ChatLoopAdapter


def _parse_json_arg(value: str) -> dict:
    if not value.strip():
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin chat-loop adapter for claude_pseudo_intelligence_core.")
    parser.add_argument("--memory-dir", default=str(ROOT_DIR / ".local-memory"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a session and build guidance context.")
    prepare.add_argument("--session-id", default="")
    prepare.add_argument("--task-scope", default="")
    prepare.add_argument("--task-title", default="")
    prepare.add_argument("--issuer", default="")
    prepare.add_argument("--raw-text", default="")
    prepare.add_argument("--raw-text-file", default="")
    prepare.add_argument("--company-profile-json", default="")
    prepare.add_argument("--system-message", default="")
    prepare.add_argument("--user-message", default="")
    prepare.add_argument("--prompt", default="")
    prepare.add_argument("--metadata-json", default="")

    feedback = subparsers.add_parser("feedback", help="Persist corrective user feedback for the current session.")
    feedback.add_argument("--session-id", required=True)
    feedback.add_argument("--user-message", default="")
    feedback.add_argument("--assistant-message", required=True)
    feedback.add_argument("--user-feedback", required=True)
    feedback.add_argument("--extracted-corrections-json", default="")
    feedback.add_argument("--tags-json", default="")
    feedback.add_argument("--metadata-json", default="")

    accept = subparsers.add_parser("accept", help="Persist the accepted response and optional training example.")
    accept.add_argument("--session-id", required=True)
    accept.add_argument("--title", default="")
    accept.add_argument("--task-type", default="generic")
    accept.add_argument("--user-message", default="")
    accept.add_argument("--assistant-message", default="")
    accept.add_argument("--accepted-output", default="")
    accept.add_argument("--feedback", default="")
    accept.add_argument("--output-json", default="")
    accept.add_argument("--model-id", default="")
    accept.add_argument("--policy-version", default="")
    accept.add_argument("--accepted-by", default="human")
    accept.add_argument("--tags-json", default="")
    accept.add_argument("--temperature", type=float)
    accept.add_argument("--metadata-json", default="")
    accept.add_argument("--no-training-example", action="store_true")
    accept.add_argument("--keep-session-open", action="store_true")

    summary = subparsers.add_parser("session-summary", help="Read the current stored session state.")
    summary.add_argument("--session-id", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = ChatLoopAdapter(args.memory_dir)

    if args.command == "prepare":
        result = adapter.prepare(
            session_id=args.session_id,
            task_scope=args.task_scope,
            task_title=args.task_title,
            issuer=args.issuer,
            raw_text=args.raw_text,
            raw_text_file=args.raw_text_file,
            company_profile=_parse_json_arg(args.company_profile_json),
            system_message=args.system_message,
            user_message=args.user_message,
            prompt=args.prompt,
            metadata=_parse_json_arg(args.metadata_json),
        )
    elif args.command == "feedback":
        extracted = json.loads(args.extracted_corrections_json) if args.extracted_corrections_json.strip() else None
        tags = json.loads(args.tags_json) if args.tags_json.strip() else None
        result = adapter.save_feedback(
            args.session_id,
            user_message=args.user_message,
            assistant_message=args.assistant_message,
            user_feedback=args.user_feedback,
            extracted_corrections=extracted,
            tags=tags,
            metadata=_parse_json_arg(args.metadata_json),
        )
    elif args.command == "accept":
        tags = json.loads(args.tags_json) if args.tags_json.strip() else None
        result = adapter.accept_response(
            args.session_id,
            title=args.title,
            task_type=args.task_type,
            user_message=args.user_message,
            assistant_message=args.assistant_message,
            accepted_output=args.accepted_output,
            feedback=args.feedback,
            output=_parse_json_arg(args.output_json),
            create_training_example=not args.no_training_example,
            model_id=args.model_id,
            policy_version=args.policy_version,
            accepted_by=args.accepted_by,
            tags=tags,
            temperature=args.temperature,
            metadata=_parse_json_arg(args.metadata_json),
            close_session=not args.keep_session_open,
        )
    else:
        result = adapter.session_summary(args.session_id)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
