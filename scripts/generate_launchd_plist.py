from __future__ import annotations

import argparse
import plistlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a launchd plist for nightly auto-training.")
    parser.add_argument("--output-plist", default=str(ROOT_DIR / "launchd" / "com.claude-pseudo-intelligence.auto-train.plist"))
    parser.add_argument("--memory-dir", default=str(ROOT_DIR / ".local-memory"))
    parser.add_argument("--artifacts-dir", default=str(ROOT_DIR / "training_artifacts"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--hour", type=int, default=3)
    parser.add_argument("--minute", type=int, default=15)
    parser.add_argument("--minimum-new-examples", type=int, default=8)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_plist(args: argparse.Namespace) -> dict:
    artifacts_dir = Path(args.artifacts_dir)
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    program_arguments = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "auto_train.py"),
        "--memory-dir",
        str(Path(args.memory_dir)),
        "--output-dir",
        str(artifacts_dir),
        "--model",
        args.model,
        "--minimum-new-examples",
        str(args.minimum_new_examples),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
    ]
    if args.force:
        program_arguments.append("--force")
    if args.dry_run:
        program_arguments.append("--dry-run")

    return {
        "Label": "com.claude-pseudo-intelligence.auto-train",
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(ROOT_DIR),
        "RunAtLoad": False,
        "StartCalendarInterval": {
            "Hour": args.hour,
            "Minute": args.minute,
        },
        "StandardOutPath": str(logs_dir / "auto-train.stdout.log"),
        "StandardErrorPath": str(logs_dir / "auto-train.stderr.log"),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_plist)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_plist(args)
    with output_path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=False)
    print(output_path)


if __name__ == "__main__":
    main()
