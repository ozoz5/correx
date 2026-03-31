from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from claude_pseudo_intelligence import PseudoIntelligenceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export accepted training examples to MLX-LM jsonl files.")
    parser.add_argument("--memory-dir", default=str(ROOT_DIR / ".local-memory"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "training_artifacts" / "dataset"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--shuffle-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = PseudoIntelligenceService(args.memory_dir)
    report = service.export_training_dataset(
        args.output_dir,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
