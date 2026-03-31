from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from claude_pseudo_intelligence import PseudoIntelligenceService
from claude_pseudo_intelligence.mlx_trainer import MlxLoraTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an automatic MLX-LM LoRA training cycle.")
    parser.add_argument("--memory-dir", default=str(ROOT_DIR / ".local-memory"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "training_artifacts"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--minimum-new-examples", type=int, default=8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--shuffle-seed", type=int, default=7)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--fine-tune-type", default="lora")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--no-mask-prompt", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = PseudoIntelligenceService(args.memory_dir)
    training_config = MlxLoraTrainingConfig(
        model=args.model,
        data_dir=Path(args.output_dir) / "dataset",
        adapter_path=Path(args.output_dir) / "adapters" / "pending",
        iters=args.iters,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        fine_tune_type=args.fine_tune_type,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        mask_prompt=not args.no_mask_prompt,
        grad_checkpoint=args.grad_checkpoint,
    )
    report = service.run_auto_training_cycle(
        model=args.model,
        output_dir=args.output_dir,
        minimum_new_examples=args.minimum_new_examples,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        shuffle_seed=args.shuffle_seed,
        force=args.force,
        dry_run=args.dry_run,
        training_config=training_config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
