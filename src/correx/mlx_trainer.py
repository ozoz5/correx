from __future__ import annotations

import importlib.util
import subprocess  # nosec B404
import sys
from dataclasses import dataclass, field
from pathlib import Path


def mlx_lm_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


@dataclass(slots=True)
class MlxLoraTrainingConfig:
    model: str
    data_dir: str | Path
    adapter_path: str | Path
    iters: int = 600
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    fine_tune_type: str = "lora"
    learning_rate: float | None = None
    num_layers: int | None = None
    mask_prompt: bool = True
    grad_checkpoint: bool = False
    resume_adapter_file: str | Path | None = None
    steps_per_report: int | None = None
    extra_args: list[str] = field(default_factory=list)


def build_train_command(
    config: MlxLoraTrainingConfig,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        config.model,
        "--train",
        "--data",
        str(Path(config.data_dir)),
        "--adapter-path",
        str(Path(config.adapter_path)),
        "--iters",
        str(config.iters),
        "--batch-size",
        str(config.batch_size),
        "--grad-accumulation-steps",
        str(config.grad_accumulation_steps),
        "--fine-tune-type",
        config.fine_tune_type,
    ]

    if config.learning_rate is not None:
        command.extend(["--learning-rate", str(config.learning_rate)])
    if config.num_layers is not None:
        command.extend(["--num-layers", str(config.num_layers)])
    if config.mask_prompt:
        command.append("--mask-prompt")
    if config.grad_checkpoint:
        command.append("--grad-checkpoint")
    if config.resume_adapter_file:
        command.extend(["--resume-adapter-file", str(Path(config.resume_adapter_file))])
    if config.steps_per_report is not None:
        command.extend(["--steps-per-report", str(config.steps_per_report)])
    if config.extra_args:
        command.extend(config.extra_args)
    return command


def build_test_command(
    config: MlxLoraTrainingConfig,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        config.model,
        "--adapter-path",
        str(Path(config.adapter_path)),
        "--data",
        str(Path(config.data_dir)),
        "--test",
    ]


def run_command(command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )
