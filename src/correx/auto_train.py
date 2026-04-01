from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .mlx_trainer import (
    MlxLoraTrainingConfig,
    build_test_command,
    build_train_command,
    mlx_lm_available,
    run_command,
)
from .schemas import EpisodeRecord
from .training_dataset import DatasetExportReport, export_mlx_lm_dataset


@dataclass(slots=True)
class AutoTrainReport:
    status: str
    generated_at: str
    dataset_report: dict
    adapter_path: str
    new_episode_ids: list[str]
    train_command: list[str]
    test_command: list[str]
    train_stdout: str = ""
    test_stdout: str = ""
    state_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"trained_episode_ids": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"trained_episode_ids": []}
    if not isinstance(payload, dict):
        return {"trained_episode_ids": []}
    trained_ids = payload.get("trained_episode_ids")
    if not isinstance(trained_ids, list):
        trained_ids = []
    return {
        "trained_episode_ids": [str(entry) for entry in trained_ids if str(entry).strip()],
        "last_run_at": str(payload.get("last_run_at", "")),
        "last_adapter_path": str(payload.get("last_adapter_path", "")),
    }


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_auto_training_cycle(
    entries: list[EpisodeRecord],
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
) -> AutoTrainReport:
    root_dir = Path(output_dir)
    dataset_dir = root_dir / "dataset"
    state_path = root_dir / "auto_train_state.json"
    adapters_root = root_dir / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)

    dataset_report = export_mlx_lm_dataset(
        entries,
        dataset_dir,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        shuffle_seed=shuffle_seed,
        split_strategy=split_strategy,
    )

    state = _load_state(state_path)
    trained_episode_ids = set(state["trained_episode_ids"])
    new_episode_ids = [
        entry_id
        for entry_id in dataset_report.included_episode_ids
        if entry_id not in trained_episode_ids
    ]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    adapter_path = adapters_root / timestamp

    config = training_config or MlxLoraTrainingConfig(
        model=model,
        data_dir=dataset_dir,
        adapter_path=adapter_path,
    )
    config.data_dir = dataset_dir
    config.adapter_path = adapter_path
    config.model = model

    train_command = build_train_command(config)
    test_command = build_test_command(config) if dataset_report.test_examples else []
    generated_at = datetime.now().strftime("%Y/%m/%d %H:%M")

    if not force and len(new_episode_ids) < minimum_new_examples:
        report = AutoTrainReport(
            status="skipped",
            generated_at=generated_at,
            dataset_report=dataset_report.to_dict(),
            adapter_path=str(adapter_path),
            new_episode_ids=new_episode_ids,
            train_command=train_command,
            test_command=test_command,
            state_path=str(state_path),
        )
        _write_state(
            state_path,
            {
                **state,
                "last_run_at": generated_at,
                "last_status": report.status,
            },
        )
        return report

    if dry_run:
        return AutoTrainReport(
            status="dry_run",
            generated_at=generated_at,
            dataset_report=dataset_report.to_dict(),
            adapter_path=str(adapter_path),
            new_episode_ids=new_episode_ids,
            train_command=train_command,
            test_command=test_command,
            state_path=str(state_path),
        )

    if not mlx_lm_available():
        raise RuntimeError(
            "mlx_lm is not installed. Install it with `pip install \"mlx-lm[train]\"` before training."
        )

    train_result = run_command(train_command, cwd=root_dir)
    test_stdout = ""
    if test_command:
        test_result = run_command(test_command, cwd=root_dir)
        test_stdout = test_result.stdout.strip()

    updated_ids = sorted(trained_episode_ids | set(dataset_report.included_episode_ids))
    _write_state(
        state_path,
        {
            "trained_episode_ids": updated_ids,
            "last_run_at": generated_at,
            "last_adapter_path": str(adapter_path),
            "last_status": "trained",
        },
    )

    return AutoTrainReport(
        status="trained",
        generated_at=generated_at,
        dataset_report=dataset_report.to_dict(),
        adapter_path=str(adapter_path),
        new_episode_ids=new_episode_ids,
        train_command=train_command,
        test_command=test_command,
        train_stdout=train_result.stdout.strip(),
        test_stdout=test_stdout,
        state_path=str(state_path),
    )
