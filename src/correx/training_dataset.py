from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .schemas import EpisodeRecord, TrainingExample


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2).strip()


def _build_chat_record(example: TrainingExample) -> dict | None:
    accepted_output = _normalize_text(example.accepted_output)
    if not accepted_output:
        return None

    messages: list[dict] = []
    system_message = _normalize_text(example.system_message)
    user_message = _normalize_text(example.user_message)
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    if not messages:
        return None

    messages.append({"role": "assistant", "content": accepted_output})
    return {"messages": messages}


def _build_completion_record(example: TrainingExample) -> dict | None:
    prompt = _normalize_text(example.prompt)
    accepted_output = _normalize_text(example.accepted_output)
    if not prompt or not accepted_output:
        return None
    return {"prompt": prompt, "completion": accepted_output}


def build_preference_record(example: TrainingExample) -> dict | None:
    prompt = _normalize_text(example.prompt or example.user_message)
    accepted_output = _normalize_text(example.accepted_output)
    rejected_output = _normalize_text(example.rejected_output or example.draft_output)
    if not prompt or not accepted_output or not rejected_output:
        return None
    return {
        "prompt": prompt,
        "chosen": accepted_output,
        "rejected": rejected_output,
    }


def build_mlx_record(entry: EpisodeRecord) -> dict | None:
    example = entry.training_example
    if example is None or not example.accepted:
        return None
    if example.format == "completions":
        return _build_completion_record(example)
    return _build_chat_record(example)


def collect_trainable_entries(entries: list[EpisodeRecord]) -> list[tuple[str, dict]]:
    collected: list[tuple[str, dict]] = []
    for entry in entries:
        record = build_mlx_record(entry)
        if record is None:
            continue
        collected.append((entry.id, record))
    return collected


def _split_records(
    records: list[tuple[str, dict]],
    *,
    valid_ratio: float,
    test_ratio: float,
    shuffle_seed: int,
    split_strategy: str,
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]], list[tuple[str, dict]]]:
    ordered = list(records)
    if split_strategy == "chronological":
        ordered = list(reversed(ordered))
    if split_strategy == "random":
        import random

        random.Random(shuffle_seed).shuffle(ordered)
    total = len(ordered)
    if total < 5:
        return ordered, [], []

    valid_count = max(1, round(total * valid_ratio)) if valid_ratio > 0 else 0
    test_count = max(1, round(total * test_ratio)) if test_ratio > 0 else 0

    while total - valid_count - test_count < 1:
        if valid_count >= test_count and valid_count > 0:
            valid_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_end = total - valid_count - test_count
    valid_end = total - test_count
    return (
        ordered[:train_end],
        ordered[train_end:valid_end],
        ordered[valid_end:],
    )


@dataclass(slots=True)
class DatasetExportReport:
    output_dir: str
    generated_at: str
    total_examples: int
    train_examples: int
    valid_examples: int
    test_examples: int
    preference_examples: int
    split_strategy: str
    included_episode_ids: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def export_mlx_lm_dataset(
    entries: list[EpisodeRecord],
    output_dir: str | Path,
    *,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle_seed: int = 7,
    split_strategy: str = "chronological",
) -> DatasetExportReport:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    collected = collect_trainable_entries(entries)
    if not collected:
        raise ValueError("No accepted training examples are available")

    train_split, valid_split, test_split = _split_records(
        collected,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        shuffle_seed=shuffle_seed,
        split_strategy=split_strategy,
    )

    _write_jsonl(destination / "train.jsonl", [record for _, record in train_split])
    if valid_split:
        _write_jsonl(destination / "valid.jsonl", [record for _, record in valid_split])
    else:
        (destination / "valid.jsonl").unlink(missing_ok=True)
    if test_split:
        _write_jsonl(destination / "test.jsonl", [record for _, record in test_split])
    else:
        (destination / "test.jsonl").unlink(missing_ok=True)

    preference_records = [
        record
        for entry in entries
        if entry.training_example is not None
        for record in [build_preference_record(entry.training_example)]
        if record is not None
    ]
    if preference_records:
        _write_jsonl(destination / "preference.jsonl", preference_records)
    else:
        (destination / "preference.jsonl").unlink(missing_ok=True)

    report = DatasetExportReport(
        output_dir=str(destination),
        generated_at=datetime.now().strftime("%Y/%m/%d %H:%M"),
        total_examples=len(collected),
        train_examples=len(train_split),
        valid_examples=len(valid_split),
        test_examples=len(test_split),
        preference_examples=len(preference_records),
        split_strategy=split_strategy,
        included_episode_ids=[entry_id for entry_id, _ in collected],
    )
    (destination / "manifest.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
