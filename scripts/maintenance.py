#!/usr/bin/env python3
"""CORREX filesystem maintenance — rotates logs, prunes old backups.

Usage:
    python scripts/maintenance.py              # dry-run (default)
    python scripts/maintenance.py --apply      # actually delete/rotate
    python scripts/maintenance.py --rotate-log-only
    python scripts/maintenance.py --prune-bak-only
    python scripts/maintenance.py --prune-dir-only

Complements scripts/cleanup_overfitting.py (which handles data quality —
rule deduplication, confidence recalc). This script handles filesystem
hygiene — log rotation, backup pruning.

Defaults to dry-run. Use --apply to execute destructive actions.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path.home() / ".correx"
HOME = Path.home()

# Thresholds
GATE_LOG_ROTATE_BYTES = 1 * 1024 * 1024  # 1 MB
BACKUP_AGE_DAYS = 30  # prune backups older than this

# Patterns for generational backups (non-current)
# .bak is considered current (most recent auto-backup), so we keep it.
# task_*_bak / pre_*_bak / phase*_bak / step*_bak are experimental generations.
GENERATIONAL_BACKUP_PATTERNS = [
    "*.json.task_*_bak",
    "*.json.pre_*_bak",
    "*.json.phase*_bak",
    "*.json.step*_bak",
]

CORREX_BACKUP_DIR_PATTERN = ".correx_backup_*"


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────


def human_size(bytes_val: float) -> str:
    """Human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def age_days(path: Path) -> int:
    """Days since last modification."""
    if not path.exists():
        return -1
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - mtime).days


def dir_size(path: Path) -> int:
    """Recursive directory size in bytes."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


# ─────────────────────────────────────────────────────────────────────
# 1. Gate log rotation
# ─────────────────────────────────────────────────────────────────────


def rotate_gate_log(apply: bool = False) -> dict:
    """Rotate ~/.correx/gate_log.jsonl if over threshold."""
    log_path = BASE_DIR / "gate_log.jsonl"
    if not log_path.exists():
        return {"status": "skip", "reason": "no gate_log.jsonl", "action": "skipped"}

    size = log_path.stat().st_size
    if size < GATE_LOG_ROTATE_BYTES:
        return {
            "status": "skip",
            "reason": f"size {human_size(size)} < threshold {human_size(GATE_LOG_ROTATE_BYTES)}",
            "action": "skipped",
        }

    today = datetime.now().strftime("%Y%m%d")
    rotated_path = BASE_DIR / f"gate_log.jsonl.{today}"

    # If today's rotated file already exists, append an incrementing suffix
    counter = 1
    while rotated_path.exists():
        rotated_path = BASE_DIR / f"gate_log.jsonl.{today}.{counter}"
        counter += 1

    action = "rotated" if apply else "would rotate"
    if apply:
        log_path.rename(rotated_path)
        log_path.touch()

    return {
        "status": "ok",
        "action": action,
        "from": str(log_path),
        "to": str(rotated_path),
        "size": human_size(size),
    }


# ─────────────────────────────────────────────────────────────────────
# 2. Prune generational backups inside ~/.correx
# ─────────────────────────────────────────────────────────────────────


def prune_generational_backups(apply: bool = False) -> dict:
    """Prune old generational backups (.task_*_bak, .pre_*_bak, .phase*_bak, .step*_bak)."""
    candidates: list[Path] = []
    for pattern in GENERATIONAL_BACKUP_PATTERNS:
        candidates.extend(BASE_DIR.glob(pattern))

    to_prune: list[dict] = []
    total_bytes = 0
    for p in candidates:
        if not p.is_file():
            continue
        age = age_days(p)
        size = p.stat().st_size
        if age >= BACKUP_AGE_DAYS:
            to_prune.append({"path": str(p), "age_days": age, "size": size})
            total_bytes += size

    action = "pruned" if apply else "would prune"
    if apply:
        for item in to_prune:
            Path(item["path"]).unlink()

    return {
        "status": "ok",
        "action": action,
        "count": len(to_prune),
        "total_size": human_size(total_bytes),
        "items": to_prune,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. Prune old ~/.correx_backup_* directories
# ─────────────────────────────────────────────────────────────────────


def prune_backup_dirs(apply: bool = False) -> dict:
    """Prune old ~/.correx_backup_* directories."""
    candidates = list(HOME.glob(CORREX_BACKUP_DIR_PATTERN))

    to_prune: list[dict] = []
    total_bytes = 0
    for p in candidates:
        if not p.is_dir():
            continue
        age = age_days(p)
        size = dir_size(p)
        if age >= BACKUP_AGE_DAYS:
            to_prune.append({"path": str(p), "age_days": age, "size": size})
            total_bytes += size

    action = "pruned" if apply else "would prune"
    if apply:
        for item in to_prune:
            shutil.rmtree(item["path"])

    return {
        "status": "ok",
        "action": action,
        "count": len(to_prune),
        "total_size": human_size(total_bytes),
        "items": to_prune,
    }


# ─────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────


def format_report(
    rotate_result: dict,
    prune_bak_result: dict,
    prune_dir_result: dict,
    apply: bool,
) -> str:
    header = "=" * 60
    mode = "APPLY (destructive)" if apply else "DRY-RUN (no changes)"
    out = [
        header,
        f"CORREX maintenance — {mode}",
        header,
        "",
        "[1] Gate log rotation",
    ]
    if rotate_result["status"] == "skip":
        out.append(f"    skipped: {rotate_result.get('reason', '')}")
    else:
        out.append(
            f"    {rotate_result['action']}: {rotate_result['from']} "
            f"({rotate_result['size']}) → {rotate_result['to']}"
        )

    out += [
        "",
        "[2] Generational backup pruning (*.task_*_bak, *.pre_*_bak, *.phase*_bak, *.step*_bak)",
        f"    {prune_bak_result['action']}: {prune_bak_result['count']} files, "
        f"total {prune_bak_result['total_size']}",
    ]
    for item in prune_bak_result["items"][:10]:  # show top 10
        out.append(
            f"      - {item['path']} ({item['age_days']}d old, "
            f"{human_size(item['size'])})"
        )
    if len(prune_bak_result["items"]) > 10:
        out.append(f"      ... and {len(prune_bak_result['items']) - 10} more")

    out += [
        "",
        "[3] Backup directory pruning (~/.correx_backup_*)",
        f"    {prune_dir_result['action']}: {prune_dir_result['count']} dirs, "
        f"total {prune_dir_result['total_size']}",
    ]
    for item in prune_dir_result["items"]:
        out.append(
            f"      - {item['path']} ({item['age_days']}d old, "
            f"{human_size(item['size'])})"
        )

    out += ["", header]
    if not apply:
        out.append("Run with --apply to actually execute the above actions.")
        out.append(header)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="CORREX filesystem maintenance (log rotation + backup pruning)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the actions (default: dry-run)",
    )
    parser.add_argument(
        "--rotate-log-only",
        action="store_true",
        help="Only run gate log rotation",
    )
    parser.add_argument(
        "--prune-bak-only",
        action="store_true",
        help="Only prune generational file backups",
    )
    parser.add_argument(
        "--prune-dir-only",
        action="store_true",
        help="Only prune backup directories",
    )
    args = parser.parse_args()

    # Decide which steps to run
    any_only = args.rotate_log_only or args.prune_bak_only or args.prune_dir_only
    run_rotate = args.rotate_log_only or not any_only
    run_prune_bak = args.prune_bak_only or not any_only
    run_prune_dir = args.prune_dir_only or not any_only

    skip_result = {
        "status": "skip",
        "action": "skipped",
        "reason": "disabled by flags",
        "count": 0,
        "total_size": "0B",
        "items": [],
    }

    rotate_result = rotate_gate_log(apply=args.apply) if run_rotate else skip_result
    prune_bak_result = (
        prune_generational_backups(apply=args.apply) if run_prune_bak else skip_result
    )
    prune_dir_result = prune_backup_dirs(apply=args.apply) if run_prune_dir else skip_result

    report = format_report(rotate_result, prune_bak_result, prune_dir_result, apply=args.apply)
    print(report)


if __name__ == "__main__":
    main()
