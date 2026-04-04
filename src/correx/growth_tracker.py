"""Growth measurement: compare AI output quality with and without guidance."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class GrowthRun:
    """One execution of a task — either baseline (no guidance) or guided."""

    run_id: str
    case_id: str
    guidance_applied: bool
    guidance_text: str
    output: str
    score: float  # 0.0 - 1.0
    ran_at: str


@dataclass
class GrowthRecord:
    """
    A before/after pair for one GrowthCase.
    delta > 0  → guidance helped
    delta == 0 → no difference
    delta < 0  → guidance hurt
    """

    record_id: str
    case_id: str
    case_title: str
    baseline_score: float
    guided_score: float
    delta: float
    baseline_run: GrowthRun
    guided_run: GrowthRun
    recorded_at: str


class GrowthTracker:
    """
    Records evidence of AI growth by comparing scores before and after
    guidance is applied to the same task.

    The caller is responsible for running the AI and scoring the output.
    GrowthTracker only stores and analyses the results.

    Usage:
        tracker = GrowthTracker(memory_dir)

        # Run the task twice, score each output yourself (0.0-1.0)
        record = tracker.record(
            case_id="npc-dialogue-quality",
            case_title="NPCセリフの自然さ",
            task_scope="game-writing",
            baseline_output="いらっしゃい、何かお探しで？",
            baseline_score=0.4,
            guided_output="ほう…珍しいものをお探しで。名前は聞かなくていい。",
            guided_score=0.85,
            guidance_text="[applied guidance here]",
        )

        print(record.delta)   # +0.45  →  growing
        tracker.summary()     # overall growth across all cases
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.growth_dir = self.base_dir / "growth"
        self.growth_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        case_id: str,
        case_title: str,
        task_scope: str = "",
        baseline_output: str,
        baseline_score: float,
        guided_output: str,
        guided_score: float,
        guidance_text: str = "",
    ) -> GrowthRecord:
        """
        Record one before/after measurement.

        baseline_* = result when run WITHOUT guidance
        guided_*   = result when run WITH guidance
        scores are 0.0 (worst) to 1.0 (best)
        """
        now = datetime.now(timezone.utc).isoformat()

        baseline_run = GrowthRun(
            run_id=str(uuid.uuid4()),
            case_id=case_id,
            guidance_applied=False,
            guidance_text="",
            output=baseline_output,
            score=float(baseline_score),
            ran_at=now,
        )
        guided_run = GrowthRun(
            run_id=str(uuid.uuid4()),
            case_id=case_id,
            guidance_applied=True,
            guidance_text=guidance_text,
            output=guided_output,
            score=float(guided_score),
            ran_at=now,
        )
        record = GrowthRecord(
            record_id=str(uuid.uuid4()),
            case_id=case_id,
            case_title=case_title,
            baseline_score=float(baseline_score),
            guided_score=float(guided_score),
            delta=float(guided_score) - float(baseline_score),
            baseline_run=baseline_run,
            guided_run=guided_run,
            recorded_at=now,
        )
        self._save(record)
        return record

    def load_history(self, *, case_id: str | None = None) -> list[GrowthRecord]:
        """Load all growth records, optionally filtered by case_id."""
        records: list[GrowthRecord] = []
        for path in sorted(self.growth_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Skip non-growth records (e.g., session feedback files)
            if "record_id" not in data or "case_id" not in data:
                continue
            if case_id and data.get("case_id") != case_id:
                continue
            try:
                records.append(self._deserialize(data))
            except (KeyError, TypeError):
                continue
        return records

    def trend(self, case_id: str) -> list[dict]:
        """
        Return score history for one case, sorted oldest → newest.

        Each entry: { recorded_at, baseline, guided, delta }
        Growing AI shows delta increasing over time.
        """
        history = self.load_history(case_id=case_id)
        return [
            {
                "recorded_at": r.recorded_at,
                "baseline": r.baseline_score,
                "guided": r.guided_score,
                "delta": r.delta,
            }
            for r in sorted(history, key=lambda r: r.recorded_at)
        ]

    def summary(self) -> dict:
        """
        Overall growth summary across all cases.

        trend label:
          "growing"   → average delta > +0.05
          "flat"      → average delta between -0.05 and +0.05
          "degrading" → average delta < -0.05
        """
        all_records = self.load_history()
        if not all_records:
            return {"total_runs": 0, "average_delta": 0.0, "cases": []}

        by_case: dict[str, list[GrowthRecord]] = {}
        for r in all_records:
            by_case.setdefault(r.case_id, []).append(r)

        cases = []
        for cid, records in by_case.items():
            sorted_records = sorted(records, key=lambda r: r.recorded_at)
            latest = sorted_records[-1]
            avg_delta = sum(r.delta for r in records) / len(records)
            cases.append({
                "case_id": cid,
                "case_title": latest.case_title,
                "runs": len(records),
                "latest_baseline": latest.baseline_score,
                "latest_guided": latest.guided_score,
                "latest_delta": latest.delta,
                "average_delta": round(avg_delta, 4),
                "trend": (
                    "growing" if avg_delta > 0.05
                    else "flat" if avg_delta >= -0.05
                    else "degrading"
                ),
            })

        all_deltas = [r.delta for r in all_records]
        avg = sum(all_deltas) / len(all_deltas)
        return {
            "total_runs": len(all_records),
            "average_delta": round(avg, 4),
            "overall_trend": (
                "growing" if avg > 0.05
                else "flat" if avg >= -0.05
                else "degrading"
            ),
            "cases": sorted(cases, key=lambda c: c["case_id"]),
        }

    def auto_record_from_turns(
        self,
        turns: list,
        *,
        min_turns_per_side: int = 2,
    ) -> list[GrowthRecord]:
        """
        Automatically record growth from accumulated ConversationTurns.

        Groups turns by task_scope, then compares:
          - turns where guidance_applied=False  →  baseline scores
          - turns where guidance_applied=True   →  guided scores

        Records a new GrowthRecord only when both sides have enough data
        AND the average scores have changed since the last recorded delta.

        Returns the list of newly recorded GrowthRecords.
        """
        from collections import defaultdict

        # group by scope
        by_scope: dict[str, list] = defaultdict(list)
        for turn in turns:
            scope = getattr(turn, "task_scope", "") or "generic"
            by_scope[scope].append(turn)

        new_records: list[GrowthRecord] = []

        for scope, scope_turns in by_scope.items():
            baseline_turns = [
                t for t in scope_turns
                if not getattr(t, "guidance_applied", False)
                and getattr(t, "reaction_score", None) is not None
            ]
            guided_turns = [
                t for t in scope_turns
                if getattr(t, "guidance_applied", False)
                and getattr(t, "reaction_score", None) is not None
            ]

            if len(baseline_turns) < min_turns_per_side:
                continue
            if len(guided_turns) < min_turns_per_side:
                continue

            baseline_avg = sum(t.reaction_score for t in baseline_turns) / len(baseline_turns)
            guided_avg = sum(t.reaction_score for t in guided_turns) / len(guided_turns)

            # Skip if already recorded this exact delta for this scope
            existing = self.load_history(case_id=f"auto-{scope}")
            if existing:
                last = sorted(existing, key=lambda r: r.recorded_at)[-1]
                if (
                    abs(last.baseline_score - baseline_avg) < 0.01
                    and abs(last.guided_score - guided_avg) < 0.01
                ):
                    continue  # nothing new to record

            # Pick representative outputs for traceability
            baseline_output = getattr(baseline_turns[-1], "assistant_message", "") or ""
            guided_output = getattr(guided_turns[-1], "assistant_message", "") or ""
            guidance_text = getattr(guided_turns[-1], "user_message", "") or ""

            record = self.record(
                case_id=f"auto-{scope}",
                case_title=f"{scope} (auto)",
                task_scope=scope,
                baseline_output=baseline_output,
                baseline_score=round(baseline_avg, 4),
                guided_output=guided_output,
                guided_score=round(guided_avg, 4),
                guidance_text=guidance_text,
            )
            new_records.append(record)

        return new_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, record: GrowthRecord) -> None:
        target = self.growth_dir / f"{record.record_id}.json"
        tmp = target.with_suffix(".tmp")
        content = json.dumps(self._serialize(record), ensure_ascii=False, indent=2)
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)

    def _serialize(self, record: GrowthRecord) -> dict:
        return {
            "record_id": record.record_id,
            "case_id": record.case_id,
            "case_title": record.case_title,
            "baseline_score": record.baseline_score,
            "guided_score": record.guided_score,
            "delta": record.delta,
            "recorded_at": record.recorded_at,
            "baseline_run": {
                "run_id": record.baseline_run.run_id,
                "guidance_applied": False,
                "guidance_text": "",
                "output": record.baseline_run.output,
                "score": record.baseline_run.score,
                "ran_at": record.baseline_run.ran_at,
            },
            "guided_run": {
                "run_id": record.guided_run.run_id,
                "guidance_applied": True,
                "guidance_text": record.guided_run.guidance_text,
                "output": record.guided_run.output,
                "score": record.guided_run.score,
                "ran_at": record.guided_run.ran_at,
            },
        }

    def _deserialize(self, data: dict) -> GrowthRecord:
        br = data.get("baseline_run")
        gr = data.get("guided_run")
        if not br or not gr:
            # Legacy record without run details — synthesize from top-level
            br = br or {
                "run_id": f"{data.get('record_id', 'unknown')}_base",
                "output": data.get("baseline_output", ""),
                "score": data.get("baseline_score", 0.0),
                "ran_at": data.get("recorded_at", ""),
            }
            gr = gr or {
                "run_id": f"{data.get('record_id', 'unknown')}_guided",
                "guidance_text": data.get("guidance_text", ""),
                "output": data.get("guided_output", ""),
                "score": data.get("guided_score", 0.0),
                "ran_at": data.get("recorded_at", ""),
            }
        return GrowthRecord(
            record_id=data["record_id"],
            case_id=data["case_id"],
            case_title=data["case_title"],
            baseline_score=data["baseline_score"],
            guided_score=data["guided_score"],
            delta=data["delta"],
            recorded_at=data["recorded_at"],
            baseline_run=GrowthRun(
                run_id=br["run_id"],
                case_id=data["case_id"],
                guidance_applied=False,
                guidance_text="",
                output=br["output"],
                score=br["score"],
                ran_at=br["ran_at"],
            ),
            guided_run=GrowthRun(
                run_id=gr["run_id"],
                case_id=data["case_id"],
                guidance_applied=True,
                guidance_text=gr["guidance_text"],
                output=gr["output"],
                score=gr["score"],
                ran_at=gr["ran_at"],
            ),
        )
