"""Correction analytics: frequency, policy effectiveness, and reporting."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


# Default data directory
_DEFAULT_BASE_DIR = Path.home() / ".correx"


class CorrectionAnalytics:
    """
    Analyses CORREX correction data from stored JSON files.

    Reads conversation_history.json, preference_rules.json, and policies.json
    to produce frequency counts, policy effectiveness metrics, and text reports.

    Usage:
        analytics = CorrectionAnalytics()
        freq = analytics.get_correction_frequency(days=30)
        eff = analytics.get_policy_effectiveness("pol-some-id")
        print(analytics.format_report(days=14))
    """

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR

    # ------------------------------------------------------------------
    # Data loading (lazy, no caching — keeps it simple)
    # ------------------------------------------------------------------

    def _load_json(self, filename: str) -> dict:
        """Load a JSON file from base_dir. Returns empty dict on failure."""
        path = self.base_dir / filename
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _load_turns(self) -> list[dict]:
        data = self._load_json("conversation_history.json")
        return data.get("items", [])

    def _load_rules(self) -> list[dict]:
        data = self._load_json("preference_rules.json")
        return data.get("items", [])

    def _load_policies(self) -> list[dict]:
        data = self._load_json("policies.json")
        return data.get("items", [])

    # ------------------------------------------------------------------
    # Parse helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_recorded_at(raw: str) -> datetime | None:
        """Parse recorded_at which may be 'YYYY/MM/DD HH:MM' or ISO-8601."""
        if not raw:
            return None
        for fmt in (
            "%Y/%m/%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f%z",
        ):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        # Last resort: take first 16 chars as YYYY/MM/DD HH:MM
        try:
            return datetime.strptime(raw[:16], "%Y/%m/%d %H:%M")
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_correction_frequency(self, days: int = 30) -> dict:
        """
        Daily correction counts for the last *days* days.

        Returns:
            {
                "period_days": int,
                "total_turns": int,
                "total_with_corrections": int,
                "daily_counts": { "YYYY-MM-DD": int, ... },
                "scope_counts": { scope: int, ... },
            }
        """
        turns = self._load_turns()
        now = datetime.now()
        cutoff = now - timedelta(days=days)

        daily: Counter[str] = Counter()
        scopes: Counter[str] = Counter()
        total = 0
        with_corrections = 0

        for turn in turns:
            dt = self._parse_recorded_at(turn.get("recorded_at", ""))
            if dt is None or dt < cutoff:
                continue
            total += 1
            corrections = turn.get("extracted_corrections", [])
            if corrections:
                with_corrections += 1
            day_key = dt.strftime("%Y-%m-%d")
            daily[day_key] += 1
            scope = turn.get("task_scope", "") or "unknown"
            scopes[scope] += 1

        # Fill in zero-count days for a complete series
        daily_sorted: dict[str, int] = {}
        for i in range(days):
            day = (cutoff + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            daily_sorted[day] = daily.get(day, 0)

        return {
            "period_days": days,
            "total_turns": total,
            "total_with_corrections": with_corrections,
            "daily_counts": daily_sorted,
            "scope_counts": dict(scopes.most_common()),
        }

    def get_policy_effectiveness(self, policy_id: str) -> dict:
        """
        Measure how well a policy's rules are followed vs violated.

        Cross-references:
          policy -> source_rule_ids -> preference_rules -> source_turn_ids -> turns

        A turn's reaction_score determines follow/violate:
          >= 0.6  -> followed (user was satisfied)
          < 0.4   -> violated (user was dissatisfied)
          0.4-0.6 -> neutral

        Returns:
            {
                "policy_id": str,
                "policy_title": str,
                "rule_count": int,
                "turn_count": int,
                "followed": int,
                "violated": int,
                "neutral": int,
                "follow_rate": float,   # 0.0 - 1.0
                "avg_reaction_score": float,
                "per_rule": [ { "rule_id", "turn_count", "avg_score" }, ... ],
            }
        """
        policies = self._load_policies()
        rules = self._load_rules()
        turns = self._load_turns()

        # Find the target policy
        policy = None
        for p in policies:
            if p.get("id") == policy_id:
                policy = p
                break
        if policy is None:
            return {
                "policy_id": policy_id,
                "policy_title": "",
                "error": f"Policy '{policy_id}' not found",
            }

        # Build turn lookup by id
        turn_by_id: dict[str, dict] = {t["id"]: t for t in turns if "id" in t}

        # Build rule lookup
        rule_by_id: dict[str, dict] = {r["id"]: r for r in rules if "id" in r}

        source_rule_ids = policy.get("source_rule_ids", [])
        followed = 0
        violated = 0
        neutral = 0
        scores: list[float] = []
        per_rule: list[dict] = []

        for rule_id in source_rule_ids:
            rule = rule_by_id.get(rule_id)
            if rule is None:
                continue
            turn_ids = rule.get("source_turn_ids", [])
            rule_scores: list[float] = []
            for tid in turn_ids:
                turn = turn_by_id.get(tid)
                if turn is None:
                    continue
                rs = turn.get("reaction_score")
                if rs is None:
                    continue
                rs = float(rs)
                rule_scores.append(rs)
                scores.append(rs)
                if rs >= 0.6:
                    followed += 1
                elif rs < 0.4:
                    violated += 1
                else:
                    neutral += 1
            per_rule.append({
                "rule_id": rule_id,
                "turn_count": len(rule_scores),
                "avg_score": (
                    round(sum(rule_scores) / len(rule_scores), 3)
                    if rule_scores else 0.0
                ),
            })

        total = followed + violated + neutral
        return {
            "policy_id": policy_id,
            "policy_title": policy.get("title", ""),
            "rule_count": len(source_rule_ids),
            "turn_count": total,
            "followed": followed,
            "violated": violated,
            "neutral": neutral,
            "follow_rate": round(followed / total, 3) if total else 0.0,
            "avg_reaction_score": (
                round(sum(scores) / len(scores), 3) if scores else 0.0
            ),
            "per_rule": per_rule,
        }

    def format_report(self, *, days: int = 30) -> str:
        """
        Human-readable text report combining frequency and policy data.

        Args:
            days: lookback window for correction frequency.

        Returns:
            Multi-line string suitable for terminal display.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("CORREX Correction Analytics Report")
        lines.append("=" * 60)

        # --- Frequency section ---
        freq = self.get_correction_frequency(days=days)
        lines.append("")
        lines.append(f"--- Correction Frequency (last {days} days) ---")
        lines.append(f"Total turns:            {freq['total_turns']}")
        lines.append(f"Turns with corrections: {freq['total_with_corrections']}")

        daily = freq["daily_counts"]
        active_days = sum(1 for v in daily.values() if v > 0)
        lines.append(f"Active days:            {active_days} / {days}")

        if freq["scope_counts"]:
            lines.append("")
            lines.append("Top scopes:")
            for scope, count in list(freq["scope_counts"].items())[:5]:
                lines.append(f"  {scope}: {count}")

        # --- Policy effectiveness section ---
        policies = self._load_policies()
        active_policies = [p for p in policies if p.get("maturity") == "active"]

        if active_policies:
            lines.append("")
            lines.append("--- Policy Effectiveness ---")
            for policy in active_policies:
                pid = policy.get("id", "")
                eff = self.get_policy_effectiveness(pid)
                title = eff.get("policy_title", pid)
                total = eff.get("turn_count", 0)
                if total == 0:
                    lines.append(f"  [{pid}] {title}: no linked turns")
                    continue
                rate = eff.get("follow_rate", 0.0)
                avg = eff.get("avg_reaction_score", 0.0)
                lines.append(
                    f"  [{pid}] {title}"
                )
                lines.append(
                    f"    turns={total}  followed={eff['followed']}  "
                    f"violated={eff['violated']}  neutral={eff['neutral']}  "
                    f"follow_rate={rate:.1%}  avg_score={avg:.2f}"
                )
        else:
            lines.append("")
            lines.append("--- Policy Effectiveness ---")
            lines.append("  No active policies found.")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
