import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correx import CorrexService
from correx.rule_lifecycle import MAX_HISTORY, transition_status
from correx.schemas import PreferenceRule


def _make_rule(status: str = "candidate") -> PreferenceRule:
    return PreferenceRule(
        id="pref-test",
        statement="余白を作れ",
        normalized_statement="余白を作れ",
        instruction="余白を作れ",
        status=status,
        evidence_count=1,
        first_recorded_at="2026/07/01 00:00",
        last_recorded_at="2026/07/01 00:00",
    )


class TransitionStatusTest(unittest.TestCase):
    def test_dict_rule_records_event_and_changes_status(self):
        rule = {"status": "candidate", "statement": "余白を作れ"}
        transition_status(rule, "promoted", "evaluate: success")
        self.assertEqual("promoted", rule["status"])
        self.assertEqual(1, len(rule["status_history"]))
        event = rule["status_history"][0]
        self.assertEqual("candidate", event["from"])
        self.assertEqual("promoted", event["to"])
        self.assertEqual("evaluate: success", event["reason"])
        self.assertIn("writer_pid", event)
        self.assertIn("at", event)

    def test_same_status_is_noop(self):
        rule = {"status": "promoted"}
        transition_status(rule, "promoted", "no change")
        self.assertNotIn("status_history", rule)

    def test_history_capped(self):
        rule = {"status": "a0"}
        for i in range(MAX_HISTORY + 10):
            transition_status(rule, f"a{i + 1}", "churn")
        self.assertEqual(MAX_HISTORY, len(rule["status_history"]))
        self.assertEqual(f"a{MAX_HISTORY + 10}", rule["status"])

    def test_dataclass_rule_records_event(self):
        rule = _make_rule("promoted")
        transition_status(rule, "demoted", "stale_retention")
        self.assertEqual("demoted", rule.status)
        self.assertEqual(1, len(rule.status_history))
        self.assertEqual("promoted", rule.status_history[0]["from"])

    def test_reason_truncated(self):
        rule = {"status": "candidate"}
        transition_status(rule, "dormant", "x" * 500)
        self.assertEqual(120, len(rule["status_history"][0]["reason"]))


class GuidanceAdoptionTest(unittest.TestCase):
    def test_record_and_stats_roundtrip(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir))
            first = service.record_guidance_adoption(
                guidance_id="g-1",
                adopted_rule_ids=["pref-a", "pref-b"],
                rejected_rule_ids=["pref-c"],
                reason="pref-c は無関係なスコープだった",
                task_scope="correx_development",
            )
            self.assertTrue(first["ok"])
            self.assertEqual(2, first["adopted"])
            self.assertEqual(1, first["rejected"])

            second = service.record_guidance_adoption(
                guidance_id="g-2",
                adopted_rule_ids=["pref-a"],
                rejected_rule_ids=["pref-b"],
            )
            stats = second["rule_stats"]
            self.assertEqual({"adopted": 2, "rejected": 0}, stats["pref-a"])
            self.assertEqual({"adopted": 1, "rejected": 1}, stats["pref-b"])

            records = service.history.load_guidance_adoptions()
            self.assertEqual(2, len(records))
            self.assertEqual("g-2", records[0]["guidance_id"])
            self.assertIn("writer_pid", records[0])

    def test_blank_ids_filtered(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir))
            result = service.record_guidance_adoption(
                guidance_id="g-3",
                adopted_rule_ids=["", "  ", "pref-x"],
                rejected_rule_ids=None,
            )
            self.assertEqual(1, result["adopted"])
            self.assertEqual(0, result["rejected"])


if __name__ == "__main__":
    unittest.main()
