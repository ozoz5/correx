"""Tests for build_guidance_context's ``verbose`` flag (Medium-mode diet).

verbose=False (default, added 2026-04-25): trims policy analogy, caps
prohibition laws at top 15 by coverage, slims rule entries inside
inference_trace. verbose=True: full legacy payload.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correx import CorrexService
from correx.schemas import Policy


def _seed_active_policy(service: CorrexService) -> None:
    policy = Policy(
        id="test-policy-analogy-check",
        title="テストポリシー",
        core="核の内容",
        why="理由の内容",
        analogy="類推の文章 — verbose=False で消えるべき",
        opposite="反対の文",
        limits="限界の文",
        maturity="active",
        evidence_count=99,
    )
    service.history.write_policies([policy])


def _seed_prohibition_laws(service: CorrexService, count: int = 30) -> None:
    # Varied covers length so top-k sort has something to sort on.
    laws = [
        {
            "law": f"禁止法理{i:02d}",
            "covers": list(range(count - i)),  # law 00 has most, law N-1 fewest
        }
        for i in range(count)
    ]
    service.history.write_ghost_universal_laws(laws)


def _seed_conversation_turn(service: CorrexService) -> None:
    service.save_conversation_turn(
        task_scope="proposal_summary",
        user_message="提案書要約を書け",
        assistant_message="弊社目線の提案を書いた",
        user_feedback="貴社視点で書け。具体業務を入れろ。",
    )


class PolicyAnalogyDropTest(unittest.TestCase):
    def test_verbose_false_omits_analogy_line(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            _seed_active_policy(service)
            text = service.build_guidance_context(
                task_title="提案書要約",
                raw_text="物流管理案件",
                task_scope="proposal_summary",
                verbose=False,
            )
            self.assertIn("核: 核の内容", text)
            self.assertIn("なぜ: 理由の内容", text)
            self.assertIn("反対: 反対の文", text)
            self.assertIn("限界: 限界の文", text)
            self.assertNotIn("類推:", text)

    def test_verbose_true_includes_analogy_line(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            _seed_active_policy(service)
            text = service.build_guidance_context(
                task_title="提案書要約",
                raw_text="物流管理案件",
                task_scope="proposal_summary",
                verbose=True,
            )
            self.assertIn("類推: 類推の文章", text)


class ProhibitionLawCapTest(unittest.TestCase):
    def test_verbose_false_caps_prohibition_laws_at_15(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            _seed_prohibition_laws(service, count=30)
            text = service.build_guidance_context(
                task_title="t",
                raw_text="t",
                task_scope="s",
                verbose=False,
            )
            prohibition_lines = [
                line for line in text.splitlines() if "禁止法理" in line and ". " in line
            ]
            self.assertLessEqual(len(prohibition_lines), 15)
            # Top-k by covers should keep law 00 (largest covers).
            self.assertTrue(any("禁止法理00" in line for line in prohibition_lines))
            # And should drop law 29 (smallest covers).
            self.assertFalse(any("禁止法理29" in line for line in prohibition_lines))

    def test_verbose_true_keeps_all_prohibition_laws(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            _seed_prohibition_laws(service, count=30)
            text = service.build_guidance_context(
                task_title="t",
                raw_text="t",
                task_scope="s",
                verbose=True,
            )
            prohibition_lines = [
                line for line in text.splitlines() if "禁止法理" in line and ". " in line
            ]
            self.assertGreaterEqual(len(prohibition_lines), 30)


class RuleEntrySlimTest(unittest.TestCase):
    def _get_trace(self, service: CorrexService, *, verbose: bool) -> dict:
        _seed_conversation_turn(service)
        result = service.build_guidance_context(
            task_title="提案書要約",
            raw_text="物流管理案件",
            task_scope="proposal_summary",
            return_trace=True,
            verbose=verbose,
        )
        return result["inference_trace"]

    def test_verbose_false_slims_selected_rule_entries(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            trace = self._get_trace(service, verbose=False)
            selected = trace["selected_rules"]
            if not selected:
                self.skipTest("no selected rules produced from seed turn")
            for entry in selected:
                self.assertEqual(entry["reason"], "")
                self.assertLessEqual(len(entry.get("applies_when_tags", [])), 3)
                self.assertLessEqual(len(entry.get("top_context_keywords", [])), 3)
                self.assertLessEqual(len(entry.get("top_context_tags", [])), 3)
                matches = entry.get("latent_context_matches")
                self.assertIsInstance(matches, dict)
                self.assertIn("count", matches)
                self.assertIn("max_posterior", matches)

    def test_verbose_true_keeps_full_rule_entries(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            trace = self._get_trace(service, verbose=True)
            selected = trace["selected_rules"]
            if not selected:
                self.skipTest("no selected rules produced from seed turn")
            # At least one entry keeps a non-empty reason (legacy shape).
            self.assertTrue(any(entry.get("reason", "") for entry in selected))
            # latent_context_matches stays a list in verbose=True.
            for entry in selected:
                matches = entry.get("latent_context_matches")
                self.assertIsInstance(matches, list)


class PayloadSizeComparisonTest(unittest.TestCase):
    def test_verbose_false_payload_is_smaller(self):
        with TemporaryDirectory() as temp_dir:
            service = CorrexService(Path(temp_dir) / "memory")
            _seed_active_policy(service)
            _seed_prohibition_laws(service, count=30)
            _seed_conversation_turn(service)

            medium = service.build_guidance_context(
                task_title="提案書要約",
                raw_text="物流管理案件",
                task_scope="proposal_summary",
                return_trace=True,
                verbose=False,
            )
            legacy = service.build_guidance_context(
                task_title="提案書要約",
                raw_text="物流管理案件",
                task_scope="proposal_summary",
                return_trace=True,
                verbose=True,
            )
            medium_size = len(
                json.dumps(
                    {
                        "guidance_context": medium["guidance_context"],
                        "inference_trace": medium["inference_trace"],
                    },
                    ensure_ascii=False,
                )
            )
            legacy_size = len(
                json.dumps(
                    {
                        "guidance_context": legacy["guidance_context"],
                        "inference_trace": legacy["inference_trace"],
                    },
                    ensure_ascii=False,
                )
            )
            self.assertLess(medium_size, legacy_size)


if __name__ == "__main__":
    unittest.main()
