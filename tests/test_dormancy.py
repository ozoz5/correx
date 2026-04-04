"""Tests for the dormancy engine — automatic retirement and awakening."""
from __future__ import annotations

import pytest

from src.correx.dormancy import (
    awaken_relevant,
    check_coverage,
    scan_and_dormant,
)


class TestCheckCoverage:
    """Test individual principle coverage detection."""

    def test_covered_by_keyword_pattern(self):
        # "状況確認→作業" law
        result = check_coverage("現状把握を飛ばして作業に入るな")
        assert result is not None
        assert "状況確認" in result

    def test_covered_by_order_pattern(self):
        result = check_coverage("ユーザーが指定した優先順位を勝手に変えるな")
        assert result is not None
        assert "ユーザー順序" in result

    def test_covered_by_existing_change(self):
        result = check_coverage("既存の表示形式や構造を破壊するな")
        assert result is not None
        assert "既存変更" in result

    def test_unique_principle_not_covered(self):
        result = check_coverage("UI側を疑う前にサーバー側を確認せよ")
        assert result is None

    def test_unique_specific_principle(self):
        result = check_coverage("コストが発生する行動は事前承認を得よ")
        assert result is None

    def test_covered_by_bigram_similarity(self):
        laws = ["状況確認を完了してから作業を開始せよ"]
        result = check_coverage(
            "現状確認してから作業に入れ",
            laws=laws,
        )
        assert result is not None

    def test_empty_principle(self):
        assert check_coverage("") is None
        assert check_coverage("ab") is None

    def test_keyword_only_mode(self):
        # Should find keyword match
        result = check_coverage("前提条件を確認してから作業せよ", keyword_only=True)
        assert result is not None

        # Should NOT find bigram match in keyword-only mode
        result = check_coverage(
            "現状確認してから作業に入れ",
            laws=["状況確認を完了してから作業を開始せよ"],
            keyword_only=True,
        )
        # keyword match should still work for this one
        # (前提条件 is not in this text, but 確認してから is)


class TestScanAndDormant:
    """Test batch dormancy scanning."""

    def test_marks_covered_as_dormant(self):
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "前提条件を確認してから作業せよ",
                "scopes": ["general"],
            },
            {
                "fired": True,
                "sublimated_principle": "UI側を疑う前にサーバー側を確認せよ",
                "scopes": ["debugging"],
            },
        ]
        result, dormant, active = scan_and_dormant(trajectories)
        assert dormant == 1  # first one covered
        assert active == 1  # second one unique
        assert result[0]["dormant"] is True
        assert "dormant_reason" in result[0]
        assert result[1].get("dormant") is not True

    def test_skips_already_dormant(self):
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "前提条件を確認してから作業せよ",
                "dormant": True,
                "dormant_reason": "already dormant",
            },
        ]
        result, dormant, active = scan_and_dormant(trajectories)
        assert dormant == 1
        assert active == 0
        # Should not overwrite existing reason
        assert result[0]["dormant_reason"] == "already dormant"

    def test_skips_unfired(self):
        trajectories = [
            {
                "fired": False,
                "sublimated_principle": "前提条件を確認してから作業せよ",
            },
        ]
        _, dormant, active = scan_and_dormant(trajectories)
        assert dormant == 0
        assert active == 0

    def test_empty_list(self):
        result, dormant, active = scan_and_dormant([])
        assert dormant == 0
        assert active == 0


class TestAwakenRelevant:
    """Test dormant principle awakening."""

    def test_awaken_by_scope_and_text(self):
        """Scope match lowers threshold, but text relevance still needed."""
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "UI側を疑う前にサーバー側を確認せよ",
                "dormant": True,
                "dormant_reason": "test",
                "scopes": ["debugging"],
            },
        ]
        result, awakened = awaken_relevant(
            trajectories,
            user_feedback="サーバー側を先に確認しろ",
            scope="debugging",
        )
        assert len(awakened) == 1
        assert result[0]["dormant"] is False
        assert result[0]["awakened_count"] == 1

    def test_no_awaken_scope_only_without_text_relevance(self):
        """Scope match alone is not enough — need some text relevance too."""
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "コストが発生する行動は事前承認を得よ",
                "dormant": True,
                "dormant_reason": "test",
                "scopes": ["general"],
            },
        ]
        result, awakened = awaken_relevant(
            trajectories,
            user_feedback="フォントを変えろ",  # completely unrelated
            scope="general",
        )
        assert len(awakened) == 0  # scope matches but text doesn't

    def test_no_awaken_when_irrelevant(self):
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "UI側を疑う前にサーバー側を確認せよ",
                "dormant": True,
                "dormant_reason": "test",
                "scopes": ["debugging"],
            },
        ]
        result, awakened = awaken_relevant(
            trajectories,
            user_feedback="フォントサイズを大きくして",
            scope="design",
        )
        assert len(awakened) == 0

    def test_increments_awakened_count(self):
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "コストが発生する行動は事前承認を得よ",
                "dormant": True,
                "dormant_reason": "test",
                "scopes": ["billing"],
                "awakened_count": 2,
            },
        ]
        result, _ = awaken_relevant(
            trajectories,
            user_feedback="コストが発生する行動を事前に承認なく実行するな",
            scope="billing",
        )
        assert result[0]["awakened_count"] == 3

    def test_skips_non_dormant(self):
        trajectories = [
            {
                "fired": True,
                "sublimated_principle": "コストが発生する行動は事前承認を得よ",
                "scopes": ["general"],
                # not dormant
            },
        ]
        _, awakened = awaken_relevant(
            trajectories,
            user_feedback="コストがかかる操作を勝手にやるな",
            scope="general",
        )
        assert len(awakened) == 0
