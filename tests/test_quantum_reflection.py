"""Tests for Quantum Save reflection layer (save = load 同時実行).

CorrexService.compute_quantum_reflection は save_conversation_turn の戻り値に
類似過去ターン・再発カウント・メタ警告を埋め込む反射層。Hook や能動 Pull に
頼らず、書き込みのついでに過去を見せる構造。
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correx import CorrexService


class QuantumReflectionTest(unittest.TestCase):
    def test_reflection_empty_when_first_turn(self):
        """初回保存時は類似過去がないので similar_past_turns は空."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            turn = service.save_conversation_turn(
                task_scope="test",
                user_message="最初のメッセージ",
                user_feedback="OK",
                reaction_score_override=0.7,
            )
            reflection = service.compute_quantum_reflection(turn)
            self.assertIsInstance(reflection, dict)
            self.assertEqual(reflection.get("similar_past_turns", []), [])

    def test_reflection_finds_similar(self):
        """類似テキストの過去ターンが検出される."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            service.save_conversation_turn(
                task_scope="public_strategy",
                user_message="そろそろ公開について考えたい",
                user_feedback="公開はまだ早い、保留してくれ",
                reaction_score_override=0.2,
            )
            turn = service.save_conversation_turn(
                task_scope="public_strategy",
                user_message="公開の話を再度持ち出したい",
                user_feedback="まだ公開保留",
                reaction_score_override=0.2,
            )
            reflection = service.compute_quantum_reflection(turn)
            similar = reflection.get("similar_past_turns", [])
            self.assertGreaterEqual(len(similar), 1)
            self.assertGreater(similar[0]["similarity"], 0.15)

    def test_reflection_scope_recurrence_count(self):
        """同じ task_scope の再発カウントが取れる."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            for i in range(3):
                service.save_conversation_turn(
                    task_scope="public_strategy",
                    user_message=f"公開について {i}",
                    user_feedback="まだ早い",
                    reaction_score_override=0.2,
                )
            turn = service.save_conversation_turn(
                task_scope="public_strategy",
                user_message="また公開の話",
                user_feedback="保留",
                reaction_score_override=0.2,
            )
            reflection = service.compute_quantum_reflection(turn)
            self.assertGreaterEqual(reflection.get("scope_recurrence_count", 0), 3)

    def test_reflection_excludes_self(self):
        """自分自身は類似ターンに含まれない."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            turn = service.save_conversation_turn(
                task_scope="test",
                user_message="単独メッセージ",
                user_feedback="単独",
                reaction_score_override=0.7,
            )
            reflection = service.compute_quantum_reflection(turn)
            similar_ids = [s["turn_id"] for s in reflection.get("similar_past_turns", [])]
            self.assertNotIn(turn.id, similar_ids)

    def test_reflection_empty_query_returns_empty(self):
        """user_message も user_feedback も空なら空 dict を返す."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            turn = service.save_conversation_turn(
                task_scope="test",
                user_message="",
                user_feedback="",
                reaction_score_override=0.5,
            )
            reflection = service.compute_quantum_reflection(turn)
            self.assertEqual(reflection, {})

    def test_reflection_meta_warning_triggers(self):
        """高類似ターン (>=0.3) が 2 件以上で meta_warning が出る."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            # 同じパターンを 3 回保存 (互いに完全一致 → similarity 1.0)
            for i in range(3):
                service.save_conversation_turn(
                    task_scope="commit_inflation",
                    user_message="commit を 19 件積んだので公開戦略を考えたい",
                    user_feedback="commit 数と使用感を混同するな",
                    reaction_score_override=0.2,
                )
            turn = service.save_conversation_turn(
                task_scope="commit_inflation",
                user_message="commit を 19 件積んだので公開戦略を考えたい",
                user_feedback="commit 数と使用感を混同するな",
                reaction_score_override=0.2,
            )
            reflection = service.compute_quantum_reflection(turn)
            high_sim = [
                s for s in reflection.get("similar_past_turns", [])
                if s["similarity"] >= 0.3
            ]
            self.assertGreaterEqual(len(high_sim), 2)
            self.assertIn("meta_warning", reflection)

    def test_reflection_similarity_threshold_filters(self):
        """類似度が threshold 未満のターンはフィルタされる."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            service.save_conversation_turn(
                task_scope="topic_a",
                user_message="全く違う話題",
                user_feedback="関係ないコメント",
                reaction_score_override=0.5,
            )
            turn = service.save_conversation_turn(
                task_scope="topic_b",
                user_message="量子セーブの設計",
                user_feedback="鏡の哲学",
                reaction_score_override=0.5,
            )
            reflection = service.compute_quantum_reflection(
                turn, similarity_threshold=0.9
            )
            # 高い閾値では類似なしの想定
            self.assertEqual(reflection.get("similar_past_turns", []), [])

    def test_reflection_limit_caps_results(self):
        """limit 引数で返す類似ターン数を制限できる."""
        with TemporaryDirectory() as tmp:
            service = CorrexService(tmp)
            for i in range(5):
                service.save_conversation_turn(
                    task_scope="test",
                    user_message="同じメッセージ",
                    user_feedback="同じフィードバック",
                    reaction_score_override=0.5,
                )
            turn = service.save_conversation_turn(
                task_scope="test",
                user_message="同じメッセージ",
                user_feedback="同じフィードバック",
                reaction_score_override=0.5,
            )
            reflection = service.compute_quantum_reflection(turn, limit=2)
            self.assertLessEqual(len(reflection.get("similar_past_turns", [])), 2)


if __name__ == "__main__":
    unittest.main()
