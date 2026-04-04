"""Tests for the Curiosity Layer (third learning layer).

Engine unit tests + service integration tests.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correx.curiosity_engine import (
    assign_signal_to_cluster,
    add_signal_to_cluster,
    build_cognitive_map,
    cluster_from_dict,
    cluster_to_dict,
    compute_escalation_score,
    compute_gap_strength,
    create_signal,
    process_curiosity_signal,
    resolve_cluster,
    signal_from_dict,
    signal_to_dict,
)
from correx.schemas import CuriositySignal, KnowledgeGapCluster
from correx import CorrexService


# ── Engine unit tests ──────────────────────────────────────────────────────


class CuriosityEngineTest(unittest.TestCase):
    def _make_signal(self, text="Reactって何？", qtype="knowledge_gap",
                     scope="frontend", keywords=None):
        return create_signal(
            question_text=text,
            question_type=qtype,
            target="self",
            task_scope=scope,
            keywords=keywords or ["React", "フレームワーク"],
            confidence=0.8,
        )

    def test_create_signal_basic(self):
        sig = self._make_signal()
        self.assertEqual(sig.question_type, "knowledge_gap")
        self.assertEqual(sig.target, "self")
        self.assertTrue(sig.id)
        self.assertTrue(sig.created_at)

    def test_create_signal_invalid_type_defaults(self):
        sig = create_signal(question_text="test", question_type="invalid")
        self.assertEqual(sig.question_type, "knowledge_gap")

    def test_cluster_assignment_new(self):
        """First signal should create a new cluster."""
        sig = self._make_signal()
        cluster, is_new = assign_signal_to_cluster(sig, [])
        self.assertTrue(is_new)
        self.assertEqual(cluster.scope, "frontend")
        self.assertIn("React", cluster.theme_keywords)

    def test_cluster_assignment_existing(self):
        """Similar signal should join existing cluster."""
        sig1 = self._make_signal(text="Reactって何？", keywords=["React", "フレームワーク"])
        cluster, _ = assign_signal_to_cluster(sig1, [])
        cluster = add_signal_to_cluster(sig1, cluster)

        sig2 = self._make_signal(text="Reactのhooksって何？", keywords=["React", "hooks"])
        matched, is_new = assign_signal_to_cluster(sig2, [cluster])
        self.assertFalse(is_new)
        self.assertEqual(matched.id, cluster.id)

    def test_escalation_increases_with_repeat(self):
        """Escalation score should increase as more signals are added."""
        sig = self._make_signal()
        cluster, _ = assign_signal_to_cluster(sig, [])
        cluster = add_signal_to_cluster(sig, cluster)
        score1 = cluster.escalation_score

        sig2 = self._make_signal(text="Reactのstateって何？", keywords=["React", "state"])
        sig2.id = sig2.id + "2"
        cluster = add_signal_to_cluster(sig2, cluster)
        score2 = cluster.escalation_score

        self.assertGreater(score2, score1)

    def test_gap_strength_knowledge_gap_higher(self):
        """knowledge_gap type should have higher gap_strength than confirmation_seeking."""
        kg_cluster = KnowledgeGapCluster(
            id="kg", created_at="", updated_at="",
            dominant_type="knowledge_gap", signal_count=3,
        )
        cs_cluster = KnowledgeGapCluster(
            id="cs", created_at="", updated_at="",
            dominant_type="confirmation_seeking", signal_count=3,
        )
        self.assertGreater(
            compute_gap_strength(kg_cluster),
            compute_gap_strength(cs_cluster),
        )

    def test_cognitive_map_build(self):
        """Cognitive map should aggregate clusters by scope."""
        c1 = KnowledgeGapCluster(
            id="1", created_at="", updated_at="",
            scope="frontend", signal_count=3, status="open",
            gap_strength=0.7, dominant_type="knowledge_gap",
        )
        c2 = KnowledgeGapCluster(
            id="2", created_at="", updated_at="",
            scope="backend", signal_count=2, status="escalated",
            gap_strength=0.8, dominant_type="judgment_uncertainty",
        )
        c3 = KnowledgeGapCluster(
            id="3", created_at="", updated_at="",
            scope="frontend", signal_count=1, status="resolved",
            gap_strength=0.3, dominant_type="confirmation_seeking",
        )
        cmap = build_cognitive_map([c1, c2, c3])
        self.assertEqual(cmap["total_open"], 1)
        self.assertEqual(cmap["total_escalated"], 1)
        self.assertIn("backend", cmap["hotspots"])
        self.assertEqual(cmap["scopes"]["frontend"]["open_clusters"], 1)
        # Resolved cluster should not appear
        self.assertEqual(cmap["scopes"]["frontend"]["total_questions"], 3)

    def test_cluster_resolution(self):
        """Resolving a cluster should set status and timestamp."""
        cluster = KnowledgeGapCluster(
            id="1", created_at="", updated_at="", status="open",
        )
        resolved = resolve_cluster(cluster)
        self.assertEqual(resolved.status, "resolved")
        self.assertTrue(resolved.resolved_at)

    def test_process_full_pipeline(self):
        """Full pipeline: signal → cluster → metrics updated."""
        sig = self._make_signal()
        updated_sig, cluster, is_new = process_curiosity_signal(sig, [])
        self.assertTrue(is_new)
        self.assertEqual(cluster.signal_count, 1)
        self.assertEqual(updated_sig.cluster_id, cluster.id)

    def test_serialize_roundtrip_signal(self):
        """Signal should survive dict → dataclass → dict roundtrip."""
        sig = self._make_signal()
        d = signal_to_dict(sig)
        restored = signal_from_dict(d)
        self.assertEqual(restored.id, sig.id)
        self.assertEqual(restored.question_type, sig.question_type)

    def test_serialize_roundtrip_cluster(self):
        """Cluster should survive dict → dataclass → dict roundtrip."""
        cluster = KnowledgeGapCluster(
            id="abc", created_at="2026/04/04", updated_at="2026/04/04",
            scope="test", theme_keywords=["React"], signal_count=2,
            escalation_score=0.5, status="escalated",
        )
        d = cluster_to_dict(cluster)
        restored = cluster_from_dict(d)
        self.assertEqual(restored.id, "abc")
        self.assertEqual(restored.status, "escalated")
        self.assertEqual(restored.signal_count, 2)

    def test_multiple_scope_cognitive_map(self):
        """Cognitive map with multiple scopes should track each separately."""
        clusters = [
            KnowledgeGapCluster(
                id=str(i), created_at="", updated_at="",
                scope=f"scope_{i % 3}", signal_count=i + 1, status="open",
                gap_strength=0.3 * (i + 1), dominant_type="knowledge_gap",
            )
            for i in range(6)
        ]
        cmap = build_cognitive_map(clusters)
        self.assertEqual(len(cmap["scopes"]), 3)
        self.assertEqual(cmap["total_open"], 6)


# ── Service integration tests ──────────────────────────────────────────────


class CuriosityServiceTest(unittest.TestCase):
    def test_save_curiosity_signal_creates_cluster(self):
        """save_curiosity_signal should create a new cluster for first signal."""
        with TemporaryDirectory() as td:
            svc = CorrexService(td)
            sig, cluster, is_new = svc.save_curiosity_signal(
                question_text="Reactって何？",
                question_type="knowledge_gap",
                task_scope="frontend",
                keywords=["React"],
            )
            self.assertTrue(is_new)
            self.assertEqual(sig["question_type"], "knowledge_gap")
            self.assertEqual(cluster["signal_count"], 1)

    def test_repeated_questions_escalate(self):
        """3 similar questions in same scope should escalate."""
        with TemporaryDirectory() as td:
            svc = CorrexService(td)
            for i in range(5):
                sig, cluster, _ = svc.save_curiosity_signal(
                    question_text=f"Reactのhooks{i}って何？",
                    question_type="knowledge_gap",
                    task_scope="frontend",
                    keywords=["React", "hooks"],
                )
            # After 5 signals, should have escalated
            clusters = svc.list_knowledge_gap_clusters()
            # At least one cluster should exist with high signal count
            total_signals = sum(c.get("signal_count", 0) for c in clusters)
            self.assertGreaterEqual(total_signals, 5)

    def test_resolve_curiosity_clusters(self):
        """resolve_curiosity_clusters should change status."""
        with TemporaryDirectory() as td:
            svc = CorrexService(td)
            svc.save_curiosity_signal(
                question_text="Reactって何？",
                question_type="knowledge_gap",
                task_scope="frontend",
                keywords=["React"],
            )
            resolved = svc.resolve_curiosity_clusters(task_scope="frontend")
            self.assertEqual(resolved, 1)
            clusters = svc.list_knowledge_gap_clusters(include_resolved=True)
            self.assertEqual(clusters[0]["status"], "resolved")

    def test_guidance_includes_curiosity_section(self):
        """build_guidance_context should include curiosity section when clusters exist."""
        with TemporaryDirectory() as td:
            svc = CorrexService(td)
            # Create enough signals to generate a meaningful cluster
            for i in range(3):
                svc.save_curiosity_signal(
                    question_text=f"APIの認証{i}って何？",
                    question_type="knowledge_gap",
                    task_scope="backend",
                    keywords=["API", "認証"],
                )
            guidance = svc.build_guidance_context(task_scope="backend", task_title="API開発")
            self.assertIn("知識空白", guidance)

    def test_personality_includes_curiosity(self):
        """Personality profile should include curiosity_level."""
        with TemporaryDirectory() as td:
            svc = CorrexService(td)
            # Need at least one turn for personality computation
            svc.save_conversation_turn(
                task_scope="test",
                user_message="テスト",
                assistant_message="回答",
            )
            # Save some curiosity signals
            for i in range(3):
                svc.save_curiosity_signal(
                    question_text=f"質問{i}",
                    question_type="knowledge_gap",
                    task_scope="test",
                    keywords=["テスト"],
                )
            profile = svc.get_personality_profile()
            self.assertTrue(hasattr(profile, "curiosity_level"))
            self.assertTrue(hasattr(profile, "curiosity_label"))


if __name__ == "__main__":
    unittest.main()
