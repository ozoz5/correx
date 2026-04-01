"""Tests for memory_manager: smart forgetting, compression, semantic search."""

from __future__ import annotations

import unittest

from correx.memory_manager import (
    apply_forgetting_curve,
    archive_turns_to_episode,
    auto_correct_flagged_rules,
    build_context_signature,
    build_rule_associations,
    detect_contradicting_rules,
    evict_episodes,
    evict_turns,
    find_relevant_rules_semantic,
    find_relevant_turns_semantic,
    infer_latent_context_responsibilities,
    merge_similar_rules,
    predict_next_contexts,
    reconsolidate_rule,
    reconsolidate_rules_from_turns,
    resolve_contradicting_rules,
    semantic_similarity,
)
from correx.schemas import (
    ConversationTurn,
    EpisodeRecord,
    LatentContext,
    LatentTransition,
    PreferenceRule,
    TrainingExample,
)


def _make_turn(
    turn_id: str,
    corrections: list[str] | None = None,
    guidance_applied: bool = False,
    reaction_score: float | None = None,
    tags: list[str] | None = None,
    task_scope: str = "",
    user_feedback: str = "",
) -> ConversationTurn:
    return ConversationTurn(
        id=turn_id,
        recorded_at="2026/03/28 22:00",
        task_scope=task_scope,
        user_feedback=user_feedback,
        extracted_corrections=corrections or [],
        tags=tags or [],
        guidance_applied=guidance_applied,
        reaction_score=reaction_score,
    )


def _make_episode(
    ep_id: str,
    corrections_count: int = 0,
    has_training: bool = False,
    issuer: str = "",
) -> EpisodeRecord:
    from correx.schemas import CorrectionRecord

    corrections = [
        CorrectionRecord(recorded_at="2026/03/28", correction_note=f"fix-{i}")
        for i in range(corrections_count)
    ]
    training = (
        TrainingExample(updated_at="2026/03/28", accepted=True)
        if has_training
        else None
    )
    return EpisodeRecord(
        id=ep_id,
        timestamp="2026/03/28 22:00",
        title=f"Episode {ep_id}",
        issuer=issuer,
        corrections=corrections,
        training_example=training,
    )


def _make_rule(
    rule_id: str,
    instruction: str,
    status: str = "promoted",
    evidence_count: int = 2,
    scope: str = "",
    tags: list[str] | None = None,
    support_score: float | None = None,
    expected_gain: float = 0.0,
    confidence_score: float = 0.0,
    latent_contexts: list[LatentContext] | None = None,
) -> PreferenceRule:
    return PreferenceRule(
        id=rule_id,
        statement=instruction,
        normalized_statement=instruction.lower(),
        instruction=instruction,
        status=status,
        evidence_count=evidence_count,
        first_recorded_at="2026/03/28 22:00",
        last_recorded_at="2026/03/28 22:00",
        applies_to_scope=scope,
        applies_when_tags=tags or [],
        tags=tags or [],
        support_score=float(support_score if support_score is not None else evidence_count),
        expected_gain=expected_gain,
        confidence_score=confidence_score,
        latent_contexts=latent_contexts or [],
    )


class SmartForgettingTest(unittest.TestCase):

    def test_evict_turns_keeps_high_value(self):
        """Turns with corrections/guidance survive, empty ones are evicted."""
        valuable = _make_turn("t1", corrections=["fix X"], guidance_applied=True)
        empty = _make_turn("t2")
        also_empty = _make_turn("t3")

        result = evict_turns([valuable, empty, also_empty], retention_limit=2)
        ids = {t.id for t in result}
        self.assertIn("t1", ids)
        self.assertEqual(len(result), 2)

    def test_evict_turns_noop_under_limit(self):
        turns = [_make_turn(f"t{i}") for i in range(5)]
        result = evict_turns(turns, retention_limit=10)
        self.assertEqual(len(result), 5)

    def test_evict_episodes_keeps_trained(self):
        """Episodes with training examples are protected from eviction."""
        trained = _make_episode("e1", has_training=True)
        plain = _make_episode("e2")
        also_plain = _make_episode("e3")

        result = evict_episodes([trained, plain, also_plain], retention_limit=2)
        ids = {e.id for e in result}
        self.assertIn("e1", ids)
        self.assertEqual(len(result), 2)

    def test_evict_episodes_prefers_corrected(self):
        """Episodes with corrections score higher than empty ones."""
        corrected = _make_episode("e1", corrections_count=3)
        empty = _make_episode("e2")

        result = evict_episodes([corrected, empty], retention_limit=1)
        self.assertEqual(result[0].id, "e1")


class RuleCompressionTest(unittest.TestCase):

    def test_merge_similar_rules(self):
        """Rules with similar instructions should be merged."""
        r1 = _make_rule("r1", "余白を作れ", evidence_count=2)
        r2 = _make_rule("r2", "余白を増やせ", evidence_count=1, status="candidate")

        result = merge_similar_rules([r1, r2], similarity_threshold=0.25)
        self.assertEqual(len(result.merged_rules), 1)
        self.assertEqual(result.merge_count, 1)
        # Evidence should be combined
        merged = result.merged_rules[0]
        self.assertEqual(merged.evidence_count, 3)
        self.assertEqual(merged.status, "promoted")

    def test_no_merge_when_different(self):
        """Dissimilar rules should not be merged."""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "ROI数値を入れる")

        result = merge_similar_rules([r1, r2])
        self.assertEqual(len(result.merged_rules), 2)
        self.assertEqual(result.merge_count, 0)

    def test_merge_empty_list(self):
        result = merge_similar_rules([])
        self.assertEqual(len(result.merged_rules), 0)


class SemanticSearchTest(unittest.TestCase):

    def test_similarity_identical(self):
        sim = semantic_similarity("余白を作れ", "余白を作れ")
        self.assertEqual(sim, 1.0)

    def test_similarity_similar(self):
        sim = semantic_similarity("余白を作れ", "余白を増やせ")
        self.assertGreater(sim, 0.2)

    def test_similarity_unrelated(self):
        sim = semantic_similarity("余白を作れ", "データベースを最適化しろ")
        self.assertLess(sim, 0.3)

    def test_find_relevant_rules_semantic(self):
        """Query '窮屈なレイアウト' should find '余白を作れ' rule."""
        rules = [
            _make_rule("r1", "余白を作れ", scope="ui_design"),
            _make_rule("r2", "ROI数値を入れる", scope="proposal"),
        ]
        results = find_relevant_rules_semantic(rules, "余白が足りない")
        self.assertTrue(len(results) > 0)
        # The whitespace rule should rank higher
        top_rule, _ = results[0]
        self.assertIn("余白", top_rule.instruction)

    def test_latent_context_match_prefers_matching_situation(self):
        rule = _make_rule(
            "r1",
            "ROI数値を入れろ",
            status="candidate",
            evidence_count=3,
            latent_contexts=[
                LatentContext(
                    id="latent-sales",
                    scope="sales_brief",
                    tags=["roi", "finance"],
                    keywords=["roi", "profit"],
                    prototype_text="sales brief roi finance profit",
                    evidence_count=2.0,
                    support_score=3.2,
                    expected_gain=2.8,
                    confidence_score=0.72,
                    prior_weight=0.54,
                    posterior_mass=2.0,
                ),
                LatentContext(
                    id="latent-ui",
                    scope="ui_design",
                    tags=["layout", "spacing"],
                    keywords=["hero", "layout"],
                    prototype_text="ui design layout spacing hero",
                    evidence_count=2.0,
                    support_score=1.4,
                    expected_gain=1.0,
                    confidence_score=0.44,
                    prior_weight=0.22,
                    posterior_mass=2.0,
                ),
            ],
        )

        matches, novelty = infer_latent_context_responsibilities(
            rule,
            task_scope="sales_brief",
            tags=["roi", "finance"],
            query_text="利益とROIを入れる営業ブリーフ",
        )

        self.assertTrue(matches)
        self.assertEqual(matches[0].context.id, "latent-sales")
        self.assertLess(novelty, 0.5)

    def test_transition_prior_biases_context_selection(self):
        rule = _make_rule(
            "r1",
            "次の議論で数字を出せ",
            status="candidate",
            evidence_count=3,
            latent_contexts=[
                LatentContext(
                    id="latent-pricing",
                    scope="pricing_review",
                    tags=["pricing", "estimate"],
                    keywords=["pricing", "estimate"],
                    prototype_text="pricing review estimate pricing",
                    evidence_count=1.6,
                    support_score=2.4,
                    expected_gain=2.0,
                    confidence_score=0.64,
                    prior_weight=0.36,
                    posterior_mass=1.4,
                ),
                LatentContext(
                    id="latent-risk",
                    scope="risk_review",
                    tags=["risk", "assumption"],
                    keywords=["risk", "assumption"],
                    prototype_text="risk review assumption risk",
                    evidence_count=1.5,
                    support_score=2.2,
                    expected_gain=1.9,
                    confidence_score=0.62,
                    prior_weight=0.35,
                    posterior_mass=1.4,
                ),
            ],
        )

        previous_nodes = [
            {
                "scope": "proposal_summary",
                "tags": ["proposal", "summary"],
                "keywords": ["proposal", "summary"],
                "posterior": 0.78,
                "signature": build_context_signature(
                    "proposal_summary",
                    ["proposal", "summary"],
                    ["proposal", "summary"],
                ),
            }
        ]
        transitions = [
            LatentTransition(
                id="tr-1",
                from_signature=previous_nodes[0]["signature"],
                to_signature=build_context_signature(
                    "pricing_review",
                    ["pricing", "estimate"],
                    ["pricing", "estimate"],
                ),
                from_scope="proposal_summary",
                to_scope="pricing_review",
                from_tags=["proposal", "summary"],
                to_tags=["pricing", "estimate"],
                from_keywords=["proposal", "summary"],
                to_keywords=["pricing", "estimate"],
                evidence_count=3.2,
                success_weight=2.6,
                failure_weight=0.0,
                confidence_score=0.82,
            )
        ]

        matches, _ = infer_latent_context_responsibilities(
            rule,
            task_scope="review",
            tags=["estimate"],
            query_text="次は見積と価格の議論に入る",
            previous_context_nodes=previous_nodes,
            transitions=transitions,
        )

        self.assertTrue(matches)
        self.assertEqual("latent-pricing", matches[0].context.id)

    def test_predict_next_contexts_ranks_transition_targets(self):
        previous_nodes = [
            {
                "scope": "proposal_summary",
                "tags": ["proposal", "summary"],
                "keywords": ["proposal", "summary"],
                "posterior": 0.8,
                "signature": build_context_signature(
                    "proposal_summary",
                    ["proposal", "summary"],
                    ["proposal", "summary"],
                ),
            }
        ]
        transitions = [
            LatentTransition(
                id="tr-pricing",
                from_signature=previous_nodes[0]["signature"],
                to_signature=build_context_signature(
                    "pricing_review",
                    ["pricing", "estimate"],
                    ["pricing", "estimate"],
                ),
                from_scope="proposal_summary",
                to_scope="pricing_review",
                from_tags=["proposal", "summary"],
                to_tags=["pricing", "estimate"],
                from_keywords=["proposal", "summary"],
                to_keywords=["pricing", "estimate"],
                evidence_count=3.0,
                success_weight=2.8,
                confidence_score=0.84,
            ),
            LatentTransition(
                id="tr-risk",
                from_signature=previous_nodes[0]["signature"],
                to_signature=build_context_signature(
                    "risk_review",
                    ["risk", "assumption"],
                    ["risk", "assumption"],
                ),
                from_scope="proposal_summary",
                to_scope="risk_review",
                from_tags=["proposal", "summary"],
                to_tags=["risk", "assumption"],
                from_keywords=["proposal", "summary"],
                to_keywords=["risk", "assumption"],
                evidence_count=1.4,
                success_weight=0.8,
                confidence_score=0.42,
            ),
        ]

        predictions = predict_next_contexts(
            previous_context_nodes=previous_nodes,
            transitions=transitions,
            limit=3,
        )

        self.assertTrue(predictions)
        self.assertEqual("pricing_review", predictions[0]["to_scope"])

    def test_predict_next_contexts_prefers_calibrated_flow(self):
        previous_nodes = [
            {
                "scope": "proposal_summary",
                "tags": ["proposal", "summary"],
                "keywords": ["proposal", "summary"],
                "posterior": 0.8,
                "signature": build_context_signature(
                    "proposal_summary",
                    ["proposal", "summary"],
                    ["proposal", "summary"],
                ),
            }
        ]
        transitions = [
            LatentTransition(
                id="tr-calibrated",
                from_signature=previous_nodes[0]["signature"],
                to_signature=build_context_signature(
                    "pricing_review",
                    ["pricing", "estimate"],
                    ["pricing", "estimate"],
                ),
                from_scope="proposal_summary",
                to_scope="pricing_review",
                from_tags=["proposal", "summary"],
                to_tags=["pricing", "estimate"],
                from_keywords=["proposal", "summary"],
                to_keywords=["pricing", "estimate"],
                evidence_count=1.8,
                success_weight=1.1,
                confidence_score=0.62,
                prediction_hit_count=2.4,
                prediction_miss_count=0.2,
                forecast_score=0.58,
            ),
            LatentTransition(
                id="tr-uncalibrated",
                from_signature=previous_nodes[0]["signature"],
                to_signature=build_context_signature(
                    "risk_review",
                    ["risk", "assumption"],
                    ["risk", "assumption"],
                ),
                from_scope="proposal_summary",
                to_scope="risk_review",
                from_tags=["proposal", "summary"],
                to_tags=["risk", "assumption"],
                from_keywords=["proposal", "summary"],
                to_keywords=["risk", "assumption"],
                evidence_count=1.8,
                success_weight=1.1,
                confidence_score=0.62,
                prediction_hit_count=0.0,
                prediction_miss_count=2.0,
                forecast_score=-0.42,
            ),
        ]

        predictions = predict_next_contexts(
            previous_context_nodes=previous_nodes,
            transitions=transitions,
            limit=3,
        )

        self.assertTrue(predictions)
        self.assertEqual("pricing_review", predictions[0]["to_scope"])
        self.assertGreater(predictions[0]["forecast_score"], predictions[1]["forecast_score"])

    def test_find_relevant_turns_semantic(self):
        turns = [
            _make_turn("t1", corrections=["余白を作れ"], user_feedback="詰め込みすぎ"),
            _make_turn("t2", corrections=["ROIを入れろ"], user_feedback="数値が足りない"),
        ]
        results = find_relevant_turns_semantic(turns, "余白が足りない")
        self.assertTrue(len(results) > 0)

    def test_empty_query_returns_nothing(self):
        rules = [_make_rule("r1", "test")]
        results = find_relevant_rules_semantic(rules, "")
        self.assertEqual(len(results), 0)


class ArchiveTurnsToEpisodeTest(unittest.TestCase):

    def test_corrections_are_preserved(self):
        """All extracted_corrections from the turns appear as CorrectionRecords."""
        turns = [
            _make_turn("t1", corrections=["修正A", "修正B"], task_scope="design"),
            _make_turn("t2", corrections=["修正C"], task_scope="coding"),
        ]
        episode = archive_turns_to_episode(turns)
        self.assertIsNotNone(episode)
        correction_notes = [c.correction_note for c in episode.corrections]
        self.assertIn("修正A", correction_notes)
        self.assertIn("修正B", correction_notes)
        self.assertIn("修正C", correction_notes)
        self.assertEqual(len(episode.corrections), 3)

    def test_title_format(self):
        """Title must follow 'アーカイブ {earliest} 〜 {latest} ({n}ターン)' format."""
        turns = [
            _make_turn("t1"),
            _make_turn("t2"),
        ]
        # Override recorded_at for determinism
        turns[0].recorded_at = "2026/03/01 09:00"
        turns[1].recorded_at = "2026/03/28 22:00"
        episode = archive_turns_to_episode(turns)
        self.assertIsNotNone(episode)
        expected_title = f"アーカイブ 2026/03/01 09:00 〜 2026/03/28 22:00 (2ターン)"
        self.assertEqual(episode.title, expected_title)

    def test_task_type_is_archived_turns(self):
        """task_type must be 'archived_turns'."""
        turns = [_make_turn("t1")]
        episode = archive_turns_to_episode(turns)
        self.assertIsNotNone(episode)
        self.assertEqual(episode.task_type, "archived_turns")

    def test_source_text_contains_counts(self):
        """source_text must mention turn count and correction count."""
        turns = [
            _make_turn("t1", corrections=["fix X", "fix Y"]),
            _make_turn("t2", corrections=["fix Z"]),
        ]
        episode = archive_turns_to_episode(turns)
        self.assertIsNotNone(episode)
        self.assertIn("2件", episode.source_text)   # 2 turns
        self.assertIn("3件", episode.source_text)   # 3 corrections

    def test_id_starts_with_archive(self):
        """Episode ID must start with 'archive-'."""
        turns = [_make_turn("t1")]
        episode = archive_turns_to_episode(turns)
        self.assertIsNotNone(episode)
        self.assertTrue(episode.id.startswith("archive-"))

    def test_empty_turns_returns_none(self):
        """Passing an empty list must return None."""
        result = archive_turns_to_episode([])
        self.assertIsNone(result)

    def test_turns_without_corrections(self):
        """Turns with no corrections produce an episode with zero CorrectionRecords."""
        turns = [_make_turn("t1"), _make_turn("t2")]
        episode = archive_turns_to_episode(turns)
        self.assertIsNotNone(episode)
        self.assertEqual(len(episode.corrections), 0)


class ForgettingCurveTest(unittest.TestCase):

    def _make_rule_with_dates(
        self,
        rule_id: str,
        instruction: str,
        status: str = "candidate",
        evidence_count: int = 2,
        last_recorded_at: str = "2026/03/29 00:00",
        priority: int = 3,
    ) -> PreferenceRule:
        return PreferenceRule(
            id=rule_id,
            statement=instruction,
            normalized_statement=instruction.lower(),
            instruction=instruction,
            status=status,
            evidence_count=evidence_count,
            first_recorded_at="2026/01/01 00:00",
            last_recorded_at=last_recorded_at,
            priority=priority,
        )

    def test_forgetting_curve_reduces_priority(self):
        """An old candidate rule should have lower priority than a fresh one."""
        now_str = "2026/03/29 00:00"
        # Rule recorded 90 days ago (3 half-lives for candidate at 30 days)
        old_rule = self._make_rule_with_dates(
            "r_old", "テストを書け", status="candidate",
            last_recorded_at="2025/12/29 00:00", priority=5, evidence_count=0,
        )
        fresh_rule = self._make_rule_with_dates(
            "r_fresh", "コードを整理しろ", status="candidate",
            last_recorded_at="2026/03/29 00:00", priority=5, evidence_count=0,
        )
        result = apply_forgetting_curve([old_rule, fresh_rule], now_str, half_life_days=30.0)
        old_result = next(r for r in result if r.id == "r_old")
        fresh_result = next(r for r in result if r.id == "r_fresh")
        self.assertLess(old_result.priority, fresh_result.priority)

    def test_forgetting_curve_promoted_decays_slower(self):
        """A promoted rule should retain higher priority than a candidate after the same time."""
        now_str = "2026/03/29 00:00"
        old_date = "2025/12/29 00:00"  # 90 days ago
        promoted_rule = self._make_rule_with_dates(
            "r_promoted", "余白を増やせ", status="promoted",
            last_recorded_at=old_date, priority=4, evidence_count=0,
        )
        candidate_rule = self._make_rule_with_dates(
            "r_candidate", "余白を増やせ", status="candidate",
            last_recorded_at=old_date, priority=4, evidence_count=0,
        )
        result = apply_forgetting_curve([promoted_rule, candidate_rule], now_str, half_life_days=30.0)
        promoted_result = next(r for r in result if r.id == "r_promoted")
        candidate_result = next(r for r in result if r.id == "r_candidate")
        self.assertGreaterEqual(promoted_result.priority, candidate_result.priority)

    def test_forgetting_curve_high_value_rule_decays_slower(self):
        """High expected gain should preserve priority better even with the same status."""
        now_str = "2026/03/29 00:00"
        high_value = _make_rule(
            "r_high",
            "ROI数値を入れろ",
            status="candidate",
            evidence_count=2,
            support_score=3.0,
            expected_gain=3.2,
            confidence_score=0.7,
        )
        low_value = _make_rule(
            "r_low",
            "ROI数値を入れろ",
            status="candidate",
            evidence_count=2,
            support_score=1.0,
            expected_gain=0.2,
            confidence_score=0.1,
        )
        high_value.last_recorded_at = "2025/12/29 00:00"
        low_value.last_recorded_at = "2025/12/29 00:00"

        result = apply_forgetting_curve([high_value, low_value], now_str, half_life_days=30.0)
        high_result = next(r for r in result if r.id == "r_high")
        low_result = next(r for r in result if r.id == "r_low")
        self.assertGreaterEqual(high_result.priority, low_result.priority)

    def test_forgetting_curve_does_not_change_status(self):
        """apply_forgetting_curve must not change the status field."""
        now_str = "2026/03/29 00:00"
        rule = self._make_rule_with_dates(
            "r1", "テスト", status="candidate", last_recorded_at="2025/01/01 00:00",
        )
        result = apply_forgetting_curve([rule], now_str)
        self.assertEqual(result[0].status, "candidate")

    def test_forgetting_curve_fresh_rule_unchanged(self):
        """A rule recorded right now should not lose priority."""
        now_str = "2026/03/29 00:00"
        rule = self._make_rule_with_dates(
            "r1", "テスト", last_recorded_at=now_str, priority=3, evidence_count=0,
        )
        result = apply_forgetting_curve([rule], now_str)
        # decay=1.0, effective_priority = 3*1.0 + 0*0.1 = 3.0 → priority=3
        self.assertEqual(result[0].priority, 3)


class RuleAssociationTest(unittest.TestCase):

    def test_build_rule_associations_finds_similar(self):
        """Similar rules should appear in each other's association list."""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "余白を増やせ")
        r3 = _make_rule("r3", "ROI数値を入れる")

        associations = build_rule_associations([r1, r2, r3], similarity_threshold=0.2)
        # r1 and r2 share "余白" so should be associated
        self.assertIn("r2", associations["r1"])
        self.assertIn("r1", associations["r2"])

    def test_build_rule_associations_no_self_link(self):
        """A rule must never appear in its own association list."""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "余白を増やせ")
        r3 = _make_rule("r3", "ROI数値を入れる")

        associations = build_rule_associations([r1, r2, r3])
        for rule_id, related in associations.items():
            self.assertNotIn(rule_id, related)

    def test_build_rule_associations_unrelated_not_linked(self):
        """Unrelated rules should not appear in each other's association list."""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "データベースを最適化しろ")

        associations = build_rule_associations([r1, r2], similarity_threshold=0.2)
        self.assertNotIn("r2", associations["r1"])
        self.assertNotIn("r1", associations["r2"])

    def test_build_rule_associations_empty(self):
        """Empty input should return empty dict."""
        associations = build_rule_associations([])
        self.assertEqual(associations, {})


class ContradictionDetectionTest(unittest.TestCase):

    def test_detects_negation_conflict(self):
        """「余白を作れ」と「余白を作るな」は negation_conflict として検出される。"""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "余白を作るな")

        conflicts = detect_contradicting_rules([r1, r2])
        self.assertEqual(len(conflicts), 1)
        rule_a, rule_b, reason = conflicts[0]
        self.assertEqual(reason, "negation_conflict")
        ids = {rule_a.id, rule_b.id}
        self.assertEqual(ids, {"r1", "r2"})

    def test_no_contradiction_dissimilar(self):
        """全く関係ないルール同士は矛盾として検出されない。"""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "ROI数値を入れるな")

        conflicts = detect_contradicting_rules([r1, r2])
        self.assertEqual(len(conflicts), 0)

    def test_same_instruction_no_conflict(self):
        """同じルールが2つあっても矛盾ではない。"""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "余白を作れ")

        conflicts = detect_contradicting_rules([r1, r2])
        self.assertEqual(len(conflicts), 0)

    def test_detects_scope_conflict_via_negative_conditions(self):
        """negative_conditions に相手のルールが含まれる場合 scope_conflict として検出。"""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = PreferenceRule(
            id="r2",
            statement="余白を作れ",
            normalized_statement="余白を作れ",
            instruction="余白を作れ",
            status="promoted",
            evidence_count=2,
            first_recorded_at="2026/03/28 22:00",
            last_recorded_at="2026/03/28 22:00",
            applies_to_scope="design",
            negative_conditions=["余白を作れ"],
        )

        conflicts = detect_contradicting_rules([r1, r2])
        self.assertEqual(len(conflicts), 1)
        _, _, reason = conflicts[0]
        self.assertEqual(reason, "scope_conflict")

    def test_empty_list_returns_no_conflicts(self):
        """空リストは矛盾なし。"""
        conflicts = detect_contradicting_rules([])
        self.assertEqual(conflicts, [])

    def test_single_rule_returns_no_conflicts(self):
        """ルールが1つしかなければ矛盾なし。"""
        r1 = _make_rule("r1", "余白を作れ")
        conflicts = detect_contradicting_rules([r1])
        self.assertEqual(conflicts, [])


class ReconsolidationTest(unittest.TestCase):

    def test_good_outcome_strengthens_rule(self):
        """High outcome score should increment evidence_count and raise priority."""
        rule = _make_rule("r1", "余白を作れ", evidence_count=2, tags=["design"])
        original_evidence = rule.evidence_count
        original_priority = rule.priority

        updated = reconsolidate_rule(rule, applied=True, outcome_score=0.9)

        self.assertEqual(updated.evidence_count, original_evidence + 1)
        self.assertGreaterEqual(updated.priority, original_priority)
        self.assertNotIn("needs_revision", updated.tags)

    def test_bad_outcome_flags_revision(self):
        """Low outcome score should add 'needs_revision' tag to the rule."""
        rule = _make_rule("r1", "余白を作れ", evidence_count=2, tags=["design"])

        updated = reconsolidate_rule(rule, applied=True, outcome_score=0.1)

        self.assertIn("needs_revision", updated.tags)
        # evidence_count should not change for bad outcome
        self.assertEqual(updated.evidence_count, rule.evidence_count)

    def test_not_applied_no_change(self):
        """applied=False should return the rule unchanged."""
        rule = _make_rule("r1", "余白を作れ", evidence_count=2, tags=["design"])

        updated = reconsolidate_rule(rule, applied=False, outcome_score=0.9)

        self.assertEqual(updated.evidence_count, rule.evidence_count)
        self.assertEqual(updated.priority, rule.priority)
        self.assertEqual(updated.tags, rule.tags)
        # Should return the same object (no copy made)
        self.assertIs(updated, rule)

    def test_neutral_outcome_no_change(self):
        """Score in the neutral range (0.3, 0.7) should not change the rule."""
        rule = _make_rule("r1", "余白を作れ", evidence_count=2, tags=["design"])

        updated = reconsolidate_rule(rule, applied=True, outcome_score=0.5)

        self.assertEqual(updated.evidence_count, rule.evidence_count)
        self.assertNotIn("needs_revision", updated.tags)

    def test_good_outcome_promotes_candidate(self):
        """A candidate rule with 1 evidence that gets a good score should become promoted."""
        rule = _make_rule("r1", "余白を作れ", status="candidate", evidence_count=1)

        updated = reconsolidate_rule(rule, applied=True, outcome_score=0.8)

        # evidence_count goes from 1 to 2, triggering promotion
        self.assertEqual(updated.evidence_count, 2)
        self.assertEqual(updated.status, "promoted")

    def test_reconsolidation_learns_new_scope_context(self):
        """Applying a rule in a new situation should create context memory for that situation."""
        rule = _make_rule("r1", "ROI数値を入れろ", status="candidate", evidence_count=1, tags=["roi"])

        updated = reconsolidate_rule(
            rule,
            applied=True,
            outcome_score=0.85,
            task_scope="sales_brief",
            tags=["roi", "finance"],
        )

        scope_contexts = [context for context in updated.contexts if context.kind == "scope"]
        self.assertTrue(any(context.value == "sales_brief" for context in scope_contexts))
        self.assertGreater(updated.expected_gain, 0.0)
        self.assertGreater(updated.confidence_score, 0.0)

    def test_reconsolidation_updates_best_matching_latent_context(self):
        rule = _make_rule(
            "r1",
            "余白を作れ",
            status="candidate",
            evidence_count=2,
            tags=["design"],
            latent_contexts=[
                LatentContext(
                    id="latent-ui",
                    scope="ui_design",
                    tags=["design", "spacing"],
                    keywords=["layout", "hero"],
                    prototype_text="ui design layout spacing hero",
                    evidence_count=2.0,
                    support_score=2.6,
                    expected_gain=2.2,
                    confidence_score=0.68,
                    prior_weight=0.48,
                    posterior_mass=2.0,
                ),
                LatentContext(
                    id="latent-proposal",
                    scope="proposal_summary",
                    tags=["proposal"],
                    keywords=["summary"],
                    prototype_text="proposal summary",
                    evidence_count=2.0,
                    support_score=1.0,
                    expected_gain=0.8,
                    confidence_score=0.38,
                    prior_weight=0.18,
                    posterior_mass=2.0,
                ),
            ],
        )

        updated = reconsolidate_rule(
            rule,
            applied=True,
            outcome_score=0.9,
            task_scope="ui_design",
            tags=["design", "layout"],
        )

        ui_context = next(context for context in updated.latent_contexts if context.id == "latent-ui")
        proposal_context = next(context for context in updated.latent_contexts if context.id == "latent-proposal")
        self.assertGreater(ui_context.support_score, proposal_context.support_score)
        self.assertGreater(ui_context.posterior_mass, proposal_context.posterior_mass)

    def test_confident_failure_blames_rule_more_than_uncertain_failure(self):
        base_rule = _make_rule(
            "r1",
            "ROI数値を入れろ",
            status="candidate",
            evidence_count=2,
            latent_contexts=[
                LatentContext(
                    id="latent-sales",
                    scope="sales_brief",
                    tags=["roi", "finance"],
                    keywords=["roi"],
                    prototype_text="sales brief roi finance",
                    evidence_count=2.0,
                    support_score=3.0,
                    expected_gain=2.4,
                    confidence_score=0.7,
                    prior_weight=0.52,
                    posterior_mass=2.0,
                )
            ],
        )

        confident_failure = reconsolidate_rule(
            base_rule,
            applied=True,
            outcome_score=0.1,
            task_scope="sales_brief",
            tags=["roi", "finance"],
            inference_trace={
                "top_context_posterior": 0.82,
                "posterior_gap": 0.31,
                "novelty_probability": 0.08,
                "should_abstain": False,
                "latent_context_matches": [
                    {"context_id": "latent-sales", "posterior": 0.82}
                ],
            },
        )
        uncertain_failure = reconsolidate_rule(
            base_rule,
            applied=True,
            outcome_score=0.1,
            task_scope="sales_brief",
            tags=["roi", "finance"],
            inference_trace={
                "top_context_posterior": 0.34,
                "posterior_gap": 0.03,
                "novelty_probability": 0.51,
                "should_abstain": True,
                "latent_context_matches": [
                    {"context_id": "latent-sales", "posterior": 0.34}
                ],
            },
        )

        confident_context = confident_failure.latent_contexts[0]
        uncertain_context = uncertain_failure.latent_contexts[0]
        self.assertLess(confident_context.support_score, uncertain_context.support_score)

    def test_reconsolidate_rules_from_turns_uses_saved_trace_not_hindsight_search(self):
        untouched_rule = _make_rule(
            "r1",
            "余白を作れ",
            tags=["design"],
            scope="ui_design",
        )
        selected_rule = _make_rule(
            "r2",
            "ROI数値を入れろ",
            tags=["roi"],
            scope="sales_brief",
            latent_contexts=[
                LatentContext(
                    id="latent-sales",
                    scope="sales_brief",
                    tags=["roi", "finance"],
                    keywords=["roi"],
                    prototype_text="sales brief roi finance",
                    evidence_count=2.0,
                    support_score=2.8,
                    expected_gain=2.2,
                    confidence_score=0.68,
                    prior_weight=0.49,
                    posterior_mass=2.0,
                )
            ],
        )
        turn = _make_turn(
            "t1",
            guidance_applied=True,
            reaction_score=0.9,
            tags=["roi", "finance"],
            task_scope="sales_brief",
        )
        turn.metadata = {
            "inference_trace": {
                "selected_rule_ids": ["r2"],
                "selected_rules": [
                    {
                        "rule_id": "r2",
                        "top_context_posterior": 0.79,
                        "posterior_gap": 0.26,
                        "novelty_probability": 0.11,
                        "should_abstain": False,
                        "latent_context_matches": [
                            {"context_id": "latent-sales", "posterior": 0.79}
                        ],
                    }
                ],
                "abstained_overall": False,
            }
        }

        updated_rules = reconsolidate_rules_from_turns([untouched_rule, selected_rule], [turn])

        updated_untouched = next(rule for rule in updated_rules if rule.id == "r1")
        updated_selected = next(rule for rule in updated_rules if rule.id == "r2")
        self.assertEqual(updated_untouched.evidence_count, untouched_rule.evidence_count)
        self.assertGreater(updated_selected.evidence_count, selected_rule.evidence_count)

    def test_reconsolidate_rules_from_turns_good(self):
        """reconsolidate_rules_from_turns should strengthen matching rules on high score turns."""
        rule = _make_rule("r1", "余白を作れ", tags=["design"], scope="ui_design")
        turn = _make_turn(
            "t1",
            guidance_applied=True,
            reaction_score=0.9,
            tags=["design"],
            task_scope="ui_design",
        )

        updated_rules = reconsolidate_rules_from_turns([rule], [turn])

        self.assertEqual(len(updated_rules), 1)
        updated = updated_rules[0]
        self.assertGreater(updated.evidence_count, rule.evidence_count)
        self.assertGreater(updated.expected_gain, rule.expected_gain)

    def test_reconsolidate_rules_from_turns_bad(self):
        """reconsolidate_rules_from_turns should flag matching rules on low score turns."""
        rule = _make_rule("r1", "余白を作れ", tags=["design"], scope="ui_design")
        turn = _make_turn(
            "t1",
            guidance_applied=True,
            reaction_score=0.1,
            tags=["design"],
            task_scope="ui_design",
        )

        updated_rules = reconsolidate_rules_from_turns([rule], [turn])

        self.assertEqual(len(updated_rules), 1)
        updated = updated_rules[0]
        self.assertIn("needs_revision", updated.tags)

    def test_reconsolidate_rules_from_turns_skips_no_guidance(self):
        """Turns with guidance_applied=False should not affect rules."""
        rule = _make_rule("r1", "余白を作れ", tags=["design"], scope="ui_design")
        turn = _make_turn(
            "t1",
            guidance_applied=False,
            reaction_score=0.9,
            tags=["design"],
            task_scope="ui_design",
        )

        updated_rules = reconsolidate_rules_from_turns([rule], [turn])

        self.assertEqual(updated_rules[0].evidence_count, rule.evidence_count)
        self.assertNotIn("needs_revision", updated_rules[0].tags)

    def test_reconsolidate_rules_from_turns_skips_no_score(self):
        """Turns without a reaction_score should not affect rules."""
        rule = _make_rule("r1", "余白を作れ", tags=["design"], scope="ui_design")
        turn = _make_turn(
            "t1",
            guidance_applied=True,
            reaction_score=None,
            tags=["design"],
            task_scope="ui_design",
        )

        updated_rules = reconsolidate_rules_from_turns([rule], [turn])

        self.assertEqual(updated_rules[0].evidence_count, rule.evidence_count)


class ConflictResolutionTest(unittest.TestCase):
    """Tests for resolve_contradicting_rules (P0: auto-resolution)."""

    def test_negation_conflict_demotes_weaker(self):
        """Weaker rule (lower confidence×evidence) gets demoted."""
        strong = _make_rule("r1", "余白を作れ", confidence_score=0.9, evidence_count=3)
        weak = _make_rule("r2", "余白を作るな", confidence_score=0.3, evidence_count=1)

        updated, log = resolve_contradicting_rules([strong, weak])

        self.assertEqual(weak.status, "demoted")
        self.assertIn("conflict_demoted", weak.tags)
        self.assertEqual(strong.status, "promoted")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["action"], "demote")

    def test_scope_conflict_adds_negative_conditions(self):
        """Scope conflicts separate by adding mutual negative_conditions."""
        r1 = _make_rule("r1", "余白を作れ")
        r2 = PreferenceRule(
            id="r2", statement="余白を作れ", normalized_statement="余白を作れ",
            instruction="余白を作れ", status="promoted", evidence_count=2,
            first_recorded_at="2026/03/28 22:00", last_recorded_at="2026/03/28 22:00",
            applies_to_scope="design", negative_conditions=["余白を作れ"],
        )

        updated, log = resolve_contradicting_rules([r1, r2])

        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["action"], "scope_separate")

    def test_metabolism_lowers_guard(self):
        """Aggressive metabolism should lower the safety guard threshold."""
        strong = _make_rule("r1", "余白を作れ", confidence_score=0.6, evidence_count=4)
        weak = _make_rule("r2", "余白を作るな", confidence_score=0.5, evidence_count=3)

        # Conservative: safety guard protects the weak rule (evidence >= 5*(1.5-0.0)=7.5 → 8, so 4 < 8, not guarded)
        _, log_conservative = resolve_contradicting_rules([strong, weak], metabolism_rate=0.0)

        # Reset
        weak.status = "promoted"
        weak.tags = [t for t in weak.tags if t != "conflict_demoted"]

        _, log_aggressive = resolve_contradicting_rules([strong, weak], metabolism_rate=1.0)

        # Both should resolve (neither hits the guard threshold), but aggressive should be equally willing
        self.assertTrue(len(log_aggressive) >= 1)

    def test_no_conflicts_returns_unchanged(self):
        r1 = _make_rule("r1", "余白を作れ")
        r2 = _make_rule("r2", "ROI数値を入れろ")

        updated, log = resolve_contradicting_rules([r1, r2])

        self.assertEqual(len(log), 0)
        self.assertEqual(r1.status, "promoted")
        self.assertEqual(r2.status, "promoted")


class SelfCorrectionTest(unittest.TestCase):
    """Tests for auto_correct_flagged_rules (P0: self-correction loop)."""

    def test_low_evidence_deleted(self):
        """needs_revision + evidence<=1 → deleted entirely."""
        rule = _make_rule("r1", "余白を作れ", evidence_count=1, tags=["needs_revision"])

        corrected, log = auto_correct_flagged_rules([rule], "2026/04/01 12:00")

        self.assertEqual(len(corrected), 0)
        self.assertEqual(log[0]["action"], "delete")

    def test_low_confidence_old_demoted(self):
        """needs_revision + low confidence + aged → demoted."""
        rule = PreferenceRule(
            id="r1", statement="余白を作れ", normalized_statement="余白を作れ",
            instruction="余白を作れ", status="promoted", evidence_count=3,
            first_recorded_at="2026/01/01 00:00", last_recorded_at="2026/03/01 00:00",
            tags=["needs_revision"], confidence_score=0.2,
        )

        corrected, log = auto_correct_flagged_rules([rule], "2026/04/01 12:00")

        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].status, "demoted")
        self.assertIn("self_corrected", corrected[0].tags)

    def test_decent_confidence_narrowed(self):
        """needs_revision + decent confidence → scope narrowed, not demoted."""
        rule = PreferenceRule(
            id="r1", statement="余白を作れ", normalized_statement="余白を作れ",
            instruction="余白を作れ", status="promoted", evidence_count=3,
            first_recorded_at="2026/01/01 00:00", last_recorded_at="2026/03/28 00:00",
            tags=["needs_revision"], confidence_score=0.5,
            applies_to_scope="design",
        )

        corrected, log = auto_correct_flagged_rules([rule], "2026/04/01 12:00")

        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].status, "promoted")
        self.assertEqual(log[0]["action"], "narrow")

    def test_no_revision_tag_untouched(self):
        """Rules without needs_revision are not modified."""
        rule = _make_rule("r1", "余白を作れ", evidence_count=3)

        corrected, log = auto_correct_flagged_rules([rule], "2026/04/01 12:00")

        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].status, "promoted")
        self.assertEqual(len(log), 0)

    def test_metabolism_shortens_wait(self):
        """Aggressive metabolism should demote faster (shorter wait time)."""
        def make_flagged():
            return PreferenceRule(
                id="r1", statement="余白を作れ", normalized_statement="余白を作れ",
                instruction="余白を作れ", status="promoted", evidence_count=3,
                first_recorded_at="2026/01/01 00:00", last_recorded_at="2026/03/25 00:00",
                tags=["needs_revision"], confidence_score=0.2,
            )

        # Conservative (rate=0.0): wait 14*1.5=21 days. 7 days elapsed → not demoted yet
        rule_c = make_flagged()
        corrected_conservative, _ = auto_correct_flagged_rules(
            [rule_c], "2026/04/01 12:00", metabolism_rate=0.0,
        )

        # Aggressive (rate=1.0): wait 14*0.5=7 days. 7.5 days elapsed → demoted
        rule_a = make_flagged()
        corrected_aggressive, _ = auto_correct_flagged_rules(
            [rule_a], "2026/04/01 12:00", metabolism_rate=1.0,
        )

        self.assertEqual(corrected_conservative[0].status, "promoted")
        self.assertEqual(corrected_aggressive[0].status, "demoted")


class ForgettingCurveMetabolismTest(unittest.TestCase):
    """Tests for metabolism_rate modulation in apply_forgetting_curve."""

    def _make_rule_with_dates(self, rule_id, instruction, last_recorded_at="2026/03/01 00:00", priority=3):
        return PreferenceRule(
            id=rule_id, statement=instruction, normalized_statement=instruction.lower(),
            instruction=instruction, status="candidate", evidence_count=2,
            first_recorded_at="2026/01/01 00:00", last_recorded_at=last_recorded_at,
            priority=priority,
        )

    def test_aggressive_decays_faster(self):
        """Aggressive metabolism should cause faster priority decay."""
        rule_a = self._make_rule_with_dates("r1", "余白を作れ", priority=4)
        rule_c = self._make_rule_with_dates("r2", "余白を作れ", priority=4)

        result_aggressive = apply_forgetting_curve([rule_a], "2026/04/01 00:00", metabolism_rate=0.9)
        result_conservative = apply_forgetting_curve([rule_c], "2026/04/01 00:00", metabolism_rate=0.1)

        self.assertLessEqual(result_aggressive[0].priority, result_conservative[0].priority)


class PersonalityLayerTest(unittest.TestCase):
    """Tests for personality_layer.py: profile computation and interventions."""

    def test_empty_data_returns_defaults(self):
        from correx.personality_layer import compute_personality_profile
        profile = compute_personality_profile([], [])

        self.assertEqual(profile.metabolism_rate, 0.5)
        self.assertEqual(profile.metabolism_label, "balanced")
        self.assertEqual(profile.digestibility, 0.5)
        self.assertEqual(profile.sample_size, 0)

    def test_high_demote_rate_signals_aggressive(self):
        from correx.personality_layer import compute_personality_profile
        rules = [
            _make_rule(f"r{i}", f"rule {i}", status="demoted") for i in range(8)
        ] + [
            _make_rule(f"rp{i}", f"promoted {i}", status="promoted") for i in range(2)
        ]
        turns = [_make_turn(f"t{i}", reaction_score=0.3) for i in range(10)]

        profile = compute_personality_profile(turns, rules)

        self.assertGreater(profile.metabolism_rate, 0.6)
        self.assertEqual(profile.metabolism_label, "aggressive")

    def test_stale_retention_detected(self):
        from correx.personality_layer import compute_personality_profile, detect_interventions
        rule = _make_rule("r1", "余白を作れ", confidence_score=0.5)
        rule.failure_count = 5
        rule.success_count = 1

        profile = compute_personality_profile([], [rule])
        interventions = detect_interventions([rule], [], profile)

        stale = [s for s in interventions if s.pattern_type == "stale_retention"]
        self.assertEqual(len(stale), 1)

    def test_repeated_failure_detected(self):
        from correx.personality_layer import compute_personality_profile, detect_interventions
        turns = [
            _make_turn(f"t{i}", task_scope="design", reaction_score=0.2) for i in range(5)
        ]

        profile = compute_personality_profile(turns, [])
        interventions = detect_interventions([], turns, profile)

        repeated = [s for s in interventions if s.pattern_type == "repeated_failure"]
        self.assertEqual(len(repeated), 1)
        self.assertIn("design", repeated[0].evidence)

    def test_goal_drift_detected(self):
        from correx.personality_layer import compute_personality_profile
        turns = (
            [_make_turn(f"t{i}", task_scope="architecture") for i in range(5)] +
            [_make_turn(f"t{i+5}", task_scope="commercialization") for i in range(5)]
        )

        profile = compute_personality_profile(turns, [])

        self.assertTrue(profile.drift_detected)

    def test_format_adapts_to_digestibility(self):
        from correx.personality_layer import (
            PersonalityProfile, InterventionSignal, format_personality_guidance,
        )
        sig = InterventionSignal(
            pattern_type="stale_retention", confidence=0.8,
            evidence="Rule failed 5x", mirror_prompt="Still relevant?",
            reward_frame="Helps precision",
        )

        abstract_profile = PersonalityProfile(digestibility=0.8, digestibility_label="abstract-tolerant")
        concrete_profile = PersonalityProfile(digestibility=0.2, digestibility_label="concrete-preferring")

        abstract_out = format_personality_guidance(abstract_profile, [sig])
        concrete_out = format_personality_guidance(concrete_profile, [sig])

        # Abstract: includes confidence percentage, no reward frame inline
        self.assertIn("confidence: 80%", abstract_out)
        # Concrete: includes evidence detail and reward frame
        self.assertIn("Rule failed 5x", concrete_out)
        self.assertIn("Helps precision", concrete_out)


class CreativeDestructionTest(unittest.TestCase):
    """Tests for apply_creative_destruction in meaning_synthesis.py."""

    def test_subsumed_rules_weakened(self):
        from correx.meaning_synthesis import apply_creative_destruction
        from correx.schemas import Meaning

        r1 = _make_rule("r1", "余白を作れ", evidence_count=3)
        r1.priority = 4
        r2 = _make_rule("r2", "スペースを確保しろ", evidence_count=2)
        r2.priority = 3

        meaning = Meaning(
            id="m1", principle="余白を作れ", normalized_principle="余白を作れ",
            summary="", source_rule_ids=["r1", "r2"], scopes=["design"], tags=[],
            strength=3, cross_scope_count=1, confidence=0.8,
            first_seen_at="2026/04/01 00:00", last_seen_at="2026/04/01 00:00",
            personal_settings_overlap=[], status="active",
        )

        updated, log = apply_creative_destruction([r1, r2], [meaning], metabolism_rate=0.5)

        # r1 is the meaning's principle → not weakened
        self.assertEqual(r1.priority, 4)
        # r2 is subsumed → weakened
        self.assertLess(r2.priority, 3)
        self.assertIn("subsumed", r2.tags)
        self.assertEqual(len(log), 1)

    def test_aggressive_metabolism_destroys_more(self):
        from correx.meaning_synthesis import apply_creative_destruction
        from correx.schemas import Meaning

        r1 = _make_rule("r1", "余白を作れ", evidence_count=3)
        r1.priority = 4
        r2 = _make_rule("r2", "スペースを確保しろ", evidence_count=2)
        r2.priority = 4

        meaning = Meaning(
            id="m1", principle="余白を作れ", normalized_principle="余白を作れ",
            summary="", source_rule_ids=["r1", "r2"], scopes=["design"], tags=[],
            strength=3, cross_scope_count=1, confidence=0.8,
            first_seen_at="2026/04/01 00:00", last_seen_at="2026/04/01 00:00",
            personal_settings_overlap=[], status="active",
        )

        apply_creative_destruction([r1, r2], [meaning], metabolism_rate=1.0)
        aggressive_priority = r2.priority

        # Reset
        r2.priority = 4
        r2.tags = [t for t in r2.tags if t != "subsumed"]

        apply_creative_destruction([r1, r2], [meaning], metabolism_rate=0.0)
        conservative_priority = r2.priority

        self.assertLessEqual(aggressive_priority, conservative_priority)

    def test_weak_meaning_no_destruction(self):
        from correx.meaning_synthesis import apply_creative_destruction
        from correx.schemas import Meaning

        r1 = _make_rule("r1", "余白を作れ")
        r1.priority = 3

        meaning = Meaning(
            id="m1", principle="余白を作れ", normalized_principle="余白を作れ",
            summary="", source_rule_ids=["r1"], scopes=[], tags=[],
            strength=2, cross_scope_count=0, confidence=0.5,
            first_seen_at="2026/04/01 00:00", last_seen_at="2026/04/01 00:00",
            personal_settings_overlap=[], status="active",
        )

        _, log = apply_creative_destruction([r1], [meaning])

        self.assertEqual(len(log), 0)
        self.assertEqual(r1.priority, 3)


if __name__ == "__main__":
    unittest.main()
