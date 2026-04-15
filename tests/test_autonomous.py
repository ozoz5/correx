"""Tests for the Autonomous Intelligence Engine (Phase 2).

Tests the LLM-free thinking loop: tick, perceive, retrieve, predict, verify, modulate.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correx import CorrexService
from correx.autonomous import AutonomousEngine, Event


class EngineInitTest(unittest.TestCase):
    """Test engine initialization and basic tick."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_tick_empty_data(self):
        """Tick should work even with no data at all."""
        engine = AutonomousEngine(self.svc.history)
        result = engine.tick()  # self-reflection mode
        self.assertEqual(result.cycle_count, 1)
        self.assertIsNotNone(result.decision)

    def test_tick_with_event(self):
        """Tick should process an event."""
        engine = AutonomousEngine(self.svc.history)
        event = Event(type="correction", scope="testing", keywords=["pytest", "unit"])
        result = engine.tick(event)
        self.assertEqual(result.cycle_count, 1)
        self.assertIsNone(result.verification_error)  # no previous prediction

    def test_cycle_count_increments(self):
        engine = AutonomousEngine(self.svc.history)
        engine.tick()
        engine.tick()
        engine.tick()
        self.assertEqual(engine.cycle_count, 3)

    def test_get_state(self):
        engine = AutonomousEngine(self.svc.history)
        engine.tick()
        state = engine.get_state()
        self.assertEqual(state["cycle_count"], 1)
        self.assertIn("state", state)


class RetrieveTest(unittest.TestCase):
    """Test rule retrieval and tension resolution."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        # Seed some rules
        self.svc.history.write_preference_rules_raw([
            {
                "id": "rule-1",
                "statement": "テストを先に書け",
                "applies_to_scope": "testing",
                "tags": ["testing", "tdd"],
                "evidence_count": 5,
                "confidence_score": 0.8,
            },
            {
                "id": "rule-2",
                "statement": "コードレビューを忘れるな",
                "applies_to_scope": "review",
                "tags": ["review"],
                "evidence_count": 3,
                "confidence_score": 0.6,
            },
            {
                "id": "rule-3",
                "statement": "一般的なルール",
                "applies_to_scope": "general",
                "tags": ["general"],
                "evidence_count": 10,
                "confidence_score": 0.9,
            },
        ])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_retrieve_by_scope(self):
        engine = AutonomousEngine(self.svc.history)
        event = Event(type="correction", scope="testing", keywords=["testing"])
        result = engine.tick(event)
        rule_ids = [r.get("id") for r in result.decision.applicable_rules]
        self.assertIn("rule-1", rule_ids)  # scope match
        self.assertIn("rule-3", rule_ids)  # general always applies
        self.assertNotIn("rule-2", rule_ids)  # different scope

    def test_general_rules_always_apply(self):
        engine = AutonomousEngine(self.svc.history)
        event = Event(type="question", scope="unknown_scope")
        result = engine.tick(event)
        rule_ids = [r.get("id") for r in result.decision.applicable_rules]
        self.assertIn("rule-3", rule_ids)


class PredictionVerificationTest(unittest.TestCase):
    """Test the predict → verify → update loop."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _seed_transitions(self):
        import json
        t_file = Path(self.tmpdir.name) / "context_transitions.json"
        t_file.write_text(json.dumps([{
            "id": "trans-1", "from_signature": "sig-a", "to_signature": "sig-b",
            "from_scope": "design", "to_scope": "implementation",
            "from_tags": [], "to_tags": [], "from_keywords": [], "to_keywords": [],
            "evidence_count": 5, "confidence_score": 0.8,
            "success_weight": 0, "failure_weight": 0,
            "prediction_hit_count": 0, "prediction_miss_count": 0,
            "forecast_score": 0, "last_seen_at": "",
        }]))

    def test_prediction_with_transitions(self):
        """If transitions exist, engine should predict next scope."""
        self._seed_transitions()

        engine = AutonomousEngine(self.svc.history)
        event = Event(type="correction", scope="design")
        result = engine.tick(event)
        self.assertIsNotNone(result.prediction)
        self.assertEqual(result.prediction.scope, "implementation")

    def test_verification_on_second_tick(self):
        """Second tick should verify the previous prediction."""
        self._seed_transitions()

        engine = AutonomousEngine(self.svc.history)
        # Tick 1: predict "implementation" after "design"
        engine.tick(Event(type="correction", scope="design"))
        # Tick 2: actual is "implementation" → low error
        result2 = engine.tick(Event(type="correction", scope="implementation"))
        self.assertIsNotNone(result2.verification_error)
        self.assertLess(result2.verification_error, 0.5)  # good prediction

    def test_bad_prediction_high_error(self):
        self._seed_transitions()

        engine = AutonomousEngine(self.svc.history)
        engine.tick(Event(type="correction", scope="design"))
        # Actual is "testing" not "implementation" → high error
        result2 = engine.tick(Event(type="correction", scope="testing"))
        self.assertIsNotNone(result2.verification_error)
        self.assertGreater(result2.verification_error, 0.5)  # bad prediction


class JourneyAwakenTest(unittest.TestCase):
    """Test journey awakening within the engine."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        self.svc.save_journey(
            where="https://docs.python.org",
            impression=["python", "asyncio", "coroutine"],
            valence=0.9,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_journey_awakened_in_tick(self):
        engine = AutonomousEngine(self.svc.history)
        event = Event(
            type="question",
            scope="backend",
            keywords=["python", "asyncio"],
        )
        result = engine.tick(event)
        self.assertGreater(len(result.decision.awakened_journeys), 0)


class ModulationTest(unittest.TestCase):
    """Test cross-layer modulation effects."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        self.svc.history.write_preference_rules_raw([
            {
                "id": "rule-mod",
                "statement": "asyncioのテスト",
                "applies_to_scope": "backend",
                "tags": ["python"],
                "evidence_count": 3,
                "confidence_score": 0.7,
            },
        ])
        self.svc.save_journey(
            where="https://docs.python.org",
            impression=["python", "asyncio"],
            scope="backend",
            valence=0.9,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_deja_vu_boosts_confidence(self):
        engine = AutonomousEngine(self.svc.history)
        event = Event(
            type="correction",
            scope="backend",
            keywords=["python", "asyncio"],
        )
        result = engine.tick(event)
        self.assertIn("boosted", result.modulations)


class GhostCheckTest(unittest.TestCase):
    """Test ghost trajectory resonance detection."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        self.svc.history.write_ghost_trajectories([
            {
                "id": "traj-1",
                "theme": "テスト 確認 不足",
                "scopes": ["testing"],
                "cumulative_pe": 0.9,
                "firing_threshold": 1.0,
                "fired": False,
                "ghost_ids": ["g1", "g2"],
            },
        ])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_near_firing_detected(self):
        engine = AutonomousEngine(self.svc.history)
        event = Event(
            type="correction",
            scope="testing",
            keywords=["テスト", "確認"],
        )
        result = engine.tick(event)
        self.assertIsNotNone(result.modulations)


class ServiceIntegrationTest(unittest.TestCase):
    """Test service-level integration (run_autonomous_tick, get_engine_state)."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_autonomous_tick_via_service(self):
        result = self.svc.run_autonomous_tick()
        self.assertIn("cycle", result)
        self.assertEqual(result["cycle"], 1)

    def test_run_with_event(self):
        result = self.svc.run_autonomous_tick(
            event_type="correction",
            scope="testing",
            keywords=["python"],
        )
        self.assertIn("rules_count", result)

    def test_get_engine_state_via_service(self):
        self.svc.run_autonomous_tick()
        state = self.svc.get_engine_state()
        self.assertEqual(state["cycle_count"], 1)

    def test_engine_lazy_init(self):
        """Engine should be created on first use, not at construction."""
        self.assertIsNone(self.svc._autonomous_engine)
        self.svc.run_autonomous_tick()
        self.assertIsNotNone(self.svc._autonomous_engine)


class SynthesisTest(unittest.TestCase):
    """Test periodic synthesis (every 10 ticks)."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        self.svc.history.write_preference_rules_raw([
            {
                "id": "rule-s1",
                "statement": "テスト",
                "applies_to_scope": "testing",
                "tags": [],
                "evidence_count": 2,
                "confidence_score": 0.5,
            },
        ])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_synthesis_runs_at_10th_tick(self):
        engine = AutonomousEngine(self.svc.history)
        for _ in range(10):
            engine.tick()
        state = engine.get_state()
        self.assertIn("scope_coverage", state["state"])
        self.assertIn("last_synthesis", state["state"])


class PlasticityTest(unittest.TestCase):
    """Phase 3: Test instinct plasticity — policies become mutable with enough contradictions."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        # Seed an active policy
        import json
        policies_file = Path(self.tmpdir.name) / "policies.json"
        policies_file.write_text(json.dumps([{
            "id": "pol-1", "title": "理解が行動に先行する",
            "core": "確信のない行動は状況を悪化させる",
            "why": "過去の事故から", "maturity": "active",
            "scopes": ["testing"], "evidence_count": 85,
            "source_rule_ids": [], "source_ghost_ids": [], "source_law_ids": [],
            "tags": [], "analogy": "", "opposite": "", "limits": "",
            "created_at": "", "updated_at": "", "approved_by": "auto",
        }]))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_corrections_accumulate_contradictions(self):
        engine = AutonomousEngine(self.svc.history)
        for _ in range(10):
            engine.tick(Event(type="correction", scope="testing", keywords=["test"]))
        state = engine.get_state()
        contradictions = state["state"].get("policy_contradictions", {})
        self.assertGreater(len(contradictions), 0)
        self.assertEqual(contradictions.get("pol-1:testing", 0), 10)

    def test_plasticity_candidate_at_threshold(self):
        from correx.autonomous import POLICY_PLASTICITY_THRESHOLD
        engine = AutonomousEngine(self.svc.history)
        engine._state["policy_contradictions"] = {"pol-1:testing": POLICY_PLASTICITY_THRESHOLD - 1}
        engine.tick(Event(type="correction", scope="testing", keywords=["test"]))
        state = engine.get_state()
        candidates = state["state"].get("plastic_candidates", [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["policy_id"], "pol-1")

    def test_no_plasticity_below_threshold(self):
        engine = AutonomousEngine(self.svc.history)
        for _ in range(5):
            engine.tick(Event(type="correction", scope="testing", keywords=["test"]))
        state = engine.get_state()
        candidates = state["state"].get("plastic_candidates", [])
        self.assertEqual(len(candidates), 0)

    def test_non_correction_events_dont_count(self):
        engine = AutonomousEngine(self.svc.history)
        for _ in range(100):
            engine.tick(Event(type="praise", scope="testing", keywords=["test"]))
        state = engine.get_state()
        contradictions = state["state"].get("policy_contradictions", {})
        self.assertEqual(contradictions.get("pol-1:testing", 0), 0)


class StochasticResonanceTest(unittest.TestCase):
    """Phase 4: Test stochastic resonance (noise-injected retrieval)."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        self.svc.history.write_preference_rules_raw([
            {
                "id": "rule-sr1",
                "statement": "テストルール",
                "applies_to_scope": "general",
                "tags": ["python", "testing"],
                "evidence_count": 5,
                "confidence_score": 0.8,
            },
        ])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_stochastic_injection_happens_eventually(self):
        """Over many ticks, stochastic injection should fire at least once."""
        import random
        random.seed(42)
        engine = AutonomousEngine(self.svc.history)
        for _ in range(50):
            engine.tick(Event(type="question", scope="testing", tags=["testing"], keywords=["testing"]))
        injections = engine._state.get("stochastic_injections", 0)
        self.assertGreater(injections, 0)


class EngramCompetitionTest(unittest.TestCase):
    """Phase 4: Test engram competition (similar journeys compete)."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)
        self.svc.save_journey(
            where="https://strong.example.com",
            impression=["python", "asyncio", "networking"],
            valence=0.9,
            connected_turn_id="turn-1",
        )
        self.svc.save_journey(
            where="https://weak.example.com",
            impression=["python", "asyncio", "networking", "extra"],
            valence=0.2,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_weaker_journey_identified_as_loser(self):
        engine = AutonomousEngine(self.svc.history)
        for _ in range(10):
            engine.tick()
        state = engine.get_state()
        losers = state["state"].get("engram_competition_losers", [])
        self.assertEqual(len(losers), 1)
        journeys = self.svc.history.load_journeys()
        weak_j = [j for j in journeys if "weak" in j.get("where", "")]
        self.assertIn(weak_j[0]["id"], losers)

    def test_no_competition_when_dissimilar(self):
        self.svc.save_journey(
            where="https://different.example.com",
            impression=["rust", "cargo", "tokio"],
            valence=0.1,
        )
        engine = AutonomousEngine(self.svc.history)
        for _ in range(10):
            engine.tick()
        state = engine.get_state()
        losers = state["state"].get("engram_competition_losers", [])
        journeys = self.svc.history.load_journeys()
        diff_j = [j for j in journeys if "different" in j.get("where", "")]
        self.assertNotIn(diff_j[0]["id"], losers)


if __name__ == "__main__":
    unittest.main()
