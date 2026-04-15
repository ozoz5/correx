"""Tests for Journey Memory — Phase 1 brain-science enhancements.

Tests: SWR tag, Labile Window, déjà vu banding, SWR forgetting protection.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correx import CorrexService


class JourneySaveTest(unittest.TestCase):
    """Test save_journey with SWR tag computation."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_swr_tag_with_connected_turn(self):
        j = self.svc.save_journey(
            where="https://example.com/docs",
            valence=0.8,
            connected_turn_id="turn-123",
        )
        # swr = 0.8 * 1.0 = 0.8
        self.assertAlmostEqual(j["swr_tag"], 0.8, places=2)

    def test_swr_tag_without_connected_turn(self):
        j = self.svc.save_journey(
            where="https://example.com/random",
            valence=0.8,
        )
        # swr = 0.8 * 0.5 = 0.4
        self.assertAlmostEqual(j["swr_tag"], 0.4, places=2)

    def test_swr_tag_zero_valence(self):
        j = self.svc.save_journey(
            where="https://example.com/junk",
            valence=0.0,
            connected_turn_id="turn-456",
        )
        self.assertAlmostEqual(j["swr_tag"], 0.0, places=2)

    def test_labile_until_initially_empty(self):
        j = self.svc.save_journey(where="https://example.com")
        self.assertEqual(j["labile_until"], "")


class SimilarityBandTest(unittest.TestCase):
    """Test déjà vu banding classification."""

    def test_irrelevant(self):
        self.assertEqual(CorrexService._similarity_band(0.0), "irrelevant")
        self.assertEqual(CorrexService._similarity_band(0.14), "irrelevant")

    def test_weak_association(self):
        self.assertEqual(CorrexService._similarity_band(0.15), "weak_association")
        self.assertEqual(CorrexService._similarity_band(0.34), "weak_association")

    def test_deja_vu(self):
        self.assertEqual(CorrexService._similarity_band(0.35), "deja_vu")
        self.assertEqual(CorrexService._similarity_band(0.64), "deja_vu")

    def test_direct_match(self):
        self.assertEqual(CorrexService._similarity_band(0.65), "direct_match")
        self.assertEqual(CorrexService._similarity_band(1.0), "direct_match")


class AwakenJourneysTest(unittest.TestCase):
    """Test awakening with labile window and band classification."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _save_journey(self, where="https://example.com", impression=None, **kw):
        return self.svc.save_journey(
            where=where,
            impression=impression or ["python", "testing", "pytest"],
            **kw,
        )

    def test_awaken_sets_labile_window(self):
        j = self._save_journey()
        # Force dormant
        journeys = self.svc.history.load_journeys()
        journeys[0]["dormant"] = True
        self.svc.history.write_journeys(journeys)

        results = self.svc.awaken_journeys(
            context_keywords=["python", "testing", "pytest"],
        )
        self.assertTrue(len(results) > 0)
        # Check labile_until was set on the awakened journey
        awakened_j = results[0]["journey"]
        self.assertTrue(awakened_j.get("labile_until", ""))
        # Should be ~30min from now
        labile_dt = datetime.fromisoformat(awakened_j["labile_until"])
        now = datetime.now(timezone.utc)
        delta = (labile_dt - now).total_seconds()
        self.assertGreater(delta, 1700)  # > ~28 min
        self.assertLess(delta, 1900)  # < ~32 min

    def test_awaken_returns_band(self):
        self._save_journey(impression=["python", "testing", "pytest", "unit"])
        results = self.svc.awaken_journeys(
            context_keywords=["python", "testing"],
        )
        self.assertTrue(len(results) > 0)
        self.assertIn("band", results[0])
        self.assertIn(results[0]["band"], ["weak_association", "deja_vu", "direct_match"])

    def test_irrelevant_band_filtered_out(self):
        self._save_journey(impression=["rust", "cargo", "tokio"])
        results = self.svc.awaken_journeys(
            context_keywords=["python", "django"],
        )
        # No overlap → irrelevant → not returned
        self.assertEqual(len(results), 0)

    def test_non_dormant_no_labile(self):
        """Non-dormant journeys should not get labile_until set."""
        self._save_journey()
        results = self.svc.awaken_journeys(
            context_keywords=["python", "testing", "pytest"],
        )
        # Journey was not dormant, so labile_until stays empty
        if results:
            awakened_j = results[0]["journey"]
            self.assertEqual(awakened_j.get("labile_until", ""), "")


class UpdateJourneyTest(unittest.TestCase):
    """Test labile window update mechanism."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _save_and_awaken(self):
        j = self.svc.save_journey(
            where="https://docs.python.org",
            impression=["python", "stdlib", "asyncio"],
            valence=0.6,
        )
        # Force dormant then awaken
        journeys = self.svc.history.load_journeys()
        journeys[0]["dormant"] = True
        self.svc.history.write_journeys(journeys)
        self.svc.awaken_journeys(context_keywords=["python", "stdlib", "asyncio"])
        return j["id"]

    def test_update_within_labile_window(self):
        jid = self._save_and_awaken()
        result = self.svc.update_journey(
            journey_id=jid,
            impression=["python", "stdlib", "asyncio", "coroutine"],
            valence=0.9,
        )
        self.assertTrue(result["ok"])
        self.assertGreater(result["labile_remaining_sec"], 0)

        # Verify updated data
        journeys = self.svc.history.load_journeys()
        updated = [j for j in journeys if j["id"] == jid][0]
        self.assertIn("coroutine", updated["impression"])
        self.assertAlmostEqual(updated["valence"], 0.9, places=2)
        # SWR recomputed (no connected turn → 0.9 * 0.5 = 0.45)
        self.assertAlmostEqual(updated["swr_tag"], 0.45, places=2)

    def test_update_without_labile_state(self):
        j = self.svc.save_journey(
            where="https://example.com",
            impression=["test"],
        )
        result = self.svc.update_journey(
            journey_id=j["id"],
            valence=0.9,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "not_labile")

    def test_update_after_window_expired(self):
        jid = self._save_and_awaken()
        # Manually expire the window
        journeys = self.svc.history.load_journeys()
        for j in journeys:
            if j["id"] == jid:
                j["labile_until"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        self.svc.history.write_journeys(journeys)

        result = self.svc.update_journey(journey_id=jid, valence=0.9)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "window_closed")

    def test_update_not_found(self):
        result = self.svc.update_journey(journey_id="journey-nonexistent", valence=0.5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "not_found")


class DormantJourneysTest(unittest.TestCase):
    """Test dormancy scan with SWR protection and labile expiry."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.svc = CorrexService(base_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_old_journey(self, days_old=40, swr_tag=0.0, **kw):
        j = self.svc.save_journey(
            where="https://old-site.com",
            impression=["old", "data"],
            journey_type="wander",
            **kw,
        )
        # Backdate
        old_dt = datetime.now(timezone.utc) - timedelta(days=days_old)
        journeys = self.svc.history.load_journeys()
        for jj in journeys:
            if jj["id"] == j["id"]:
                jj["when"] = old_dt.strftime("%Y/%m/%d %H:%M")
                jj["dormant"] = True
                jj["swr_tag"] = swr_tag
        self.svc.history.write_journeys(journeys)
        return j["id"]

    def test_swr_protects_from_forgetting(self):
        """High SWR tag prevents forgetting even after max_idle_days."""
        jid = self._make_old_journey(days_old=60, swr_tag=0.8)
        result = self.svc.dormant_journeys(max_idle_days=30)
        # Should NOT be forgotten
        journeys = self.svc.history.load_journeys()
        protected = [j for j in journeys if j["id"] == jid][0]
        self.assertFalse(protected.get("forgotten", False))
        self.assertEqual(result["forgotten"], 0)

    def test_low_swr_gets_forgotten(self):
        """Low SWR tag allows normal forgetting."""
        jid = self._make_old_journey(days_old=60, swr_tag=0.3)
        result = self.svc.dormant_journeys(max_idle_days=30)
        journeys = self.svc.history.load_journeys()
        forgotten_j = [j for j in journeys if j["id"] == jid][0]
        self.assertTrue(forgotten_j.get("forgotten", False))
        self.assertEqual(result["forgotten"], 1)

    def test_labile_window_expired_cleared(self):
        """Expired labile windows are cleaned up during scan."""
        j = self.svc.save_journey(where="https://example.com", impression=["test"])
        journeys = self.svc.history.load_journeys()
        expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        journeys[0]["labile_until"] = expired
        self.svc.history.write_journeys(journeys)

        result = self.svc.dormant_journeys(max_idle_days=30)
        self.assertEqual(result["labile_expired"], 1)
        # Verify cleared
        journeys = self.svc.history.load_journeys()
        self.assertEqual(journeys[0]["labile_until"], "")

    def test_business_journey_never_forgotten(self):
        """Business journeys are not forgettable regardless of age."""
        j = self.svc.save_journey(
            where="https://important.com",
            impression=["critical"],
            journey_type="business",
        )
        # Backdate
        journeys = self.svc.history.load_journeys()
        old_dt = datetime.now(timezone.utc) - timedelta(days=90)
        journeys[0]["when"] = old_dt.strftime("%Y/%m/%d %H:%M")
        journeys[0]["dormant"] = True
        self.svc.history.write_journeys(journeys)

        self.svc.dormant_journeys(max_idle_days=30)
        journeys = self.svc.history.load_journeys()
        self.assertFalse(journeys[0].get("forgotten", False))


if __name__ == "__main__":
    unittest.main()
