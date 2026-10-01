"""Publication claims must not become fabricated point-in-time proof."""

from datetime import datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1_predictor.early.historical_evidence import assess_recovered_timing, publication_timing


def dt(value):
    return datetime.fromisoformat(value)


class HistoricalEvidenceTests(unittest.TestCase):
    cutoff = dt("2023-03-02T11:30:00+00:00")

    def test_bahrain_entry_list_is_after_cutoff(self):
        published = dt("2023-03-03T12:05:00+01:00")
        self.assertEqual(publication_timing(self.cutoff, published, published), "after_cutoff")

    def test_same_day_without_time_is_ambiguous(self):
        self.assertEqual(publication_timing(self.cutoff, dt("2023-03-02T00:00:00+00:00"), dt("2023-03-03T00:00:00+00:00")), "ambiguous")

    def test_exact_cutoff_is_inclusive_across_timezones(self):
        published = dt("2023-03-02T14:30:00+03:00")
        self.assertEqual(publication_timing(self.cutoff, published, published), "before_or_at_cutoff")

    def test_unknown_is_not_before(self):
        self.assertEqual(publication_timing(self.cutoff, None, None), "unknown")

    def test_invalid_bounds_fail_closed(self):
        for start, end in [(self.cutoff, None), (self.cutoff, dt("2023-03-01T00:00:00+00:00")), (datetime(2023, 3, 1), self.cutoff)]:
            with self.assertRaises(ValueError):
                publication_timing(self.cutoff, start, end)

    def test_recovery_never_authorizes_inputs(self):
        earlier = dt("2023-03-01T00:00:00+00:00")
        report = assess_recovered_timing(cutoff=self.cutoff, retrieved_at=dt("2026-09-30T00:00:00+00:00"), earliest=earlier, latest=earlier)
        self.assertEqual(report["claimed_publication_timing"], "before_or_at_cutoff")
        self.assertTrue(report["retrieved_after_cutoff"])
        self.assertFalse(report["point_in_time_bytes_verified"])
        self.assertFalse(report["authorizes_forecast_inputs"])

    def test_future_publication_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_recovered_timing(cutoff=self.cutoff, retrieved_at=self.cutoff, earliest=self.cutoff, latest=dt("2023-03-03T00:00:00+00:00"))


if __name__ == "__main__":
    unittest.main()
