"""Synthetic publication claims; no fixture is historical evidence."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.early.research_features import publication_upper_bound, publication_lower_bound, _research_evidence
from f1_predictor.stage3.sources import SourceStore


class ResearchPublicationTests(unittest.TestCase):
    def claim(self, value="2018-01-31", **extra):
        return dict(reviewed_by="unit-test-fixture", applicability={"research_publication_date":
            dict(date=value, basis="document", timezone="unknown")}, **extra)

    def test_unknown_timezone_date_uses_latest_possible_end_of_day(self):
        self.assertEqual(publication_upper_bound(self.claim()), datetime(2018, 2, 1, 12, tzinfo=timezone.utc))

    def test_date_lower_bound_uses_earliest_possible_timezone(self):
        self.assertEqual(publication_lower_bound(self.claim()), datetime(2018, 1, 30, 10, tzinfo=timezone.utc))

    def test_midnight_metadata_does_not_shorten_date_uncertainty(self):
        self.assertEqual(publication_upper_bound(self.claim(published_at="2018-01-31T00:00:00Z",
            publication_basis="provider_metadata")), publication_upper_bound(self.claim()))

    def test_date_claim_cannot_override_later_exact_publication(self):
        self.assertEqual(publication_upper_bound(self.claim(published_at="2018-02-02T13:00:00Z",
            publication_basis="document")), datetime(2018, 2, 2, 13, tzinfo=timezone.utc))

    def test_unknown_unreviewed_and_malformed_claims_fail(self):
        for record in ({}, {"reviewed_by":"fixture"}, self.claim("2018-02-30"),
                       self.claim("20180131"), self.claim("2018-W05-3"),
                       {**self.claim(), "reviewed_by":None},
                       {**self.claim(), "applicability":{"research_publication_date":{"date":"2018-01-31"}}}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                publication_upper_bound(record)

    def test_date_boundary_and_live_observation_gate_are_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SourceStore(Path(folder))
            ref = store.capture("f1", "https://www.formula1.com/unit-test/date", b"Synthetic date-only source",
                                document_kind="event_lineup")
            scope = dict(race_id=1, year=2018, round=1, grand_prix_id="fixture", circuit_id="fixture", session="event",
                         **self.claim()["applicability"])
            ref = store.review(ref, applicability=scope, reviewed_by="unit-test-fixture", notes="Synthetic date claim")
            with self.assertRaisesRegex(ValueError, "uncertainty"):
                _research_evidence(store, ref, "2018-02-01T11:59:59Z", frozenset())
            record = _research_evidence(store, ref, "2018-02-01T12:00:00Z", frozenset())
            self.assertIsNone(record["published_at"])
            with self.assertRaisesRegex(ValueError, "after cutoff"):
                store.available(ref, "2018-02-02T12:00:00Z", 10**12)

    def test_late_or_unapproved_support_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SourceStore(Path(folder))
            for uri, stamp in [("https://www.formula1.com/unit-test/support", "2018-03-01T00:00:00Z"),
                               ("https://example.com/support", "2018-01-01T00:00:00Z")]:
                support = store.capture("f1", uri, b"Synthetic supporting claim", published_at=stamp,
                    publication_basis="document", reviewed_by="unit-test-fixture", document_kind="engine_supply")
                root = store.capture("f1", "https://www.formula1.com/unit-test/root", b"Synthetic root claim",
                    published_at="2018-01-01T00:00:00Z", publication_basis="document", document_kind="event_lineup")
                root = store.review(root, applicability=dict(race_id=1, year=2018, round=1,
                    grand_prix_id="fixture", circuit_id="fixture", session="event"),
                    reviewed_by="unit-test-fixture", notes="Synthetic support test", supporting_observations=[support])
                with self.subTest(uri=uri), self.assertRaises(ValueError):
                    _research_evidence(store, root, "2018-02-01T00:00:00Z", frozenset())


if __name__ == "__main__":
    unittest.main()
