"""Technical fixtures exercise the builder; they are not reviewed race rosters."""

import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.early.policy import CANDIDATE_FEATURES, EXCLUDED_FEATURES
from f1_predictor.early.research_features import ResearchRequest, build_research_features, load_research_tables
from f1_predictor.stage3.sources import SourceStore


class ResearchFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.store = SourceStore(Path(cls.temp.name))
        ref = cls.store.capture("f1", "https://www.formula1.com/unit-test/roster", b"Synthetic test identities, not historical proof",
                                document_kind="event_lineup", reviewed_by="unit-test-fixture",
                                published_at="2023-03-01T00:00:00+00:00", publication_basis="document")
        ref = cls.store.review(ref, applicability={
            "race_id": 1080, "year": 2023, "round": 1, "grand_prix_id": "bahrain", "circuit_id": "bahrain", "session": "event",
            "driver_constructor_pairs": [["max-verstappen", "red-bull"], ["sergio-perez", "red-bull"], ["unseen-test-driver", "unseen-test-team"]]},
            reviewed_by="unit-test-fixture", notes="Synthetic fixture; no historical evidence claim")
        checksum = cls.store.read(ref)[0]["sha256"]
        cls.tables = load_research_tables(ROOT / "data/raw")
        cls.request = ResearchRequest(
            race_id=1080, fp1_start="2023-03-03T11:30:00+00:00", cutoff="2023-03-02T11:30:00+00:00",
            entries=tuple({"driver_id": driver, "constructor_id": team,
                           "engine_manufacturer_id": engine, "evidence": [ref]} for driver, team, engine in [
                ("max-verstappen", "red-bull", "honda-rbpt"),
                ("sergio-perez", "red-bull", "honda-rbpt"),
                ("unseen-test-driver", "unseen-test-team", None)]),
            reviewed_by="unit-test-fixture", review_notes="Synthetic three-entry field; not a real roster review",
            roster_complete=True, unresolved_conflicts=(), source_hashes={ref: checksum})
        cls.batch = build_research_features(cls.request, cls.tables, cls.store)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_stream_loader_excludes_test_years_and_never_opens_processed_splits(self):
        self.assertLessEqual(self.tables["races"].year.max(), 2023)
        ids = set(self.tables["races"].id)
        for table in ("results", "qualifying", "grid", "driver_standings", "constructor_standings"):
            self.assertTrue(set(self.tables[table].raceId) <= ids)
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotIn(path.name, {"train.csv", "validation.csv", "test.csv", "race_features.csv"})
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            load_research_tables(ROOT / "data/raw")

    def test_schema_and_history_boundary(self):
        frame = self.batch.frame
        self.assertEqual(frame.columns.tolist()[4:], list(CANDIDATE_FEATURES))
        self.assertEqual(len(CANDIDATE_FEATURES), 125)
        self.assertFalse(EXCLUDED_FEATURES & set(frame.columns))
        self.assertFalse({"race_winner", "points_finish", "podium_finish", "finish_order", "prediction_eligible"} & set(frame.columns))
        self.assertNotIn(1080, self.batch.historical_race_ids)
        dates = self.tables["races"].set_index("id").loc[list(self.batch.historical_race_ids), "date"]
        self.assertLess(pd.to_datetime(dates).max(), pd.Timestamp("2023-03-02"))
        self.assertEqual(set(frame.split), {"validation"})
        self.assertEqual(self.batch.evidence_mode, "retrospective_research")

    def test_existing_history_definitions_match_validation_reference(self):
        expected = pd.read_csv(ROOT / "data/processed/stage2a/validation.csv")
        expected = expected[expected.race_id.eq(1080)].set_index("driver_id")
        actual = self.batch.frame.set_index("driver_id")
        columns = [c for c in CANDIDATE_FEATURES if c.startswith(("driver_", "constructor_", "circuit_historical")) and c not in ("driver_id", "driver_nationality_country_id", "constructor_id")]
        for driver in ("max-verstappen", "sergio-perez"):
            np.testing.assert_allclose(actual.loc[driver, columns].to_numpy(float), expected.loc[driver, columns].to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_current_and_future_outcomes_grid_qualifying_standings_cannot_change_features(self):
        tables = copy.deepcopy(self.tables)
        for name in ("results", "qualifying", "grid", "driver_standings", "constructor_standings"):
            tables[name] = tables[name][tables[name].raceId.isin(self.batch.historical_race_ids)].copy()
        actual = build_research_features(self.request, tables, self.store).frame
        pd.testing.assert_frame_equal(actual, self.batch.frame)

    def test_announced_entrants_not_in_current_results_are_retained(self):
        frame = self.batch.frame.set_index("driver_id")
        self.assertIn("unseen-test-driver", frame.index)
        self.assertEqual(frame.loc["unseen-test-driver", "driver_prior_entries"], 0)
        self.assertTrue(pd.isna(frame.loc["unseen-test-driver", "driver_career_win_rate"]))
        self.assertTrue(pd.isna(frame.loc["unseen-test-driver", "engine_manufacturer_id"]))
        self.assertEqual(frame.loc["max-verstappen", "current_constructor_entry_count"], 2)
        self.assertTrue(any(x["driver_id"] == "unseen-test-driver" for x in self.batch.missingness))

    def test_roster_order_does_not_change_features(self):
        result = build_research_features(replace(self.request, entries=tuple(reversed(self.request.entries))), self.tables, self.store)
        pd.testing.assert_frame_equal(result.frame, self.batch.frame)

    def test_announced_team_switch_and_rookies_use_only_prior_history(self):
        people = [('george-russell', 'mercedes', 'mercedes'),
                  ('valtteri-bottas', 'mercedes', 'mercedes'),
                  ('jack-aitken', 'williams', 'mercedes'),
                  ('nicholas-latifi', 'williams', 'mercedes'),
                  ('pietro-fittipaldi', 'haas', 'ferrari')]
        ref = self.store.capture('f1', 'https://www.formula1.com/unit-test/team-switch',
            b'Synthetic substitution fixture; not historical source proof', document_kind='event_lineup',
            published_at='2020-12-02T07:06:15+00:00', publication_basis='document', reviewed_by='unit-test-fixture')
        ref = self.store.review(ref, applicability=dict(race_id=1034, year=2020, round=16,
            grand_prix_id='sakhir', circuit_id='bahrain', session='event',
            driver_constructor_pairs=[[d, t] for d, t, _ in people]),
            reviewed_by='unit-test-fixture', notes='Synthetic identity fixture, not a roster attestation')
        request = ResearchRequest(1034, '2020-12-04T13:30:00+00:00', '2020-12-03T13:30:00+00:00',
            tuple(dict(driver_id=d, constructor_id=t, engine_manufacturer_id=e, evidence=[ref]) for d, t, e in people),
            'unit-test-fixture', 'Synthetic partial field, not historical proof', True, (),
            {ref: self.store.read(ref)[0]['sha256']})
        batch = build_research_features(request, self.tables, self.store)
        frame = batch.frame.set_index('driver_id')
        self.assertEqual(frame.loc['george-russell', 'constructor_id'], 'mercedes')
        self.assertEqual(frame.loc['george-russell', 'driver_constructor_prior_entries'], 0)
        self.assertGreater(frame.loc['george-russell', 'driver_prior_entries'], 0)
        self.assertEqual(frame.loc['george-russell', 'current_constructor_entry_count'], 2)
        for driver in ('jack-aitken', 'pietro-fittipaldi'):
            self.assertEqual(frame.loc[driver, 'driver_prior_entries'], 0)
            self.assertTrue(pd.isna(frame.loc[driver, 'driver_career_win_rate']))
        columns = [c for c in CANDIDATE_FEATURES if c.startswith('constructor_')]
        pd.testing.assert_series_equal(frame.loc['george-russell', columns], frame.loc['valtteri-bottas', columns], check_names=False)
        prior_only = copy.deepcopy(self.tables)
        for name in ('results', 'qualifying', 'grid', 'driver_standings', 'constructor_standings'):
            prior_only[name] = prior_only[name][prior_only[name].raceId.isin(batch.historical_race_ids)]
        pd.testing.assert_frame_equal(build_research_features(request, prior_only, self.store).frame, batch.frame)

    def test_block_incomplete_conflicting_unreviewed_or_duplicate_rosters(self):
        for changes in ({"roster_complete": False}, {"unresolved_conflicts": ("seat disputed",)}, {"reviewed_by": ""}, {"entries": self.request.entries * 2}):
            with self.assertRaises(ValueError):
                build_research_features(replace(self.request, **changes), self.tables, self.store)

    def test_no_current_session_fields_in_entries(self):
        entries = copy.deepcopy(self.request.entries)
        entries[0]["grid_position"] = 1
        with self.assertRaises(ValueError):
            build_research_features(replace(self.request, entries=entries), self.tables, self.store)

    def test_cannot_be_promoted_to_live_forecast(self):
        with self.assertRaisesRegex(ValueError, "prospective"):
            build_research_features(replace(self.request, evidence_mode="prospective"), self.tables, self.store)

    def test_reject_hash_mismatch_and_test_year(self):
        reference = next(iter(self.request.source_hashes))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_research_features(replace(self.request, source_hashes={reference: "0" * 64}), self.tables, self.store)
        tables = copy.deepcopy(self.tables)
        tables["races"].loc[tables["races"].id.eq(1080), "year"] = 2024
        with self.assertRaisesRegex(ValueError, "2014-2023"):
            build_research_features(self.request, tables, self.store)

    def test_missing_latest_standings_blocks_zero_substitution(self):
        tables = copy.deepcopy(self.tables)
        latest = self.batch.historical_race_ids[-1]
        tables["driver_standings"] = tables["driver_standings"][tables["driver_standings"].raceId.ne(latest)]
        with self.assertRaisesRegex(ValueError, "lacks driver_standings"):
            build_research_features(self.request, tables, self.store)

    def test_roster_cannot_come_from_results_or_late_entry_list(self):
        for kind, published in [("race_results", "2023-03-01T00:00:00+00:00"),
                                ("entry_list", "2023-03-03T11:05:00+00:00")]:
            ref = self.store.capture("f1", "https://www.formula1.com/unit-test/" + kind, b"Rejected test evidence",
                                     document_kind=kind, reviewed_by="unit-test-fixture",
                                     published_at=published, publication_basis="document")
            entries = tuple({**e, "evidence": [ref]} for e in self.request.entries)
            checksum = self.store.read(ref)[0]["sha256"]
            with self.assertRaises(ValueError):
                build_research_features(replace(self.request, entries=entries, source_hashes={ref: checksum}), self.tables, self.store)

    def test_unapproved_source_authority_is_rejected(self):
        ref = self.store.capture("f1", "https://example.com/untrusted", b"Test-only unapproved source",
                                 document_kind="event_lineup", reviewed_by="unit-test-fixture",
                                 published_at="2023-03-01T00:00:00+00:00", publication_basis="document")
        entries = tuple({**e, "evidence": [ref]} for e in self.request.entries)
        checksum = self.store.read(ref)[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "Official roster"):
            build_research_features(replace(self.request, entries=entries, source_hashes={ref: checksum}), self.tables, self.store)

    def test_missing_earlier_race_and_duplicate_current_identity_fail(self):
        tables = copy.deepcopy(self.tables)
        tables["results"] = tables["results"][tables["results"].raceId.ne(self.batch.historical_race_ids[-1])]
        with self.assertRaisesRegex(ValueError, "lack results"):
            build_research_features(self.request, tables, self.store)
        tables = copy.deepcopy(self.tables)
        tables["races"] = pd.concat([tables["races"], tables["races"][tables["races"].id.eq(1080)]])
        with self.assertRaisesRegex(ValueError, "one matching"):
            build_research_features(self.request, tables, self.store)

    def test_wrong_event_and_wrong_constructor_review_are_rejected(self):
        tables = copy.deepcopy(self.tables)
        tables["races"].loc[tables["races"].id.eq(1080), "circuitId"] = "jeddah"
        with self.assertRaisesRegex(ValueError, "Wrong event"):
            build_research_features(self.request, tables, self.store)
        entries = copy.deepcopy(self.request.entries)
        entries[0]["constructor_id"] = "ferrari"
        with self.assertRaisesRegex(ValueError, "entrant/team"):
            build_research_features(replace(self.request, entries=entries), self.tables, self.store)


if __name__ == "__main__":
    unittest.main()
