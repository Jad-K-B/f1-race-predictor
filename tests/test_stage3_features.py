import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.stage2a import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_eligibility_audit
from f1_predictor.stage3.sources import SourceStore, import_f1db
from f1_predictor.stage3.replay import replay_snapshot
from f1_predictor.stage3.features import build_features, historical_tables


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.store = SourceStore(Path(cls.temp.name))
        cls.history = import_f1db(cls.store, ROOT / "data/raw", "test-local-pinned")
        cls.audit = load_eligibility_audit()
        cls.reference = pd.read_csv(ROOT / "data/processed/stage2a/validation.csv")
        cls.snapshot = replay_snapshot(cls.store, cls.history, int(cls.reference.iloc[0].race_id), cls.audit)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_feature_parity_2023_first_and_withdrawal_races(self):
        for race_id in (int(self.reference.iloc[0].race_id), 1094):
            snapshot = replay_snapshot(self.store, self.history, race_id, self.audit)
            actual = build_features(snapshot, self.store).frame.set_index("driver_id").sort_index()
            expected = self.reference[(self.reference.race_id == race_id) & self.reference.prediction_eligible.eq(1)].set_index("driver_id").sort_index()
            self.assertEqual(list(actual.index), list(expected.index))
            for column in NUMERIC_FEATURES:
                np.testing.assert_allclose(actual[column].to_numpy(float), expected[column].to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True, err_msg=column)
            for column in set(CATEGORICAL_FEATURES) - {"driver_id"}:
                self.assertEqual(actual[column].fillna("missing").tolist(), expected[column].fillna("missing").tolist(), column)

    def test_current_results_withheld_and_future_history_excluded(self):
        tables = historical_tables(self.store, self.snapshot)
        self.assertNotIn(self.snapshot.race["race_id"], set(tables["results"].raceId))
        self.assertLess(pd.to_datetime(tables["races"].date).max(), pd.Timestamp(self.snapshot.race["race_date"]))
        expected = build_features(self.snapshot, self.store).frame
        with patch("f1_predictor.stage3.features.historical_tables", return_value=tables):
            pd.testing.assert_frame_equal(build_features(self.snapshot, self.store).frame, expected)
        self.assertFalse({"points_finish", "race_winner", "finish_order", "result_status"} & set(expected.columns))
        self.assertEqual(expected.columns.tolist()[2:], FEATURE_COLUMNS)

    def test_unknown_rookie_substitution_and_new_circuit(self):
        entries = [dict(e) for e in self.snapshot.entries]
        entries[0] = {**entries[0], "driver_id": "unseen-rookie", "constructor_id": "new-constructor"}
        snapshot = replace(self.snapshot, entries=entries, race={**self.snapshot.race, "circuit_id": "new-circuit"})
        result = build_features(snapshot, self.store).frame.set_index("driver_id").loc["unseen-rookie"]
        self.assertEqual(result.driver_prior_entries, 0)
        self.assertEqual(result.constructor_prior_races, 0)
        self.assertEqual(result.circuit_prior_races, 0)
        self.assertTrue(pd.isna(result.driver_career_win_rate))

    def test_grid_validation_and_no_early_model(self):
        with self.assertRaises(ValueError):
            build_features(replace(self.snapshot, kind="provisional_grid"), self.store)
        entries = [dict(e) for e in self.snapshot.entries]
        entries[1]["grid_position"] = entries[0]["grid_position"]
        with self.assertRaisesRegex(ValueError, "Duplicate numeric"):
            build_features(replace(self.snapshot, entries=entries), self.store)

    def test_pit_lane_missing_grid_and_penalty(self):
        entries = [dict(e) for e in self.snapshot.entries]
        entries[0].update(grid_position=None, pit_lane_start=True, has_grid_penalty=True, grid_penalty_positions=5)
        snapshot = replace(self.snapshot, entries=entries)
        result = build_features(snapshot, self.store).frame.set_index("driver_id").loc[entries[0]["driver_id"]]
        self.assertEqual(result.effective_grid_position, result.grid_field_size + 1)
        self.assertEqual(result.missing_final_grid, 1)
        self.assertEqual(result.grid_penalty_positions, 5)
        entries[0].update(pit_lane_start=False)
        result = build_features(replace(self.snapshot, entries=entries), self.store).frame
        self.assertIn(entries[0]["driver_id"], set(result.driver_id))

    def test_known_withdrawal_after_grid_is_not_a_future_dns_lookup(self):
        entries = [dict(e) for e in self.snapshot.entries]
        withdrawn = entries[0]["driver_id"]
        entries[0].update(eligible=False, eligibility_reason="Unit test: pre-start withdrawal decision")
        result = build_features(replace(self.snapshot, entries=entries), self.store).frame
        self.assertEqual(len(result), len(entries) - 1)
        self.assertNotIn(withdrawn, set(result.driver_id))
