"""Public checks for external model feature sets and race-grouped search."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.stage2a import FEATURE_COLUMNS, FORBIDDEN_CURRENT_RACE_FEATURES
from f1_predictor.stage2b_data import EXTERNAL_FEATURE_SETS, IDENTITY_FEATURES
from f1_predictor.stage2b_external import (
    _ordered_for_ranking,
    _ranking_relevance,
    external_classifier_candidates,
    external_ranker_candidates,
)


class ExternalModelTests(unittest.TestCase):
    def test_feature_sets_are_allowlisted_and_exclude_outcomes(self):
        self.assertEqual(set(EXTERNAL_FEATURE_SETS), {"compact", "id_free", "full"})
        self.assertEqual([len(EXTERNAL_FEATURE_SETS[k]) for k in ("compact", "id_free", "full")], [67, 144, 147])
        self.assertEqual(EXTERNAL_FEATURE_SETS["full"], FEATURE_COLUMNS)
        for features in EXTERNAL_FEATURE_SETS.values():
            self.assertEqual(len(features), len(set(features)))
            self.assertFalse(set(features) & FORBIDDEN_CURRENT_RACE_FEATURES)
        for name in ("compact", "id_free"):
            self.assertFalse(set(EXTERNAL_FEATURE_SETS[name]) & IDENTITY_FEATURES)

    def test_candidate_search_is_bounded(self):
        classifiers = external_classifier_candidates()
        rankers = external_ranker_candidates()
        self.assertEqual(len(classifiers), 15)
        self.assertEqual(len(rankers), 6)
        self.assertEqual(
            {candidate["family"] for candidate in classifiers},
            {"logistic_regression", "random_forest", "xgboost"},
        )

    def test_ranker_groups_races_and_prefers_better_finishes(self):
        frame = pd.DataFrame({
            "race_id": [2, 1, 2, 1],
            "race_date": ["2020-02-01", "2020-01-01"] * 2,
            "round": [2, 1, 2, 1],
            "driver_id": ["b", "b", "a", "a"],
            "finish_order": [2, 2, 1, 1],
        })
        ordered = _ordered_for_ranking(frame)
        self.assertEqual(ordered["race_id"].tolist(), [1, 1, 2, 2])
        relevance = _ranking_relevance(ordered)
        for race_id in ordered["race_id"].unique():
            rows = np.flatnonzero(ordered["race_id"].to_numpy() == race_id)
            self.assertEqual(
                int(np.argmax(relevance[rows])),
                int(np.argmin(ordered.iloc[rows]["finish_order"].to_numpy())),
            )

    def test_public_aggregate_results_cover_both_test_seasons(self):
        metrics = json.loads((ROOT / "artifacts/stage2b/final_test/test_metrics.json").read_text())
        self.assertEqual(set(metrics), {"overall", "2024", "2025"})
        source = (ROOT / "src/f1_predictor/stage2b_external.py").read_text()
        self.assertNotIn("test.csv", source.lower())


if __name__ == "__main__":
    unittest.main()
