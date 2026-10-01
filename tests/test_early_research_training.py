"""Split and preprocessing guards for retrospective early-model experiments."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1_predictor.early.policy import CANDIDATE_FEATURES, EXCLUDED_FEATURES
from f1_predictor.early.research_training import (
    FEATURE_SETS, Baseline, Candidate, classifier_candidates, feature_groups,
    fit_classifier, fit_ranker, load_research_cohort, make_preprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_COHORT = ROOT / "data/stage3/evidence-private/early-research/cohort-2014-2023-v1"


class EarlyResearchTrainingTests(unittest.TestCase):
    def test_feature_sets_have_no_current_race_grid_or_qualifying(self):
        self.assertEqual(tuple(FEATURE_SETS["full"]), CANDIDATE_FEATURES)
        self.assertEqual(set(FEATURE_SETS), {"compact", "id_free", "full"})
        for selected in FEATURE_SETS.values():
            self.assertTrue(selected)
            self.assertFalse(set(selected) & EXCLUDED_FEATURES)
            self.assertTrue(set(selected) <= set(CANDIDATE_FEATURES))
        self.assertEqual(
            {candidate.family for candidate in classifier_candidates()},
            {"logistic", "random_forest", "xgboost"},
        )

    def test_preprocessor_fits_train_and_handles_new_categories_and_nulls(self):
        categorical, numeric = feature_groups("compact")
        train = pd.DataFrame({
            **{name: ["known", "known"] for name in categorical},
            **{name: [1.0, np.nan] for name in numeric},
        })
        future = pd.DataFrame({
            **{name: ["new"] for name in categorical},
            **{name: [np.nan] for name in numeric},
        })
        preprocessor = make_preprocessor("compact", scale=True)
        fitted = preprocessor.fit_transform(train)
        transformed = preprocessor.transform(future)
        self.assertEqual(fitted.shape[1], transformed.shape[1])
        fitted_values = fitted.toarray() if hasattr(fitted, "toarray") else fitted
        future_values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        self.assertTrue(np.isfinite(fitted_values).all())
        self.assertTrue(np.isfinite(future_values).all())

    def test_estimators_reject_validation_rows(self):
        validation = pd.DataFrame({"split": ["validation"], "year": [2023]})
        with self.assertRaisesRegex(ValueError, "training seasons only"):
            fit_classifier(Candidate("logistic", "compact", {"C": 1.0}),
                           validation, "race_winner")
        with self.assertRaisesRegex(ValueError, "training seasons only"):
            fit_ranker(Candidate("xgboost_ranker", "compact", {}),
                       validation)

    def test_recent_form_rank_score_places_stronger_driver_first(self):
        training = pd.DataFrame({
            "race_id": [1, 1], "driver_id": ["strong", "weak"],
            "split": ["train", "train"], "year": [2022, 2022],
            "driver_recent_5_avg_finish": [2.0, 18.0],
            "constructor_recent_5_avg_finish": [3.0, 17.0],
            "race_winner": [1, 0], "podium_finish": [1, 0],
            "points_finish": [1, 0],
        })
        baseline = Baseline.fit(training, "recent_form")
        _, rank_scores = baseline.predict(training)
        self.assertLess(rank_scores[0], rank_scores[1])

    @unittest.skipUnless(PRIVATE_COHORT.exists(), "Private historical cohort is unavailable")
    def test_private_cohort_has_original_validation_blocks(self):
        cohort = load_research_cohort(PRIVATE_COHORT)
        self.assertEqual(len(cohort.frame), 3744)
        self.assertEqual(cohort.block("train").race_id.nunique(), 162)
        self.assertEqual(cohort.block("selection").race_id.nunique(), 8)
        self.assertEqual(cohort.block("calibration").race_id.nunique(), 7)
        self.assertEqual(cohort.block("report").race_id.nunique(), 7)
        self.assertTrue(cohort.block("report").year.eq(2023).all())


if __name__ == "__main__":
    unittest.main()
