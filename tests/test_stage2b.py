"""Leakage, preprocessing, estimator, and consistency tests for Stage 2B."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_predictor.stage2a import FEATURE_COLUMNS  # noqa: E402
from f1_predictor.stage2b_data import (  # noqa: E402
    FEATURE_SETS,
    NON_FEATURE_COLUMNS,
    TEMPORAL_FOLDS,
    SplitSafePreprocessor,
    load_stage2b_training_data,
    temporal_fold_frames,
    validation_protocol_split,
)
from f1_predictor.stage2b_metrics import (  # noqa: E402
    TARGET_PROBABILITY_COLUMNS,
    probability_consistency_metrics,
    reconcile_probabilities,
)
from f1_predictor.stage2b_evaluation import (  # noqa: E402
    _bootstrap_interval,
    _calibration_rows,
    audit_frozen_stage2b,
    evaluate_locked_test,
)
from f1_predictor.stage2b_models import (  # noqa: E402
    GradientBoostedStumpClassifier,
    GradientBoostedStumpRanker,
    LogisticRegressionGD,
    PlattCalibrator,
    RidgeRanker,
    calibrator_from_dict,
    model_from_dict,
)


class Stage2BDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage2a_dir = PROJECT_ROOT / "data" / "processed" / "stage2a"
        cls.train, cls.validation, cls.hashes = load_stage2b_training_data(
            cls.stage2a_dir
        )

    def test_loader_never_opens_locked_test_split(self) -> None:
        with patch(
            "f1_predictor.stage2b_data.pd.read_csv", wraps=pd.read_csv
        ) as mocked_read:
            train, validation, hashes = load_stage2b_training_data(self.stage2a_dir)
        opened = [Path(call.args[0]).name for call in mocked_read.call_args_list]
        self.assertEqual(opened, ["train.csv", "validation.csv"])
        self.assertEqual(set(hashes), {"train.csv", "validation.csv"})
        self.assertTrue(train["prediction_eligible"].eq(1).all())
        self.assertTrue(validation["prediction_eligible"].eq(1).all())

    def test_final_test_evaluation_is_approval_gated_before_reads(self) -> None:
        with patch("f1_predictor.stage2b_evaluation.pd.read_csv") as mocked_read:
            with self.assertRaisesRegex(PermissionError, "explicit"):
                evaluate_locked_test(
                    self.stage2a_dir,
                    PROJECT_ROOT / "artifacts" / "stage2b" / "validation",
                    PROJECT_ROOT / "artifacts" / "stage2b" / "test",
                    confirm_locked_test=False,
                )
        mocked_read.assert_not_called()

    def test_frozen_audit_uses_only_train_validation_and_saved_artifacts(self) -> None:
        external_artifacts = PROJECT_ROOT / "artifacts" / "stage2b" / "external_validation"
        if not (external_artifacts / "manifest.json").exists():
            self.skipTest("Frozen artifact audit requires the private release")
        model_names = json.loads(
            (external_artifacts / "manifest.json").read_text(encoding="utf-8")
        )["model_artifacts_sha256"]
        if not all((external_artifacts / "models" / name).exists() for name in model_names):
            self.skipTest("Locally generated external joblib artifacts are unavailable")
        with patch(
            "f1_predictor.stage2b_evaluation.pd.read_csv", wraps=pd.read_csv
        ) as mocked_read:
            audit = audit_frozen_stage2b(
                self.stage2a_dir,
                PROJECT_ROOT / "artifacts" / "stage2b" / "validation",
                external_artifacts,
            )
        opened = [Path(call.args[0]).name for call in mocked_read.call_args_list]
        self.assertNotIn("test.csv", opened)
        self.assertEqual(audit["status"], "verified_before_test_unlock")
        self.assertFalse(audit["test_data_read_during_audit"])

    def test_feature_sets_are_allowlisted_and_outcomes_are_excluded(self) -> None:
        for features in FEATURE_SETS.values():
            self.assertTrue(set(features) <= set(FEATURE_COLUMNS))
            self.assertFalse(set(features) & NON_FEATURE_COLUMNS)

    def test_temporal_folds_are_expanding_and_race_grouped(self) -> None:
        previous_rows = 0
        for fold in TEMPORAL_FOLDS:
            fit, score = temporal_fold_frames(self.train, fold)
            self.assertGreater(len(fit), previous_rows)
            self.assertEqual(set(score["year"]), {fold.validation_year})
            self.assertLess(fit["year"].max(), score["year"].min())
            self.assertFalse(set(fit["race_id"]) & set(score["race_id"]))
            previous_rows = len(fit)

    def test_validation_protocol_blocks_are_chronological(self) -> None:
        calibration, selection, report = validation_protocol_split(self.validation, 8, 7)
        self.assertEqual(calibration["race_id"].nunique(), 8)
        self.assertEqual(selection["race_id"].nunique(), 7)
        self.assertEqual(report["race_id"].nunique(), 7)
        self.assertLess(
            pd.to_datetime(calibration["race_date"]).max(),
            pd.to_datetime(selection["race_date"]).min(),
        )
        self.assertLess(
            pd.to_datetime(selection["race_date"]).max(),
            pd.to_datetime(report["race_date"]).min(),
        )

    def test_preprocessor_fits_train_only_and_handles_unknowns(self) -> None:
        columns = ["driver_id", "effective_grid_position", "q3_pct_off_best"]
        preprocessor = SplitSafePreprocessor(columns).fit(self.train)
        original = preprocessor.to_dict()
        transformed = preprocessor.transform(self.validation)
        self.assertEqual(transformed.shape[0], len(self.validation))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(original, preprocessor.to_dict())

        unseen = self.validation.head(1).copy()
        unseen["driver_id"] = "never-seen-driver"
        unknown_matrix = preprocessor.transform(unseen)
        unknown_index = preprocessor.output_feature_names.index(
            "driver_id==__UNKNOWN__"
        )
        self.assertEqual(unknown_matrix[0, unknown_index], 1.0)
        with self.assertRaisesRegex(ValueError, "training rows"):
            SplitSafePreprocessor(columns).fit(self.validation)


class Stage2BModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(240, 8))
        signal = 1.2 * self.x[:, 0] - 0.8 * self.x[:, 1]
        self.y = (signal + rng.normal(scale=0.7, size=240) > 0).astype(float)
        self.rank_y = np.clip(0.5 - 0.2 * signal + rng.normal(scale=0.1, size=240), 0, 1)

    def test_native_estimators_produce_finite_predictions(self) -> None:
        classifiers = [
            LogisticRegressionGD(max_iter=80),
            GradientBoostedStumpClassifier(
                n_estimators=12, max_features=5, max_bins=5, min_leaf=8
            ),
        ]
        for model in classifiers:
            probabilities = model.fit(self.x, self.y).predict_proba(self.x)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())

        rankers = [
            RidgeRanker(alpha=1.0),
            GradientBoostedStumpRanker(
                n_estimators=12, max_features=5, max_bins=5, min_leaf=8
            ),
        ]
        for model in rankers:
            scores = model.fit(self.x, self.rank_y).predict_score(self.x)
            self.assertTrue(np.isfinite(scores).all())

    def test_platt_calibration_is_finite(self) -> None:
        raw = np.linspace(0.01, 0.99, len(self.y))
        calibrated = PlattCalibrator().fit(raw, self.y).predict(raw)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated > 0) & (calibrated < 1)).all())

    def test_models_and_calibrator_round_trip(self) -> None:
        fitted_models = [
            LogisticRegressionGD(max_iter=40).fit(self.x, self.y),
            GradientBoostedStumpClassifier(
                n_estimators=5, max_features=4, max_bins=4, min_leaf=8
            ).fit(self.x, self.y),
            RidgeRanker(alpha=1.0).fit(self.x, self.rank_y),
            GradientBoostedStumpRanker(
                n_estimators=5, max_features=4, max_bins=4, min_leaf=8
            ).fit(self.x, self.rank_y),
        ]
        for model in fitted_models:
            restored = model_from_dict(model.to_dict())
            np.testing.assert_allclose(
                model.predict_score(self.x), restored.predict_score(self.x)
            )

        raw = np.linspace(0.01, 0.99, len(self.y))
        calibrator = PlattCalibrator().fit(raw, self.y)
        restored_calibrator = calibrator_from_dict(calibrator.to_dict())
        np.testing.assert_allclose(
            calibrator.predict(raw), restored_calibrator.predict(raw)
        )


@unittest.skipUnless(
    (PROJECT_ROOT / "artifacts/stage2b/final_test/manifest.json").exists(),
    "Detailed final-test artifacts remain private",
)
class Stage2BFinalArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = PROJECT_ROOT / "artifacts" / "stage2b" / "final_test"
        cls.manifest = json.loads(
            (cls.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        cls.predictions = pd.read_csv(
            cls.output_dir / "test_predictions.csv", low_memory=False
        )

    def test_final_artifacts_are_sealed_and_test_evaluated_once(self) -> None:
        self.assertEqual(
            self.manifest["test_status"],
            "evaluated_once_after_explicit_approval",
        )
        self.assertEqual(self.manifest["rows"], 959)
        self.assertEqual(self.manifest["races"], 48)
        self.assertEqual(self.manifest["years"], [2024, 2025])
        for name, expected in self.manifest["artifacts_sha256"].items():
            digest = hashlib.sha256((self.output_dir / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)
        for relative, expected in self.manifest["evaluation_code_sha256"].items():
            digest = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)

    def test_final_predictions_have_complete_races_targets_and_quotas(self) -> None:
        self.assertEqual(len(self.predictions), 959)
        self.assertFalse(
            self.predictions.duplicated(["race_id", "driver_id"]).any()
        )
        for _, race in self.predictions.groupby("race_id"):
            self.assertEqual(int(race["race_winner"].sum()), 1)
            self.assertEqual(int(race["podium_finish"].sum()), 3)
            self.assertEqual(int(race["points_finish"].sum()), 10)
            for method in ("external_selected", "numpy_selected"):
                p_win = race[f"p_race_winner__{method}"]
                p_podium = race[f"p_podium_finish__{method}"]
                p_points = race[f"p_points_finish__{method}"]
                self.assertAlmostEqual(float(p_win.sum()), 1.0, places=9)
                self.assertAlmostEqual(float(p_podium.sum()), 3.0, places=9)
                self.assertAlmostEqual(float(p_points.sum()), 10.0, places=9)
                self.assertTrue((p_win <= p_podium + 1e-9).all())
                self.assertTrue((p_podium <= p_points + 1e-9).all())

    def test_final_metrics_are_separated_from_validation_and_cover_seasons(self) -> None:
        metrics = json.loads(
            (self.output_dir / "test_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(metrics), {"overall", "2024", "2025"})
        snapshot = json.loads(
            (self.output_dir / "validation_results_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            snapshot["scope"], "held_out_2023_validation_report_block_only"
        )
        report = (self.output_dir / "FINAL_TEST_REPORT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Validation results (2023 report block)", report)
        self.assertIn("Final test probability results", report)

    def test_final_evaluation_refuses_to_overwrite_without_reading_data(self) -> None:
        with patch("f1_predictor.stage2b_evaluation.pd.read_csv") as mocked_read:
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                evaluate_locked_test(
                    PROJECT_ROOT / "data" / "processed" / "stage2a",
                    PROJECT_ROOT / "artifacts" / "stage2b" / "validation",
                    self.output_dir,
                    external_artifact_dir=(
                        PROJECT_ROOT / "artifacts" / "stage2b" / "external_validation"
                    ),
                    confirm_locked_test=True,
                )
        mocked_read.assert_not_called()


class Stage2BConsistencyTests(unittest.TestCase):
    def test_reconciliation_enforces_hierarchy_and_race_sums(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": np.repeat([1, 2], 20),
                "driver_id": [f"d{index}" for index in range(40)],
            }
        )
        rng = np.random.default_rng(11)
        probabilities = pd.DataFrame(
            {
                TARGET_PROBABILITY_COLUMNS["points_finish"]: rng.uniform(size=40),
                TARGET_PROBABILITY_COLUMNS["podium_finish"]: rng.uniform(size=40),
                TARGET_PROBABILITY_COLUMNS["race_winner"]: rng.uniform(size=40),
            }
        )
        reconciled = reconcile_probabilities(frame, probabilities)
        p_win = reconciled[TARGET_PROBABILITY_COLUMNS["race_winner"]]
        p_podium = reconciled[TARGET_PROBABILITY_COLUMNS["podium_finish"]]
        p_points = reconciled[TARGET_PROBABILITY_COLUMNS["points_finish"]]
        self.assertTrue((p_win <= p_podium + 1e-7).all())
        self.assertTrue((p_podium <= p_points + 1e-7).all())
        sums = reconciled.assign(race_id=frame["race_id"]).groupby("race_id").sum()
        self.assertTrue(
            np.allclose(sums[TARGET_PROBABILITY_COLUMNS["race_winner"]], 1.0)
        )
        self.assertTrue(
            np.allclose(sums[TARGET_PROBABILITY_COLUMNS["podium_finish"]], 3.0)
        )
        self.assertTrue(
            np.allclose(sums[TARGET_PROBABILITY_COLUMNS["points_finish"]], 10.0)
        )
        metrics = probability_consistency_metrics(
            frame, reconciled, np.arange(len(frame), dtype=float)
        )
        self.assertLess(metrics["hierarchy_violation_rate"], 1e-9)
        self.assertLess(metrics["mean_abs_win_sum_error"], 1e-7)
        self.assertLess(metrics["mean_abs_podium_sum_error"], 1e-7)
        self.assertLess(metrics["mean_abs_points_sum_error"], 1e-7)

    def test_uncertainty_and_calibration_summaries_are_deterministic(self) -> None:
        values = np.array([0.1, 0.2, 0.4, 0.8])
        first = _bootstrap_interval(values, 200, np.random.default_rng(2026))
        second = _bootstrap_interval(values, 200, np.random.default_rng(2026))
        self.assertEqual(first, second)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

        frame = pd.DataFrame(
            {
                "race_id": [1, 1, 2, 2],
                "points_finish": [1, 0, 1, 0],
                "podium_finish": [1, 0, 0, 0],
                "race_winner": [1, 0, 0, 1],
            }
        )
        probabilities = pd.DataFrame(
            {
                "p_points_finish": [0.8, 0.2, 0.7, 0.3],
                "p_podium_finish": [0.6, 0.1, 0.2, 0.1],
                "p_race_winner": [0.7, 0.1, 0.2, 0.8],
            }
        )
        rows = _calibration_rows(
            frame,
            np.arange(len(frame)),
            "synthetic",
            {"model": probabilities},
        )
        for target in ("points_finish", "podium_finish", "race_winner"):
            self.assertEqual(
                sum(row["count"] for row in rows if row["target"] == target),
                len(frame),
            )


if __name__ == "__main__":
    unittest.main()
