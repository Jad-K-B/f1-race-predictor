"""Research bundle contracts; these tests never evaluate the sealed test set."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.bundle import (
    ResearchReplayBundle, package_research_bundle, verify_research_bundle,
)
from f1_predictor.early.research_training import load_research_cohort

PRIVATE = ROOT / "data/stage3/evidence-private/early-research"
BUNDLE = PRIVATE / "replay-bundle-v4"
COHORT = PRIVATE / "cohort-2014-2023-v1"
EVALUATION = PRIVATE / "model-evaluation-v2"


class EarlyResearchBundleTests(unittest.TestCase):
    def test_public_output_and_existing_output_are_refused_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "private"):
                package_research_bundle(PRIVATE / "model-selection-v2", EVALUATION,
                                        Path(folder) / "public-bundle")
        if BUNDLE.exists():
            with self.assertRaises(FileExistsError):
                package_research_bundle(PRIVATE / "model-selection-v2", EVALUATION,
                                        BUNDLE)

    @unittest.skipUnless(BUNDLE.exists(), "Private research bundle is unavailable")
    def test_bundle_is_explicitly_not_a_production_release(self):
        manifest = verify_research_bundle(BUNDLE)
        self.assertEqual(manifest["purpose"], "2023_validation_replay_only")
        self.assertIs(manifest["production_release_authorized"], False)
        self.assertIs(manifest["prospective_prediction_authorized"], False)
        self.assertIs(manifest["base_model_fit_performed"], False)
        self.assertIs(manifest["model_selection_performed"], False)
        self.assertIs(manifest["sealed_2024_2025_loaded"], False)
        self.assertFalse(hasattr(ResearchReplayBundle(BUNDLE), "predict"))

    @unittest.skipUnless(BUNDLE.exists() and COHORT.exists(), "Private research artifacts are unavailable")
    def test_saved_selection_predictions_replay_without_fit(self):
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRanker

        frame = load_research_cohort(COHORT).block("selection")
        expected = pd.read_csv(EVALUATION / "validation_predictions.csv")
        expected = expected[expected.validation_block.eq("selection")]
        bundle = ResearchReplayBundle(BUNDLE)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("unexpected classifier fit")), \
                patch.object(XGBRanker, "fit", side_effect=AssertionError("unexpected ranker fit")):
            actual = bundle.replay_2023_block(frame, "selection")
        self.assertEqual(len(actual), 8 * 20 * 7)
        keys = ["validation_block", "method", "race_id", "driver_id"]
        pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_exact=False, rtol=0, atol=1e-12,
        )

    @unittest.skipUnless(BUNDLE.exists() and COHORT.exists(), "Private research artifacts are unavailable")
    def test_future_incomplete_and_wrong_schema_are_rejected(self):
        frame = load_research_cohort(COHORT).block("selection")
        bundle = ResearchReplayBundle(BUNDLE)
        future = frame.copy()
        future["year"] = 2026
        with self.assertRaisesRegex(ValueError, "other races"):
            bundle.replay_2023_block(future, "selection")
        with self.assertRaisesRegex(ValueError, "other races"):
            bundle.replay_2023_block(frame.iloc[1:], "selection")
        with self.assertRaisesRegex(ValueError, "feature schema"):
            bundle.replay_2023_block(frame.drop(columns="driver_recent_5_avg_finish"), "selection")
        with self.assertRaisesRegex(ValueError, "complete original"):
            bundle.replay_2023_block(frame, "unknown")

    @unittest.skipUnless(BUNDLE.exists(), "Private research bundle is unavailable")
    def test_bundle_tampering_is_detected_before_unpickling(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copy)
            config = copy / "configuration.json"
            config.write_bytes(config.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "changed frozen artifact"):
                ResearchReplayBundle(copy)


if __name__ == "__main__":
    unittest.main()
