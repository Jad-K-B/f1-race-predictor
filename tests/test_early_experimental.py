"""Private experimental-release tests; synthetic inputs are not forecasts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.experimental import (
    ExperimentalEarlyPredictor, archive_private_forecast,
    prepare_experimental_release, replay_private_forecast, verify_experimental_release,
)
from f1_predictor.early.policy import CANDIDATE_FEATURES
from f1_predictor.stage2a import CATEGORICAL_FEATURES
from f1_predictor.stage3.release import sha256

SOURCE = ROOT / "data/stage3/evidence-private/early-research/replay-bundle-v4"


@unittest.skipUnless(SOURCE.exists(), "Private reviewed research bundle unavailable")
class ExperimentalEarlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.private_patch = patch("f1_predictor.early.experimental.PRIVATE_ROOT", cls.root)
        cls.private_patch.start()
        cls.addClassCleanup(cls.private_patch.stop)
        cls.release = prepare_experimental_release(SOURCE, cls.root / "release")
        cls.predictor = ExperimentalEarlyPredictor(cls.release)

    def test_models_and_settings_are_exact_copies_and_private(self):
        manifest = verify_experimental_release(self.release)
        self.assertEqual(manifest["status"], "experimental_prepared_private")
        for key in ("deployment_authorized", "public_release_authorized",
                    "sealed_2024_2025_loaded", "base_model_fit_performed",
                    "model_selection_performed"):
            self.assertIs(manifest[key], False)
        self.assertEqual(manifest["research_manifest_sha256"], sha256(SOURCE / "manifest.json"))
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(digest, sha256(SOURCE / name))
        with self.assertRaises(FileExistsError):
            prepare_experimental_release(SOURCE, self.release)
        with patch("f1_predictor.early.experimental.PRIVATE_ROOT", self.root / "another-private"):
            with self.assertRaisesRegex(ValueError, "private"):
                prepare_experimental_release(SOURCE, self.root / "public")

    def test_tampered_model_is_refused_before_unpickling(self):
        target = self.release / "models/baseline__prior_rate.joblib"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "changed frozen artifact"):
                ExperimentalEarlyPredictor(self.release)
        finally:
            target.write_bytes(original)

    def _synthetic_capture(self) -> Path:
        capture = self.root / "synthetic-capture"
        capture.mkdir(exist_ok=True)
        rows = []
        for index in range(20):
            row = {name: (None if name in CATEGORICAL_FEATURES else np.nan)
                   for name in CANDIDATE_FEATURES}
            row.update(driver_id=f"synthetic-driver-{index:02d}",
                       constructor_id=f"synthetic-team-{index // 2:02d}",
                       grand_prix_id="synthetic-event", circuit_id="synthetic-circuit",
                       year=2026, round=1)
            rows.append({"race_id": 9999, "race_date": "2026-10-04",
                         "cutoff": "2026-10-01T10:00:00Z", **row})
        frame = pd.DataFrame(rows)[["race_id", "race_date", "cutoff", *CANDIDATE_FEATURES]]
        numeric = [name for name in CANDIDATE_FEATURES if name not in CATEGORICAL_FEATURES]
        frame[numeric] = frame[numeric].astype(float)
        frame[list(CATEGORICAL_FEATURES)] = frame[list(CATEGORICAL_FEATURES)].astype("str")
        (capture / "features.json").write_text(json.dumps({
            "columns": frame.columns.tolist(), "dtypes": frame.dtypes.astype(str).to_dict(),
            "records": json.loads(frame.to_json(orient="records", double_precision=15)),
            "missingness": [],
        }), encoding="utf-8")
        (capture / "snapshot.json").write_text(json.dumps({
            "snapshot_id": "synthetic-test-only", "cutoff": "2026-10-01T10:00:00Z",
            "race": {"race_id": 9999, "year": 2026, "round": 1,
                     "grand_prix_id": "synthetic-event", "circuit_id": "synthetic-circuit",
                     "fp1_start": "2026-10-02T10:00:00Z",
                     "qualifying_start": "2026-10-03T10:00:00Z",
                     "race_start": "2026-10-04T10:00:00Z"},
            "entries": [], "sources": {}, "reviewed_by": "synthetic-unit-test",
        }), encoding="utf-8")
        (capture / "manifest.json").write_text("{}", encoding="utf-8")
        return capture

    def test_synthetic_prediction_consistency_and_timing(self):
        capture = self._synthetic_capture()
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRanker

        with patch("f1_predictor.early.experimental.inspect_early_capture",
                   return_value={"completed_at": "2026-10-01T10:05:00Z"}), \
                patch("f1_predictor.early.experimental.replay_early_capture") as replay, \
                patch("f1_predictor.early.experimental.now",
                      return_value="2026-10-01T10:06:00Z"), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("unexpected fit")), \
                patch.object(XGBRanker, "fit", side_effect=AssertionError("unexpected fit")):
            result = self.predictor.predict_capture(capture)
            replay.assert_called_once_with(capture)
            self.assertEqual(result["publication_status"], "private_unpublished")
            self.assertEqual(result["forecast_stage"], "early")
            self.assertEqual(len(result["methods"]), 7)
            self.assertIsNone(result["methods"]["prior_rate"]["predicted_order"])
            selected = result["methods"]["selected_combination"]["drivers"]
            for target, family in self.predictor.config["chosen_families_on_selection"].items():
                family_rows = result["methods"][family]["drivers"]
                self.assertEqual(
                    [row["probabilities"]["raw"][target] for row in selected],
                    [row["probabilities"]["raw"][target] for row in family_rows],
                )
            for method in result["methods"].values():
                self.assertEqual(len(method["drivers"]), 20)
                sums = method["diagnostics"]["probability_sums"]
                for target, expected in (("race_winner", 1), ("podium_finish", 3),
                                         ("points_finish", 10)):
                    self.assertAlmostEqual(sums[target], expected, places=7)
                for driver in method["drivers"]:
                    probability = driver["probabilities"]["reconciled"]
                    self.assertLessEqual(probability["race_winner"], probability["podium_finish"] + 1e-9)
                    self.assertLessEqual(probability["podium_finish"], probability["points_finish"] + 1e-9)
                    self.assertNotIn("starting_position", driver)
                    self.assertNotIn("qualifying_position", driver)
            archived = archive_private_forecast(self.predictor, capture, self.root / "forecast")
            self.assertEqual(json.loads((archived / "manifest.json").read_text())["status"],
                             "complete_private_unpublished")
            with self.assertRaises(FileExistsError):
                archive_private_forecast(self.predictor, capture, archived)
        with patch("f1_predictor.early.experimental.inspect_early_capture",
                   return_value={"completed_at": "2026-10-01T10:05:00Z"}), \
                patch("f1_predictor.early.experimental.replay_early_capture"):
            self.assertEqual(replay_private_forecast(self.predictor, capture, archived)["status"],
                             "exact_private_forecast_replay")
            saved = archived / "forecast.json"
            original = saved.read_bytes()
            try:
                saved.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "changed frozen artifact"):
                    replay_private_forecast(self.predictor, capture, archived)
            finally:
                saved.write_bytes(original)
        for timestamp in ("2026-10-01T09:59:59Z", "2026-10-02T10:00:00Z"):
            with patch("f1_predictor.early.experimental.inspect_early_capture",
                       return_value={"completed_at": "2026-10-01T10:05:00Z"}), \
                    patch("f1_predictor.early.experimental.replay_early_capture"), \
                    patch("f1_predictor.early.experimental.now", return_value=timestamp):
                with self.assertRaisesRegex(ValueError, "after cutoff and before FP1"):
                    self.predictor.predict_capture(capture)

    def test_unverified_capture_fails_closed(self):
        capture = self._synthetic_capture()
        with self.assertRaises(ValueError):
            self.predictor.predict_capture(capture)

    def test_cli_help_and_public_destination_refusal(self):
        cli = ROOT / "scripts/early_experimental.py"
        python = sys.executable
        for command in ((), ("prepare",), ("verify",), ("forecast",), ("replay",)):
            result = subprocess.run([python, "-B", str(cli), *command, "--help"],
                                    cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([
            python, "-B", str(cli), "prepare", "--research-bundle", str(SOURCE),
            "--output", str(self.root.parent / "public-bundle"),
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root.parent / "public-bundle").exists())


if __name__ == "__main__":
    unittest.main()
