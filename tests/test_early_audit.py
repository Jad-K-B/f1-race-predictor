"""Offline audit never trains, evaluates or opens any processed split."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.early.audit import inspect_inputs, write_audit
from f1_predictor.stage3.release import sha256


class EarlyAuditTests(unittest.TestCase):
    def test_real_metadata_census_is_deterministic_and_does_not_open_splits(self):
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.name in {"train.csv", "validation.csv", "test.csv", "race_features.csv", "f1db-races-race-results.csv"}:
                raise AssertionError(f"Forbidden audit input: {path}")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded), patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            report = inspect_inputs(ROOT)
            self.assertEqual(report, inspect_inputs(ROOT))
        self.assertEqual(report["research_races"], 204)
        self.assertEqual(report["feature_decisions"], {"candidate_requires_evidence": 124, "exclude": 22, "rebuild": 1})
        self.assertEqual(report["evidence_inventory"]["validated_early_evidence_packets"], 0)
        self.assertFalse(any(report["activity"].values()))
        self.assertEqual(len(report["eligibility_reaudit"]), 6)
        self.assertTrue(all(row["early_status"] == "unresolved" for row in report["eligibility_reaudit"]))

    def test_reports_are_hashed_and_cannot_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            write_audit(ROOT, output)
            manifest = json.loads((output / "manifest.json").read_text())
            for name, expected in manifest["artifacts_sha256"].items():
                self.assertEqual(sha256(output / name), expected)
            with self.assertRaises(FileExistsError):
                write_audit(ROOT, output)

    def test_missing_cache_is_not_silently_treated_as_a_successful_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not exist"):
                inspect_inputs(ROOT, Path(directory) / "absent")


if __name__ == "__main__":
    unittest.main()
