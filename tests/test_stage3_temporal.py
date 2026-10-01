import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.stage2a import load_eligibility_audit
from f1_predictor.stage3.sources import SourceStore, import_f1db
from f1_predictor.stage3.replay import replay_snapshot
from f1_predictor.stage3.features import historical_tables
from f1_predictor.stage3.archive import archive_snapshot


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.store = SourceStore(Path(cls.temp.name) / "cache")
        cls.history = import_f1db(cls.store, ROOT / "data/raw", "test-pit")
        cls.snapshot = replay_snapshot(cls.store, cls.history, 1080, load_eligibility_audit())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_current_outcome_fields_rejected_in_entries(self):
        entries = [dict(e) for e in self.snapshot.entries]
        entries[0]["finish_order"] = 1
        with self.assertRaisesRegex(ValueError, "unlabeled allowlist"):
            replace(self.snapshot, entries=entries).validate(self.store)

    def test_latest_history_cannot_be_backdated_to_old_cutoff(self):
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            historical_tables(self.store, replace(self.snapshot, evidence_mode="prospective"))

    def test_late_publication_and_wrong_authority(self):
        with patch("f1_predictor.stage3.sources.now", return_value="2026-01-01T12:00:00Z"):
            with self.assertRaisesRegex(ValueError, "future"):
                self.store.capture("fia", "https://www.fia.com/document", b"x", published_at="2026-01-01T13:00:00Z")
        live = replace(self.snapshot, evidence_mode="prospective", cutoff="2099-01-01T10:00:00Z",
                       race={**self.snapshot.race, "race_start": "2099-01-01T12:00:00Z"})
        with self.assertRaises(ValueError):
            live.validate(self.store)

    def test_no_post_start_forecast_or_fake_early_features(self):
        early = replace(self.snapshot, kind="pre_weekend")
        with self.assertRaisesRegex(ValueError, "Pre-weekend"):
            archive_snapshot(early, self.store, Path(self.temp.name) / "forecasts")
        snapshot = replace(self.snapshot, race={**self.snapshot.race, "race_start": "2023-03-04T00:00:00Z"})
        with self.assertRaisesRegex(ValueError, "precede race start"):
            snapshot.validate(self.store)

    def test_actual_live_observations_have_point_in_time_provenance(self):
        cache = ROOT / "data/stage3/cache"
        observations = [p for p in cache.rglob("*.json") if p.is_file()]
        checked = 0
        store = SourceStore(cache)
        for path in observations:
            record = json.loads(path.read_text())
            if record.get("provider") not in ("jolpica", "openf1"):
                continue
            ref = record["observation_id"]
            store.available(ref, record["retrieved_at"], 0)
            self.assertIsNone(record["published_at"])
            checked += 1
        if not checked:
            self.skipTest("Run the live readiness command to collect real observations")
        self.assertGreaterEqual(checked, 2)
