"""Synthetic future-event fixtures; never live race evidence or model validation."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.contracts import EarlySnapshot
from f1_predictor.early.archive import (
    capture_early_snapshot, inspect_early_capture, replay_early_capture,
)
from f1_predictor.early.live_features import (
    _build_from_tables, _completed_tables, _source_tables, build_live_features,
)
from f1_predictor.early.policy import CANDIDATE_FEATURES, EXCLUDED_FEATURES
from f1_predictor.early.research_features import load_research_tables
from f1_predictor.stage2a import CATEGORICAL_FEATURES, RAW_FILES
from f1_predictor.stage3.sources import SourceStore


class EarlyLiveFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_research_tables(ROOT / "data/raw")
        cls.tables = {name: table.copy() for name, table in cls.tables.items()}
        template = cls.tables["races"][cls.tables["races"].id.eq(1080)].copy()
        assert len(template) == 1
        template.loc[:, "id"] = 9999
        template.loc[:, "year"] = 2026
        template.loc[:, "round"] = 1
        template.loc[:, "date"] = "2026-10-04"
        cls.tables["races"] = pd.concat([cls.tables["races"], template], ignore_index=True)
        cls.snapshot = EarlySnapshot(
            snapshot_id="synthetic-future-only", cutoff="2026-10-01T10:00:00Z",
            race={"race_id": 9999, "year": 2026, "round": 1,
                  "grand_prix_id": "bahrain", "circuit_id": "bahrain",
                  "fp1_start": "2026-10-02T10:00:00Z",
                  "qualifying_start": "2026-10-03T10:00:00Z",
                  "race_start": "2026-10-04T10:00:00Z"},
            entries=[
                {"driver_id": "max-verstappen", "constructor_id": "red-bull",
                 "engine_manufacturer_id": None, "status": "announced", "evidence": [], "reason": "synthetic"},
                {"driver_id": "unit-rookie", "constructor_id": "red-bull",
                 "engine_manufacturer_id": None, "status": "announced", "evidence": [], "reason": "synthetic"},
                {"driver_id": "unit-withdrawn", "constructor_id": "red-bull",
                 "engine_manufacturer_id": None, "status": "withdrawn", "evidence": [], "reason": "synthetic"},
            ], sources={}, reviewed_by="unit-test"
        )
        cls.completed = set(cls.tables["results"].raceId.astype(int))

    def test_feature_schema_rookie_and_withdrawal(self):
        batch = _build_from_tables(self.snapshot, self.tables, self.completed)
        self.assertEqual(batch.frame.columns.tolist(), [
            "race_id", "race_date", "cutoff", *CANDIDATE_FEATURES
        ])
        self.assertEqual(len(CANDIDATE_FEATURES), 125)
        self.assertFalse(EXCLUDED_FEATURES & set(batch.frame.columns))
        self.assertEqual(set(batch.frame.driver_id), {"max-verstappen", "unit-rookie"})
        rookie = batch.frame.set_index("driver_id").loc["unit-rookie"]
        self.assertEqual(rookie.driver_prior_entries, 0)
        self.assertEqual(rookie.current_constructor_entry_count, 2)
        self.assertTrue(pd.isna(rookie.driver_career_win_rate))
        self.assertNotIn(9999, batch.historical_race_ids)
        self.assertEqual(batch.evidence_mode, "prospective")

    def test_current_race_outcome_is_never_a_feature(self):
        original = _build_from_tables(self.snapshot, self.tables, self.completed).frame
        injected = {name: table.copy() for name, table in self.tables.items()}
        result = injected["results"].iloc[[0]].copy()
        result.loc[:, "raceId"] = 9999
        result.loc[:, "year"] = 2026
        injected["results"] = pd.concat([injected["results"], result], ignore_index=True)
        pd.testing.assert_frame_equal(
            original, _build_from_tables(self.snapshot, injected, self.completed).frame
        )

    @unittest.skipUnless(
        (ROOT / "data/stage3/evidence-private/early-research/cohort-2014-2023-v1/features.csv").exists(),
        "Private reviewed research cohort is unavailable",
    )
    def test_historical_feature_values_match_reviewed_research_cohort(self):
        cohort = pd.read_csv(
            ROOT / "data/stage3/evidence-private/early-research/cohort-2014-2023-v1/features.csv",
            low_memory=False,
        )
        reference = cohort[cohort.race_id.eq(1080)].set_index("driver_id")
        entries = [{"driver_id": driver, "constructor_id": row.constructor_id,
                    "engine_manufacturer_id": row.engine_manufacturer_id,
                    "status": "announced", "evidence": [], "reason": "synthetic parity test"}
                   for driver, row in reference.iterrows()]
        snapshot = replace(
            self.snapshot, race={"race_id": 1080, "year": 2023, "round": 1,
                                 "grand_prix_id": "bahrain", "circuit_id": "bahrain",
                                 "fp1_start": "2023-03-03T11:30:00Z",
                                 "qualifying_start": "2023-03-04T15:00:00Z",
                                 "race_start": "2023-03-05T18:00:00+03:00"},
            cutoff=reference.cutoff.iloc[0], entries=entries,
        )
        actual = _build_from_tables(snapshot, self.tables, self.completed).frame.set_index("driver_id")
        self.assertEqual(set(actual.index), set(reference.index))
        numeric = [name for name in CANDIDATE_FEATURES if name not in CATEGORICAL_FEATURES]
        np.testing.assert_allclose(actual.loc[reference.index, numeric].to_numpy(float),
                                   reference[numeric].to_numpy(float), rtol=1e-10, atol=1e-10,
                                   equal_nan=True)
        for name in CATEGORICAL_FEATURES:
            if name == "driver_id":
                continue
            self.assertEqual(actual.loc[reference.index, name].fillna("<missing>").tolist(),
                             reference[name].fillna("<missing>").tolist(), name)

    def test_unfinished_prior_event_and_same_day_result_fail(self):
        tables = {name: table.copy() for name, table in self.tables.items()}
        previous = tables["races"].loc[tables["races"].id.eq(9999)].copy()
        previous.loc[:, "id"] = 9998
        previous.loc[:, "date"] = "2026-09-27"
        tables["races"] = pd.concat([tables["races"], previous], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "lack completed results"):
            _completed_tables(self.snapshot, tables, self.completed)
        tables["races"] = tables["races"][tables["races"].id.ne(9998)]
        same_day = previous.copy()
        same_day.loc[:, "date"] = "2026-10-01"
        tables["races"] = pd.concat([tables["races"], same_day], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Same-day results"):
            _completed_tables(self.snapshot, tables, self.completed | {9998})

    def test_manifest_requires_cutoff_validated_tables_and_no_current_result(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SourceStore(Path(folder))
            with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T09:00:00Z"):
                refs = {name: store.capture("f1db", f"f1db:unit/{filename}",
                                            b"raceId\n1\n" if name == "results" else b"id\n1\n")
                        for name, filename in RAW_FILES.items()}
                payload = json.dumps({"version": "unit", "tables": refs,
                                      "result_race_ids": [1]}).encode()
                manifest = store.capture("f1db", "f1db:unit/manifest", payload)
            snapshot = replace(self.snapshot, sources={"history": manifest, "registry": manifest})
            tables, ids = _source_tables(snapshot, store)
            self.assertEqual(set(tables), set(RAW_FILES))
            self.assertEqual(ids, {1})
            with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T10:01:00Z"):
                late = store.capture("f1db", "f1db:unit/late", b"id\n2\n")
                late_payload = json.dumps({"version": "unit", "tables": {**refs, "results": late},
                                           "result_race_ids": [1]}).encode()
                late_manifest = store.capture("f1db", "f1db:unit/late-manifest", late_payload)
            with self.assertRaisesRegex(ValueError, "after cutoff"):
                _source_tables(replace(snapshot, sources={**snapshot.sources, "history": late_manifest}), store)
            with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T09:00:00Z"):
                current = store.capture("f1db", "f1db:unit/current-result",
                                        json.dumps({"version": "unit", "tables": refs,
                                                    "result_race_ids": [1, 9999]}).encode())
            with self.assertRaisesRegex(ValueError, "current-race result"):
                _source_tables(replace(snapshot, sources={**snapshot.sources, "history": current}), store)
            with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T09:00:00Z"):
                omitted = store.capture("f1db", "f1db:unit/omitted-result",
                                        json.dumps({"version": "unit", "tables": refs,
                                                    "result_race_ids": [2]}).encode())
            with self.assertRaisesRegex(ValueError, "ledger differs"):
                _source_tables(replace(snapshot, sources={**snapshot.sources, "history": omitted}), store)

    def test_public_builder_requires_reviewed_as_of_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SourceStore(Path(folder))
            entries = [dict(entry) for entry in self.snapshot.entries[:2]]
            identity = {key: self.snapshot.race[key] for key in (
                "race_id", "year", "round", "grand_prix_id", "circuit_id"
            )}
            scope = {**identity, "session": "event"}
            raw = self.tables["races"].loc[self.tables["races"].id.eq(9999)].iloc[0]
            with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T09:00:00Z"):
                entry_ref = store.capture("fia", "https://www.fia.com/unit-test/entry-list",
                                          b"SYNTHETIC ENTRY LIST", document_kind="entry_list")
                entry_ref = store.review(
                    entry_ref, applicability={**scope, "driver_constructor_pairs": [
                        [entry["driver_id"], entry["constructor_id"]] for entry in entries
                    ]}, reviewed_by="unit-test", notes="Synthetic event pairing"
                )
                for entry in entries:
                    entry["evidence"] = [entry_ref]
                schedule_ref = store.capture("fia", "https://www.fia.com/unit-test/schedule",
                                             b"SYNTHETIC SCHEDULE", document_kind="event_schedule")
                schedule_ref = store.review(schedule_ref, applicability={
                    **scope, **{key: self.snapshot.race[key] for key in (
                        "fp1_start", "qualifying_start", "race_start"
                    )}, "qualifying_format": raw.qualifyingFormat,
                    "is_sprint_weekend": False,
                }, reviewed_by="unit-test", notes="Synthetic exact session review")
                roster_packet = {"policy_version": self.snapshot.policy_version,
                                 "race_id": self.snapshot.race["race_id"],
                                 "entries": entries, "unresolved_conflicts": []}
                roster_ref = store.capture("formation", "formation:synthetic-roster",
                                           json.dumps(roster_packet).encode(),
                                           document_kind="early_roster_review")
                roster_ref = store.review(
                    roster_ref, applicability={**scope, "roster_complete": True,
                                                "entrant_ids": [entry["driver_id"] for entry in entries],
                                                "engine_manufacturer_by_driver": {
                                                    entry["driver_id"]: None for entry in entries
                                                }},
                    reviewed_by="unit-test", notes="Synthetic complete field",
                    supporting_observations=[entry_ref],
                )
                refs = {name: store.capture("f1db", f"f1db:synthetic/{filename}",
                                            self.tables[name].to_csv(index=False).encode())
                        for name, filename in RAW_FILES.items()}
                manifest = store.capture("f1db", "f1db:synthetic/manifest", json.dumps({
                    "version": "synthetic-as-of", "tables": refs,
                    "result_race_ids": sorted(self.completed),
                }).encode())
            snapshot = replace(self.snapshot, entries=entries, sources={
                "schedule": schedule_ref, "roster": roster_ref,
                "history": manifest, "registry": manifest,
            })
            batch = build_live_features(snapshot, store)
            self.assertEqual(len(batch.frame), 2)
            self.assertNotIn(9999, batch.historical_race_ids)
            with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T09:00:00Z"):
                bad_schedule = store.review(
                    schedule_ref,
                    applicability={**store.read(schedule_ref)[0]["applicability"],
                                   "is_sprint_weekend": True},
                    reviewed_by="unit-test", notes="Synthetic conflicting sprint designation",
                )
                bad_roster = store.review(
                    roster_ref,
                    applicability={**store.read(roster_ref)[0]["applicability"],
                                   "engine_manufacturer_by_driver": {
                                       entries[0]["driver_id"]: "invented-engine",
                                       entries[1]["driver_id"]: None,
                                   }},
                    reviewed_by="unit-test", notes="Synthetic conflicting engine mapping",
                    supporting_observations=[entry_ref],
                )
            with self.assertRaisesRegex(ValueError, "sprint format"):
                build_live_features(replace(snapshot, sources={
                    **snapshot.sources, "schedule": bad_schedule,
                }), store)
            with self.assertRaisesRegex(ValueError, "engine identity"):
                build_live_features(replace(snapshot, sources={
                    **snapshot.sources, "roster": bad_roster,
                }), store)
            output = Path(folder) / "captures"
            with self.assertRaisesRegex(ValueError, "remain private"):
                capture_early_snapshot(snapshot, store, output)
            private_patch = patch("f1_predictor.early.archive.PRIVATE_ROOT", Path(folder))
            private_patch.start()
            self.addCleanup(private_patch.stop)
            with patch("f1_predictor.early.archive.now", return_value="2026-10-01T09:59:59Z"):
                with self.assertRaisesRegex(ValueError, "after cutoff"):
                    capture_early_snapshot(snapshot, store, output)
            with patch("f1_predictor.early.archive.now", return_value="2026-10-02T10:00:00Z"):
                with self.assertRaisesRegex(ValueError, "before FP1"):
                    capture_early_snapshot(snapshot, store, output)
            with patch("f1_predictor.early.archive.now", return_value="2026-10-01T10:05:00Z"):
                archived = capture_early_snapshot(snapshot, store, output)
            self.assertEqual(inspect_early_capture(archived)["prediction_status"],
                             "capture_only_no_model_output")
            self.assertEqual(replay_early_capture(archived)["status"], "exact_capture_replay")
            self.assertFalse((archived / "predictions.json").exists())
            with patch("f1_predictor.early.archive.now", return_value="2026-10-01T10:06:00Z"):
                with self.assertRaises(FileExistsError):
                    capture_early_snapshot(snapshot, store, output)
            features = archived / "features.json"
            features.write_bytes(features.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "changed frozen artifact"):
                inspect_early_capture(archived)


if __name__ == "__main__":
    unittest.main()
