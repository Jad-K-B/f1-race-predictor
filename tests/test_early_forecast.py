"""Synthetic contract fixtures only; these are never forecasts or source evidence."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.early.contracts import EarlySnapshot
from f1_predictor.early.policy import (CANDIDATE_FEATURES, EXCLUDED_FEATURES,
    POLICY_VERSION, feature_lineage, scheduled_cutoff, validate_feature_names)
from f1_predictor.stage2a import FEATURE_COLUMNS
from f1_predictor.stage3.sources import SourceStore


class EarlyContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SourceStore(Path(self.temp.name))
        self.clock = patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T09:30:00Z")
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.race = {"race_id": 9999, "year": 2026, "round": 1, "grand_prix_id": "unit-test",
                     "circuit_id": "unit-test-circuit", "fp1_start": "2026-10-02T10:00:00Z",
                     "qualifying_start": "2026-10-03T10:00:00Z", "race_start": "2026-10-04T10:00:00Z"}
        self.pairs = [["unit-driver-a", "unit-team"], ["unit-driver-b", "unit-team"]]
        self.scope = {**self.race, "session": "event", "driver_constructor_pairs": self.pairs}
        entry_source = self.document("entry_list")
        self.entries = [{"driver_id": driver, "constructor_id": team, "engine_manufacturer_id": None,
                         "status": "announced", "evidence": [entry_source],
                         "reason": "SYNTHETIC UNIT TEST ONLY"} for driver, team in self.pairs]
        registry = self.store.capture("f1db", "f1db:unit/registry", b"synthetic registry")
        table = self.store.capture("f1db", "f1db:unit/races", b"synthetic history")
        history = self.store.capture("f1db", "f1db:unit/manifest",
                                     json.dumps({"version": "unit-only", "tables": {"races": table}}).encode())
        self.snapshot = EarlySnapshot("unit-only", "2026-10-01T10:00:00Z", self.race,
                                      self.entries, {"schedule": self.document("calendar"),
                                                     "roster": self.roster(self.entries),
                                                     "registry": registry, "history": history}, "unit-reviewer")

    def document(self, kind, *, provider="fia", url=None, scope=None, **capture):
        ref = self.store.capture(provider, url or f"https://www.fia.com/unit-test-only/{kind}",
                                 b"SYNTHETIC TEST ONLY; NOT RACE EVIDENCE", document_kind=kind, **capture)
        return self.store.review(ref, applicability=scope or self.scope, reviewed_by="unit-reviewer",
                                 notes="Synthetic fixture; not a claim about a real event")

    def roster(self, entries, conflicts=None, complete=True):
        packet = {"policy_version": POLICY_VERSION, "race_id": self.race["race_id"],
                  "entries": entries, "unresolved_conflicts": conflicts or []}
        ref = self.store.capture("formation", "formation:unit-early-roster", json.dumps(packet).encode(),
                                 document_kind="early_roster_review")
        scope = {**self.scope, "roster_complete": complete, "entrant_ids": [e["driver_id"] for e in entries]}
        return self.store.review(ref, applicability=scope, reviewed_by="unit-reviewer",
                                 notes="Synthetic complete roster", supporting_observations=
                                 sorted({ref for entry in entries for ref in entry["evidence"]}))

    def with_entries(self, entries, **review):
        return replace(self.snapshot, entries=entries,
                       sources={**self.snapshot.sources, "roster": self.roster(entries, **review)})

    def test_valid_announced_field_has_no_final_grid_eligibility(self):
        self.snapshot.validate(self.store)
        self.assertNotIn("eligible", self.entries[0])
        self.assertEqual(EarlySnapshot.from_dict(self.snapshot.to_dict()), self.snapshot)

    def test_all_147_features_classified_without_changing_frozen_order(self):
        rows = feature_lineage()
        self.assertEqual([row["name"] for row in rows], FEATURE_COLUMNS)
        self.assertEqual(len(rows), 147)
        self.assertEqual(sum(row["early_decision"] == "exclude" for row in rows), 22)
        self.assertEqual(sum(row["early_decision"] == "rebuild" for row in rows), 1)
        self.assertEqual(sum(row["early_decision"] == "candidate_requires_evidence" for row in rows), 124)
        validate_feature_names(CANDIDATE_FEATURES)

    def test_feature_allowlist_rejects_all_current_session_and_outcome_fields(self):
        for field in EXCLUDED_FEATURES | {"race_winner", "finish_order", "result_status", "prediction_eligible", "weather"}:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_feature_names(["driver_prior_entries", field])
        for names in ([], ["driver_id", "driver_id"]):
            with self.assertRaises(ValueError):
                validate_feature_names(names)
        validate_feature_names(["driver_previous_grid", "driver_recent_5_avg_qualifying"])

    def test_snapshot_rejects_qualifying_grid_or_outcome_injection(self):
        for field in ("grid_position", "qualifying_position", "finish_order", "eligible", "race_winner"):
            entries = copy.deepcopy(self.entries)
            entries[0][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "allowlist"):
                self.with_entries(entries).validate(self.store)

    def test_session_order_and_cutoff_are_strict(self):
        for cutoff in ("2026-10-01T10:01:00Z", "2026-10-03T10:01:00Z", "2026-10-01T10:00:00"):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                replace(self.snapshot, cutoff=cutoff).validate(self.store)
        race = {**self.race, "qualifying_start": "2026-10-02T09:00:00Z"}
        with self.assertRaisesRegex(ValueError, "chronology"):
            replace(self.snapshot, race=race).validate(self.store)

    def test_cutoff_uses_utc_across_offset_and_calendar_boundaries(self):
        self.assertEqual(scheduled_cutoff("2026-01-01T01:00:00+03:00"), "2025-12-30T22:00:00+00:00")

    def test_wrong_event_pairing_and_incomplete_roster_fail_closed(self):
        ref = self.document("calendar", scope={**self.scope, "race_id": 9998})
        with self.assertRaisesRegex(ValueError, "applicability"):
            replace(self.snapshot, sources={**self.snapshot.sources, "schedule": ref}).validate(self.store)
        entries = copy.deepcopy(self.entries)
        entries[0]["constructor_id"] = "wrong-team"
        with self.assertRaisesRegex(ValueError, "pairing"):
            self.with_entries(entries).validate(self.store)
        with self.assertRaisesRegex(ValueError, "completeness"):
            self.with_entries(self.entries, complete=False).validate(self.store)

    def test_conflict_and_unreviewed_entry_changes_fail(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.with_entries(self.entries, conflicts=["unresolved substitute"]).validate(self.store)
        entries = copy.deepcopy(self.entries)
        entries[0]["status"] = "withdrawn"
        with self.assertRaisesRegex(ValueError, "reviewed bytes"):
            replace(self.snapshot, entries=entries).validate(self.store)

    def test_missing_session_does_not_exclude_announced_driver(self):
        self.snapshot.validate(self.store)
        entries = copy.deepcopy(self.entries)
        entries[0]["status"] = "withdrawn"
        entries[0]["reason"] = "No qualifying or grid record"
        with self.assertRaisesRegex(ValueError, "Withdrawal"):
            self.with_entries(entries).validate(self.store)

    def test_pre_cutoff_withdrawal_can_be_recorded_without_rewriting_field(self):
        entries = copy.deepcopy(self.entries)
        entries[0]["status"] = "withdrawn"
        entries[0]["evidence"].append(self.document("withdrawal"))
        self.with_entries(entries).validate(self.store)
        entries[0]["status"] = "announced"
        with self.assertRaisesRegex(ValueError, "Withdrawal"):
            self.with_entries(entries).validate(self.store)

    def test_later_withdrawal_cannot_enter_early_snapshot(self):
        entries = copy.deepcopy(self.entries)
        with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T10:01:00Z"):
            entries[0]["evidence"].append(self.document("withdrawal"))
            entries[0]["status"] = "withdrawn"
            late = self.with_entries(entries)
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            late.validate(self.store)
        self.snapshot.validate(self.store)

    def test_late_observation_cannot_be_backdated_by_old_publication(self):
        with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T10:01:00Z"):
            late = self.document("calendar", published_at="2026-09-01T00:00:00Z", publication_basis="document")
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            replace(self.snapshot, sources={**self.snapshot.sources, "schedule": late}).validate(self.store)

    def test_unknown_publication_is_accepted_only_after_observation(self):
        self.assertIsNone(self.store.read(self.snapshot.sources["schedule"])[0]["published_at"])
        self.snapshot.validate(self.store)
        legacy = self.document("calendar", published_at="2026-09-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "verified basis"):
            replace(self.snapshot, sources={**self.snapshot.sources, "schedule": legacy}).validate(self.store)

    def test_results_and_unapproved_hosts_cannot_be_entry_evidence(self):
        for ref in (self.document("results"), self.document("entry_list", url="https://unapproved.invalid/list")):
            entries = copy.deepcopy(self.entries)
            entries[0]["evidence"] = [ref]
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.with_entries(entries).validate(self.store)

    def test_team_route_requires_season_registration_and_event_lineup(self):
        season = self.document("season_registration")
        lineup = self.document("event_lineup", provider="team", url="https://unit-team.test/lineup")
        entries = copy.deepcopy(self.entries)
        for entry in entries:
            entry["evidence"] = [season]
        with self.assertRaisesRegex(ValueError, "season registration plus"):
            self.with_entries(entries).validate(self.store)
        for entry in entries:
            entry["evidence"].append(lineup)
        snapshot = self.with_entries(entries)
        with self.assertRaisesRegex(ValueError, "hosts"):
            snapshot.validate(self.store)
        snapshot.validate(self.store, approved_team_hosts=frozenset({"unit-team.test"}))

    def test_stale_schedule_cannot_be_refreshed_by_review(self):
        with patch("f1_predictor.stage3.sources.now", return_value="2026-09-28T10:00:00Z"):
            stale = self.document("calendar", url="https://www.fia.com/unit-test-only/stale-calendar")
        reviewed = self.store.review(stale, applicability=self.scope, reviewed_by="unit-reviewer", notes="Still old bytes")
        with self.assertRaisesRegex(ValueError, "stale"):
            replace(self.snapshot, sources={**self.snapshot.sources, "schedule": reviewed}).validate(self.store)

    def test_future_historical_table_is_rejected_even_with_earlier_manifest(self):
        with patch("f1_predictor.stage3.sources.now", return_value="2026-10-01T10:01:00Z"):
            table = self.store.capture("f1db", "f1db:unit/later-table", b"later correction")
        manifest = self.store.capture("f1db", "f1db:unit/earlier-manifest",
                                     json.dumps({"version": "unit", "tables": {"results": table}}).encode())
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            replace(self.snapshot, sources={**self.snapshot.sources, "history": manifest}).validate(self.store)

    def test_results_cannot_masquerade_as_schedule(self):
        ref = self.document("results")
        with self.assertRaisesRegex(ValueError, "schedule"):
            replace(self.snapshot, sources={**self.snapshot.sources, "schedule": ref}).validate(self.store)

    def test_new_rookie_requires_evidence_and_unseen_identity_is_supported(self):
        entries = copy.deepcopy(self.entries)
        entries[0]["driver_id"] = "unit-unseen-rookie"
        with self.assertRaisesRegex(ValueError, "pairing"):
            self.with_entries(entries).validate(self.store)
        scope = {**self.scope, "driver_constructor_pairs": [["unit-unseen-rookie", "unit-team"]]}
        ref = self.document("entry_list", url="https://www.fia.com/unit-test-only/substitution", scope=scope)
        entries[0]["evidence"] = [ref]
        self.with_entries(entries).validate(self.store)

    def test_changed_session_timestamp_must_match_reviewed_schedule(self):
        race = {**self.race, "qualifying_start": "2026-10-03T11:00:00Z"}
        with self.assertRaisesRegex(ValueError, "exact session"):
            replace(self.snapshot, race=race).validate(self.store)

    def test_duplicate_entrant_and_ambiguous_status_are_rejected(self):
        entries = copy.deepcopy(self.entries)
        entries.append(copy.deepcopy(entries[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.with_entries(entries).validate(self.store)
        entries = copy.deepcopy(self.entries)
        entries[0]["status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.with_entries(entries).validate(self.store)

    def test_publication_requires_cutoff_before_generation_and_fp1_deadline(self):
        self.snapshot.validate_publication_time("2026-10-01T10:00:01Z", "2026-10-01T10:05:00Z")
        for generated, published in (("2026-10-01T09:59:59Z", "2026-10-01T10:05:00Z"),
                                     ("2026-10-01T10:00:01Z", "2026-10-02T10:00:00Z"),
                                     ("2026-10-01T10:05:00Z", "2026-10-01T10:00:01Z")):
            with self.subTest(published=published), self.assertRaises(ValueError):
                self.snapshot.validate_publication_time(generated, published)


if __name__ == "__main__":
    unittest.main()
