import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.stage3.archive import export_evidence, inspect_evidence
from f1_predictor.stage3.contracts import RaceSnapshot
from f1_predictor.stage3.sources import SourceStore


class ProspectiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SourceStore(Path(self.temp.name) / "cache")
        self.clock = patch("f1_predictor.stage3.sources.now", return_value="2026-09-27T10:00:00Z")
        self.clock.start()
        self.addCleanup(self.clock.stop)
        fixture = json.loads((ROOT / "tests/fixtures/replay-1080-snapshot.json").read_text())
        race = {**fixture["race"], "race_id": 9999, "year": 2026, "round": 16,
                "race_date": "2026-10-04", "race_start": "2026-10-04T07:00:00Z"}
        self.scope = {k: race[k] for k in ("race_id", "year", "round", "grand_prix_id", "circuit_id")}
        entries = [dict(e) for e in fixture["entries"][:2]]
        self.ids = [e["driver_id"] for e in entries]
        sources = {role: self.review(role) for role in ("schedule", "entries", "eligibility", "qualifying", "grid")}
        sources["registry"] = self.store.capture("test", "fixture:registry", b"unit test only")
        for i, entry in enumerate(entries):
            entry.update(eligibility_source=sources["eligibility"], eligible=True, grid_position=i + 1,
                         pit_lane_start=False, has_grid_penalty=False, grid_penalty_positions=0)
        self.snapshot = RaceSnapshot("unit-test-only", "confirmed_grid", "2026-09-27T10:00:01Z",
                                     race, entries, sources, sources["registry"], "unit-test-reviewer")

    def review(self, role, **changes):
        session = {"schedule": "event", "entries": "event", "eligibility": "race", "qualifying": "qualifying", "grid": "race"}[role]
        scope = {**self.scope, "session": session, "entrant_ids": self.ids,
                 "race_start": "2026-10-04T07:00:00Z", **changes}
        kind = {"schedule": "calendar", "entries": "entry_list", "eligibility": "eligibility_review",
                "grid": "final_grid", "qualifying": "qualifying"}[role]
        source = self.store.capture("fia", f"https://www.fia.com/unit-test-only/{role}",
                                    b"unit test only; not race evidence", document_kind=kind)
        return self.store.review(source, applicability=scope, reviewed_by="unit-test-reviewer",
                                 notes="Synthetic contract fixture, never used for forecasting")

    def test_unknown_publication_is_honest_and_accepted_after_observation(self):
        self.snapshot.validate(self.store)
        record, _ = self.store.read(self.snapshot.sources["grid"])
        self.assertIsNone(record["published_at"])

    def test_wrong_event_session_start_and_incomplete_entrant_review_fail_closed(self):
        for role, changes in (("grid", {"race_id": 9998}), ("qualifying", {"session": "race"}),
                              ("schedule", {"race_start": "2026-10-04T08:00:00Z"}),
                              ("eligibility", {"entrant_ids": self.ids[:1]})):
            with self.subTest(role=role):
                sources = {**self.snapshot.sources, role: self.review(role, **changes)}
                with self.assertRaises(ValueError):
                    replace(self.snapshot, sources=sources).validate(self.store)

    def test_review_cannot_refresh_stale_bytes_or_backdate_availability(self):
        original = self.snapshot.sources["grid"]
        with patch("f1_predictor.stage3.sources.now", return_value="2026-09-28T10:00:00Z"):
            reviewed = self.store.review(original, applicability={**self.scope, "session": "race"},
                                         reviewed_by="test", notes="Later review")
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            self.store.available(reviewed, self.snapshot.cutoff, 3600)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.available(reviewed, "2026-09-28T10:00:01Z", 3600)
        self.assertEqual(self.store.read(reviewed)[0]["retrieved_at"], self.store.read(original)[0]["retrieved_at"])

    def test_http_modified_is_not_publication_and_legacy_ambiguous_time_rejected(self):
        source = self.store.capture("fia", "https://www.fia.com/unit-test-header", b"fixture",
                                    http_last_modified="Sat, 26 Sep 2026 10:00:00 GMT")
        self.assertIsNone(self.store.read(source)[0]["published_at"])
        with self.assertRaises(ValueError):
            self.store.capture("fia", "uri", b"fixture", publication_basis="http_last_modified")
        source = self.store.capture("fia", "https://www.fia.com/unit-test-legacy", b"fixture",
                                    published_at="2026-09-26T10:00:00Z", document_kind="final_grid")
        sources = {**self.snapshot.sources, "grid": source}
        with self.assertRaisesRegex(ValueError, "verified basis"):
            replace(self.snapshot, sources=sources).validate(self.store)

    def test_eligible_driver_also_requires_authoritative_eligibility_reference(self):
        entries = [dict(e) for e in self.snapshot.entries]
        entries[0]["eligibility_source"] = self.snapshot.sources["qualifying"]
        with self.assertRaises(ValueError):
            replace(self.snapshot, entries=entries).validate(self.store)

    def test_provisional_grid_document_cannot_confirm_grid(self):
        source = self.store.capture("fia", "https://www.fia.com/unit-test-provisional", b"fixture",
                                    document_kind="provisional_grid")
        reviewed = self.store.review(source, applicability={**self.scope, "session": "race"},
                                     reviewed_by="test", notes="Unit test provisional document")
        with self.assertRaisesRegex(ValueError, "Wrong FIA document kind"):
            replace(self.snapshot, sources={**self.snapshot.sources, "grid": reviewed}).validate(self.store)

    def test_private_export_preserves_revision_support_and_history_closure(self):
        old = self.store.capture("test", "fixture:revision", b"old")
        new = self.store.capture("test", "fixture:revision", b"new")
        table = self.store.capture("f1db", "f1db:fixture/races.csv", b"fixture")
        history = self.store.capture("f1db", "f1db:fixture/manifest", json.dumps({"tables": {"races": table}}).encode())
        review = self.store.review(new, applicability={**self.scope, "session": "race"},
                                   reviewed_by="test", notes="Unit test", supporting_observations=[history])
        destination = Path(self.temp.name) / "export"
        report = export_evidence(self.store, [review], destination)
        self.assertEqual(set(report["observations"]), {old, new, review, history, table})
        for reference in report["observations"]:
            self.assertEqual(SourceStore(destination / "sources").read(reference), self.store.read(reference))
        with self.assertRaises(FileExistsError):
            export_evidence(self.store, [review], destination)
        blob = self.store.read(table)[0]["blob"]
        (destination / "sources" / blob).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            inspect_evidence(destination)

    def test_later_amendment_cannot_be_added_to_earlier_review(self):
        with patch("f1_predictor.stage3.sources.now", return_value="2026-09-28T10:00:00Z"):
            later = self.store.capture("fia", "https://www.fia.com/unit-test-amendment", b"fixture")
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            self.store.review(self.snapshot.sources["grid"], applicability={**self.scope, "session": "race"},
                              reviewed_by="test", notes="fixture", supporting_observations=[later])
