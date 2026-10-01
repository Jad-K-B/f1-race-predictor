import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.stage3.sources import SourceStore, Jolpica, OpenF1, fetch_json, utc


class SourceTests(unittest.TestCase):
    def test_observation_revision_and_content_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SourceStore(Path(directory))
            with patch("f1_predictor.stage3.sources.now", return_value="2026-01-01T00:00:00Z"):
                first = store.capture("test", "https://source", b"old")
            with patch("f1_predictor.stage3.sources.now", return_value="2026-01-02T00:00:00Z"):
                second = store.capture("test", "https://source", b"new")
            record, data = store.read(second)
            self.assertEqual(data, b"new")
            self.assertEqual(record["previous_observation"], first)
            self.assertEqual(store.read(first)[1], b"old")
            with self.assertRaisesRegex(ValueError, "after cutoff"):
                store.available(second, "2026-01-01T12:00:00Z", 86400)
            with self.assertRaisesRegex(ValueError, "stale"):
                store.available(first, "2026-01-03T00:00:00Z", 86400)
            (Path(directory) / record["blob"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                store.read(second)

    def test_metadata_tampering_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SourceStore(Path(directory))
            key = store.capture("test", "uri", b"data")
            path = Path(directory) / key
            data = json.loads(path.read_text())
            data["first_observed_at"] = "1950-01-01T00:00:00Z"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                store.read(key)
            with self.assertRaises(ValueError):
                store.read("../../outside")

    def test_network_allowlist_and_no_implicit_latest(self):
        store = SourceStore(Path("unused"))
        with self.assertRaises(ValueError):
            fetch_json(store, "jolpica", "https://example.com/secret")
        with self.assertRaises(ValueError):
            OpenF1(store).fetch("starting_grid", session_key="latest")
        with self.assertRaises(ValueError):
            Jolpica(store).fetch(2026, "grid")
        with self.assertRaises(ValueError):
            utc("2026-01-01")

    def test_jolpica_pagination(self):
        pages = [({"MRData": {"limit": "100", "total": "150", "offset": str(i)}}, str(i)) for i in (0, 100)]
        with patch("f1_predictor.stage3.sources.fetch_json", side_effect=pages), patch("f1_predictor.stage3.sources.time.sleep"):
            result, ids = Jolpica(SourceStore(Path("unused"))).fetch(2023, "results")
        self.assertEqual(len(result), 2)
        self.assertEqual(ids, ["0", "100"])
