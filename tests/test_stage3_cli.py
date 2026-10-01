import json
import io
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.stage3.sources import SourceStore, import_f1db
from f1_predictor.stage3.draft import draft_snapshot
from f1_predictor.stage3.contracts import RaceSnapshot


class CLITests(unittest.TestCase):
    def test_source_network_failure_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [str(ROOT / "scripts/stage3.py"), "--cache", directory,
                         "fetch-openf1", "--endpoint", "starting_grid", "--session-key", "11377"]
            stderr = io.StringIO()
            with patch.object(sys, "argv", arguments), redirect_stderr(stderr), patch(
                "f1_predictor.stage3.sources.OpenF1.fetch", side_effect=URLError("source unavailable")
            ), self.assertRaises(SystemExit) as raised:
                runpy.run_path(arguments[0], run_name="__main__")
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("source unavailable", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_help_and_no_accidental_training_command(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/stage3.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("draft-snapshot", result.stdout)
        self.assertIn("capture", result.stdout)
        self.assertNotIn("train-model", result.stdout)

    def test_draft_never_guesses_roster_or_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SourceStore(Path(directory))
            history = import_f1db(store, ROOT / "data/raw", "draft-test")
            packet = draft_snapshot(store, history, 1165, "draft", "pre_weekend", [{"driver_id": "unseen-driver", "constructor_id": "reviewed-team"}])
            self.assertIsNone(packet["entries"][0]["eligible"])
            self.assertIsNone(packet["entries"][0]["grid_position"])
            self.assertEqual(packet["reviewed_by"], "")
            with self.assertRaises(ValueError):
                RaceSnapshot.from_dict(packet).validate(store)
