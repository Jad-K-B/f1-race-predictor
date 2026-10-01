"""CLI smoke and fail-closed checks; no live source requests."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/early_capture.py"


class EarlyCaptureCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(CLI), *map(str, args)],
                              cwd=ROOT, text=True, capture_output=True, check=False)

    def test_commands_are_available_without_network(self):
        for command in ((), ("capture",), ("inspect",), ("replay",)):
            with self.subTest(command=command):
                result = self.run_cli(*command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unlabeled", self.run_cli("--help").stdout.lower())

    def test_invalid_snapshot_cannot_create_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "bad.json"
            snapshot.write_text("{}", encoding="utf-8")
            output = Path(folder) / "capture"
            result = self.run_cli("capture", "--snapshot", snapshot,
                                  "--store", Path(folder) / "sources",
                                  "--output-root", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_incomplete_archive_is_not_inspectable(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli("inspect", "--archive", folder)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
