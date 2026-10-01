"""Capture or replay a private, unlabeled early-race snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.archive import (
    capture_early_snapshot, inspect_early_capture, replay_early_capture,
)
from f1_predictor.early.contracts import EarlySnapshot
from f1_predictor.stage3.sources import SourceStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Archive reviewed source bytes and unlabeled features before FP1")
    capture.add_argument("--snapshot", type=Path, required=True)
    capture.add_argument("--store", type=Path, required=True)
    capture.add_argument("--output-root", type=Path, default=ROOT / "data/stage3/evidence-private/early-captures")
    for name in ("inspect", "replay"):
        command = commands.add_parser(name, help=f"{name.capitalize()} an existing private early capture")
        command.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        snapshot = EarlySnapshot.from_dict(json.loads(args.snapshot.read_text(encoding="utf-8")))
        directory = capture_early_snapshot(snapshot, SourceStore(args.store), args.output_root)
        manifest = inspect_early_capture(directory)
        result = {"archive": str(directory), "snapshot_id": manifest["snapshot_id"],
                  "prediction_status": manifest["prediction_status"],
                  "generated_at": manifest["generated_at"]}
    elif args.command == "inspect":
        manifest = inspect_early_capture(args.archive)
        result = {"snapshot_id": manifest["snapshot_id"],
                  "prediction_status": manifest["prediction_status"],
                  "generated_at": manifest["generated_at"],
                  "files_verified": len(manifest["files_sha256"])}
    else:
        result = replay_early_capture(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
