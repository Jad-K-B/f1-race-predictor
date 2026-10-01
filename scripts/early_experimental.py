"""Private experimental early release preparation and inference; never publishes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.experimental import (
    ExperimentalEarlyPredictor, archive_private_forecast,
    prepare_experimental_release, replay_private_forecast, verify_experimental_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and inspect private experimental early forecasts")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Copy verified research models into a private release")
    prepare.add_argument("--research-bundle", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify a private experimental release")
    verify.add_argument("--bundle", type=Path, required=True)
    forecast = commands.add_parser("forecast", help="Archive a private prediction from a verified capture")
    forecast.add_argument("--bundle", type=Path, required=True)
    forecast.add_argument("--capture", type=Path, required=True)
    forecast.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="Reproduce a saved private forecast offline")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--capture", type=Path, required=True)
    replay.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"private_bundle": str(prepare_experimental_release(
            args.research_bundle, args.output))}))
    elif args.command == "verify":
        manifest = verify_experimental_release(args.bundle)
        print(json.dumps({"status": manifest["status"],
                          "public_release_authorized": manifest["public_release_authorized"]}))
    elif args.command == "forecast":
        predictor = ExperimentalEarlyPredictor(args.bundle)
        print(json.dumps({"private_archive": str(archive_private_forecast(
            predictor, args.capture, args.output))}))
    else:
        predictor = ExperimentalEarlyPredictor(args.bundle)
        print(json.dumps(replay_private_forecast(predictor, args.capture, args.archive)))


if __name__ == "__main__":
    main()
