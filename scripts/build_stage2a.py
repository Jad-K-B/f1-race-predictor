"""Command-line entry point for the Stage 2A dataset build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_predictor.stage2a import (  # noqa: E402
    Stage2AConfig,
    build_stage2a_dataset,
    write_stage2a_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe Stage 2A dataset")
    parser.add_argument(
        "--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "stage2a",
    )
    args = parser.parse_args()

    config = Stage2AConfig()
    dataset = build_stage2a_dataset(args.raw_dir, config=config)
    metadata = write_stage2a_outputs(
        dataset, args.raw_dir, args.output_dir, config=config
    )
    print(json.dumps(metadata["dataset"], indent=2))
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
