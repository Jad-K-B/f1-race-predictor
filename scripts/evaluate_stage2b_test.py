"""Run the explicitly approved, one-time Stage 2B locked test evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_predictor.stage2b_evaluation import evaluate_locked_test  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked Stage 2B test split")
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Required explicit confirmation that final test evaluation is approved.",
    )
    parser.add_argument(
        "--stage2a-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "stage2a",
    )
    parser.add_argument(
        "--validation-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage2b" / "validation",
    )
    parser.add_argument(
        "--external-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage2b" / "external_validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage2b" / "final_test",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    args = parser.parse_args()
    if not args.confirm_locked_test:
        parser.error("--confirm-locked-test is required after explicit user approval")
    result = evaluate_locked_test(
        args.stage2a_dir,
        args.validation_artifact_dir,
        args.output_dir,
        external_artifact_dir=args.external_artifact_dir,
        confirm_locked_test=True,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
