"""Run the scikit-learn/XGBoost Stage 2B validation comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_predictor.stage2b_external import (  # noqa: E402
    ExternalStage2BConfig,
    run_external_stage2b_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare scikit-learn and XGBoost models on the three-block 2023 "
            "validation protocol without reading the locked test split"
        )
    )
    parser.add_argument(
        "--stage2a-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "stage2a",
    )
    parser.add_argument(
        "--numpy-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage2b" / "validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage2b" / "external_validation",
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()

    result = run_external_stage2b_validation(
        stage2a_dir=args.stage2a_dir,
        numpy_artifact_dir=args.numpy_artifact_dir,
        output_dir=args.output_dir,
        config=ExternalStage2BConfig(n_jobs=args.n_jobs),
    )
    summary = {
        "test_status": result["manifest"]["test_status"],
        "runtime": result["manifest"]["runtime"],
        "selected_families": result["manifest"]["selected_families"],
        "selected_ranker": result["manifest"]["selected_ranker"],
        "selected_reconciliation_alpha": result["manifest"][
            "selected_reconciliation_alpha"
        ],
        "held_out_validation": result["metrics"]["report_block"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
