"""Train and validate Stage 2B without opening the locked test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from f1_predictor.stage2b_training import (  # noqa: E402
    Stage2BConfig,
    run_stage2b_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune Stage 2B on train-era folds and evaluate 2023 validation"
    )
    parser.add_argument(
        "--stage2a-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "stage2a",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stage2b" / "validation",
    )
    args = parser.parse_args()
    result = run_stage2b_validation(args.stage2a_dir, args.output_dir, Stage2BConfig())
    summary = {
        "test_status": result["manifest"]["test_status"],
        "selected_binary_models": result["manifest"]["selected_binary_models"],
        "selected_ranking_model": result["manifest"]["selected_ranking_model"],
        "selected_calibrators": result["manifest"]["selected_calibrators"],
        "selected_reconciliation_alpha": result["manifest"][
            "selected_reconciliation_alpha"
        ],
        "final_pipeline_held_out_validation": result["metrics"][
            "final_pipeline_held_out_validation"
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
