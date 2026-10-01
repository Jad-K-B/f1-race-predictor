"""Build private retrospective early features and separate outcomes, offline."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.research_dataset import write_dataset
from f1_predictor.stage3.sources import SourceStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = (ROOT / "data/stage3/evidence-private").resolve()
    if not args.output.resolve().is_relative_to(private):
        parser.error("Research features and raw sources must stay in the private evidence directory")
    try:
        report = write_dataset(ROOT, args.packets, SourceStore(args.source_store), args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({k: report[k] for k in ("rows", "races", "feature_count", "blocks", "training_roster_coverage_complete")}, indent=2))
