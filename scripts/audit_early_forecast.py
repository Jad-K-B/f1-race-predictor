"""Generate an exclusive-create offline Early v1 policy/availability audit."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_predictor.early.audit import write_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, help="Existing private Stage 3 cache; read-only, no fetches")
    args = parser.parse_args()
    try:
        result = write_audit(ROOT, args.output, args.source_store)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "research_races": result["research_races"],
                      "features": result["feature_decisions"],
                      "evidence": result["evidence_inventory"], "output": str(args.output)}, indent=2))
