"""Compatibility diagnostics, never model selection or a new test evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..stage2a import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, load_eligibility_audit
from .features import build_features
from .replay import replay_snapshot
from .release import write_json, sha256
from .sources import SourceStore, now


def validate_2023_replay(root: Path, store: SourceStore, history_source: str, output: Path) -> dict:
    path = root / "data/processed/stage2a/validation.csv"
    # Read feature values and eligibility only. No prediction outcomes are needed.
    reference = pd.read_csv(path, usecols=["race_id", "prediction_eligible", *FEATURE_COLUMNS])
    audit = load_eligibility_audit()
    report = {"generated_at": now(), "purpose": "feature compatibility only, not model selection",
              "evidence_mode": "approximate_retrospective", "validation_file_sha256": sha256(path),
              "history_source": history_source, "feature_count": len(FEATURE_COLUMNS), "races": []}
    for race_id in reference.race_id.unique():
        record = {"race_id": int(race_id), "mismatches": []}
        try:
            snapshot = replay_snapshot(store, history_source, int(race_id), audit)
            actual = build_features(snapshot, store).frame.set_index("driver_id").sort_index()
            expected = reference[reference.race_id.eq(race_id) & reference.prediction_eligible.eq(1)].set_index("driver_id").sort_index()
            record["eligible_rows"] = len(actual)
            if actual.index.tolist() != expected.index.tolist():
                record["mismatches"].append("entrant_roster")
            else:
                for column in NUMERIC_FEATURES:
                    if not np.allclose(actual[column].to_numpy(float), expected[column].to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True):
                        record["mismatches"].append(column)
                for column in set(CATEGORICAL_FEATURES) - {"driver_id"}:
                    if actual[column].fillna("__missing__").tolist() != expected[column].fillna("__missing__").tolist():
                        record["mismatches"].append(column)
        except (ValueError, KeyError) as error:
            record["error"] = str(error)
        report["races"].append(record)
    report["passed"] = all(not r["mismatches"] and "error" not in r for r in report["races"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
