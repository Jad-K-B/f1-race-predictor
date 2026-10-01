"""Private, research-only replay bundle. No fitting or prospective release API."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from ..stage2a import BINARY_TARGET_COLUMNS
from ..stage2b_metrics import TARGET_PROBABILITY_COLUMNS, reconcile_probabilities
from ..stage2b_models import calibrator_from_dict
from ..stage3.release import checked, sha256, write_json
from .policy import CANDIDATE_FEATURES
from .research_training import (
    Candidate, FEATURE_SETS, RESEARCH_MODEL_SCHEMA, classifier_probabilities,
    rank_scores, ranker_candidates,
)

BUNDLE_SCHEMA = "early-private-research-bundle-v1"
METHODS = ("logistic", "random_forest", "xgboost", "prior_rate",
           "recent_form", "constructor_form", "selected_combination")
BLOCKS = ("selection", "calibration", "report")


def _environment() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def _verify_source(directory: Path, manifest: dict) -> None:
    if (manifest.get("schema_version") != RESEARCH_MODEL_SCHEMA or
            manifest.get("sealed_2024_2025_loaded") is not False):
        raise ValueError("Expected unchanged early research artifact")
    for name, digest in manifest["files_sha256"].items():
        checked(directory / name, digest)


def package_research_bundle(selection: Path, evaluation: Path,
                            output: Path) -> Path:
    """Copy previously fitted artifacts; never fit, tune, or publish a model."""
    root = Path(__file__).resolve().parents[3]
    private = (root / "data/stage3/evidence-private/early-research").resolve()
    if not output.resolve().is_relative_to(private):
        raise ValueError("Research bundle must remain in the private early-research archive")
    if output.exists():
        raise FileExistsError("Research bundles cannot be overwritten")
    selected = json.loads((selection / "manifest.json").read_text(encoding="utf-8"))
    evaluated = json.loads((evaluation / "manifest.json").read_text(encoding="utf-8"))
    _verify_source(selection, selected)
    _verify_source(evaluation, evaluated)
    if (selected.get("production_release_authorized") is not False or
            selected.get("calibration_and_report_labels_used") is not False or
            evaluated.get("selection_manifest_sha256") != sha256(selection / "manifest.json") or
            evaluated.get("dataset_manifest_sha256") != selected.get("dataset_manifest_sha256") or
            selected.get("feature_sets") != {key: list(value) for key, value in FEATURE_SETS.items()}):
        raise ValueError("Research selection, calibration, or feature schema changed")
    if selected["environment"] != evaluated["environment"] or selected["environment"] != _environment():
        raise ValueError("Pinned early research environment differs")
    for name, digest in evaluated["code_sha256"].items():
        checked(root / name, digest)
    calibration = json.loads((evaluation / "calibrators.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(evaluation / "validation_predictions.csv",
                              usecols=["race_id", "validation_block", "method"])
    if set(predictions.method) != set(METHODS) or set(predictions.validation_block) != set(BLOCKS):
        raise ValueError("Incomplete research comparison methods or blocks")
    blocks = {
        block: sorted(int(value) for value in predictions.loc[
            predictions.validation_block.eq(block), "race_id"
        ].unique()) for block in BLOCKS
    }
    if [len(blocks[block]) for block in BLOCKS] != [8, 7, 7]:
        raise ValueError("Original validation race blocks changed")
    for method in METHODS[:-1]:
        if method not in calibration or set(calibration[method]) != set(BINARY_TARGET_COLUMNS):
            raise ValueError("Missing research calibrator")
        for target in BINARY_TARGET_COLUMNS:
            record = calibration[method][target]
            if record["chosen"] not in ("identity", "platt") or record["candidate"]["type"] != "platt":
                raise ValueError("Unexpected calibration selection")
    models = selection / "models"
    required_models = [
        f"{target}__{family}.joblib"
        for target in BINARY_TARGET_COLUMNS
        for family in METHODS[:3]
    ] + [
        f"baseline__{name}.joblib" for name in METHODS[3:6]
    ]
    ranker, = [item for item in ranker_candidates()
               if item.name == selected["chosen_ranker_on_selection"]]
    required_models.append(f"finish_order__{ranker.feature_set}.joblib")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    for name in required_models:
        shutil.copyfile(models / name, output / "models" / name)
    shutil.copyfile(evaluation / "calibrators.json", output / "calibrators.json")
    write_json(output / "configuration.json", {
        "feature_order": list(CANDIDATE_FEATURES),
        "feature_sets": selected["feature_sets"],
        "selected_candidates": selected["selected_candidates"],
        "chosen_families_on_selection": selected["chosen_families_on_selection"],
        "chosen_ranker_on_selection": selected["chosen_ranker_on_selection"],
        "validation_races": blocks,
        "calibration_policy": evaluated["calibration_policy"],
        "reconciliation_policy": evaluated["reconciliation_policy"],
        "methods": list(METHODS),
    })
    write_json(output / "manifest.json", {
        "schema_version": BUNDLE_SCHEMA,
        "purpose": "2023_validation_replay_only",
        "production_release_authorized": False,
        "prospective_prediction_authorized": False,
        "sealed_2024_2025_loaded": False,
        "base_model_fit_performed": False,
        "model_selection_performed": False,
        "dataset_manifest_sha256": selected["dataset_manifest_sha256"],
        "selection_manifest_sha256": sha256(selection / "manifest.json"),
        "evaluation_manifest_sha256": sha256(evaluation / "manifest.json"),
        "environment": _environment(),
        "code_sha256": {
            **evaluated["code_sha256"],
            "src/f1_predictor/early/bundle.py": sha256(root / "src/f1_predictor/early/bundle.py"),
            "src/f1_predictor/stage3/release.py": sha256(root / "src/f1_predictor/stage3/release.py"),
        },
        "files_sha256": {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    })
    verify_research_bundle(output)
    return output


def verify_research_bundle(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    code = manifest.get("code_sha256", {})
    if (manifest.get("schema_version") != BUNDLE_SCHEMA or
            manifest.get("purpose") != "2023_validation_replay_only" or
            manifest.get("production_release_authorized") is not False or
            manifest.get("prospective_prediction_authorized") is not False or
            manifest.get("sealed_2024_2025_loaded") is not False or
            manifest.get("environment") != _environment() or
            not {"src/f1_predictor/early/bundle.py",
                 "src/f1_predictor/stage3/release.py"} <= set(code)):
        raise ValueError("Research bundle cannot be used in this environment or purpose")
    root = Path(__file__).resolve().parents[3]
    for name, digest in code.items():
        checked(root / name, digest)
    for name, digest in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Research bundle member escapes directory")
        checked(path, digest)
    config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    if (config["feature_order"] != list(CANDIDATE_FEATURES) or
            config["feature_sets"] != {key: list(value) for key, value in FEATURE_SETS.items()} or
            config["methods"] != list(METHODS)):
        raise ValueError("Research bundle feature or method contract changed")
    return manifest


class ResearchReplayBundle:
    """Validation replay only; there is intentionally no future-race predict method."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = verify_research_bundle(directory)
        self.config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
        self.calibration = json.loads((directory / "calibrators.json").read_text(encoding="utf-8"))
        self.models = {
            target: {family: joblib.load(directory / "models" / f"{target}__{family}.joblib")
                     for family in METHODS[:3]}
            for target in BINARY_TARGET_COLUMNS
        }
        self.baselines = {
            method: joblib.load(directory / "models" / f"baseline__{method}.joblib")
            for method in METHODS[3:6]
        }
        self.ranker_candidate, = [item for item in ranker_candidates()
                                  if item.name == self.config["chosen_ranker_on_selection"]]
        self.rank_preprocessor, self.ranker = joblib.load(
            directory / "models" / f"finish_order__{self.ranker_candidate.feature_set}.joblib"
        )

    def replay_2023_block(self, frame: pd.DataFrame, block: str) -> pd.DataFrame:
        if block not in BLOCKS or frame.empty:
            raise ValueError("One complete original 2023 validation block required")
        required = {"race_id", "driver_id", "year", "split", "validation_block", *CANDIDATE_FEATURES}
        if (not frame.columns.is_unique or not required <= set(frame.columns) or
                tuple(name for name in frame.columns if name in CANDIDATE_FEATURES)
                != CANDIDATE_FEATURES):
            raise ValueError("Research validation feature schema/order differs")
        allowed = self.config["validation_races"][block]
        if (set(frame.race_id) != set(allowed) or
                frame[["race_id", "driver_id"]].duplicated().any() or
                not frame.groupby("race_id").size().eq(20).all() or
                not frame.year.eq(2023).all() or
                not frame.split.eq("validation").all() or
                not frame.validation_block.eq(block).all()):
            raise ValueError("Research bundle refuses other races, splits, or feature schemas")
        features = frame[["race_id", *CANDIDATE_FEATURES]].copy()
        rank = rank_scores(self.rank_preprocessor, self.ranker,
                           self.ranker_candidate, features)
        raw: dict[str, dict[str, np.ndarray]] = {}
        ranks: dict[str, np.ndarray] = {}
        for method in METHODS[:-1]:
            if method in self.baselines:
                raw[method], ranks[method] = self.baselines[method].predict(features)
            else:
                raw[method] = {
                    target: classifier_probabilities(
                        self.models[target][method],
                        Candidate(**self.config["selected_candidates"][target][method]),
                        features,
                    ) for target in BINARY_TARGET_COLUMNS
                }
                ranks[method] = rank
        chosen = self.config["chosen_families_on_selection"]
        raw["selected_combination"] = {
            target: raw[chosen[target]][target] for target in BINARY_TARGET_COLUMNS
        }
        ranks["selected_combination"] = rank
        rows = []
        for method in METHODS:
            source_method = lambda target: chosen[target] if method == "selected_combination" else method
            calibrated = {}
            for target in BINARY_TARGET_COLUMNS:
                record = self.calibration[source_method(target)][target]
                values = raw[method][target]
                calibrated[target] = (calibrator_from_dict(record["candidate"]).predict(values)
                                      if record["chosen"] == "platt" else values)
            stage_frames = {
                "raw": pd.DataFrame({TARGET_PROBABILITY_COLUMNS[target]: raw[method][target]
                                     for target in BINARY_TARGET_COLUMNS}),
                "calibrated": pd.DataFrame({TARGET_PROBABILITY_COLUMNS[target]: calibrated[target]
                                            for target in BINARY_TARGET_COLUMNS}),
            }
            stage_frames["reconciled"] = reconcile_probabilities(
                features[["race_id"]], stage_frames["calibrated"]
            )
            for index, row in enumerate(features.itertuples()):
                item = {"race_id": int(row.race_id), "driver_id": row.driver_id,
                        "validation_block": block, "method": method,
                        "rank_score": float(ranks[method][index])}
                for stage, values in stage_frames.items():
                    for target in BINARY_TARGET_COLUMNS:
                        column = TARGET_PROBABILITY_COLUMNS[target]
                        item[f"{stage}_{column}"] = float(values.iloc[index][column])
                rows.append(item)
        return pd.DataFrame(rows)
