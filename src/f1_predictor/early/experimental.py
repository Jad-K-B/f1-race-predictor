"""Private experimental early inference from an immutable prospective capture."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..stage2a import BINARY_TARGET_COLUMNS
from ..stage2b_metrics import TARGET_PROBABILITY_COLUMNS, reconcile_probabilities
from ..stage2b_models import calibrator_from_dict
from ..stage3.release import checked, sha256, write_json
from ..stage3.sources import now, utc
from .archive import PRIVATE_ROOT, inspect_early_capture, replay_early_capture
from .bundle import METHODS, _environment, verify_research_bundle
from .contracts import EarlySnapshot
from .policy import CANDIDATE_FEATURES, EXCLUDED_FEATURES
from .research_training import (
    Candidate, FEATURE_SETS, classifier_probabilities, rank_scores, ranker_candidates,
)

RELEASE_SCHEMA = "early-experimental-private-release-v1"
FORECAST_SCHEMA = "early-experimental-private-forecast-v1"


def _private(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError("Experimental early artifacts must remain private")
    return resolved


def prepare_experimental_release(research_bundle: Path, output: Path) -> Path:
    """Copy the reviewed models unchanged; no fit, tuning, or public authorization."""
    output = _private(output)
    if output.exists():
        raise FileExistsError("Experimental releases are immutable")
    source = verify_research_bundle(research_bundle)
    config = json.loads((research_bundle / "configuration.json").read_text(encoding="utf-8"))
    if source["sealed_2024_2025_loaded"] is not False or config["methods"] != list(METHODS):
        raise ValueError("Unapproved research model source")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(research_bundle / "models", output / "models")
    for name in ("calibrators.json", "configuration.json"):
        shutil.copyfile(research_bundle / name, output / name)
    root = Path(__file__).resolve().parents[3]
    write_json(output / "manifest.json", {
        "schema_version": RELEASE_SCHEMA,
        "status": "experimental_prepared_private",
        "deployment_authorized": False,
        "public_release_authorized": False,
        "sealed_2024_2025_loaded": False,
        "base_model_fit_performed": False,
        "model_selection_performed": False,
        "research_manifest_sha256": sha256(research_bundle / "manifest.json"),
        "selection_manifest_sha256": source["selection_manifest_sha256"],
        "evaluation_manifest_sha256": source["evaluation_manifest_sha256"],
        "environment": _environment(),
        "code_sha256": {
            **source["code_sha256"],
            "src/f1_predictor/early/experimental.py": sha256(Path(__file__)),
        },
        "files_sha256": {
            file.relative_to(output).as_posix(): sha256(file)
            for file in sorted(output.rglob("*")) if file.is_file()
        },
    })
    verify_experimental_release(output)
    return output


def verify_experimental_release(directory: Path) -> dict[str, Any]:
    directory = _private(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != RELEASE_SCHEMA or
            manifest.get("status") != "experimental_prepared_private" or
            any(manifest.get(key) is not False for key in (
                "deployment_authorized", "public_release_authorized", "sealed_2024_2025_loaded",
                "base_model_fit_performed", "model_selection_performed")) or
            manifest.get("environment") != _environment()):
        raise ValueError("Unverified or incompatible experimental release")
    root = Path(__file__).resolve().parents[3]
    required_code = {
        "src/f1_predictor/early/bundle.py",
        "src/f1_predictor/early/research_training.py",
        "src/f1_predictor/early/experimental.py",
        "src/f1_predictor/stage2b_metrics.py",
    }
    if not required_code <= set(manifest["code_sha256"]):
        raise ValueError("Experimental release code provenance is incomplete")
    for name, digest in manifest["code_sha256"].items():
        checked((root / name).resolve(), digest)
    config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    ranker_candidate, = [item for item in ranker_candidates()
                         if item.name == config["chosen_ranker_on_selection"]]
    expected_files = {"configuration.json", "calibrators.json"}
    expected_files.update(f"models/{target}__{family}.joblib"
                          for target in BINARY_TARGET_COLUMNS for family in METHODS[:3])
    expected_files.update(f"models/baseline__{name}.joblib" for name in METHODS[3:6])
    expected_files.add(f"models/finish_order__{ranker_candidate.feature_set}.joblib")
    if set(manifest["files_sha256"]) != expected_files:
        raise ValueError("Experimental release member list changed")
    for name, digest in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Experimental release member escapes directory")
        checked(path, digest)
    if (config["feature_order"] != list(CANDIDATE_FEATURES) or
            config["feature_sets"] != {key: list(value) for key, value in FEATURE_SETS.items()} or
            config["methods"] != list(METHODS)):
        raise ValueError("Experimental feature schema or method set changed")
    return manifest


class ExperimentalEarlyPredictor:
    """Load only a verified private release; accept only a replayed live capture."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = verify_experimental_release(directory)
        self.config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
        self.calibration = json.loads((directory / "calibrators.json").read_text(encoding="utf-8"))
        self.models = {
            target: {family: joblib.load(directory / "models" / f"{target}__{family}.joblib")
                     for family in METHODS[:3]}
            for target in BINARY_TARGET_COLUMNS
        }
        for target in BINARY_TARGET_COLUMNS:
            self.models[target]["random_forest"].named_steps["estimator"].n_jobs = 1
        self.baselines = {
            name: joblib.load(directory / "models" / f"baseline__{name}.joblib")
            for name in METHODS[3:6]
        }
        ranker_candidate, = [item for item in ranker_candidates()
                             if item.name == self.config["chosen_ranker_on_selection"]]
        self.ranker_candidate = ranker_candidate
        self.rank_preprocessor, self.ranker = joblib.load(
            directory / "models" / f"finish_order__{ranker_candidate.feature_set}.joblib"
        )

    def predict_capture(self, capture: Path) -> dict:
        """Generate a new private prediction at the actual current time."""
        return self._predict_capture_at(capture, now())

    def _predict_capture_at(self, capture: Path, timestamp: str) -> dict:
        capture = _private(capture)
        inspection = inspect_early_capture(capture)
        replay_early_capture(capture)
        snapshot = EarlySnapshot.from_dict(json.loads((capture / "snapshot.json").read_text(encoding="utf-8")))
        if not utc(snapshot.cutoff) <= utc(timestamp) < utc(snapshot.race["fp1_start"]):
            raise ValueError("Experimental forecast must be computed after cutoff and before FP1")
        if utc(timestamp) < utc(inspection["completed_at"]):
            raise ValueError("Forecast timestamp precedes completed capture")
        saved = json.loads((capture / "features.json").read_text(encoding="utf-8"))
        frame = pd.DataFrame(saved["records"], columns=saved["columns"]).astype(saved["dtypes"])
        if (frame.columns.tolist() != ["race_id", "race_date", "cutoff", *CANDIDATE_FEATURES] or
                set(frame.columns) & EXCLUDED_FEATURES or
                frame.empty or frame[["race_id", "driver_id"]].duplicated().any() or
                not frame.race_id.eq(snapshot.race["race_id"]).all()):
            raise ValueError("Prospective early feature schema or event identity changed")
        features = frame[["race_id", *CANDIDATE_FEATURES]].reset_index(drop=True)
        rank = rank_scores(self.rank_preprocessor, self.ranker, self.ranker_candidate, features)
        raw: dict[str, dict[str, np.ndarray]] = {}
        scores: dict[str, np.ndarray] = {}
        for method in METHODS[:-1]:
            if method in self.baselines:
                raw[method], scores[method] = self.baselines[method].predict(features)
            else:
                raw[method] = {
                    target: classifier_probabilities(
                        self.models[target][method],
                        Candidate(**self.config["selected_candidates"][target][method]),
                        features,
                    ) for target in BINARY_TARGET_COLUMNS
                }
                scores[method] = rank
        chosen = self.config["chosen_families_on_selection"]
        raw["selected_combination"] = {
            target: raw[chosen[target]][target] for target in BINARY_TARGET_COLUMNS
        }
        scores["selected_combination"] = rank
        methods = {}
        for method in METHODS:
            calibrated = {}
            for target in BINARY_TARGET_COLUMNS:
                source_method = chosen[target] if method == "selected_combination" else method
                record = self.calibration[source_method][target]
                values = raw[method][target]
                calibrated[target] = (calibrator_from_dict(record["candidate"]).predict(values)
                                      if record["chosen"] == "platt" else values)
            stages = {
                "raw": pd.DataFrame({TARGET_PROBABILITY_COLUMNS[target]: raw[method][target]
                                     for target in BINARY_TARGET_COLUMNS}),
                "calibrated": pd.DataFrame({TARGET_PROBABILITY_COLUMNS[target]: calibrated[target]
                                            for target in BINARY_TARGET_COLUMNS}),
            }
            stages["reconciled"] = reconcile_probabilities(
                features[["race_id"]], stages["calibrated"]
            )
            if any(not np.isfinite(values.to_numpy(float)).all() or
                   ((values.to_numpy(float) < -1e-9) | (values.to_numpy(float) > 1 + 1e-9)).any()
                   for values in stages.values()):
                raise ValueError("Non-finite or out-of-range early probability")
            ranking_available = method != "prior_rate"
            order = (np.argsort(scores[method], kind="mergesort").tolist()
                     if ranking_available else [])
            positions = {index: place + 1 for place, index in enumerate(order)}
            rows = []
            for index, driver in enumerate(features.driver_id):
                rows.append({
                    "driver_id": str(driver),
                    "constructor_id": str(features.constructor_id.iloc[index]),
                    "predicted_position": positions.get(index),
                    "rank_score": float(scores[method][index]) if ranking_available else None,
                    "probabilities": {
                        stage: {target: float(values.iloc[index][TARGET_PROBABILITY_COLUMNS[target]])
                                for target in BINARY_TARGET_COLUMNS}
                        for stage, values in stages.items()
                    },
                })
            reconciled = stages["reconciled"]
            sums = {target: float(reconciled[TARGET_PROBABILITY_COLUMNS[target]].sum())
                    for target in BINARY_TARGET_COLUMNS}
            expected = {"race_winner": 1, "podium_finish": min(3, len(features)),
                        "points_finish": min(10, len(features))}
            if any(abs(sums[target] - expected[target]) > 1e-7 for target in expected):
                raise ValueError("Race-level probability reconciliation failed")
            win_column = TARGET_PROBABILITY_COLUMNS["race_winner"]
            highest = int(np.argmax(reconciled[win_column].to_numpy(float)))
            methods[method] = {
                "drivers": rows,
                "predicted_order": [str(features.driver_id.iloc[index]) for index in order]
                                   if ranking_available else None,
                "diagnostics": {
                    "probability_sums": sums,
                    "rank_p1_differs_from_highest_win_probability":
                        bool(order[0] != highest) if ranking_available else None,
                },
            }
        return {
            "schema_version": FORECAST_SCHEMA,
            "forecast_stage": "early",
            "experimental": True,
            "publication_status": "private_unpublished",
            "prediction_time": timestamp,
            "cutoff": snapshot.cutoff,
            "snapshot_id": snapshot.snapshot_id,
            "race": snapshot.race,
            "capture_manifest_sha256": sha256(capture / "manifest.json"),
            "release_manifest_sha256": sha256(self.directory / "manifest.json"),
            "missingness": saved["missingness"],
            "limitations": ["No qualifying or starting-grid inputs", "Retrospective training vintages are not fully point-in-time verified"],
            "methods": methods,
        }


def archive_private_forecast(predictor: ExperimentalEarlyPredictor, capture: Path,
                             output: Path) -> Path:
    output = _private(output)
    if output.exists():
        raise FileExistsError("Experimental forecasts are immutable")
    forecast = predictor.predict_capture(capture)
    if utc(now()) >= utc(forecast["race"]["fp1_start"]):
        raise ValueError("FP1 began during inference; do not archive a late forecast")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "forecast.json", forecast)
    write_json(output / "manifest.json", {
        "schema_version": FORECAST_SCHEMA,
        "status": "complete_private_unpublished",
        "forecast_sha256": sha256(output / "forecast.json"),
        "release_manifest_sha256": forecast["release_manifest_sha256"],
        "capture_manifest_sha256": forecast["capture_manifest_sha256"],
    })
    return output


def replay_private_forecast(predictor: ExperimentalEarlyPredictor, capture: Path,
                            archive: Path) -> dict:
    """Reproduce archived bytes offline; never create a new prospective claim."""
    archive = _private(archive)
    capture = _private(capture)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != FORECAST_SCHEMA or
            manifest.get("status") != "complete_private_unpublished"):
        raise ValueError("Not a completed private experimental forecast")
    checked(archive / "forecast.json", manifest["forecast_sha256"])
    saved = json.loads((archive / "forecast.json").read_text(encoding="utf-8"))
    if (saved.get("publication_status") != "private_unpublished" or
            saved.get("experimental") is not True or
            saved.get("capture_manifest_sha256") != sha256(capture / "manifest.json") or
            saved.get("release_manifest_sha256") != sha256(predictor.directory / "manifest.json") or
            manifest["capture_manifest_sha256"] != saved["capture_manifest_sha256"] or
            manifest["release_manifest_sha256"] != saved["release_manifest_sha256"]):
        raise ValueError("Archive does not bind this capture and experimental release")
    actual = predictor._predict_capture_at(capture, saved["prediction_time"])
    if actual != saved:
        changed = [key for key in saved if actual.get(key) != saved[key]]
        if "methods" in changed:
            changed.extend(name for name in METHODS if actual["methods"][name] != saved["methods"][name])
        raise ValueError(f"Experimental forecast differs from deterministic replay: {changed}")
    return {"status": "exact_private_forecast_replay", "snapshot_id": saved["snapshot_id"],
            "publication_status": "private_unpublished", "network_used": False}
