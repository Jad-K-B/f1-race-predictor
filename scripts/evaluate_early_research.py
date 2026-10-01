"""Calibrate frozen research selections and report on final 2023 block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from f1_predictor.early.research_training import (
    Candidate, classifier_probabilities, load_research_cohort,
    rank_scores, ranker_candidates, RESEARCH_MODEL_SCHEMA
)
from f1_predictor.stage2a import BINARY_TARGET_COLUMNS
from f1_predictor.stage2b_metrics import (
    TARGET_PROBABILITY_COLUMNS, average_precision, brier_score,
    expected_calibration_error, log_loss, probability_consistency_metrics,
    ranking_metrics, reconcile_probabilities, roc_auc
)
from f1_predictor.stage2b_models import PlattCalibrator

METHODS = ("logistic", "random_forest", "xgboost",
           "prior_rate", "recent_form", "constructor_form")
STAGES = ("raw", "calibrated", "reconciled")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    return dict(
        log_loss=log_loss(y, probability),
        brier=brier_score(y, probability),
        roc_auc=roc_auc(y, probability),
        average_precision=average_precision(y, probability),
        ece_10=expected_calibration_error(y, probability),
        prevalence=float(np.mean(y)),
        mean_prediction=float(np.mean(probability)),
    )


def rank_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    mask = frame.ranking_available.eq(1).to_numpy()
    return ranking_metrics(frame.loc[mask].reset_index(drop=True),
                           np.asarray(scores, dtype=float)[mask])


def race_bootstrap(frame: pd.DataFrame, probability: np.ndarray,
                   target: str, *, seed: int = 2026) -> dict:
    work = frame[["race_id", target]].copy()
    work["probability"] = np.asarray(probability, dtype=float)
    losses = np.asarray([
        log_loss(race[target].to_numpy(int), race.probability.to_numpy(float))
        for _, race in work.groupby("race_id", sort=False)
    ])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(losses, size=(2000, len(losses)), replace=True).mean(axis=1)
    return dict(races=len(losses), mean_race_log_loss=float(losses.mean()),
                bootstrap_95pct=[float(value) for value in np.quantile(sampled, [0.025, 0.975])],
                note="Descriptive race bootstrap over seven 2023 races, not a prospective uncertainty guarantee.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("A new research evaluation output and summary are required")
    cohort = load_research_cohort(args.cohort)
    chosen_manifest = json.loads((args.selection / "manifest.json").read_text())
    if (chosen_manifest.get("schema_version") != RESEARCH_MODEL_SCHEMA or
            chosen_manifest.get("dataset_manifest_sha256") != sha(args.cohort / "manifest.json") or
            chosen_manifest.get("calibration_and_report_labels_used") is not False or
            chosen_manifest.get("sealed_2024_2025_loaded") is not False):
        raise ValueError("Selection artifact is not compatible with the research cohort")
    for name, digest in chosen_manifest["files_sha256"].items():
        if sha(args.selection / name) != digest:
            raise ValueError(f"Selection artifact hash mismatch: {name}")
    candidates = {
        target: {
            family: Candidate(**item)
            for family, item in chosen_manifest["selected_candidates"][target].items()
        } for target in BINARY_TARGET_COLUMNS
    }
    models = {
        target: {
            family: joblib.load(args.selection / "models" / f"{target}__{family}.joblib")
            for family in ("logistic", "random_forest", "xgboost")
        } for target in BINARY_TARGET_COLUMNS
    }
    baselines = {
        name: joblib.load(args.selection / "models" / f"baseline__{name}.joblib")
        for name in ("prior_rate", "recent_form", "constructor_form")
    }
    rank_candidate, = [
        item for item in ranker_candidates()
        if item.name == chosen_manifest["chosen_ranker_on_selection"]
    ]
    rank_preprocessor, ranker = joblib.load(
        args.selection / "models" / f"finish_order__{rank_candidate.feature_set}.joblib"
    )
    calibration = cohort.block("calibration")
    raw_calibration: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        if method in baselines:
            raw_calibration[method], _ = baselines[method].predict(calibration)
        else:
            raw_calibration[method] = {
                target: classifier_probabilities(
                    models[target][method], candidates[target][method], calibration
                ) for target in BINARY_TARGET_COLUMNS
            }
    calibrators, calibrator_records = {}, {}
    for method in METHODS:
        calibrators[method], calibrator_records[method] = {}, {}
        for target in BINARY_TARGET_COLUMNS:
            raw = raw_calibration[method][target]
            labels = calibration[target].to_numpy(int)
            candidate = PlattCalibrator(l2=0.01).fit(raw, labels)
            corrected = candidate.predict(raw)
            raw_loss = log_loss(labels, raw)
            corrected_loss = log_loss(labels, corrected)
            use_platt = (0.0 <= candidate.slope <= 5.0 and
                         corrected_loss < raw_loss - 0.001)
            calibrators[method][target] = candidate if use_platt else None
            calibrator_records[method][target] = dict(
                chosen="platt" if use_platt else "identity",
                candidate=candidate.to_dict(),
                raw_calibration_log_loss=raw_loss,
                platt_calibration_log_loss=corrected_loss,
            )
    chosen_families = chosen_manifest["chosen_families_on_selection"]
    method_frames: dict[str, dict[str, dict]] = {}
    prediction_rows = []
    for block in ("selection", "calibration", "report"):
        frame = cohort.block(block)
        rank_values = rank_scores(rank_preprocessor, ranker, rank_candidate, frame)
        raw_values, calibrated_values, ranking_values = {}, {}, {}
        for method in METHODS:
            if method in baselines:
                raw, baseline_rank = baselines[method].predict(frame)
                ranking_values[method] = baseline_rank
            else:
                raw = {
                    target: classifier_probabilities(
                        models[target][method], candidates[target][method], frame
                    ) for target in BINARY_TARGET_COLUMNS
                }
                ranking_values[method] = rank_values
            raw_values[method] = raw
            calibrated_values[method] = {
                target: (
                    calibrators[method][target].predict(raw[target])
                    if calibrators[method][target] else raw[target]
                ) for target in BINARY_TARGET_COLUMNS
            }
        raw_values["selected_combination"] = {
            target: raw_values[chosen_families[target]][target]
            for target in BINARY_TARGET_COLUMNS
        }
        calibrated_values["selected_combination"] = {
            target: calibrated_values[chosen_families[target]][target]
            for target in BINARY_TARGET_COLUMNS
        }
        ranking_values["selected_combination"] = rank_values
        method_frames[block] = {}
        for method in (*METHODS, "selected_combination"):
            raw = pd.DataFrame({
                TARGET_PROBABILITY_COLUMNS[target]: raw_values[method][target]
                for target in BINARY_TARGET_COLUMNS
            })
            corrected = pd.DataFrame({
                TARGET_PROBABILITY_COLUMNS[target]: calibrated_values[method][target]
                for target in BINARY_TARGET_COLUMNS
            })
            reconciled = reconcile_probabilities(frame[["race_id"]], corrected)
            method_frames[block][method] = dict(
                frame=frame, raw=raw, calibrated=corrected,
                reconciled=reconciled, rank_scores=ranking_values[method]
            )
            for index, row in enumerate(frame.itertuples()):
                prediction = dict(
                    race_id=int(row.race_id), driver_id=row.driver_id,
                    validation_block=block, method=method,
                    rank_score=float(ranking_values[method][index])
                )
                for stage, values in (
                    ("raw", raw), ("calibrated", corrected),
                    ("reconciled", reconciled)
                ):
                    for target in BINARY_TARGET_COLUMNS:
                        prediction[f"{stage}_{TARGET_PROBABILITY_COLUMNS[target]}"] = float(
                            values.iloc[index][TARGET_PROBABILITY_COLUMNS[target]]
                        )
                prediction_rows.append(prediction)
    report_frame = cohort.block("report")
    report_metrics, uncertainty, consistency = {}, {}, {}
    for method, item in method_frames["report"].items():
        report_metrics[method], uncertainty[method], consistency[method] = {}, {}, {}
        for stage in STAGES:
            values = item[stage]
            report_metrics[method][stage] = {
                target: metrics(
                    report_frame[target].to_numpy(int),
                    values[TARGET_PROBABILITY_COLUMNS[target]].to_numpy(float)
                ) for target in BINARY_TARGET_COLUMNS
            }
            if method != "prior_rate":
                consistency[method][stage] = probability_consistency_metrics(
                    report_frame[["race_id"]], values, item["rank_scores"]
                )
        uncertainty[method] = {
            target: race_bootstrap(
                report_frame,
                item["reconciled"][TARGET_PROBABILITY_COLUMNS[target]].to_numpy(float),
                target
            ) for target in BINARY_TARGET_COLUMNS
        }
    ranking_report = {
        "xgboost_ranker": rank_metrics(report_frame,
                                       method_frames["report"]["selected_combination"]["rank_scores"]),
        "recent_form": rank_metrics(report_frame,
                                    method_frames["report"]["recent_form"]["rank_scores"]),
        "constructor_form": rank_metrics(report_frame,
                                         method_frames["report"]["constructor_form"]["rank_scores"]),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("calibrators.json", calibrator_records),
        ("report_metrics.json", report_metrics),
        ("ranking_report.json", ranking_report),
        ("consistency_report.json", consistency),
        ("uncertainty_report.json", uncertainty),
    ):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    pd.DataFrame(prediction_rows).to_csv(
        output / "validation_predictions.csv", index=False, lineterminator="\n"
    )
    result = dict(
        schema_version=RESEARCH_MODEL_SCHEMA,
        evidence_mode="retrospective_research",
        point_in_time_verified=False,
        production_release_authorized=False,
        selection_manifest_sha256=sha(args.selection / "manifest.json"),
        dataset_manifest_sha256=sha(args.cohort / "manifest.json"),
        selected_families_on_first_2023_block=chosen_families,
        selected_ranker_on_first_2023_block=chosen_manifest["chosen_ranker_on_selection"],
        calibration_races=7, report_races=7,
        report_metrics=report_metrics, ranking_report=ranking_report,
        consistency_report=consistency, uncertainty_report=uncertainty,
        limitations=[
            "Twenty 2014-2022 training races were excluded for unresolved early fields.",
            "Historical source bytes and original publication times are not point-in-time verified.",
            "Seven report races give wide uncertainty; this is retrospective research.",
            "No 2024-2025 test set or prospective result was used."
        ],
        frozen_post_qualifying_release_modified=False,
        sealed_2024_2025_loaded=False,
    )
    output_manifest = dict(
        schema_version=RESEARCH_MODEL_SCHEMA,
        selection_manifest_sha256=result["selection_manifest_sha256"],
        dataset_manifest_sha256=result["dataset_manifest_sha256"],
        calibration_policy="Platt when nonnegative slope <=5 and calibration log loss improves by 0.001; otherwise identity.",
        reconciliation_policy="Deterministic bounded projection to per-race sums 1/3/10 with nested probabilities.",
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         pandas=pd.__version__, sklearn=sklearn.__version__,
                         xgboost=xgboost.__version__, joblib=joblib.__version__),
        code_sha256={name: sha(ROOT / name) for name in (
            "src/f1_predictor/early/research_training.py",
            "scripts/train_early_research.py",
            "scripts/evaluate_early_research.py",
            "src/f1_predictor/stage2b_metrics.py",
            "src/f1_predictor/stage2b_models.py",
        )},
        files_sha256={
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
        sealed_2024_2025_loaded=False,
    )
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output_manifest, stream, indent=2, sort_keys=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "selected": chosen_families,
        "report_reconciled": {
            method: {
                target: values["log_loss"]
                for target, values in stages["reconciled"].items()
            } for method, stages in report_metrics.items()
        },
        "ranking_report": ranking_report,
        "summary": str(args.summary),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
