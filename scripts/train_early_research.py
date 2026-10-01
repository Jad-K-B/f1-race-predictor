"""Select provisional early models using 2014-2022 and 2023 block 1 only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    Baseline, classifier_probabilities, fit_classifier, fit_ranker,
    load_research_cohort, rank_scores, ranker_candidates,
    select_classifier_candidates, temporal_classifier_cv, temporal_ranker_cv,
    RESEARCH_MODEL_SCHEMA
)
from f1_predictor.stage2a import BINARY_TARGET_COLUMNS
from f1_predictor.stage2b_metrics import (
    average_precision, brier_score, log_loss, ranking_metrics, roc_auc
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_metric(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    mask = frame.ranking_available.eq(1).to_numpy()
    ranked = frame.loc[mask].reset_index(drop=True)
    return ranking_metrics(ranked, np.asarray(scores)[mask])


def probability_metric(frame: pd.DataFrame, target: str,
                       probability: np.ndarray) -> dict:
    labels = frame[target].to_numpy(int)
    return {
        "log_loss": log_loss(labels, probability),
        "brier": brier_score(labels, probability),
        "roc_auc": roc_auc(labels, probability),
        "average_precision": average_precision(labels, probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.n_jobs < 1 or args.output.exists():
        raise ValueError("Positive CPU count and a new output directory required")
    cohort = load_research_cohort(args.cohort)
    train, selection = cohort.block("train"), cohort.block("selection")
    print("Classifier temporal folds", flush=True)
    classifier_cv = temporal_classifier_cv(cohort, n_jobs=args.n_jobs)
    selected = select_classifier_candidates(classifier_cv)
    print("Ranker temporal folds", flush=True)
    ranker_cv = temporal_ranker_cv(cohort, n_jobs=args.n_jobs)
    models, selection_rows, selection_predictions = {}, [], []
    for target in BINARY_TARGET_COLUMNS:
        models[target] = {}
        for family in ("logistic", "random_forest", "xgboost"):
            candidate = selected[target][family]
            model = fit_classifier(candidate, train, target, n_jobs=args.n_jobs)
            probability = classifier_probabilities(model, candidate, selection)
            metric = probability_metric(selection, target, probability)
            selection_rows.append({
                "target": target, "family": family, "candidate": candidate.name,
                "feature_set": candidate.feature_set, **metric
            })
            models[target][family] = model
            selection_predictions.extend({
                "race_id": int(row.race_id), "driver_id": row.driver_id,
                "target": target, "family": family, "raw_probability": float(value)
            } for row, value in zip(selection.itertuples(), probability))
        print(f"Selected family candidates for {target}", flush=True)
    selection_metrics = pd.DataFrame(selection_rows)
    chosen_families = {}
    for target in BINARY_TARGET_COLUMNS:
        rows = selection_metrics[selection_metrics.target.eq(target)]
        chosen_families[target] = rows.sort_values(
            ["log_loss", "family"], kind="mergesort"
        ).iloc[0].family
    rankers, ranker_rows, ranker_predictions = {}, [], []
    for candidate in ranker_candidates():
        preprocessor, ranker = fit_ranker(candidate, train, n_jobs=args.n_jobs)
        scores = rank_scores(preprocessor, ranker, candidate, selection)
        metrics = rank_metric(selection, scores)
        ranker_rows.append({
            "candidate": candidate.name, "feature_set": candidate.feature_set,
            **metrics
        })
        rankers[candidate.feature_set] = (preprocessor, ranker)
        ranker_predictions.extend({
            "race_id": int(row.race_id), "driver_id": row.driver_id,
            "method": candidate.name, "rank_score": float(value)
        } for row, value in zip(selection.itertuples(), scores))
    ranker_selection = pd.DataFrame(ranker_rows)
    chosen_ranker = ranker_selection.sort_values(
        ["ndcg_3", "mae", "candidate"],
        ascending=[False, True, True], kind="mergesort"
    ).iloc[0].candidate
    baseline_models, baseline_metrics = {}, []
    for method in ("prior_rate", "recent_form", "constructor_form"):
        baseline = Baseline.fit(train, method)
        baseline_models[method] = baseline
        probabilities, scores = baseline.predict(selection)
        for target in BINARY_TARGET_COLUMNS:
            baseline_metrics.append({
                "method": method, "target": target,
                **probability_metric(selection, target, probabilities[target])
            })
        if method != "prior_rate":
            baseline_metrics.append({
                "method": method, "target": "finish_order",
                **rank_metric(selection, scores)
            })
        selection_predictions.extend({
            "race_id": int(row.race_id), "driver_id": row.driver_id,
            "target": target, "family": method,
            "raw_probability": float(probabilities[target][index])
        } for index, row in enumerate(selection.itertuples())
          for target in BINARY_TARGET_COLUMNS)
        ranker_predictions.extend({
            "race_id": int(row.race_id), "driver_id": row.driver_id,
            "method": method, "rank_score": float(value)
        } for row, value in zip(selection.itertuples(), scores))
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    for target, families in models.items():
        for family, model in families.items():
            joblib.dump(model, model_dir / f"{target}__{family}.joblib")
    for feature_set, model in rankers.items():
        joblib.dump(model, model_dir / f"finish_order__{feature_set}.joblib")
    for method, baseline in baseline_models.items():
        joblib.dump(baseline, model_dir / f"baseline__{method}.joblib")
    for name, frame in [
        ("classifier_cv.csv", classifier_cv),
        ("ranker_cv.csv", ranker_cv),
        ("selection_metrics.csv", selection_metrics),
        ("ranker_selection.csv", ranker_selection),
        ("baseline_selection.csv", pd.DataFrame(baseline_metrics)),
        ("selection_predictions.csv", pd.DataFrame(selection_predictions)),
        ("selection_ranks.csv", pd.DataFrame(ranker_predictions)),
    ]:
        frame.to_csv(output / name, index=False, lineterminator="\n")
    selected_payload = {
        target: {family: asdict(candidate) for family, candidate in families.items()}
        for target, families in selected.items()
    }
    manifest = {
        "schema_version": RESEARCH_MODEL_SCHEMA,
        "evidence_mode": "retrospective_research",
        "production_release_authorized": False,
        "dataset_manifest_sha256": sha(args.cohort / "manifest.json"),
        "training_races": int(train.race_id.nunique()),
        "selection_races": int(selection.race_id.nunique()),
        "calibration_and_report_labels_used": False,
        "sealed_2024_2025_loaded": False,
        "seed": 2026, "n_jobs": args.n_jobs,
        "folds": list(map(list, ((2017, 2018), (2019, 2020), (2021, 2022)))),
        "selected_candidates": selected_payload,
        "chosen_families_on_selection": chosen_families,
        "chosen_ranker_on_selection": chosen_ranker,
        "feature_sets": {key: list(value) for key, value in cohort.feature_sets.items()},
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__
        },
        "files_sha256": {
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        }
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(json.dumps({
        "selection_models": chosen_families,
        "ranker": chosen_ranker,
        "selection_metrics": selection_metrics.to_dict(orient="records"),
        "output": str(output)
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
