"""Chronological Stage 2B training, calibration, and validation orchestration."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stage2a import BINARY_TARGET_COLUMNS
from .stage2b_data import (
    FEATURE_SETS,
    TEMPORAL_FOLDS,
    SplitSafePreprocessor,
    load_stage2b_training_data,
    normalized_finish_target,
    temporal_fold_frames,
    validation_protocol_split,
)
from .stage2b_metrics import (
    TARGET_PROBABILITY_COLUMNS,
    binary_metrics,
    evaluate_probability_frame,
    log_loss,
    probability_consistency_metrics,
    ranking_metrics,
    reconcile_probabilities,
)
from .stage2b_models import (
    GradientBoostedStumpClassifier,
    GradientBoostedStumpRanker,
    LogisticRegressionGD,
    PlattCalibrator,
    RidgeRanker,
    make_calibrator,
    sigmoid,
)


@dataclass(frozen=True)
class Stage2BConfig:
    random_seed: int = 2026
    calibration_races: int = 8
    method_selection_races: int = 7
    bootstrap_repetitions: int = 400


BINARY_MODEL_CONFIGS = [
    {
        "name": "logistic_core_l2_0.005",
        "model": "logistic",
        "feature_set": "core",
        "parameters": {
            "l2": 0.005,
            "learning_rate": 0.03,
            "max_iter": 240,
        },
    },
    {
        "name": "logistic_all_l2_0.02",
        "model": "logistic",
        "feature_set": "all",
        "parameters": {
            "l2": 0.02,
            "learning_rate": 0.025,
            "max_iter": 240,
        },
    },
    {
        "name": "boosted_stumps_core",
        "model": "boosted_stumps",
        "feature_set": "core",
        "parameters": {
            "n_estimators": 45,
            "learning_rate": 0.08,
            "l2": 1.0,
            "max_bins": 8,
            "max_features": 32,
            "min_leaf": 20,
        },
    },
]

RANKING_MODEL_CONFIGS = [
    {
        "name": "ridge_core_alpha_1",
        "model": "ridge",
        "feature_set": "core",
        "parameters": {"alpha": 1.0},
    },
    {
        "name": "ridge_all_alpha_10",
        "model": "ridge",
        "feature_set": "all",
        "parameters": {"alpha": 10.0},
    },
    {
        "name": "boosted_stumps_rank_core",
        "model": "boosted_stumps_rank",
        "feature_set": "core",
        "parameters": {
            "n_estimators": 60,
            "learning_rate": 0.08,
            "max_bins": 8,
            "max_features": 32,
            "min_leaf": 20,
        },
    },
]


def _binary_model(config: dict[str, Any], seed: int) -> Any:
    parameters = dict(config["parameters"])
    if config["model"] == "logistic":
        return LogisticRegressionGD(**parameters)
    if config["model"] == "boosted_stumps":
        parameters["random_state"] = seed
        return GradientBoostedStumpClassifier(**parameters)
    raise ValueError(f"Unknown binary model: {config['model']}")


def _ranking_model(config: dict[str, Any], seed: int) -> Any:
    parameters = dict(config["parameters"])
    if config["model"] == "ridge":
        return RidgeRanker(**parameters)
    if config["model"] == "boosted_stumps_rank":
        parameters["random_state"] = seed
        return GradientBoostedStumpRanker(**parameters)
    raise ValueError(f"Unknown ranking model: {config['model']}")


def _fit_preprocessor(
    fit: pd.DataFrame, score: pd.DataFrame, feature_set: str
) -> tuple[SplitSafePreprocessor, np.ndarray, np.ndarray]:
    preprocessor = SplitSafePreprocessor(FEATURE_SETS[feature_set])
    x_fit = preprocessor.fit_transform(fit)
    x_score = preprocessor.transform(score)
    return preprocessor, x_fit, x_score


def tune_binary_models(
    train: pd.DataFrame, config: Stage2BConfig
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for target in BINARY_TARGET_COLUMNS:
        for candidate_index, candidate in enumerate(BINARY_MODEL_CONFIGS):
            for fold_index, fold in enumerate(TEMPORAL_FOLDS):
                fit, score = temporal_fold_frames(train, fold)
                _, x_fit, x_score = _fit_preprocessor(
                    fit, score, candidate["feature_set"]
                )
                model = _binary_model(
                    candidate, config.random_seed + 100 * candidate_index + fold_index
                )
                model.fit(x_fit, fit[target].to_numpy(float))
                probabilities = model.predict_proba(x_score)
                metrics = binary_metrics(score[target].to_numpy(float), probabilities)
                rows.append(
                    {
                        "target": target,
                        "candidate": candidate["name"],
                        "feature_set": candidate["feature_set"],
                        "fold": fold.name,
                        "fit_rows": len(fit),
                        "score_rows": len(score),
                        **metrics,
                    }
                )
        target_results = pd.DataFrame(row for row in rows if row["target"] == target)
        summary = (
            target_results.groupby(["candidate", "feature_set"], as_index=False)
            .agg(mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"))
            .sort_values(["mean_log_loss", "mean_brier", "candidate"])
        )
        selected_name = str(summary.iloc[0]["candidate"])
        selected[target] = next(
            candidate for candidate in BINARY_MODEL_CONFIGS if candidate["name"] == selected_name
        )
    return pd.DataFrame(rows), selected


def tune_ranking_models(
    train: pd.DataFrame, config: Stage2BConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(RANKING_MODEL_CONFIGS):
        for fold_index, fold in enumerate(TEMPORAL_FOLDS):
            fit, score = temporal_fold_frames(train, fold)
            _, x_fit, x_score = _fit_preprocessor(fit, score, candidate["feature_set"])
            model = _ranking_model(
                candidate, config.random_seed + 500 + 100 * candidate_index + fold_index
            )
            model.fit(x_fit, normalized_finish_target(fit))
            predicted = model.predict_score(x_score)
            metrics = ranking_metrics(score, predicted)
            rows.append(
                {
                    "candidate": candidate["name"],
                    "feature_set": candidate["feature_set"],
                    "fold": fold.name,
                    "fit_rows": len(fit),
                    "score_rows": len(score),
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["candidate", "feature_set"], as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_spearman=("spearman", "mean"))
        .sort_values(["mean_mae", "mean_spearman", "candidate"], ascending=[True, False, True])
    )
    selected_name = str(summary.iloc[0]["candidate"])
    selected = next(
        candidate for candidate in RANKING_MODEL_CONFIGS if candidate["name"] == selected_name
    )
    return results, selected


def _grid_scores(frame: pd.DataFrame) -> np.ndarray:
    grid = pd.to_numeric(frame["effective_grid_position"], errors="coerce")
    fallback = pd.to_numeric(frame["qualifying_position"], errors="coerce")
    race_max = frame.groupby("race_id")["driver_id"].transform("size") + 1
    return grid.fillna(fallback).fillna(race_max).to_numpy(float)


def _qualifying_scores(frame: pd.DataFrame) -> np.ndarray:
    qualifying = pd.to_numeric(frame["qualifying_position"], errors="coerce")
    return qualifying.fillna(pd.Series(_grid_scores(frame), index=frame.index)).to_numpy(float)


def _recent_form_scores(frame: pd.DataFrame) -> np.ndarray:
    grid = pd.to_numeric(frame["grid_position_normalized"], errors="coerce").fillna(1.0)
    recent = pd.to_numeric(
        frame["driver_recent_5_avg_finish"], errors="coerce"
    ).fillna(pd.to_numeric(frame["effective_grid_position"], errors="coerce"))
    field = frame.groupby("race_id")["driver_id"].transform("size").astype(float)
    recent_normalized = recent.fillna(field) / field
    standings = pd.to_numeric(
        frame["driver_championship_position_pre_race"], errors="coerce"
    ).fillna(field)
    standings_normalized = standings / field
    return (0.6 * grid + 0.25 * recent_normalized + 0.15 * standings_normalized).to_numpy(
        float
    )


def _quota_probabilities(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame(index=range(len(frame)))
    score_series = pd.Series(np.asarray(scores, dtype=float), index=frame.index)
    for target, quota, temperature in (
        ("race_winner", 1.0, 0.65),
        ("podium_finish", 3.0, 1.5),
        ("points_finish", 10.0, 4.0),
    ):
        values = np.zeros(len(frame), dtype=float)
        for _, indices in frame.groupby("race_id", sort=False).groups.items():
            positions = frame.index.get_indexer(indices)
            race_scores = score_series.loc[indices].to_numpy(float)
            strengths = np.exp(
                np.clip(-(race_scores - np.nanmin(race_scores)) / temperature, -30.0, 30.0)
            )
            raw = strengths / strengths.sum() * min(quota, len(positions))
            values[positions] = np.clip(raw, 1e-8, 1.0 - 1e-8)
        output[TARGET_PROBABILITY_COLUMNS[target]] = values
    return reconcile_probabilities(frame, output)


def _grid_frequency_probabilities(
    train: pd.DataFrame, score: pd.DataFrame
) -> pd.DataFrame:
    train_grid = np.rint(_grid_scores(train)).astype(int)
    score_grid = np.rint(_grid_scores(score)).astype(int)
    output = pd.DataFrame(index=range(len(score)))
    for target in BINARY_TARGET_COLUMNS:
        prior = float(train[target].mean())
        aggregate = (
            pd.DataFrame({"grid": train_grid, "target": train[target].to_numpy(float)})
            .groupby("grid")["target"]
            .agg(["sum", "count"])
        )
        rates = ((aggregate["sum"] + 8.0 * prior) / (aggregate["count"] + 8.0)).to_dict()
        output[TARGET_PROBABILITY_COLUMNS[target]] = np.array(
            [float(rates.get(int(grid), prior)) for grid in score_grid]
        )
    return reconcile_probabilities(score, output)


def validation_baselines(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    ranking = {
        "final_grid": _grid_scores(validation),
        "qualifying": _qualifying_scores(validation),
        "recent_form": _recent_form_scores(validation),
    }
    probability = {
        "uniform_field_quota": _quota_probabilities(
            validation,
            validation.groupby("race_id")["driver_id"].transform("size").to_numpy(float)
            * 0.0,
        ),
        "final_grid_rule": _quota_probabilities(validation, ranking["final_grid"]),
        "train_grid_frequency": _grid_frequency_probabilities(train, validation),
    }
    return ranking, probability


def _fit_selected_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_binary: dict[str, dict[str, Any]],
    selected_ranking: dict[str, Any],
    config: Stage2BConfig,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
    np.ndarray,
]:
    binary_bundles: dict[str, Any] = {}
    binary_predictions: dict[str, np.ndarray] = {}
    for index, target in enumerate(BINARY_TARGET_COLUMNS):
        candidate = selected_binary[target]
        preprocessor, x_train, x_validation = _fit_preprocessor(
            train, validation, candidate["feature_set"]
        )
        model = _binary_model(candidate, config.random_seed + 1000 + index)
        model.fit(x_train, train[target].to_numpy(float))
        binary_predictions[target] = model.predict_proba(x_validation)
        binary_bundles[target] = {
            "configuration": candidate,
            "preprocessor": preprocessor,
            "model": model,
        }

    rank_preprocessor, x_train, x_validation = _fit_preprocessor(
        train, validation, selected_ranking["feature_set"]
    )
    rank_model = _ranking_model(selected_ranking, config.random_seed + 2000)
    rank_model.fit(x_train, normalized_finish_target(train))
    rank_predictions = rank_model.predict_score(x_validation)
    rank_bundle = {
        "configuration": selected_ranking,
        "preprocessor": rank_preprocessor,
        "model": rank_model,
    }
    return binary_bundles, binary_predictions, rank_bundle, rank_predictions


def _probability_frame(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TARGET_PROBABILITY_COLUMNS[target]: np.asarray(values, dtype=float)
            for target, values in predictions.items()
        }
    )


def _select_calibrators(
    validation: pd.DataFrame,
    raw_predictions: dict[str, np.ndarray],
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    calibration_positions = validation.index.get_indexer(calibration.index)
    selection_positions = validation.index.get_indexer(selection.index)
    calibrated = pd.DataFrame(index=range(len(validation)))
    selected_calibrators: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    for target in BINARY_TARGET_COLUMNS:
        raw = raw_predictions[target]
        candidates: list[tuple[float, str, Any]] = []
        for name in ("none", "platt", "isotonic"):
            calibrator = make_calibrator(name).fit(
                raw[calibration_positions], calibration[target].to_numpy(float)
            )
            selected_probabilities = calibrator.predict(raw[selection_positions])
            score = log_loss(selection[target].to_numpy(float), selected_probabilities)
            comparison.append(
                {"target": target, "calibrator": name, "selection_log_loss": score}
            )
            candidates.append((score, name, calibrator))
        _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        selected_calibrators[target] = chosen
        calibrated[TARGET_PROBABILITY_COLUMNS[target]] = chosen.predict(raw)
    return selected_calibrators, calibrated, comparison


def _rank_derived_probabilities(
    validation: pd.DataFrame,
    rank_scores: np.ndarray,
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, PlattCalibrator]]:
    calibration_positions = validation.index.get_indexer(calibration.index)
    base = sigmoid(-5.0 * (np.asarray(rank_scores, dtype=float) - 0.5))
    output = pd.DataFrame(index=range(len(validation)))
    calibrators: dict[str, PlattCalibrator] = {}
    for target in BINARY_TARGET_COLUMNS:
        calibrator = PlattCalibrator().fit(
            base[calibration_positions], calibration[target].to_numpy(float)
        )
        calibrators[target] = calibrator
        output[TARGET_PROBABILITY_COLUMNS[target]] = calibrator.predict(base)
    return output, calibrators


def _select_reconciliation(
    validation: pd.DataFrame,
    calibrated: pd.DataFrame,
    rank_probabilities: pd.DataFrame,
    rank_scores: np.ndarray,
    selection: pd.DataFrame,
) -> tuple[float, pd.DataFrame, list[dict[str, Any]]]:
    selection_positions = validation.index.get_indexer(selection.index)
    comparisons: list[dict[str, Any]] = []
    choices: list[tuple[float, float]] = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = (1.0 - alpha) * calibrated + alpha * rank_probabilities
        reconciled = reconcile_probabilities(validation, blended)
        losses = []
        for target in BINARY_TARGET_COLUMNS:
            column = TARGET_PROBABILITY_COLUMNS[target]
            losses.append(
                log_loss(
                    selection[target].to_numpy(float),
                    reconciled.iloc[selection_positions][column].to_numpy(float),
                )
            )
        mean_loss = float(np.mean(losses))
        consistency = probability_consistency_metrics(
            selection.reset_index(drop=True),
            reconciled.iloc[selection_positions].reset_index(drop=True),
            np.asarray(rank_scores)[selection_positions],
        )
        comparisons.append(
            {
                "rank_blend_alpha": alpha,
                "selection_mean_log_loss": mean_loss,
                **consistency,
            }
        )
        choices.append((mean_loss, alpha))
    _, selected_alpha = min(choices, key=lambda item: (item[0], item[1]))
    final = reconcile_probabilities(
        validation,
        (1.0 - selected_alpha) * calibrated + selected_alpha * rank_probabilities,
    )
    return selected_alpha, final, comparisons


def _race_bootstrap_intervals(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    race_ids = frame["race_id"].drop_duplicates().to_numpy()
    grouped = {race_id: np.flatnonzero(frame["race_id"].to_numpy() == race_id) for race_id in race_ids}
    samples: dict[str, dict[str, list[float]]] = {
        target: {"log_loss": [], "brier": []} for target in BINARY_TARGET_COLUMNS
    }
    for _ in range(repetitions):
        selected_races = rng.choice(race_ids, size=len(race_ids), replace=True)
        indices = np.concatenate([grouped[race_id] for race_id in selected_races])
        for target in BINARY_TARGET_COLUMNS:
            column = TARGET_PROBABILITY_COLUMNS[target]
            metric = binary_metrics(
                frame[target].to_numpy(float)[indices],
                probabilities[column].to_numpy(float)[indices],
            )
            samples[target]["log_loss"].append(metric["log_loss"])
            samples[target]["brier"].append(metric["brier"])
    output: dict[str, dict[str, dict[str, float]]] = {}
    for target, metrics in samples.items():
        output[target] = {}
        for name, values in metrics.items():
            output[target][name] = {
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def run_stage2b_validation(
    stage2a_dir: Path,
    output_dir: Path,
    config: Stage2BConfig | None = None,
) -> dict[str, Any]:
    """Train/tune on train-era data and evaluate only the 2023 validation split."""

    config = config or Stage2BConfig()
    train, validation, source_hashes = load_stage2b_training_data(stage2a_dir)
    calibration, selection, report = validation_protocol_split(
        validation,
        calibration_races=config.calibration_races,
        method_selection_races=config.method_selection_races,
    )

    binary_cv, selected_binary = tune_binary_models(train, config)
    ranking_cv, selected_ranking = tune_ranking_models(train, config)
    binary_bundles, raw_predictions, rank_bundle, rank_predictions = _fit_selected_models(
        train, validation, selected_binary, selected_ranking, config
    )

    ranking_baselines, probability_baselines = validation_baselines(train, validation)
    report_positions = validation.index.get_indexer(report.index)
    report_reset = report.reset_index(drop=True)
    baseline_metrics = {
        "full_validation": {
            "ranking": {
                name: ranking_metrics(validation, scores)
                for name, scores in ranking_baselines.items()
            },
            "probability": {
                name: evaluate_probability_frame(validation, probabilities)
                for name, probabilities in probability_baselines.items()
            },
        },
        "held_out_validation": {
            "ranking": {
                name: ranking_metrics(report_reset, scores[report_positions])
                for name, scores in ranking_baselines.items()
            },
            "probability": {
                name: evaluate_probability_frame(
                    report_reset,
                    probabilities.iloc[report_positions].reset_index(drop=True),
                )
                for name, probabilities in probability_baselines.items()
            },
        },
    }
    raw_frame = _probability_frame(raw_predictions)
    raw_model_metrics = evaluate_probability_frame(validation, raw_frame)
    rank_model_metrics = ranking_metrics(validation, rank_predictions)

    calibrators, calibrated, calibration_comparison = _select_calibrators(
        validation, raw_predictions, calibration, selection
    )
    rank_probabilities, rank_calibrators = _rank_derived_probabilities(
        validation, rank_predictions, calibration
    )
    selected_alpha, reconciled, reconciliation_comparison = _select_reconciliation(
        validation,
        calibrated,
        rank_probabilities,
        rank_predictions,
        selection,
    )
    report_probabilities = reconciled.iloc[report_positions].reset_index(drop=True)
    final_selection_metrics = {
        "probability": evaluate_probability_frame(report_reset, report_probabilities),
        "ranking": ranking_metrics(
            report_reset, rank_predictions[report_positions]
        ),
        "consistency": probability_consistency_metrics(
            report_reset,
            report_probabilities,
            rank_predictions[report_positions],
        ),
        "bootstrap_95": _race_bootstrap_intervals(
            report_reset,
            report_probabilities,
            config.bootstrap_repetitions,
            config.random_seed + 3000,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    binary_cv.to_csv(output_dir / "binary_temporal_cv.csv", index=False, lineterminator="\n")
    ranking_cv.to_csv(output_dir / "ranking_temporal_cv.csv", index=False, lineterminator="\n")
    pd.DataFrame(calibration_comparison).to_csv(
        output_dir / "calibration_selection.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(reconciliation_comparison).to_csv(
        output_dir / "reconciliation_selection.csv", index=False, lineterminator="\n"
    )

    validation_predictions = validation[
        [
            "race_id",
            "race_date",
            "year",
            "round",
            "driver_id",
            "finish_order",
            *BINARY_TARGET_COLUMNS,
        ]
    ].reset_index(drop=True)
    validation_predictions["rank_score"] = rank_predictions
    validation_predictions["predicted_finish_order"] = (
        pd.Series(rank_predictions)
        .groupby(validation["race_id"].reset_index(drop=True))
        .rank(method="first", ascending=True)
        .astype(int)
    )
    validation_predictions["predicted_podium_position"] = validation_predictions[
        "predicted_finish_order"
    ].where(validation_predictions["predicted_finish_order"].le(3))
    for target in BINARY_TARGET_COLUMNS:
        column = TARGET_PROBABILITY_COLUMNS[target]
        validation_predictions[f"raw_{column}"] = raw_frame[column]
        validation_predictions[column] = reconciled[column]
    validation_predictions["validation_role"] = "held_out_report"
    validation_predictions.loc[calibration.index, "validation_role"] = "calibration_fit"
    validation_predictions.loc[selection.index, "validation_role"] = "method_selection"
    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False, lineterminator="\n"
    )

    model_bundle = {
        "binary_models": {
            target: {
                "configuration": bundle["configuration"],
                "preprocessor": bundle["preprocessor"].to_dict(),
                "model": bundle["model"].to_dict(),
                "validation_calibrator": calibrators[target].to_dict(),
            }
            for target, bundle in binary_bundles.items()
        },
        "ranking_model": {
            "configuration": rank_bundle["configuration"],
            "preprocessor": rank_bundle["preprocessor"].to_dict(),
            "model": rank_bundle["model"].to_dict(),
        },
        "rank_probability_calibrators": {
            target: calibrator.to_dict()
            for target, calibrator in rank_calibrators.items()
        },
        "reconciliation": {
            "method": "validation_selected_rank_blend_then_nested_bounded_projection",
            "rank_blend_alpha": selected_alpha,
            "constraints": {
                "race_probability_sums": {
                    "race_winner": 1,
                    "podium_finish": 3,
                    "points_finish": 10,
                },
                "row_hierarchy": "p_win <= p_podium <= p_points",
            },
        },
    }
    (output_dir / "model_bundle.json").write_text(
        json.dumps(_json_ready(model_bundle), indent=2) + "\n", encoding="utf-8"
    )

    metrics = {
        "baselines": baseline_metrics,
        "selected_model_raw_validation": {
            "probability": raw_model_metrics,
            "ranking": rank_model_metrics,
        },
        "final_pipeline_held_out_validation": final_selection_metrics,
        "calibration_comparison": calibration_comparison,
        "reconciliation_comparison": reconciliation_comparison,
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "stage": "2B-validation",
        "test_status": "locked_not_loaded_or_evaluated",
        "configuration": asdict(config),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "estimator_stack": "project-native NumPy implementations",
            "compiled_ml_dependency_note": (
                "scikit-learn and XGBoost wheels were blocked by Windows Application "
                "Control, so Stage 2B uses deterministic NumPy estimators."
            ),
        },
        "source_files_sha256": source_hashes,
        "rows": {
            "train_eligible": len(train),
            "validation_eligible": len(validation),
            "validation_calibration_fit": len(calibration),
            "validation_method_selection": len(selection),
            "validation_held_out_report": len(report),
        },
        "races": {
            "train": int(train["race_id"].nunique()),
            "validation": int(validation["race_id"].nunique()),
            "validation_calibration_fit": int(calibration["race_id"].nunique()),
            "validation_method_selection": int(selection["race_id"].nunique()),
            "validation_held_out_report": int(report["race_id"].nunique()),
        },
        "temporal_folds": [asdict(fold) for fold in TEMPORAL_FOLDS],
        "selected_binary_models": selected_binary,
        "selected_ranking_model": selected_ranking,
        "selected_calibrators": {
            target: calibrator.name for target, calibrator in calibrators.items()
        },
        "selected_reconciliation_alpha": selected_alpha,
        "artifacts": [
            "binary_temporal_cv.csv",
            "ranking_temporal_cv.csv",
            "calibration_selection.csv",
            "reconciliation_selection.csv",
            "validation_predictions.csv",
            "validation_metrics.json",
            "model_bundle.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "metrics": metrics}
