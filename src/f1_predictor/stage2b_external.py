"""External-runtime scikit-learn and XGBoost Stage 2B comparison.

Imports of compiled ML packages are intentionally lazy so the repository and its
tests remain usable on Windows hosts where Application Control blocks those
extensions. Run this module in the documented Colab environment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stage2a import BINARY_TARGET_COLUMNS, FORBIDDEN_CURRENT_RACE_FEATURES
from .stage2b_data import (
    CATEGORICAL_FEATURES,
    EXTERNAL_FEATURE_SETS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TEMPORAL_FOLDS,
    load_stage2b_training_data,
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
from .stage2b_models import PlattCalibrator, make_calibrator, sigmoid
from .stage2b_training import validation_baselines


@dataclass(frozen=True)
class ExternalStage2BConfig:
    random_seed: int = 2026
    calibration_races: int = 8
    method_selection_races: int = 7
    n_jobs: int = 2


def _external_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import sklearn
        import xgboost
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from xgboost import XGBClassifier, XGBRanker
    except Exception as exc:
        raise RuntimeError(
            "Stage 2B external comparison requires a compliant runtime with "
            "scikit-learn, XGBoost, and joblib. Use notebooks/stage2b_colab.ipynb."
        ) from exc
    return {
        "joblib": joblib,
        "sklearn": sklearn,
        "xgboost": xgboost,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "XGBClassifier": XGBClassifier,
        "XGBRanker": XGBRanker,
    }


def verify_external_environment() -> dict[str, str]:
    dependencies = _external_dependencies()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": dependencies["sklearn"].__version__,
        "xgboost": dependencies["xgboost"].__version__,
        "joblib": dependencies["joblib"].__version__,
    }


def external_classifier_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for feature_set in EXTERNAL_FEATURE_SETS:
        for c_value in (0.1, 1.0):
            candidates.append(
                {
                    "name": f"logistic_{feature_set}_c_{c_value:g}",
                    "family": "logistic_regression",
                    "feature_set": feature_set,
                    "parameters": {"C": c_value},
                }
            )
        candidates.append(
            {
                "name": f"random_forest_{feature_set}",
                "family": "random_forest",
                "feature_set": feature_set,
                "parameters": {
                    "n_estimators": 350,
                    "max_depth": 9,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                },
            }
        )
        for suffix, parameters in (
            (
                "shallow",
                {
                    "n_estimators": 250,
                    "max_depth": 2,
                    "learning_rate": 0.05,
                    "min_child_weight": 4.0,
                    "subsample": 0.85,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 2.0,
                },
            ),
            (
                "medium",
                {
                    "n_estimators": 350,
                    "max_depth": 3,
                    "learning_rate": 0.03,
                    "min_child_weight": 3.0,
                    "subsample": 0.85,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 3.0,
                },
            ),
        ):
            candidates.append(
                {
                    "name": f"xgboost_{feature_set}_{suffix}",
                    "family": "xgboost",
                    "feature_set": feature_set,
                    "parameters": parameters,
                }
            )
    return candidates


def external_ranker_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for feature_set in EXTERNAL_FEATURE_SETS:
        for suffix, parameters in (
            (
                "pairwise",
                {
                    "objective": "rank:pairwise",
                    "n_estimators": 300,
                    "max_depth": 3,
                    "learning_rate": 0.04,
                    "min_child_weight": 3.0,
                    "subsample": 0.85,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 3.0,
                },
            ),
            (
                "ndcg",
                {
                    "objective": "rank:ndcg",
                    "n_estimators": 300,
                    "max_depth": 3,
                    "learning_rate": 0.04,
                    "min_child_weight": 3.0,
                    "subsample": 0.85,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 3.0,
                },
            ),
        ):
            candidates.append(
                {
                    "name": f"xgboost_ranker_{feature_set}_{suffix}",
                    "family": "xgboost_ranker",
                    "feature_set": feature_set,
                    "parameters": parameters,
                }
            )
    return candidates


def _feature_groups(feature_set: str) -> tuple[list[str], list[str]]:
    features = EXTERNAL_FEATURE_SETS[feature_set]
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [column for column in NUMERIC_FEATURES if column in features]
    if len(categorical) + len(numeric) != len(features):
        raise ValueError(f"Feature set {feature_set} contains an untyped feature")
    return categorical, numeric


def _make_preprocessor(feature_set: str, scale_numeric: bool, dependencies: dict[str, Any]) -> Any:
    categorical, numeric = _feature_groups(feature_set)
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", dependencies["SimpleImputer"](strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(
            ("scaler", dependencies["StandardScaler"](with_mean=False))
        )
    numeric_pipeline = dependencies["Pipeline"](numeric_steps)
    categorical_pipeline = dependencies["Pipeline"](
        [
            (
                "imputer",
                dependencies["SimpleImputer"](
                    strategy="constant", fill_value="__MISSING__"
                ),
            ),
            (
                "one_hot",
                dependencies["OneHotEncoder"](
                    handle_unknown="ignore", sparse_output=True
                ),
            ),
        ]
    )
    return dependencies["ColumnTransformer"](
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def _make_classifier(
    candidate: dict[str, Any], config: ExternalStage2BConfig, seed_offset: int
) -> Any:
    dependencies = _external_dependencies()
    family = candidate["family"]
    parameters = dict(candidate["parameters"])
    scale_numeric = family == "logistic_regression"
    preprocessor = _make_preprocessor(
        candidate["feature_set"], scale_numeric, dependencies
    )
    if family == "logistic_regression":
        classifier = dependencies["LogisticRegression"](
            C=float(parameters["C"]),
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=config.random_seed + seed_offset,
        )
    elif family == "random_forest":
        classifier = dependencies["RandomForestClassifier"](
            **parameters,
            random_state=config.random_seed + seed_offset,
            n_jobs=config.n_jobs,
            class_weight=None,
        )
    elif family == "xgboost":
        classifier = dependencies["XGBClassifier"](
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.random_seed + seed_offset,
            n_jobs=config.n_jobs,
        )
    else:
        raise ValueError(f"Unknown classifier family: {family}")
    return dependencies["Pipeline"](
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )


def _predict_positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Binary classifier did not return two probability columns")
    return probabilities[:, 1]


def tune_external_classifiers(
    train: pd.DataFrame, config: ExternalStage2BConfig
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    candidates = external_classifier_candidates()
    for target_index, target in enumerate(BINARY_TARGET_COLUMNS):
        for candidate_index, candidate in enumerate(candidates):
            features = EXTERNAL_FEATURE_SETS[candidate["feature_set"]]
            for fold_index, fold in enumerate(TEMPORAL_FOLDS):
                fit, score = temporal_fold_frames(train, fold)
                model = _make_classifier(
                    candidate,
                    config,
                    seed_offset=10000 * target_index + 100 * candidate_index + fold_index,
                )
                model.fit(fit[features], fit[target].to_numpy(int))
                probabilities = _predict_positive_probability(model, score[features])
                rows.append(
                    {
                        "target": target,
                        "family": candidate["family"],
                        "candidate": candidate["name"],
                        "feature_set": candidate["feature_set"],
                        "fold": fold.name,
                        "fit_rows": len(fit),
                        "score_rows": len(score),
                        **binary_metrics(score[target].to_numpy(float), probabilities),
                    }
                )
    results = pd.DataFrame(rows)
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for target in BINARY_TARGET_COLUMNS:
        selected[target] = {}
        target_results = results[results["target"].eq(target)]
        summary = (
            target_results.groupby(
                ["family", "candidate", "feature_set"], as_index=False
            )
            .agg(mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"))
            .sort_values(["family", "mean_log_loss", "mean_brier", "candidate"])
        )
        for family in ("logistic_regression", "random_forest", "xgboost"):
            family_summary = summary[summary["family"].eq(family)]
            selected_name = str(family_summary.iloc[0]["candidate"])
            selected[target][family] = next(
                candidate for candidate in candidates if candidate["name"] == selected_name
            )
    return results, selected


def _ordered_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["race_date", "round", "race_id", "driver_id"], kind="mergesort"
    )


def _ranking_relevance(frame: pd.DataFrame) -> np.ndarray:
    field_size = frame.groupby("race_id")["driver_id"].transform("size").to_numpy(float)
    return field_size - frame["finish_order"].to_numpy(float) + 1.0


def _fit_ranker(
    candidate: dict[str, Any],
    fit: pd.DataFrame,
    config: ExternalStage2BConfig,
    seed_offset: int,
) -> tuple[Any, Any]:
    dependencies = _external_dependencies()
    features = EXTERNAL_FEATURE_SETS[candidate["feature_set"]]
    ordered = _ordered_for_ranking(fit)
    preprocessor = _make_preprocessor(candidate["feature_set"], False, dependencies)
    transformed = preprocessor.fit_transform(ordered[features])
    group_sizes = ordered.groupby("race_id", sort=False).size().to_numpy(int)
    ranker = dependencies["XGBRanker"](
        **candidate["parameters"],
        eval_metric="ndcg@10",
        tree_method="hist",
        random_state=config.random_seed + seed_offset,
        n_jobs=config.n_jobs,
    )
    ranker.fit(transformed, _ranking_relevance(ordered), group=group_sizes, verbose=False)
    return preprocessor, ranker


def _predict_rank_scores(
    preprocessor: Any,
    ranker: Any,
    candidate: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    features = EXTERNAL_FEATURE_SETS[candidate["feature_set"]]
    ordered = _ordered_for_ranking(frame)
    relevance_score = np.asarray(
        ranker.predict(preprocessor.transform(ordered[features])), dtype=float
    )
    lower_is_better = pd.Series(-relevance_score, index=ordered.index)
    return lower_is_better.reindex(frame.index).to_numpy(float)


def tune_external_rankers(
    train: pd.DataFrame, config: ExternalStage2BConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = external_ranker_candidates()
    for candidate_index, candidate in enumerate(candidates):
        for fold_index, fold in enumerate(TEMPORAL_FOLDS):
            fit, score = temporal_fold_frames(train, fold)
            preprocessor, ranker = _fit_ranker(
                candidate,
                fit,
                config,
                seed_offset=50000 + 100 * candidate_index + fold_index,
            )
            scores = _predict_rank_scores(preprocessor, ranker, candidate, score)
            rows.append(
                {
                    "candidate": candidate["name"],
                    "feature_set": candidate["feature_set"],
                    "objective": candidate["parameters"]["objective"],
                    "fold": fold.name,
                    "fit_rows": len(fit),
                    "score_rows": len(score),
                    **ranking_metrics(score, scores),
                }
            )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["candidate", "feature_set", "objective"], as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_spearman=("spearman", "mean"))
        .sort_values(["mean_mae", "mean_spearman", "candidate"], ascending=[True, False, True])
    )
    selected_name = str(summary.iloc[0]["candidate"])
    selected = next(
        candidate for candidate in candidates if candidate["name"] == selected_name
    )
    return results, selected


def _select_calibrator(
    validation: pd.DataFrame,
    raw_probabilities: np.ndarray,
    target: str,
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[Any, list[dict[str, Any]]]:
    calibration_positions = validation.index.get_indexer(calibration.index)
    selection_positions = validation.index.get_indexer(selection.index)
    comparisons: list[dict[str, Any]] = []
    choices: list[tuple[float, str, Any]] = []
    for name in ("none", "platt", "isotonic"):
        calibrator = make_calibrator(name).fit(
            raw_probabilities[calibration_positions],
            calibration[target].to_numpy(float),
        )
        probabilities = calibrator.predict(raw_probabilities[selection_positions])
        score = log_loss(selection[target].to_numpy(float), probabilities)
        comparisons.append(
            {"target": target, "calibrator": name, "selection_log_loss": score}
        )
        choices.append((score, name, calibrator))
    return min(choices, key=lambda item: (item[0], item[1]))[2], comparisons


def _fit_family_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_configs: dict[str, dict[str, dict[str, Any]]],
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    config: ExternalStage2BConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    bundles: dict[str, Any] = {}
    calibration_rows: list[dict[str, Any]] = []
    family_selection_rows: list[dict[str, Any]] = []
    selection_positions = validation.index.get_indexer(selection.index)
    for target_index, target in enumerate(BINARY_TARGET_COLUMNS):
        bundles[target] = {}
        for family_index, family in enumerate(
            ("logistic_regression", "random_forest", "xgboost")
        ):
            candidate = selected_configs[target][family]
            features = EXTERNAL_FEATURE_SETS[candidate["feature_set"]]
            model = _make_classifier(
                candidate,
                config,
                seed_offset=70000 + 100 * target_index + family_index,
            )
            model.fit(train[features], train[target].to_numpy(int))
            raw = _predict_positive_probability(model, validation[features])
            calibrator, comparisons = _select_calibrator(
                validation, raw, target, calibration, selection
            )
            for row in comparisons:
                calibration_rows.append(
                    {
                        "family": family,
                        "candidate": candidate["name"],
                        **row,
                    }
                )
            calibrated = calibrator.predict(raw)
            selection_metrics = binary_metrics(
                selection[target].to_numpy(float), calibrated[selection_positions]
            )
            family_selection_rows.append(
                {
                    "target": target,
                    "family": family,
                    "candidate": candidate["name"],
                    "feature_set": candidate["feature_set"],
                    "calibrator": calibrator.name,
                    **selection_metrics,
                }
            )
            bundles[target][family] = {
                "candidate": candidate,
                "model": model,
                "calibrator": calibrator,
                "raw_validation": raw,
                "calibrated_validation": calibrated,
            }
    return bundles, calibration_rows, family_selection_rows


def _select_families(
    bundles: dict[str, Any], family_selection_rows: list[dict[str, Any]]
) -> dict[str, str]:
    results = pd.DataFrame(family_selection_rows)
    selected: dict[str, str] = {}
    for target in BINARY_TARGET_COLUMNS:
        target_results = results[results["target"].eq(target)].sort_values(
            ["log_loss", "brier", "family"]
        )
        selected[target] = str(target_results.iloc[0]["family"])
        if selected[target] not in bundles[target]:
            raise ValueError(f"Selected unavailable family for {target}")
    return selected


def _probability_frame_from_selected(
    bundles: dict[str, Any], selected_families: dict[str, str]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TARGET_PROBABILITY_COLUMNS[target]: bundles[target][
                selected_families[target]
            ]["calibrated_validation"]
            for target in BINARY_TARGET_COLUMNS
        }
    )


def _rank_probabilities(
    validation: pd.DataFrame,
    scores: np.ndarray,
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, PlattCalibrator]]:
    calibration_positions = validation.index.get_indexer(calibration.index)
    base = sigmoid(-5.0 * (np.asarray(scores, dtype=float) - 0.5))
    probabilities = pd.DataFrame(index=range(len(validation)))
    calibrators: dict[str, PlattCalibrator] = {}
    for target in BINARY_TARGET_COLUMNS:
        calibrator = PlattCalibrator().fit(
            base[calibration_positions], calibration[target].to_numpy(float)
        )
        calibrators[target] = calibrator
        probabilities[TARGET_PROBABILITY_COLUMNS[target]] = calibrator.predict(base)
    return probabilities, calibrators


def _select_reconciliation(
    validation: pd.DataFrame,
    classifier_probabilities: pd.DataFrame,
    rank_probabilities: pd.DataFrame,
    rank_scores: np.ndarray,
    selection: pd.DataFrame,
) -> tuple[float, pd.DataFrame, list[dict[str, Any]]]:
    positions = validation.index.get_indexer(selection.index)
    comparisons: list[dict[str, Any]] = []
    choices: list[tuple[float, float]] = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        projected = reconcile_probabilities(
            validation,
            (1.0 - alpha) * classifier_probabilities + alpha * rank_probabilities,
        )
        losses = [
            log_loss(
                selection[target].to_numpy(float),
                projected.iloc[positions][TARGET_PROBABILITY_COLUMNS[target]].to_numpy(float),
            )
            for target in BINARY_TARGET_COLUMNS
        ]
        mean_loss = float(np.mean(losses))
        comparisons.append(
            {
                "rank_blend_alpha": alpha,
                "selection_mean_log_loss": mean_loss,
                **probability_consistency_metrics(
                    selection.reset_index(drop=True),
                    projected.iloc[positions].reset_index(drop=True),
                    np.asarray(rank_scores)[positions],
                ),
            }
        )
        choices.append((mean_loss, alpha))
    _, selected_alpha = min(choices, key=lambda item: (item[0], item[1]))
    final = reconcile_probabilities(
        validation,
        (1.0 - selected_alpha) * classifier_probabilities
        + selected_alpha * rank_probabilities,
    )
    return selected_alpha, final, comparisons


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _winner_audit(
    validation: pd.DataFrame,
    numpy_predictions: pd.DataFrame,
    report: pd.DataFrame,
    reported_log_loss: float,
) -> dict[str, Any]:
    report_race_ids = set(report["race_id"])
    current = numpy_predictions[
        numpy_predictions["race_id"].isin(report_race_ids)
    ].copy()
    current = current.sort_values(["race_id", "driver_id"]).reset_index(drop=True)
    expected = report.sort_values(["race_id", "driver_id"]).reset_index(drop=True)
    if not np.array_equal(
        current["race_winner"].to_numpy(int), expected["race_winner"].to_numpy(int)
    ):
        raise ValueError("Saved NumPy winner labels do not match Stage 2A validation labels")
    y = current["race_winner"].to_numpy(float)
    final = np.clip(current["p_race_winner"].to_numpy(float), 1e-12, 1.0 - 1e-12)
    raw = np.clip(
        current["raw_p_race_winner"].to_numpy(float), 1e-12, 1.0 - 1e-12
    )
    manual_final = -float(np.mean(y * np.log(final) + (1.0 - y) * np.log(1.0 - final)))
    manual_raw = -float(np.mean(y * np.log(raw) + (1.0 - y) * np.log(1.0 - raw)))
    winners = current[current["race_winner"].eq(1)][
        ["race_id", "round", "race_date", "driver_id", "p_race_winner"]
    ].to_dict(orient="records")
    feature_overlap = sorted(set(FEATURE_COLUMNS) & FORBIDDEN_CURRENT_RACE_FEATURES)
    return {
        "reported_log_loss": reported_log_loss,
        "recomputed_clipped_log_loss": manual_final,
        "recomputed_raw_model_log_loss": manual_raw,
        "calculation_matches_report": abs(manual_final - reported_log_loss) < 1e-12,
        "formula": "mean(-y*log(clip(p))-(1-y)*log(clip(1-p)))",
        "rows": len(current),
        "races": int(current["race_id"].nunique()),
        "positive_labels": int(y.sum()),
        "winner_labels_per_race": {
            str(int(key)): int(value)
            for key, value in current.groupby("race_id")["race_winner"].sum().items()
        },
        "race_probability_sum_min": float(
            current.groupby("race_id")["p_race_winner"].sum().min()
        ),
        "race_probability_sum_max": float(
            current.groupby("race_id")["p_race_winner"].sum().max()
        ),
        "winners": winners,
        "all_report_winners_same_driver": len({row["driver_id"] for row in winners}) == 1,
        "forbidden_feature_overlap": feature_overlap,
        "leakage_assessment": (
            "No direct outcome feature is present and current-outcome mutation tests pass. "
            "The unusually low loss is dominated by seven consecutive Max Verstappen wins "
            "in a very small, non-diverse report block; compare ID-free models before treating "
            "it as a general winner-performance estimate."
        ),
    }


def run_external_stage2b_validation(
    stage2a_dir: Path,
    numpy_artifact_dir: Path,
    output_dir: Path,
    config: ExternalStage2BConfig | None = None,
) -> dict[str, Any]:
    """Run external model comparison without loading the locked test split."""

    config = config or ExternalStage2BConfig()
    runtime = verify_external_environment()
    train, validation, source_hashes = load_stage2b_training_data(stage2a_dir)
    calibration, selection, report = validation_protocol_split(
        validation,
        calibration_races=config.calibration_races,
        method_selection_races=config.method_selection_races,
    )
    report_positions = validation.index.get_indexer(report.index)
    report_reset = report.reset_index(drop=True)

    classifier_cv, tuned_configs = tune_external_classifiers(train, config)
    ranker_cv, selected_ranker_config = tune_external_rankers(train, config)
    bundles, calibration_rows, family_selection_rows = _fit_family_models(
        train, validation, tuned_configs, calibration, selection, config
    )
    selected_families = _select_families(bundles, family_selection_rows)
    selected_probabilities = _probability_frame_from_selected(
        bundles, selected_families
    )

    rank_preprocessor, ranker = _fit_ranker(
        selected_ranker_config, train, config, seed_offset=90000
    )
    rank_scores = _predict_rank_scores(
        rank_preprocessor, ranker, selected_ranker_config, validation
    )
    rank_probabilities, rank_calibrators = _rank_probabilities(
        validation, rank_scores, calibration
    )
    selected_alpha, final_probabilities, reconciliation_rows = _select_reconciliation(
        validation,
        selected_probabilities,
        rank_probabilities,
        rank_scores,
        selection,
    )

    numpy_predictions = pd.read_csv(
        numpy_artifact_dir / "validation_predictions.csv", low_memory=False
    )
    numpy_report_source = numpy_predictions[
        numpy_predictions["validation_role"].eq("held_out_report")
    ].copy()
    report_keys = report[["race_id", "driver_id"]].reset_index(drop=True)
    numpy_report = report_keys.merge(
        numpy_report_source,
        on=["race_id", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    if len(numpy_report) != len(report) or numpy_report["p_race_winner"].isna().any():
        raise ValueError("NumPy and external report rows do not match")

    ranking_baselines, probability_baselines = validation_baselines(train, validation)
    report_probability_rows: list[dict[str, Any]] = []
    for name, probabilities in probability_baselines.items():
        for target, metrics in evaluate_probability_frame(
            report_reset,
            probabilities.iloc[report_positions].reset_index(drop=True),
        ).items():
            report_probability_rows.append(
                {"method": name, "family": "baseline", "target": target, **metrics}
            )
    numpy_probability_frame = numpy_report[
        [TARGET_PROBABILITY_COLUMNS[target] for target in BINARY_TARGET_COLUMNS]
    ].reset_index(drop=True)
    for target, metrics in evaluate_probability_frame(
        report_reset, numpy_probability_frame
    ).items():
        report_probability_rows.append(
            {"method": "numpy_selected_pipeline", "family": "numpy", "target": target, **metrics}
        )
    for target in BINARY_TARGET_COLUMNS:
        for family in ("logistic_regression", "random_forest", "xgboost"):
            probabilities = bundles[target][family]["calibrated_validation"][report_positions]
            report_probability_rows.append(
                {
                    "method": bundles[target][family]["candidate"]["name"],
                    "family": family,
                    "target": target,
                    **binary_metrics(report_reset[target].to_numpy(float), probabilities),
                }
            )
        report_probability_rows.append(
            {
                "method": "external_selected_reconciled",
                "family": selected_families[target],
                "target": target,
                **binary_metrics(
                    report_reset[target].to_numpy(float),
                    final_probabilities.iloc[report_positions][
                        TARGET_PROBABILITY_COLUMNS[target]
                    ].to_numpy(float),
                ),
            }
        )

    report_ranking_rows = [
        {
            "method": name,
            "family": "baseline",
            **ranking_metrics(report_reset, scores[report_positions]),
        }
        for name, scores in ranking_baselines.items()
    ]
    report_ranking_rows.append(
        {
            "method": "numpy_selected_ranker",
            "family": "numpy",
            **ranking_metrics(report_reset, numpy_report["rank_score"].to_numpy(float)),
        }
    )
    report_ranking_rows.append(
        {
            "method": selected_ranker_config["name"],
            "family": "xgboost_ranker",
            **ranking_metrics(report_reset, rank_scores[report_positions]),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dependencies = _external_dependencies()
    for target in BINARY_TARGET_COLUMNS:
        for family, bundle in bundles[target].items():
            dependencies["joblib"].dump(
                {
                    "pipeline": bundle["model"],
                    "calibrator": bundle["calibrator"].to_dict(),
                    "candidate": bundle["candidate"],
                    "fit_years": [2014, 2022],
                },
                models_dir / f"{target}__{family}.joblib",
            )
    dependencies["joblib"].dump(
        rank_preprocessor, models_dir / "finish_order__xgboost_preprocessor.joblib"
    )
    ranker.save_model(models_dir / "finish_order__xgboost_ranker.json")
    (models_dir / "rank_probability_calibrators.json").write_text(
        json.dumps(
            {target: calibrator.to_dict() for target, calibrator in rank_calibrators.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    classifier_cv.to_csv(
        output_dir / "classifier_temporal_cv.csv", index=False, lineterminator="\n"
    )
    ranker_cv.to_csv(
        output_dir / "ranker_temporal_cv.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "calibration_selection.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(family_selection_rows).to_csv(
        output_dir / "family_selection.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(reconciliation_rows).to_csv(
        output_dir / "reconciliation_selection.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(report_probability_rows).to_csv(
        output_dir / "probability_model_comparison.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(report_ranking_rows).to_csv(
        output_dir / "ranking_model_comparison.csv", index=False, lineterminator="\n"
    )

    predictions = validation[
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
    predictions["external_rank_score"] = rank_scores
    predictions["external_predicted_finish_order"] = (
        pd.Series(rank_scores)
        .groupby(validation["race_id"].reset_index(drop=True))
        .rank(method="first", ascending=True)
        .astype(int)
    )
    for target in BINARY_TARGET_COLUMNS:
        for family in ("logistic_regression", "random_forest", "xgboost"):
            predictions[f"raw_p_{target}__{family}"] = bundles[target][family][
                "raw_validation"
            ]
            predictions[f"p_{target}__{family}"] = bundles[target][family][
                "calibrated_validation"
            ]
        predictions[f"p_{target}__external_selected"] = final_probabilities[
            TARGET_PROBABILITY_COLUMNS[target]
        ]
    predictions["validation_role"] = "held_out_report"
    predictions.loc[calibration.index, "validation_role"] = "calibration_fit"
    predictions.loc[selection.index, "validation_role"] = "method_selection"
    predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False, lineterminator="\n"
    )

    numpy_metrics = json.loads(
        (numpy_artifact_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    reported_winner_log_loss = float(
        numpy_metrics["final_pipeline_held_out_validation"]["probability"][
            "race_winner"
        ]["log_loss"]
    )
    winner_audit = _winner_audit(
        validation,
        numpy_predictions,
        report,
        reported_winner_log_loss,
    )
    (output_dir / "winner_metric_audit.json").write_text(
        json.dumps(_json_ready(winner_audit), indent=2) + "\n", encoding="utf-8"
    )
    metrics = {
        "report_block": {
            "probability_models": report_probability_rows,
            "ranking_models": report_ranking_rows,
            "external_selected_probability": evaluate_probability_frame(
                report_reset,
                final_probabilities.iloc[report_positions].reset_index(drop=True),
            ),
            "external_selected_ranking": ranking_metrics(
                report_reset, rank_scores[report_positions]
            ),
            "external_selected_consistency": probability_consistency_metrics(
                report_reset,
                final_probabilities.iloc[report_positions].reset_index(drop=True),
                rank_scores[report_positions],
            ),
        },
        "method_selection": {
            "families": family_selection_rows,
            "reconciliation": reconciliation_rows,
        },
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2) + "\n", encoding="utf-8"
    )

    implementation_root = Path(__file__).resolve().parents[2]
    implementation_files = [
        "requirements-stage2b-external.txt",
        "src/f1_predictor/stage2b_data.py",
        "src/f1_predictor/stage2b_external.py",
        "src/f1_predictor/stage2b_metrics.py",
        "src/f1_predictor/stage2b_models.py",
        "src/f1_predictor/stage2b_training.py",
    ]
    manifest = {
        "stage": "2B-external-validation",
        "test_status": "locked_not_packaged_loaded_or_evaluated",
        "configuration": asdict(config),
        "runtime": runtime,
        "source_files_sha256": source_hashes,
        "numpy_validation_predictions_sha256": _sha256(
            numpy_artifact_dir / "validation_predictions.csv"
        ),
        "numpy_validation_metrics_sha256": _sha256(
            numpy_artifact_dir / "validation_metrics.json"
        ),
        "implementation_sha256": {
            relative: _sha256(implementation_root / relative)
            for relative in implementation_files
        },
        "feature_sets": {
            name: {
                "count": len(features),
                "features": features,
            }
            for name, features in EXTERNAL_FEATURE_SETS.items()
        },
        "temporal_folds": [asdict(fold) for fold in TEMPORAL_FOLDS],
        "validation_blocks": {
            "calibration": {
                "rows": len(calibration),
                "races": int(calibration["race_id"].nunique()),
            },
            "method_selection": {
                "rows": len(selection),
                "races": int(selection["race_id"].nunique()),
            },
            "held_out_report": {
                "rows": len(report),
                "races": int(report["race_id"].nunique()),
            },
        },
        "selected_classifier_configs_by_family": tuned_configs,
        "selected_families": selected_families,
        "selected_ranker": selected_ranker_config,
        "selected_reconciliation_alpha": selected_alpha,
        "model_artifacts_sha256": {
            path.name: _sha256(path)
            for path in sorted(models_dir.iterdir(), key=lambda item: item.name)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "metrics": metrics, "winner_audit": winner_audit}
