"""Approval-gated final evaluation for the locked 2024-2025 test split."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stage2a import BINARY_TARGET_COLUMNS
from .stage2b_data import EXTERNAL_FEATURE_SETS, SplitSafePreprocessor, _validate_split
from .stage2b_external import _predict_rank_scores, verify_external_environment
from .stage2b_metrics import (
    TARGET_PROBABILITY_COLUMNS,
    evaluate_probability_frame,
    log_loss,
    probability_consistency_metrics,
    ranking_metrics,
    reconcile_probabilities,
)
from .stage2b_models import PlattCalibrator, make_calibrator, model_from_dict, sigmoid
from .stage2b_training import validation_baselines


FINAL_RANDOM_SEED = 2026
DEFAULT_BOOTSTRAP_REPETITIONS = 2000


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


def _minimum_row(
    frame: pd.DataFrame,
    sort_columns: list[str],
    ascending: list[bool] | None = None,
) -> pd.Series:
    return frame.sort_values(
        sort_columns,
        ascending=ascending if ascending is not None else True,
        kind="mergesort",
    ).iloc[0]


def audit_frozen_stage2b(
    stage2a_dir: Path,
    numpy_artifact_dir: Path,
    external_artifact_dir: Path,
) -> dict[str, Any]:
    """Verify all frozen choices without opening the locked test file."""

    numpy_manifest = json.loads(
        (numpy_artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    external_manifest = json.loads(
        (external_artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if numpy_manifest["test_status"] != "locked_not_loaded_or_evaluated":
        raise ValueError("NumPy validation manifest does not show a locked test split")
    if (
        external_manifest["test_status"]
        != "locked_not_packaged_loaded_or_evaluated"
    ):
        raise ValueError("External validation manifest does not show a locked test split")

    data_hashes = {
        name: _sha256(stage2a_dir / name)
        for name in ("train.csv", "validation.csv")
    }
    for label, manifest in (
        ("numpy", numpy_manifest),
        ("external", external_manifest),
    ):
        if manifest["source_files_sha256"] != data_hashes:
            raise ValueError(f"{label} train/validation source hashes changed")

    project_root = Path(__file__).resolve().parents[2]
    for relative, expected in external_manifest["implementation_sha256"].items():
        if _sha256(project_root / relative) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    models_dir = external_artifact_dir / "models"
    for name, expected in external_manifest["model_artifacts_sha256"].items():
        path = models_dir / name
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"External model artifact changed or is missing: {name}")

    numpy_binary_cv = pd.read_csv(numpy_artifact_dir / "binary_temporal_cv.csv")
    numpy_selected_binary: dict[str, str] = {}
    for target in BINARY_TARGET_COLUMNS:
        summary = (
            numpy_binary_cv[numpy_binary_cv["target"].eq(target)]
            .groupby(["candidate", "feature_set"], as_index=False)
            .agg(mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"))
        )
        winner = _minimum_row(summary, ["mean_log_loss", "mean_brier", "candidate"])
        selected_name = str(numpy_manifest["selected_binary_models"][target]["name"])
        if str(winner["candidate"]) != selected_name:
            raise ValueError(f"NumPy model selection mismatch for {target}")
        numpy_selected_binary[target] = selected_name

    numpy_ranking_cv = pd.read_csv(numpy_artifact_dir / "ranking_temporal_cv.csv")
    numpy_rank_summary = (
        numpy_ranking_cv.groupby(["candidate", "feature_set"], as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_spearman=("spearman", "mean"))
    )
    numpy_rank_winner = _minimum_row(
        numpy_rank_summary,
        ["mean_mae", "mean_spearman", "candidate"],
        [True, False, True],
    )
    numpy_selected_ranker = str(numpy_manifest["selected_ranking_model"]["name"])
    if str(numpy_rank_winner["candidate"]) != numpy_selected_ranker:
        raise ValueError("NumPy ranker selection mismatch")

    numpy_calibration = pd.read_csv(numpy_artifact_dir / "calibration_selection.csv")
    numpy_selected_calibrators: dict[str, str] = {}
    for target in BINARY_TARGET_COLUMNS:
        winner = _minimum_row(
            numpy_calibration[numpy_calibration["target"].eq(target)],
            ["selection_log_loss", "calibrator"],
        )
        selected = str(numpy_manifest["selected_calibrators"][target])
        if str(winner["calibrator"]) != selected:
            raise ValueError(f"NumPy calibration selection mismatch for {target}")
        numpy_selected_calibrators[target] = selected

    numpy_reconciliation = pd.read_csv(
        numpy_artifact_dir / "reconciliation_selection.csv"
    )
    numpy_alpha = float(
        _minimum_row(
            numpy_reconciliation,
            ["selection_mean_log_loss", "rank_blend_alpha"],
        )["rank_blend_alpha"]
    )
    if numpy_alpha != float(numpy_manifest["selected_reconciliation_alpha"]):
        raise ValueError("NumPy reconciliation selection mismatch")

    external_family_results = pd.read_csv(
        external_artifact_dir / "family_selection.csv"
    )
    external_calibration = pd.read_csv(
        external_artifact_dir / "calibration_selection.csv"
    )
    external_selected_families: dict[str, str] = {}
    external_selected_calibrators: dict[str, str] = {}
    for target in BINARY_TARGET_COLUMNS:
        family_winner = _minimum_row(
            external_family_results[external_family_results["target"].eq(target)],
            ["log_loss", "brier", "family"],
        )
        family = str(family_winner["family"])
        if family != str(external_manifest["selected_families"][target]):
            raise ValueError(f"External family selection mismatch for {target}")
        calibration_winner = _minimum_row(
            external_calibration[
                external_calibration["target"].eq(target)
                & external_calibration["family"].eq(family)
            ],
            ["selection_log_loss", "calibrator"],
        )
        calibrator = str(family_winner["calibrator"])
        if str(calibration_winner["calibrator"]) != calibrator:
            raise ValueError(f"External calibration selection mismatch for {target}")
        external_selected_families[target] = family
        external_selected_calibrators[target] = calibrator

    external_ranking_cv = pd.read_csv(external_artifact_dir / "ranker_temporal_cv.csv")
    external_rank_summary = (
        external_ranking_cv.groupby(
            ["candidate", "feature_set", "objective"], as_index=False
        )
        .agg(mean_mae=("mae", "mean"), mean_spearman=("spearman", "mean"))
    )
    external_rank_winner = _minimum_row(
        external_rank_summary,
        ["mean_mae", "mean_spearman", "candidate"],
        [True, False, True],
    )
    external_selected_ranker = str(external_manifest["selected_ranker"]["name"])
    if str(external_rank_winner["candidate"]) != external_selected_ranker:
        raise ValueError("External ranker selection mismatch")

    external_reconciliation = pd.read_csv(
        external_artifact_dir / "reconciliation_selection.csv"
    )
    external_alpha = float(
        _minimum_row(
            external_reconciliation,
            ["selection_mean_log_loss", "rank_blend_alpha"],
        )["rank_blend_alpha"]
    )
    if external_alpha != float(external_manifest["selected_reconciliation_alpha"]):
        raise ValueError("External reconciliation selection mismatch")

    return {
        "status": "verified_before_test_unlock",
        "test_data_read_during_audit": False,
        "train_validation_sha256": data_hashes,
        "numpy": {
            "selected_models": numpy_selected_binary,
            "selected_calibrators": numpy_selected_calibrators,
            "selected_ranker": numpy_selected_ranker,
            "selected_reconciliation_alpha": numpy_alpha,
            "manifest_sha256": _sha256(numpy_artifact_dir / "manifest.json"),
            "model_bundle_sha256": _sha256(numpy_artifact_dir / "model_bundle.json"),
        },
        "external": {
            "selected_families": external_selected_families,
            "selected_calibrators": external_selected_calibrators,
            "selected_ranker": external_selected_ranker,
            "selected_reconciliation_alpha": external_alpha,
            "manifest_sha256": _sha256(external_artifact_dir / "manifest.json"),
            "model_artifacts_sha256": external_manifest["model_artifacts_sha256"],
        },
    }


def _predict_numpy_pipeline(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    bundle = json.loads((artifact_dir / "model_bundle.json").read_text(encoding="utf-8"))
    calibrated_test = pd.DataFrame(index=range(len(test)))
    raw_test: dict[str, np.ndarray] = {}
    for target in BINARY_TARGET_COLUMNS:
        target_bundle = bundle["binary_models"][target]
        preprocessor = SplitSafePreprocessor.from_dict(target_bundle["preprocessor"])
        model = model_from_dict(target_bundle["model"])
        validation_probabilities = model.predict_proba(preprocessor.transform(validation))
        test_probabilities = model.predict_proba(preprocessor.transform(test))
        calibrator_name = target_bundle["validation_calibrator"]["type"]
        calibrator = make_calibrator(calibrator_name).fit(
            validation_probabilities, validation[target].to_numpy(float)
        )
        raw_test[target] = test_probabilities
        calibrated_test[TARGET_PROBABILITY_COLUMNS[target]] = calibrator.predict(
            test_probabilities
        )

    rank_bundle = bundle["ranking_model"]
    rank_preprocessor = SplitSafePreprocessor.from_dict(rank_bundle["preprocessor"])
    rank_model = model_from_dict(rank_bundle["model"])
    validation_rank_scores = rank_model.predict_score(rank_preprocessor.transform(validation))
    test_rank_scores = rank_model.predict_score(rank_preprocessor.transform(test))
    validation_rank_base = sigmoid(-5.0 * (validation_rank_scores - 0.5))
    test_rank_base = sigmoid(-5.0 * (test_rank_scores - 0.5))
    rank_test_probabilities = pd.DataFrame(index=range(len(test)))
    for target in BINARY_TARGET_COLUMNS:
        calibrator_name = bundle["rank_probability_calibrators"][target]["type"]
        calibrator = make_calibrator(calibrator_name).fit(
            validation_rank_base, validation[target].to_numpy(float)
        )
        rank_test_probabilities[TARGET_PROBABILITY_COLUMNS[target]] = calibrator.predict(
            test_rank_base
        )

    alpha = float(bundle["reconciliation"]["rank_blend_alpha"])
    probabilities = reconcile_probabilities(
        test, (1.0 - alpha) * calibrated_test + alpha * rank_test_probabilities
    )
    return probabilities, test_rank_scores, raw_test


def _predict_external_pipeline(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    verify_external_environment()
    import joblib
    from xgboost import XGBRanker

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    models_dir = artifact_dir / "models"
    calibrated_test = pd.DataFrame(index=range(len(test)))
    raw_test: dict[str, np.ndarray] = {}
    for target in BINARY_TARGET_COLUMNS:
        family = str(manifest["selected_families"][target])
        saved = joblib.load(models_dir / f"{target}__{family}.joblib")
        features = EXTERNAL_FEATURE_SETS[saved["candidate"]["feature_set"]]
        validation_raw = np.asarray(
            saved["pipeline"].predict_proba(validation[features])[:, 1], dtype=float
        )
        test_raw = np.asarray(
            saved["pipeline"].predict_proba(test[features])[:, 1], dtype=float
        )
        calibrator = make_calibrator(saved["calibrator"]["type"]).fit(
            validation_raw, validation[target].to_numpy(float)
        )
        raw_test[target] = test_raw
        calibrated_test[TARGET_PROBABILITY_COLUMNS[target]] = calibrator.predict(test_raw)

    rank_candidate = manifest["selected_ranker"]
    rank_preprocessor = joblib.load(models_dir / "finish_order__xgboost_preprocessor.joblib")
    ranker = XGBRanker()
    ranker.load_model(models_dir / "finish_order__xgboost_ranker.json")
    validation_rank_scores = _predict_rank_scores(
        rank_preprocessor, ranker, rank_candidate, validation
    )
    test_rank_scores = _predict_rank_scores(rank_preprocessor, ranker, rank_candidate, test)
    validation_rank_base = sigmoid(-5.0 * (validation_rank_scores - 0.5))
    test_rank_base = sigmoid(-5.0 * (test_rank_scores - 0.5))
    rank_test_probabilities = pd.DataFrame(index=range(len(test)))
    for target in BINARY_TARGET_COLUMNS:
        calibrator = PlattCalibrator().fit(
            validation_rank_base, validation[target].to_numpy(float)
        )
        rank_test_probabilities[TARGET_PROBABILITY_COLUMNS[target]] = calibrator.predict(
            test_rank_base
        )

    alpha = float(manifest["selected_reconciliation_alpha"])
    probabilities = reconcile_probabilities(
        test, (1.0 - alpha) * calibrated_test + alpha * rank_test_probabilities
    )
    return probabilities, test_rank_scores, raw_test


def _scope_metrics(
    frame: pd.DataFrame,
    positions: np.ndarray,
    probability_methods: dict[str, pd.DataFrame],
    ranking_methods: dict[str, np.ndarray],
) -> dict[str, Any]:
    scope = frame.iloc[positions].reset_index(drop=True)
    probability = {
        method: evaluate_probability_frame(
            scope, values.iloc[positions].reset_index(drop=True)
        )
        for method, values in probability_methods.items()
    }
    ranking = {
        method: ranking_metrics(scope, np.asarray(values)[positions])
        for method, values in ranking_methods.items()
    }
    consistency = {
        method: probability_consistency_metrics(
            scope,
            probability_methods[method].iloc[positions].reset_index(drop=True),
            np.asarray(ranking_methods[method])[positions],
        )
        for method in ("external_selected", "numpy_selected")
    }
    return {
        "rows": len(scope),
        "races": int(scope["race_id"].nunique()),
        "probability": probability,
        "ranking": ranking,
        "consistency": consistency,
    }


def _bootstrap_interval(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[samples].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _uncertainty_rows(
    frame: pd.DataFrame,
    positions: np.ndarray,
    scope_name: str,
    probability_methods: dict[str, pd.DataFrame],
    ranking_methods: dict[str, np.ndarray],
    repetitions: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    scope = frame.iloc[positions].reset_index(drop=True)
    race_positions = [
        np.flatnonzero(scope["race_id"].to_numpy() == race_id)
        for race_id in scope["race_id"].drop_duplicates()
    ]
    rows: list[dict[str, Any]] = []
    for method, values in probability_methods.items():
        scoped = values.iloc[positions].reset_index(drop=True)
        for target in BINARY_TARGET_COLUMNS:
            actual = scope[target].to_numpy(float)
            predicted = scoped[TARGET_PROBABILITY_COLUMNS[target]].to_numpy(float)
            metric_values = {
                "log_loss": np.asarray(
                    [log_loss(actual[race], predicted[race]) for race in race_positions]
                ),
                "brier": np.asarray(
                    [
                        float(np.mean((predicted[race] - actual[race]) ** 2))
                        for race in race_positions
                    ]
                ),
            }
            for metric, per_race in metric_values.items():
                estimate, lower, upper = _bootstrap_interval(per_race, repetitions, rng)
                rows.append(
                    {
                        "scope": scope_name,
                        "prediction_task": target,
                        "method": method,
                        "metric": metric,
                        "race_mean_estimate": estimate,
                        "lower_95": lower,
                        "upper_95": upper,
                        "races": len(race_positions),
                    }
                )

    for method, values in ranking_methods.items():
        scoped_scores = np.asarray(values)[positions]
        race_metrics = [
            ranking_metrics(scope.iloc[race].reset_index(drop=True), scoped_scores[race])
            for race in race_positions
        ]
        for metric in ("mae", "spearman", "winner_top1", "podium_overlap"):
            estimate, lower, upper = _bootstrap_interval(
                np.asarray([item[metric] for item in race_metrics]), repetitions, rng
            )
            rows.append(
                {
                    "scope": scope_name,
                    "prediction_task": "finish_order",
                    "method": method,
                    "metric": metric,
                    "race_mean_estimate": estimate,
                    "lower_95": lower,
                    "upper_95": upper,
                    "races": len(race_positions),
                }
            )
    return rows


def _calibration_rows(
    frame: pd.DataFrame,
    positions: np.ndarray,
    scope_name: str,
    probability_methods: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    scope = frame.iloc[positions].reset_index(drop=True)
    edges = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, Any]] = []
    for method, values in probability_methods.items():
        scoped = values.iloc[positions].reset_index(drop=True)
        for target in BINARY_TARGET_COLUMNS:
            actual = scope[target].to_numpy(float)
            predicted = scoped[TARGET_PROBABILITY_COLUMNS[target]].to_numpy(float)
            bins = np.minimum(np.digitize(predicted, edges[1:-1]), 9)
            for bin_index in range(10):
                mask = bins == bin_index
                if not mask.any():
                    continue
                mean_prediction = float(predicted[mask].mean())
                observed_rate = float(actual[mask].mean())
                rows.append(
                    {
                        "scope": scope_name,
                        "method": method,
                        "target": target,
                        "bin": bin_index + 1,
                        "lower": float(edges[bin_index]),
                        "upper": float(edges[bin_index + 1]),
                        "count": int(mask.sum()),
                        "mean_prediction": mean_prediction,
                        "observed_rate": observed_rate,
                        "calibration_gap": mean_prediction - observed_rate,
                    }
                )
    return rows


def _winner_cohort_metrics(
    test: pd.DataFrame,
    probability_methods: dict[str, pd.DataFrame],
    ranking_methods: dict[str, np.ndarray],
) -> dict[str, Any]:
    winners = test[test["race_winner"].eq(1)][["race_id", "driver_id"]]
    race_to_winner = dict(zip(winners["race_id"], winners["driver_id"]))
    output: dict[str, Any] = {}
    for driver_id in sorted(set(race_to_winner.values())):
        race_ids = {
            race_id for race_id, winner in race_to_winner.items() if winner == driver_id
        }
        positions = np.flatnonzero(test["race_id"].isin(race_ids).to_numpy())
        output[str(driver_id)] = _scope_metrics(
            test, positions, probability_methods, ranking_methods
        )
        output[str(driver_id)]["seasons"] = sorted(
            int(year) for year in test.iloc[positions]["year"].unique()
        )
    return output


def _validation_snapshot(
    numpy_artifact_dir: Path,
    external_artifact_dir: Path,
) -> dict[str, Any]:
    numpy_metrics = json.loads(
        (numpy_artifact_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    external_metrics = json.loads(
        (external_artifact_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    return {
        "scope": "held_out_2023_validation_report_block_only",
        "numpy": numpy_metrics["final_pipeline_held_out_validation"],
        "external": {
            "probability": external_metrics["report_block"][
                "external_selected_probability"
            ],
            "ranking": external_metrics["report_block"]["external_selected_ranking"],
            "consistency": external_metrics["report_block"][
                "external_selected_consistency"
            ],
        },
    }


def _flatten_metric_tables(metrics: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for scope, scope_metrics in metrics.items():
        for method, targets in scope_metrics["probability"].items():
            for target, values in targets.items():
                probability_rows.append(
                    {"scope": scope, "method": method, "target": target, **values}
                )
        for method, values in scope_metrics["ranking"].items():
            ranking_rows.append({"scope": scope, "method": method, **values})
    return pd.DataFrame(probability_rows), pd.DataFrame(ranking_rows)


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_report(
    validation: dict[str, Any],
    test_metrics: dict[str, Any],
    uncertainty: pd.DataFrame,
    winner_cohorts: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    lines = [
        "# Stage 2B Final Test Evaluation",
        "",
        "The 2024-2025 test set was opened once after explicit approval. All model",
        "families, hyperparameters, preprocessing, calibration types, and race-level",
        "reconciliation settings were frozen first. No test result changed a model.",
        "",
        "## Frozen configuration",
        "",
        f"- External classifiers: `{json.dumps(audit['external']['selected_families'], sort_keys=True)}`",
        f"- External ranker: `{audit['external']['selected_ranker']}`",
        f"- External reconciliation alpha: `{audit['external']['selected_reconciliation_alpha']}`",
        f"- NumPy classifiers: `{json.dumps(audit['numpy']['selected_models'], sort_keys=True)}`",
        f"- NumPy ranker: `{audit['numpy']['selected_ranker']}`",
        f"- NumPy reconciliation alpha: `{audit['numpy']['selected_reconciliation_alpha']}`",
        "- Base models and preprocessors remain fit on 2014-2022 only. The already selected calibrator types were refit on all 2023 validation rows, as specified before test unlock.",
        "",
        "## Validation results (2023 report block)",
        "",
        "These values are the previously reported validation results, not final-test results.",
        "",
        "| Pipeline | Points log loss | Podium log loss | Winner log loss | Rank MAE | Winner top-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("External", "external"), ("NumPy", "numpy")):
        values = validation[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_number(values["probability"]["points_finish"]["log_loss"]),
                    _format_number(values["probability"]["podium_finish"]["log_loss"]),
                    _format_number(values["probability"]["race_winner"]["log_loss"]),
                    _format_number(values["ranking"]["mae"]),
                    _format_number(values["ranking"]["winner_top1"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Final test probability results",
            "",
            "Log loss is primary. Brier score, AUC, ECE, calibration slope, and mean prediction are saved in the detailed artifacts.",
        ]
    )
    for scope, values in test_metrics.items():
        lines.extend(
            [
                "",
                f"### {scope}",
                "",
                "| Method | Points log loss | Podium log loss | Winner log loss |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in (
            "external_selected",
            "numpy_selected",
            "final_grid_rule",
            "train_grid_frequency",
            "uniform_field_quota",
        ):
            method_values = values["probability"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        _format_number(method_values["points_finish"]["log_loss"]),
                        _format_number(method_values["podium_finish"]["log_loss"]),
                        _format_number(method_values["race_winner"]["log_loss"]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Final test ranking results"])
    for scope, values in test_metrics.items():
        lines.extend(
            [
                "",
                f"### {scope}",
                "",
                "| Method | MAE | Spearman | Winner top-1 | Podium overlap | Points top-10 overlap |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in ("external_selected", "numpy_selected", "final_grid", "recent_form"):
            method_values = values["ranking"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        _format_number(method_values["mae"]),
                        _format_number(method_values["spearman"]),
                        _format_number(method_values["winner_top1"]),
                        _format_number(method_values["podium_overlap"]),
                        _format_number(method_values["points_top10_overlap"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Uncertainty",
            "",
            "Intervals are deterministic 95% percentile intervals from race-level bootstrap resamples. They quantify sampling uncertainty across races and not model-refit uncertainty.",
            "",
            "| Scope | Method | Task | Metric | Race mean | 95% interval |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    chosen = uncertainty[
        uncertainty["method"].isin(["external_selected", "numpy_selected"])
        & (
            uncertainty["metric"].eq("log_loss")
            | (
                uncertainty["prediction_task"].eq("finish_order")
                & uncertainty["metric"].eq("mae")
            )
        )
    ]
    for row in chosen.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.method} | {row.prediction_task} | {row.metric} | "
            f"{_format_number(row.race_mean_estimate)} | "
            f"[{_format_number(row.lower_95)}, {_format_number(row.upper_95)}] |"
        )

    lines.extend(
        [
            "",
            "## Actual-winner cohorts",
            "",
            "| Winning driver | Races | External winner log loss | NumPy winner log loss | External top-1 | NumPy top-1 | Grid top-1 | Recent-form top-1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for driver_id, values in sorted(
        winner_cohorts.items(), key=lambda item: (-item[1]["races"], item[0])
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    driver_id,
                    str(values["races"]),
                    _format_number(values["probability"]["external_selected"]["race_winner"]["log_loss"]),
                    _format_number(values["probability"]["numpy_selected"]["race_winner"]["log_loss"]),
                    _format_number(values["ranking"]["external_selected"]["winner_top1"]),
                    _format_number(values["ranking"]["numpy_selected"]["winner_top1"]),
                    _format_number(values["ranking"]["final_grid"]["winner_top1"]),
                    _format_number(values["ranking"]["recent_form"]["winner_top1"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- The final test was not used for selection, calibration fitting, thresholding, or reconciliation tuning.",
            "- Per-season and winner-cohort samples are small; use their intervals and calibration diagnostics accordingly.",
            "- DNF drivers retain official finish order. DNS/DNP rows remain excluded only by the Stage 2A eligibility audit.",
            "- Detailed calibration bins, consistency metrics, predictions, hashes, and runtime versions accompany this report.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_locked_test(
    stage2a_dir: Path,
    validation_artifact_dir: Path,
    output_dir: Path,
    *,
    external_artifact_dir: Path | None = None,
    confirm_locked_test: bool = False,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Evaluate once after explicit approval; refuse before any split is read."""

    if not confirm_locked_test:
        raise PermissionError(
            "Locked test evaluation requires explicit confirm_locked_test=True approval"
        )
    if bootstrap_repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    external_artifact_dir = external_artifact_dir or (
        validation_artifact_dir.parent / "external_validation"
    )
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Final test evaluation artifacts already exist")

    audit = audit_frozen_stage2b(
        stage2a_dir, validation_artifact_dir, external_artifact_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pre_unlock_audit.json").write_text(
        json.dumps(_json_ready(audit), indent=2) + "\n", encoding="utf-8"
    )

    train_path = stage2a_dir / "train.csv"
    validation_path = stage2a_dir / "validation.csv"
    test_path = stage2a_dir / "test.csv"
    train = pd.read_csv(train_path, low_memory=False)
    validation = pd.read_csv(validation_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    _validate_split(train, "train")
    _validate_split(validation, "validation")
    _validate_split(test, "test")
    train = train[train["prediction_eligible"].eq(1)].reset_index(drop=True)
    validation = validation[validation["prediction_eligible"].eq(1)].reset_index(drop=True)
    test = test[test["prediction_eligible"].eq(1)].reset_index(drop=True)
    if int(train["year"].max()) >= int(validation["year"].min()):
        raise ValueError("Train and validation splits are not chronological")
    if int(validation["year"].max()) >= int(test["year"].min()):
        raise ValueError("Validation and test splits are not chronological")

    numpy_probabilities, numpy_rank_scores, numpy_raw = _predict_numpy_pipeline(
        validation, test, validation_artifact_dir
    )
    external_probabilities, external_rank_scores, external_raw = _predict_external_pipeline(
        validation, test, external_artifact_dir
    )
    ranking_baselines, probability_baselines = validation_baselines(train, test)
    probability_methods = {
        "external_selected": external_probabilities,
        "numpy_selected": numpy_probabilities,
        **probability_baselines,
    }
    ranking_methods = {
        "external_selected": external_rank_scores,
        "numpy_selected": numpy_rank_scores,
        **ranking_baselines,
    }

    scopes = {"overall": np.arange(len(test), dtype=int)}
    for year in sorted(test["year"].unique()):
        scopes[str(int(year))] = np.flatnonzero(test["year"].eq(year).to_numpy())
    test_metrics = {
        scope: _scope_metrics(test, positions, probability_methods, ranking_methods)
        for scope, positions in scopes.items()
    }

    rng = np.random.default_rng(FINAL_RANDOM_SEED)
    uncertainty_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for scope, positions in scopes.items():
        uncertainty_rows.extend(
            _uncertainty_rows(
                test,
                positions,
                scope,
                probability_methods,
                ranking_methods,
                bootstrap_repetitions,
                rng,
            )
        )
        calibration_rows.extend(
            _calibration_rows(test, positions, scope, probability_methods)
        )
    uncertainty = pd.DataFrame(uncertainty_rows)
    calibration = pd.DataFrame(calibration_rows)
    winner_cohorts = _winner_cohort_metrics(test, probability_methods, ranking_methods)
    validation_snapshot = _validation_snapshot(validation_artifact_dir, external_artifact_dir)

    predictions = test[
        [
            "race_id",
            "race_date",
            "year",
            "round",
            "driver_id",
            "finish_order",
            *BINARY_TARGET_COLUMNS,
        ]
    ].copy()
    actual_winner = test[test["race_winner"].eq(1)].set_index("race_id")["driver_id"]
    predictions["actual_winner_driver_id"] = predictions["race_id"].map(actual_winner)
    for method, scores in ranking_methods.items():
        predictions[f"rank_score__{method}"] = scores
        predictions[f"predicted_finish_order__{method}"] = (
            pd.Series(scores)
            .groupby(test["race_id"].reset_index(drop=True))
            .rank(method="first", ascending=True)
            .astype(int)
        )
    for method, probabilities in probability_methods.items():
        for target in BINARY_TARGET_COLUMNS:
            predictions[f"{TARGET_PROBABILITY_COLUMNS[target]}__{method}"] = probabilities[
                TARGET_PROBABILITY_COLUMNS[target]
            ].to_numpy(float)
    for pipeline, raw_values in (("external", external_raw), ("numpy", numpy_raw)):
        for target, values in raw_values.items():
            predictions[f"raw_p_{target}__{pipeline}"] = values

    probability_table, ranking_table = _flatten_metric_tables(test_metrics)
    report = _build_report(
        validation_snapshot, test_metrics, uncertainty, winner_cohorts, audit
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, lineterminator="\n")
    probability_table.to_csv(
        output_dir / "probability_metrics.csv", index=False, lineterminator="\n"
    )
    ranking_table.to_csv(
        output_dir / "ranking_metrics.csv", index=False, lineterminator="\n"
    )
    uncertainty.to_csv(
        output_dir / "uncertainty_intervals.csv", index=False, lineterminator="\n"
    )
    calibration.to_csv(
        output_dir / "calibration_by_bin.csv", index=False, lineterminator="\n"
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(_json_ready(test_metrics), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "winner_cohort_metrics.json").write_text(
        json.dumps(_json_ready(winner_cohorts), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation_results_snapshot.json").write_text(
        json.dumps(_json_ready(validation_snapshot), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "FINAL_TEST_REPORT.md").write_text(report, encoding="utf-8")

    runtime = verify_external_environment()
    artifact_names = [
        "pre_unlock_audit.json",
        "test_predictions.csv",
        "probability_metrics.csv",
        "ranking_metrics.csv",
        "uncertainty_intervals.csv",
        "calibration_by_bin.csv",
        "test_metrics.json",
        "winner_cohort_metrics.json",
        "validation_results_snapshot.json",
        "FINAL_TEST_REPORT.md",
    ]
    project_root = Path(__file__).resolve().parents[2]
    manifest = {
        "stage": "2B-final-test",
        "test_status": "evaluated_once_after_explicit_approval",
        "approval_gate": "confirm_locked_test=True",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": FINAL_RANDOM_SEED,
        "bootstrap_repetitions": bootstrap_repetitions,
        "runtime": {
            **runtime,
            "system_python": sys.version.split()[0],
            "platform_check": platform.platform(),
        },
        "data_sha256": {
            "train.csv": _sha256(train_path),
            "validation.csv": _sha256(validation_path),
            "test.csv": _sha256(test_path),
        },
        "rows": len(test),
        "races": int(test["race_id"].nunique()),
        "years": sorted(int(year) for year in test["year"].unique()),
        "base_models_and_preprocessing_fit_on": "2014-2022 train only",
        "calibrator_types_selected_on": "2023 calibration and method-selection blocks only",
        "calibrators_final_fit_on": "all 2023 validation rows",
        "model_and_reconciliation_selection_source": "frozen validation artifacts; no test-based changes",
        "frozen_audit": audit,
        "evaluation_code_sha256": {
            "src/f1_predictor/stage2b_evaluation.py": _sha256(Path(__file__)),
            "scripts/evaluate_stage2b_test.py": _sha256(
                project_root / "scripts" / "evaluate_stage2b_test.py"
            ),
        },
        "artifacts_sha256": {
            name: _sha256(output_dir / name) for name in artifact_names
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "validation_results": validation_snapshot,
        "test_results": test_metrics,
    }
