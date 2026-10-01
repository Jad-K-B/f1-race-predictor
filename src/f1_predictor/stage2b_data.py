"""Split-safe data loading and preprocessing for Stage 2B."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stage2a import (
    BINARY_TARGET_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OUTCOME_METADATA_COLUMNS,
    TARGET_COLUMNS,
)


CORE_CATEGORICAL_FEATURES = [
    "driver_id",
    "constructor_id",
    "engine_manufacturer_id",
    "grand_prix_id",
    "circuit_id",
    "circuit_type",
]

CORE_NUMERIC_FEATURES = [
    "year",
    "round",
    "is_sprint_weekend",
    "driver_age_years",
    "qualifying_position",
    "qualifying_position_normalized",
    "final_grid_position",
    "effective_grid_position",
    "grid_position_normalized",
    "grid_delta_from_qualifying",
    "grid_penalty_positions",
    "has_grid_penalty",
    "pit_lane_start",
    "missing_final_grid",
    "qualifying_laps",
    "reached_q2",
    "reached_q3",
    "q1_pct_off_best",
    "q2_pct_off_best",
    "q3_pct_off_best",
    "grid_field_size",
    "current_constructor_avg_grid",
    "current_constructor_best_grid",
    "current_constructor_avg_qualifying",
    "driver_prior_entries",
    "driver_career_avg_finish",
    "driver_career_points_per_start",
    "driver_career_points_finish_rate",
    "driver_career_podium_rate",
    "driver_career_win_rate",
    "driver_career_dnf_rate",
    "driver_season_prior_entries",
    "driver_season_prior_points",
    "driver_season_avg_finish",
    "driver_season_points_finish_rate",
    "driver_season_podium_rate",
    "driver_season_win_rate",
    "driver_season_dnf_rate",
    "driver_recent_5_avg_finish",
    "driver_recent_5_points_per_start",
    "driver_recent_5_points_finish_rate",
    "driver_recent_5_podium_rate",
    "driver_recent_5_win_rate",
    "driver_recent_5_dnf_rate",
    "driver_recent_5_avg_grid",
    "driver_recent_5_avg_qualifying",
    "driver_circuit_prior_entries",
    "driver_circuit_avg_finish",
    "driver_circuit_points_finish_rate",
    "driver_circuit_podium_rate",
    "driver_circuit_win_rate",
    "driver_constructor_prior_entries",
    "driver_constructor_avg_finish",
    "driver_constructor_points_per_start",
    "driver_constructor_points_finish_rate",
    "driver_constructor_podium_rate",
    "driver_constructor_win_rate",
    "driver_championship_position_pre_race",
    "driver_championship_points_pre_race",
    "constructor_prior_races",
    "constructor_career_points_per_race",
    "constructor_career_points_finish_rate",
    "constructor_career_podiums_per_race",
    "constructor_career_wins_per_race",
    "constructor_career_dnf_rate",
    "constructor_career_avg_finish",
    "constructor_season_prior_races",
    "constructor_season_prior_points",
    "constructor_season_points_per_race",
    "constructor_season_points_finish_rate",
    "constructor_season_podiums_per_race",
    "constructor_season_wins_per_race",
    "constructor_season_dnf_rate",
    "constructor_recent_5_points_per_race",
    "constructor_recent_5_podiums_per_race",
    "constructor_recent_5_wins_per_race",
    "constructor_recent_5_dnf_rate",
    "constructor_recent_5_avg_finish",
    "constructor_recent_5_avg_grid",
    "constructor_recent_5_avg_qualifying",
    "constructor_championship_position_pre_race",
    "constructor_championship_points_pre_race",
    "circuit_prior_races",
    "circuit_historical_dnf_rate",
    "circuit_historical_avg_positions_gained",
    "circuit_historical_pole_win_rate",
    "circuit_historical_avg_winner_grid",
]

FEATURE_SETS = {
    "core": CORE_CATEGORICAL_FEATURES + CORE_NUMERIC_FEATURES,
    "all": list(FEATURE_COLUMNS),
}

COMPACT_CATEGORICAL_FEATURES = [
    "grand_prix_id",
    "circuit_id",
    "circuit_type",
    "circuit_direction",
    "qualifying_format",
]

COMPACT_NUMERIC_FEATURES = [
    "year",
    "round",
    "is_sprint_weekend",
    "driver_age_years",
    "qualifying_position",
    "qualifying_position_normalized",
    "final_grid_position",
    "effective_grid_position",
    "grid_position_normalized",
    "grid_delta_from_qualifying",
    "grid_penalty_positions",
    "has_grid_penalty",
    "pit_lane_start",
    "missing_final_grid",
    "reached_q2",
    "reached_q3",
    "q1_pct_off_best",
    "q2_pct_off_best",
    "q3_pct_off_best",
    "grid_field_size",
    "driver_prior_entries",
    "driver_career_avg_finish",
    "driver_career_points_finish_rate",
    "driver_career_podium_rate",
    "driver_career_win_rate",
    "driver_career_dnf_rate",
    "driver_season_prior_entries",
    "driver_season_prior_points",
    "driver_season_avg_finish",
    "driver_season_points_finish_rate",
    "driver_season_podium_rate",
    "driver_season_win_rate",
    "driver_recent_5_avg_finish",
    "driver_recent_5_points_per_start",
    "driver_recent_5_points_finish_rate",
    "driver_recent_5_podium_rate",
    "driver_recent_5_win_rate",
    "driver_recent_5_dnf_rate",
    "driver_recent_5_avg_grid",
    "driver_recent_5_avg_qualifying",
    "driver_championship_position_pre_race",
    "driver_championship_points_pre_race",
    "constructor_season_prior_races",
    "constructor_season_prior_points",
    "constructor_season_points_per_race",
    "constructor_season_points_finish_rate",
    "constructor_season_podiums_per_race",
    "constructor_season_wins_per_race",
    "constructor_recent_5_points_per_race",
    "constructor_recent_5_podiums_per_race",
    "constructor_recent_5_wins_per_race",
    "constructor_recent_5_dnf_rate",
    "constructor_recent_5_avg_finish",
    "constructor_recent_5_avg_grid",
    "constructor_recent_5_avg_qualifying",
    "constructor_championship_position_pre_race",
    "constructor_championship_points_pre_race",
    "circuit_prior_races",
    "circuit_historical_dnf_rate",
    "circuit_historical_avg_positions_gained",
    "circuit_historical_pole_win_rate",
    "circuit_historical_avg_winner_grid",
]

IDENTITY_FEATURES = {
    "driver_id",
    "constructor_id",
    "engine_manufacturer_id",
}

EXTERNAL_FEATURE_SETS = {
    "compact": COMPACT_CATEGORICAL_FEATURES + COMPACT_NUMERIC_FEATURES,
    "id_free": [column for column in FEATURE_COLUMNS if column not in IDENTITY_FEATURES],
    "full": list(FEATURE_COLUMNS),
}

NON_FEATURE_COLUMNS = {
    "race_id",
    "race_date",
    "split",
    *TARGET_COLUMNS,
    *OUTCOME_METADATA_COLUMNS,
}


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_end_year: int
    validation_year: int


TEMPORAL_FOLDS = [
    TemporalFold("through_2017_to_2018", 2017, 2018),
    TemporalFold("through_2018_to_2019", 2018, 2019),
    TemporalFold("through_2019_to_2020", 2019, 2020),
    TemporalFold("through_2020_to_2021", 2020, 2021),
    TemporalFold("through_2021_to_2022", 2021, 2022),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage2b_training_data(
    stage2a_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load eligible training and validation rows without opening the test file."""

    train_path = stage2a_dir / "train.csv"
    validation_path = stage2a_dir / "validation.csv"
    train = pd.read_csv(train_path, low_memory=False)
    validation = pd.read_csv(validation_path, low_memory=False)
    _validate_split(train, "train")
    _validate_split(validation, "validation")
    train = train[train["prediction_eligible"].eq(1)].reset_index(drop=True)
    validation = validation[
        validation["prediction_eligible"].eq(1)
    ].reset_index(drop=True)
    if int(train["year"].max()) >= int(validation["year"].min()):
        raise ValueError("Training and validation years are not chronological")
    hashes = {"train.csv": _sha256(train_path), "validation.csv": _sha256(validation_path)}
    return train, validation, hashes


def _validate_split(frame: pd.DataFrame, expected_split: str) -> None:
    required = {
        "race_id",
        "race_date",
        "split",
        "prediction_eligible",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage 2A {expected_split} data is missing columns: {missing}")
    if not frame["split"].eq(expected_split).all():
        raise ValueError(f"Expected only {expected_split} rows")
    if frame.duplicated(["race_id", "driver_id"]).any():
        raise ValueError(f"Duplicate race/driver rows in {expected_split}")
    if set(FEATURE_COLUMNS) & NON_FEATURE_COLUMNS:
        raise ValueError("Stage 2A feature allowlist overlaps outcomes or identifiers")


def temporal_fold_frames(
    train: pd.DataFrame, fold: TemporalFold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = train[train["year"].le(fold.train_end_year)].copy()
    score = train[train["year"].eq(fold.validation_year)].copy()
    if fit.empty or score.empty:
        raise ValueError(f"Temporal fold {fold.name} has an empty partition")
    if fit["race_id"].isin(score["race_id"]).any():
        raise ValueError(f"Temporal fold {fold.name} splits a race")
    if int(fit["year"].max()) >= int(score["year"].min()):
        raise ValueError(f"Temporal fold {fold.name} is not chronological")
    return fit, score


def validation_calibration_split(
    validation: pd.DataFrame, calibration_races: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    race_order = (
        validation[["race_id", "race_date", "round"]]
        .drop_duplicates("race_id")
        .sort_values(["race_date", "round", "race_id"])
    )
    if not 1 <= calibration_races < len(race_order):
        raise ValueError("Calibration race count must leave held-out validation races")
    calibration_ids = set(race_order.iloc[:calibration_races]["race_id"])
    calibration = validation[validation["race_id"].isin(calibration_ids)].copy()
    selection = validation[~validation["race_id"].isin(calibration_ids)].copy()
    if pd.to_datetime(calibration["race_date"]).max() >= pd.to_datetime(
        selection["race_date"]
    ).min():
        raise ValueError("Calibration and selection validation blocks overlap chronologically")
    return calibration, selection


def validation_protocol_split(
    validation: pd.DataFrame,
    calibration_races: int = 8,
    method_selection_races: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological calibration, method-selection, and report blocks."""

    race_order = (
        validation[["race_id", "race_date", "round"]]
        .drop_duplicates("race_id")
        .sort_values(["race_date", "round", "race_id"])
    )
    report_races = len(race_order) - calibration_races - method_selection_races
    if calibration_races < 1 or method_selection_races < 1 or report_races < 1:
        raise ValueError("Validation protocol must leave races in all three blocks")
    calibration_ids = set(race_order.iloc[:calibration_races]["race_id"])
    selection_ids = set(
        race_order.iloc[
            calibration_races : calibration_races + method_selection_races
        ]["race_id"]
    )
    calibration = validation[validation["race_id"].isin(calibration_ids)].copy()
    selection = validation[validation["race_id"].isin(selection_ids)].copy()
    report = validation[
        ~validation["race_id"].isin(calibration_ids | selection_ids)
    ].copy()
    block_dates = [
        pd.to_datetime(block["race_date"]) for block in (calibration, selection, report)
    ]
    if block_dates[0].max() >= block_dates[1].min() or block_dates[1].max() >= block_dates[2].min():
        raise ValueError("Validation protocol blocks are not strictly chronological")
    return calibration, selection, report


def normalized_finish_target(frame: pd.DataFrame) -> np.ndarray:
    field_size = frame.groupby("race_id")["driver_id"].transform("size").to_numpy(float)
    order = frame["finish_order"].to_numpy(float)
    return np.divide(
        order - 1.0,
        np.maximum(field_size - 1.0, 1.0),
        out=np.zeros_like(order, dtype=float),
    )


class SplitSafePreprocessor:
    """Median-impute/scale numeric data and one-hot encode categoricals."""

    missing_token = "__MISSING__"
    unknown_token = "__UNKNOWN__"

    def __init__(self, feature_columns: list[str]):
        unknown = sorted(set(feature_columns) - set(FEATURE_COLUMNS))
        if unknown:
            raise ValueError(f"Features are not in the Stage 2A allowlist: {unknown}")
        self.feature_columns = list(feature_columns)
        self.categorical_columns = [
            column for column in CATEGORICAL_FEATURES if column in feature_columns
        ]
        self.numeric_columns = [
            column for column in NUMERIC_FEATURES if column in feature_columns
        ]
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.missing_indicator_columns: list[str] = []
        self.categories: dict[str, list[str]] = {}
        self.output_feature_names: list[str] = []
        self.fitted = False

    def fit(self, frame: pd.DataFrame) -> "SplitSafePreprocessor":
        if "split" in frame and not frame["split"].eq("train").all():
            raise ValueError("Preprocessing may only be fit on training rows")
        if frame.empty:
            raise ValueError("Cannot fit preprocessing on an empty frame")
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).to_numpy(float)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            self.numeric_medians[column] = median
            self.numeric_means[column] = mean
            self.numeric_scales[column] = scale if scale > 1e-12 else 1.0
            if values.isna().any():
                self.missing_indicator_columns.append(column)
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna(self.missing_token)
            categories = sorted(set(values.astype(str)))
            self.categories[column] = categories
        self.output_feature_names = list(self.numeric_columns)
        self.output_feature_names.extend(
            f"{column}__missing" for column in self.missing_indicator_columns
        )
        for column in self.categorical_columns:
            self.output_feature_names.extend(
                f"{column}=={value}"
                for value in self.categories[column] + [self.unknown_token]
            )
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Preprocessor must be fit before transform")
        blocks: list[np.ndarray] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            filled = values.fillna(self.numeric_medians[column]).to_numpy(float)
            blocks.append(
                ((filled - self.numeric_means[column]) / self.numeric_scales[column])[
                    :, None
                ]
            )
        for column in self.missing_indicator_columns:
            missing = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).isna()
            blocks.append(missing.to_numpy(float)[:, None])
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna(self.missing_token).astype(str)
            known = set(self.categories[column])
            values = values.where(values.isin(known), self.unknown_token)
            for category in self.categories[column] + [self.unknown_token]:
                blocks.append(values.eq(category).to_numpy(float)[:, None])
        if not blocks:
            raise ValueError("Preprocessor has no output features")
        return np.hstack(blocks)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise ValueError("Cannot serialize an unfitted preprocessor")
        return {
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "missing_indicator_columns": self.missing_indicator_columns,
            "categories": self.categories,
            "output_feature_names": self.output_feature_names,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitSafePreprocessor":
        instance = cls(list(payload["feature_columns"]))
        instance.numeric_medians = {
            key: float(value) for key, value in payload["numeric_medians"].items()
        }
        instance.numeric_means = {
            key: float(value) for key, value in payload["numeric_means"].items()
        }
        instance.numeric_scales = {
            key: float(value) for key, value in payload["numeric_scales"].items()
        }
        instance.missing_indicator_columns = list(payload["missing_indicator_columns"])
        instance.categories = {
            key: list(values) for key, values in payload["categories"].items()
        }
        instance.output_feature_names = list(payload["output_feature_names"])
        instance.fitted = True
        return instance
