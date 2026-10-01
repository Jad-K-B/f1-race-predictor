"""Leakage-safe Stage 2A dataset construction from local F1DB CSV files.

The prediction timestamp is after the final starting grid is confirmed and before
the race starts. Current-race qualifying and grid information are therefore
allowed; current-race outcome fields are used only to create targets.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BINARY_TARGET_COLUMNS = ["points_finish", "podium_finish", "race_winner"]
RANKING_TARGET_COLUMN = "finish_order"
TARGET_COLUMNS = BINARY_TARGET_COLUMNS + [RANKING_TARGET_COLUMN]
OUTCOME_METADATA_COLUMNS = [
    "classified_finish_position",
    "result_status",
    "prediction_eligible",
    "eligibility_reason",
    "eligibility_source",
]
DEFAULT_ELIGIBILITY_AUDIT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "stage2a_eligibility_audit.json"
)

CATEGORICAL_FEATURES = [
    "driver_id",
    "constructor_id",
    "engine_manufacturer_id",
    "grand_prix_id",
    "circuit_id",
    "circuit_layout_id",
    "circuit_type",
    "circuit_direction",
    "qualifying_format",
    "driver_nationality_country_id",
    "circuit_country_id",
]

NUMERIC_FEATURES = [
    "year",
    "round",
    "race_month",
    "course_length_km",
    "circuit_turns",
    "is_sprint_weekend",
    "driver_age_years",
    "is_driver_home_race",
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
    "current_constructor_entry_count",
    "current_constructor_avg_grid",
    "current_constructor_best_grid",
    "current_constructor_avg_qualifying",
    "grid_delta_to_constructor_avg",
    "qualifying_delta_to_constructor_avg",
    "driver_prior_entries",
    "driver_prior_classified_finishes",
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
    "driver_previous_finish",
    "driver_previous_points",
    "driver_previous_grid",
    "driver_previous_qualifying",
    "driver_circuit_prior_entries",
    "driver_circuit_avg_finish",
    "driver_circuit_points_finish_rate",
    "driver_circuit_podium_rate",
    "driver_circuit_win_rate",
    "driver_circuit_dnf_rate",
    "driver_constructor_prior_entries",
    "driver_constructor_avg_finish",
    "driver_constructor_points_per_start",
    "driver_constructor_points_finish_rate",
    "driver_constructor_podium_rate",
    "driver_constructor_win_rate",
    "driver_constructor_dnf_rate",
    "driver_championship_position_pre_race",
    "driver_championship_points_pre_race",
    "driver_previous_season_position",
    "driver_previous_season_points",
    "constructor_prior_races",
    "constructor_prior_entries",
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
    "constructor_championship_position_pre_race",
    "constructor_championship_points_pre_race",
    "constructor_previous_season_position",
    "constructor_previous_season_points",
    "circuit_prior_races",
    "circuit_historical_dnf_rate",
    "circuit_historical_avg_positions_gained",
    "circuit_historical_pole_win_rate",
    "circuit_historical_avg_winner_grid",
]

for _window in (3, 5, 10):
    NUMERIC_FEATURES.extend(
        [
            f"driver_recent_{_window}_avg_finish",
            f"driver_recent_{_window}_points_per_start",
            f"driver_recent_{_window}_points_finish_rate",
            f"driver_recent_{_window}_podium_rate",
            f"driver_recent_{_window}_win_rate",
            f"driver_recent_{_window}_dnf_rate",
            f"driver_recent_{_window}_avg_grid",
            f"driver_recent_{_window}_avg_qualifying",
            f"constructor_recent_{_window}_points_per_race",
            f"constructor_recent_{_window}_podiums_per_race",
            f"constructor_recent_{_window}_wins_per_race",
            f"constructor_recent_{_window}_dnf_rate",
            f"constructor_recent_{_window}_avg_finish",
            f"constructor_recent_{_window}_avg_grid",
            f"constructor_recent_{_window}_avg_qualifying",
        ]
    )

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
OUTPUT_COLUMNS = list(
    dict.fromkeys(
        ["race_id", "race_date", "split"]
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
        + OUTCOME_METADATA_COLUMNS
    )
)

# These source outcome fields must never appear as model features for the race
# being predicted. Historical aggregates derived from earlier races are allowed.
FORBIDDEN_CURRENT_RACE_FEATURES = {
    "finish_position",
    "position_number",
    "position_text",
    "race_points",
    "laps_completed",
    "race_time",
    "race_time_millis",
    "race_gap",
    "race_interval",
    "reason_retired",
    "positions_gained",
    "pit_stops",
    "fastest_lap",
    "driver_of_the_day",
    "grand_slam",
    "finish_order",
    "classified_finish_position",
    "result_status",
    "prediction_eligible",
    "eligibility_reason",
    "eligibility_source",
}


@dataclass(frozen=True)
class Stage2AConfig:
    """Dataset and split boundaries for Stage 2A."""

    start_year: int = 2014
    end_year: int = 2025
    train_end_year: int = 2022
    validation_year: int = 2023
    test_start_year: int = 2024

    def __post_init__(self) -> None:
        if self.start_year > self.train_end_year:
            raise ValueError("start_year must not be after train_end_year")
        if self.validation_year != self.train_end_year + 1:
            raise ValueError("validation_year must immediately follow train_end_year")
        if self.test_start_year != self.validation_year + 1:
            raise ValueError("test_start_year must immediately follow validation_year")
        if self.end_year < self.test_start_year:
            raise ValueError("end_year must include at least one test season")


RAW_FILES = {
    "races": "f1db-races.csv",
    "results": "f1db-races-race-results.csv",
    "grid": "f1db-races-starting-grid-positions.csv",
    "qualifying": "f1db-races-qualifying-results.csv",
    "driver_standings": "f1db-races-driver-standings.csv",
    "constructor_standings": "f1db-races-constructor-standings.csv",
    "drivers": "f1db-drivers.csv",
    "circuits": "f1db-circuits.csv",
}

RAW_REQUIRED_COLUMNS = {
    "races": {
        "id",
        "year",
        "round",
        "date",
        "grandPrixId",
        "qualifyingFormat",
        "sprintQualifyingFormat",
        "circuitId",
        "circuitLayoutId",
        "circuitType",
        "direction",
        "courseLength",
        "turns",
        "sprintRaceDate",
    },
    "results": {
        "raceId",
        "year",
        "round",
        "positionDisplayOrder",
        "positionNumber",
        "positionText",
        "driverId",
        "constructorId",
        "engineManufacturerId",
        "points",
        "qualificationPositionNumber",
        "gridPositionNumber",
        "gridPositionText",
        "positionsGained",
    },
    "grid": {
        "raceId",
        "driverId",
        "positionNumber",
        "positionText",
        "constructorId",
        "engineManufacturerId",
        "qualificationPositionNumber",
        "gridPenalty",
        "gridPenaltyPositions",
    },
    "qualifying": {
        "raceId",
        "driverId",
        "positionDisplayOrder",
        "positionNumber",
        "constructorId",
        "engineManufacturerId",
        "q1Millis",
        "q2Millis",
        "q3Millis",
        "laps",
    },
    "driver_standings": {
        "raceId",
        "driverId",
        "positionNumber",
        "points",
    },
    "constructor_standings": {
        "raceId",
        "constructorId",
        "positionNumber",
        "points",
    },
    "drivers": {"id", "dateOfBirth", "nationalityCountryId"},
    "circuits": {"id", "countryId"},
}

PREPROCESSING_CONTRACT = {
    "model_input_allowlist": "FEATURE_COLUMNS only",
    "fit_preprocessors_on": "train",
    "transform_only": ["validation", "test"],
    "fit_hyperparameters_on": ["train", "validation"],
    "fit_probability_calibration_on": "validation",
    "final_evaluation_only": "test",
    "categorical_unknown_policy": "must handle unseen categories without refitting",
    "missing_value_policy": "imputers must be fit on train only",
    "sequential_backtest_note": (
        "Validation and test features may use outcomes of earlier completed races "
        "in the same split, because those outcomes are known before each later race; "
        "they never use the current or a future race outcome."
    ),
}


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load only the F1DB tables required by Stage 2A."""

    tables: dict[str, pd.DataFrame] = {}
    for name, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required F1DB file is missing: {path}")
        tables[name] = pd.read_csv(path, low_memory=False)
    return tables


def load_eligibility_audit(path: Path | None = None) -> list[dict[str, Any]]:
    """Load explicit prediction-timestamp eligibility decisions."""

    audit_path = path or DEFAULT_ELIGIBILITY_AUDIT_PATH
    if not audit_path.is_file():
        raise FileNotFoundError(f"Eligibility audit is missing: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Eligibility audit must contain a records list")
    required = {
        "race_id",
        "driver_id",
        "prediction_eligible",
        "reason",
        "source_url",
    }
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Eligibility audit record is missing fields: {sorted(missing)}")
    keys = [(int(record["race_id"]), str(record["driver_id"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Eligibility audit contains duplicate race/driver decisions")
    return records


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prior_mean(
    frame: pd.DataFrame, group_columns: list[str], value_column: str
) -> pd.Series:
    return frame.groupby(group_columns, sort=False, dropna=False)[value_column].transform(
        lambda values: values.shift().expanding(min_periods=1).mean()
    )


def _prior_rolling_mean(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    window: int,
) -> pd.Series:
    return frame.groupby(group_columns, sort=False, dropna=False)[value_column].transform(
        lambda values: values.shift().rolling(window=window, min_periods=1).mean()
    )


def _prior_sum(
    frame: pd.DataFrame, group_columns: list[str], value_column: str
) -> pd.Series:
    values = frame[value_column].fillna(0)
    return values.groupby(
        [frame[column] for column in group_columns], sort=False, dropna=False
    ).cumsum() - values


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _validate_raw_tables(tables: dict[str, pd.DataFrame], config: Stage2AConfig) -> None:
    missing_tables = sorted(set(RAW_FILES) - set(tables))
    if missing_tables:
        raise ValueError(f"Required raw tables are missing: {missing_tables}")
    for name, required_columns in RAW_REQUIRED_COLUMNS.items():
        missing_columns = sorted(required_columns - set(tables[name].columns))
        if missing_columns:
            raise ValueError(
                f"Raw table {name!r} is missing required columns: {missing_columns}"
            )

    results = tables["results"]
    races = tables["races"]

    if races.duplicated(["id"]).any():
        raise ValueError("Race metadata contains duplicate race IDs")

    selected = results[results["year"].between(config.start_year, config.end_year)]
    if selected.duplicated(["raceId", "driverId"]).any():
        raise ValueError("Selected race results contain duplicate raceId/driverId records")
    if selected.empty:
        raise ValueError("No official race results exist for the configured seasons")
    if selected[["raceId", "year", "round", "driverId"]].isna().any().any():
        raise ValueError("Official race results have missing identity fields")

    winner_counts = selected.assign(
        target=_as_number(selected["positionNumber"]).eq(1)
    ).groupby("raceId")["target"].sum()
    podium_counts = selected.assign(
        target=_as_number(selected["positionNumber"]).le(3)
    ).groupby("raceId")["target"].sum()
    if not winner_counts.eq(1).all():
        raise ValueError("Every selected race must have exactly one official winner")
    if not podium_counts.eq(3).all():
        raise ValueError("Every selected race must have exactly three podium finishers")


def _prepare_history(
    tables: dict[str, pd.DataFrame], config: Stage2AConfig
) -> pd.DataFrame:
    races = tables["races"].copy()
    races = races[races["year"].le(config.end_year)].sort_values(
        ["year", "round", "date", "id"]
    )
    races["event_index"] = range(len(races))
    race_meta = races[
        [
            "id",
            "year",
            "round",
            "date",
            "grandPrixId",
            "qualifyingFormat",
            "sprintQualifyingFormat",
            "circuitId",
            "circuitLayoutId",
            "circuitType",
            "direction",
            "courseLength",
            "turns",
            "sprintRaceDate",
            "event_index",
        ]
    ].rename(
        columns={
            "id": "raceId",
            "year": "race_year",
            "round": "race_round",
            "date": "race_date",
            "grandPrixId": "grand_prix_id",
            "qualifyingFormat": "qualifying_format",
            "sprintQualifyingFormat": "sprint_qualifying_format",
            "circuitId": "circuit_id",
            "circuitLayoutId": "circuit_layout_id",
            "circuitType": "circuit_type",
            "direction": "circuit_direction",
            "courseLength": "course_length_km",
            "turns": "circuit_turns",
            "sprintRaceDate": "sprint_race_date",
        }
    )

    history = tables["results"].copy()
    history = history[history["year"].le(config.end_year)]
    # Shared-car-era data can contain two or three records for the same driver in
    # one race. Keep the driver's best classified record as the representative
    # appearance and sum any points credited across those records. The requested
    # 2014-2025 target period is already unique and is unaffected by this step.
    if history.duplicated(["raceId", "driverId"]).any():
        history = history.sort_values(
            ["raceId", "driverId", "positionDisplayOrder"], na_position="last"
        )
        credited_points = history.groupby(["raceId", "driverId"], dropna=False)[
            "points"
        ].transform(lambda values: _as_number(values).sum(min_count=1))
        history["points"] = credited_points
        history = history.drop_duplicates(["raceId", "driverId"], keep="first")
    history = history.merge(race_meta, on="raceId", how="inner", validate="many_to_one")
    if not history["year"].eq(history["race_year"]).all() or not history["round"].eq(
        history["race_round"]
    ).all():
        raise ValueError("Race result season/round does not match race metadata")

    grid = tables["grid"][
        [
            "raceId",
            "driverId",
            "positionNumber",
            "positionText",
            "constructorId",
            "engineManufacturerId",
            "qualificationPositionNumber",
            "gridPenalty",
            "gridPenaltyPositions",
        ]
    ].rename(
        columns={
            "positionNumber": "grid_position_source",
            "positionText": "grid_text_source",
            "constructorId": "grid_constructor_id",
            "engineManufacturerId": "grid_engine_id",
            "qualificationPositionNumber": "grid_qualifying_position",
            "gridPenalty": "grid_penalty_text",
            "gridPenaltyPositions": "grid_penalty_positions_source",
        }
    )
    qualifying_source = tables["qualifying"].sort_values(
        ["raceId", "driverId", "positionDisplayOrder"], na_position="last"
    )
    qualifying_source = qualifying_source.drop_duplicates(
        ["raceId", "driverId"], keep="first"
    )
    qualifying = qualifying_source[
        [
            "raceId",
            "driverId",
            "positionNumber",
            "constructorId",
            "engineManufacturerId",
            "q1Millis",
            "q2Millis",
            "q3Millis",
            "laps",
        ]
    ].rename(
        columns={
            "positionNumber": "qualifying_position_source",
            "constructorId": "qualifying_constructor_id",
            "engineManufacturerId": "qualifying_engine_id",
            "laps": "qualifying_laps",
        }
    )
    history = history.merge(
        grid, on=["raceId", "driverId"], how="left", validate="one_to_one"
    ).merge(
        qualifying, on=["raceId", "driverId"], how="left", validate="one_to_one"
    )

    history["constructor_id"] = (
        history["grid_constructor_id"]
        .combine_first(history["qualifying_constructor_id"])
        .combine_first(history["constructorId"])
    )
    history["engine_manufacturer_id"] = (
        history["grid_engine_id"]
        .combine_first(history["qualifying_engine_id"])
        .combine_first(history["engineManufacturerId"])
    )
    has_grid_record = (
        history["grid_constructor_id"].notna()
        | history["grid_text_source"].notna()
        | history["grid_position_source"].notna()
    )
    history["_has_final_grid_record"] = has_grid_record.astype("int8")
    history["final_grid_position"] = _as_number(
        history["grid_position_source"]
    ).where(has_grid_record, _as_number(history["gridPositionNumber"]))
    history["grid_text"] = history["grid_text_source"].where(
        has_grid_record, history["gridPositionText"]
    )
    history["qualifying_position"] = (
        _as_number(history["grid_qualifying_position"])
        .combine_first(_as_number(history["qualifying_position_source"]))
        .combine_first(_as_number(history["qualificationPositionNumber"]))
    )
    history["grid_penalty_positions"] = _as_number(
        history["grid_penalty_positions_source"]
    ).fillna(0)
    history["has_grid_penalty"] = (
        history["grid_penalty_text"].notna()
        | history["grid_penalty_positions"].gt(0)
    ).astype("int8")
    history["pit_lane_start"] = history["grid_text"].eq("PL").astype("int8")
    history["missing_final_grid"] = history["final_grid_position"].isna().astype("int8")

    history["finish_position_history"] = _as_number(history["positionNumber"])
    history["points_history"] = _as_number(history["points"]).fillna(0)
    history["points_finish"] = history["points_history"].gt(0).astype("int8")
    history["podium_finish"] = history["finish_position_history"].le(3).astype("int8")
    history["race_winner"] = history["finish_position_history"].eq(1).astype("int8")
    history["dnf_history"] = history["positionText"].eq("DNF").astype("int8")
    history["classified_history"] = history["finish_position_history"].notna().astype("int8")
    history["positions_gained_history"] = _as_number(history["positionsGained"])
    history["grid_position_history"] = history["final_grid_position"]
    history["qualifying_position_history"] = history["qualifying_position"]

    history = history.sort_values(["event_index", "driverId"]).reset_index(drop=True)
    return history


def _add_current_pre_race_features(
    history: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    frame = history.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame["race_month"] = frame["race_date"].dt.month
    frame["is_sprint_weekend"] = (
        frame["sprint_race_date"].notna()
        | frame["qualifying_format"].eq("SPRINT_RACE")
        | frame["sprint_qualifying_format"].notna()
    ).astype("int8")

    frame["grid_field_size"] = frame.groupby("raceId")["grid_position_source"].transform(
        lambda values: values.notna().sum()
    )
    frame["effective_grid_position"] = frame["final_grid_position"].where(
        ~frame["pit_lane_start"].eq(1), frame["grid_field_size"] + 1
    )
    frame["grid_position_normalized"] = _safe_divide(
        frame["effective_grid_position"], frame["grid_field_size"]
    )
    frame["qualifying_position_normalized"] = _safe_divide(
        frame["qualifying_position"], frame["grid_field_size"]
    )
    frame["grid_delta_from_qualifying"] = (
        frame["effective_grid_position"] - frame["qualifying_position"]
    )

    for session in ("q1", "q2", "q3"):
        milliseconds = f"{session}Millis"
        frame[milliseconds] = _as_number(frame[milliseconds])
        best = frame.groupby("raceId")[milliseconds].transform("min")
        frame[f"{session}_pct_off_best"] = _safe_divide(
            frame[milliseconds] - best, best
        )
    frame["qualifying_laps"] = _as_number(frame["qualifying_laps"])
    frame["reached_q2"] = frame["q2Millis"].notna().astype("int8")
    frame["reached_q3"] = frame["q3Millis"].notna().astype("int8")

    team_group = frame.groupby(["raceId", "constructor_id"], dropna=False)
    frame["current_constructor_entry_count"] = team_group["driverId"].transform("size")
    frame["current_constructor_avg_grid"] = team_group[
        "effective_grid_position"
    ].transform("mean")
    frame["current_constructor_best_grid"] = team_group[
        "effective_grid_position"
    ].transform("min")
    frame["current_constructor_avg_qualifying"] = team_group[
        "qualifying_position"
    ].transform("mean")
    frame["grid_delta_to_constructor_avg"] = (
        frame["effective_grid_position"] - frame["current_constructor_avg_grid"]
    )
    frame["qualifying_delta_to_constructor_avg"] = (
        frame["qualifying_position"] - frame["current_constructor_avg_qualifying"]
    )

    drivers = tables["drivers"][
        ["id", "dateOfBirth", "nationalityCountryId"]
    ].rename(
        columns={
            "id": "driverId",
            "dateOfBirth": "driver_date_of_birth",
            "nationalityCountryId": "driver_nationality_country_id",
        }
    )
    circuits = tables["circuits"][["id", "countryId"]].rename(
        columns={"id": "circuit_id", "countryId": "circuit_country_id"}
    )
    frame = frame.merge(drivers, on="driverId", how="left", validate="many_to_one")
    frame = frame.merge(circuits, on="circuit_id", how="left", validate="many_to_one")
    birth_date = pd.to_datetime(frame["driver_date_of_birth"], errors="coerce")
    frame["driver_age_years"] = (frame["race_date"] - birth_date).dt.days / 365.2425
    frame["is_driver_home_race"] = (
        frame["driver_nationality_country_id"].notna()
        & frame["circuit_country_id"].notna()
        & frame["driver_nationality_country_id"].eq(frame["circuit_country_id"])
    ).astype("int8")
    return frame


def _add_driver_history_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    driver = ["driverId"]
    frame["driver_prior_entries"] = frame.groupby(driver, sort=False).cumcount()
    frame["driver_prior_classified_finishes"] = _prior_sum(
        frame, driver, "classified_history"
    )
    frame["driver_career_avg_finish"] = _prior_mean(
        frame, driver, "finish_position_history"
    )
    frame["driver_career_points_per_start"] = _prior_mean(
        frame, driver, "points_history"
    )
    for output, source in (
        ("driver_career_points_finish_rate", "points_finish"),
        ("driver_career_podium_rate", "podium_finish"),
        ("driver_career_win_rate", "race_winner"),
        ("driver_career_dnf_rate", "dnf_history"),
    ):
        frame[output] = _prior_mean(frame, driver, source)

    frame["driver_previous_finish"] = frame.groupby(driver, sort=False)[
        "finish_position_history"
    ].shift()
    frame["driver_previous_points"] = frame.groupby(driver, sort=False)[
        "points_history"
    ].shift()
    frame["driver_previous_grid"] = frame.groupby(driver, sort=False)[
        "grid_position_history"
    ].shift()
    frame["driver_previous_qualifying"] = frame.groupby(driver, sort=False)[
        "qualifying_position_history"
    ].shift()

    recent_sources = {
        "avg_finish": "finish_position_history",
        "points_per_start": "points_history",
        "points_finish_rate": "points_finish",
        "podium_rate": "podium_finish",
        "win_rate": "race_winner",
        "dnf_rate": "dnf_history",
        "avg_grid": "grid_position_history",
        "avg_qualifying": "qualifying_position_history",
    }
    for window in (3, 5, 10):
        for suffix, source in recent_sources.items():
            frame[f"driver_recent_{window}_{suffix}"] = _prior_rolling_mean(
                frame, driver, source, window
            )

    driver_season = ["year", "driverId"]
    frame["driver_season_prior_entries"] = frame.groupby(
        driver_season, sort=False
    ).cumcount()
    frame["driver_season_prior_points"] = _prior_sum(
        frame, driver_season, "points_history"
    )
    frame["driver_season_avg_finish"] = _prior_mean(
        frame, driver_season, "finish_position_history"
    )
    for output, source in (
        ("driver_season_points_finish_rate", "points_finish"),
        ("driver_season_podium_rate", "podium_finish"),
        ("driver_season_win_rate", "race_winner"),
        ("driver_season_dnf_rate", "dnf_history"),
    ):
        frame[output] = _prior_mean(frame, driver_season, source)

    for group_columns, prefix in (
        (["driverId", "circuit_id"], "driver_circuit"),
        (["driverId", "constructor_id"], "driver_constructor"),
    ):
        frame[f"{prefix}_prior_entries"] = frame.groupby(
            group_columns, sort=False, dropna=False
        ).cumcount()
        frame[f"{prefix}_avg_finish"] = _prior_mean(
            frame, group_columns, "finish_position_history"
        )
        if prefix == "driver_constructor":
            frame[f"{prefix}_points_per_start"] = _prior_mean(
                frame, group_columns, "points_history"
            )
        for suffix, source in (
            ("points_finish_rate", "points_finish"),
            ("podium_rate", "podium_finish"),
            ("win_rate", "race_winner"),
            ("dnf_rate", "dnf_history"),
        ):
            frame[f"{prefix}_{suffix}"] = _prior_mean(frame, group_columns, source)
    return frame


def _constructor_history_features(history: pd.DataFrame) -> pd.DataFrame:
    constructor_races = (
        history.groupby(
            ["raceId", "year", "round", "event_index", "constructor_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            constructor_entries=("driverId", "size"),
            constructor_points=("points_history", "sum"),
            constructor_points_finishes=("points_finish", "sum"),
            constructor_podiums=("podium_finish", "sum"),
            constructor_wins=("race_winner", "sum"),
            constructor_dnfs=("dnf_history", "sum"),
            constructor_finish_sum=("finish_position_history", "sum"),
            constructor_classified=("classified_history", "sum"),
            constructor_race_avg_finish=("finish_position_history", "mean"),
            constructor_race_avg_grid=("grid_position_history", "mean"),
            constructor_race_avg_qualifying=("qualifying_position_history", "mean"),
        )
        .sort_values(["event_index", "constructor_id"])
        .reset_index(drop=True)
    )
    group = ["constructor_id"]
    constructor_races["constructor_prior_races"] = constructor_races.groupby(
        group, sort=False, dropna=False
    ).cumcount()
    constructor_races["constructor_prior_entries"] = _prior_sum(
        constructor_races, group, "constructor_entries"
    )
    prior_races = constructor_races["constructor_prior_races"]
    prior_entries = constructor_races["constructor_prior_entries"]
    constructor_races["constructor_career_points_per_race"] = _safe_divide(
        _prior_sum(constructor_races, group, "constructor_points"), prior_races
    )
    constructor_races["constructor_career_points_finish_rate"] = _safe_divide(
        _prior_sum(constructor_races, group, "constructor_points_finishes"), prior_entries
    )
    constructor_races["constructor_career_podiums_per_race"] = _safe_divide(
        _prior_sum(constructor_races, group, "constructor_podiums"), prior_races
    )
    constructor_races["constructor_career_wins_per_race"] = _safe_divide(
        _prior_sum(constructor_races, group, "constructor_wins"), prior_races
    )
    constructor_races["constructor_career_dnf_rate"] = _safe_divide(
        _prior_sum(constructor_races, group, "constructor_dnfs"), prior_entries
    )
    constructor_races["constructor_career_avg_finish"] = _safe_divide(
        _prior_sum(constructor_races, group, "constructor_finish_sum"),
        _prior_sum(constructor_races, group, "constructor_classified"),
    )

    rolling_sources = {
        "points_per_race": "constructor_points",
        "podiums_per_race": "constructor_podiums",
        "wins_per_race": "constructor_wins",
        "dnf_rate": "constructor_dnfs",
        "avg_finish": "constructor_race_avg_finish",
        "avg_grid": "constructor_race_avg_grid",
        "avg_qualifying": "constructor_race_avg_qualifying",
    }
    constructor_races["constructor_race_dnf_rate"] = _safe_divide(
        constructor_races["constructor_dnfs"], constructor_races["constructor_entries"]
    )
    rolling_sources["dnf_rate"] = "constructor_race_dnf_rate"
    for window in (3, 5, 10):
        for suffix, source in rolling_sources.items():
            constructor_races[f"constructor_recent_{window}_{suffix}"] = (
                _prior_rolling_mean(constructor_races, group, source, window)
            )

    season_group = ["year", "constructor_id"]
    constructor_races["constructor_season_prior_races"] = constructor_races.groupby(
        season_group, sort=False, dropna=False
    ).cumcount()
    season_races = constructor_races["constructor_season_prior_races"]
    season_starts = _prior_sum(constructor_races, season_group, "constructor_entries")
    constructor_races["constructor_season_prior_points"] = _prior_sum(
        constructor_races, season_group, "constructor_points"
    )
    constructor_races["constructor_season_points_per_race"] = _safe_divide(
        constructor_races["constructor_season_prior_points"], season_races
    )
    constructor_races["constructor_season_points_finish_rate"] = _safe_divide(
        _prior_sum(constructor_races, season_group, "constructor_points_finishes"),
        season_starts,
    )
    constructor_races["constructor_season_podiums_per_race"] = _safe_divide(
        _prior_sum(constructor_races, season_group, "constructor_podiums"), season_races
    )
    constructor_races["constructor_season_wins_per_race"] = _safe_divide(
        _prior_sum(constructor_races, season_group, "constructor_wins"), season_races
    )
    constructor_races["constructor_season_dnf_rate"] = _safe_divide(
        _prior_sum(constructor_races, season_group, "constructor_dnfs"), season_starts
    )

    keep = ["raceId", "constructor_id"] + [
        column
        for column in NUMERIC_FEATURES
        if column.startswith("constructor_")
        and not column.startswith("constructor_championship")
        and not column.startswith("constructor_previous_season")
    ]
    return constructor_races[keep]


def _circuit_history_features(history: pd.DataFrame) -> pd.DataFrame:
    circuit_races = (
        history.groupby(
            ["raceId", "event_index", "circuit_id"], as_index=False, dropna=False
        )
        .agg(
            circuit_race_dnf_rate=("dnf_history", "mean"),
            circuit_race_avg_positions_gained=("positions_gained_history", "mean"),
        )
        .sort_values(["event_index", "circuit_id"])
        .reset_index(drop=True)
    )
    winners = (
        history[history["race_winner"].eq(1)]
        .groupby("raceId", as_index=False)["final_grid_position"]
        .min()
        .rename(columns={"final_grid_position": "winner_grid"})
    )
    circuit_races = circuit_races.merge(winners, on="raceId", how="left", validate="one_to_one")
    circuit_races["pole_won_race"] = circuit_races["winner_grid"].eq(1).astype("int8")

    group = ["circuit_id"]
    circuit_races["circuit_prior_races"] = circuit_races.groupby(
        group, sort=False, dropna=False
    ).cumcount()
    for output, source in (
        ("circuit_historical_dnf_rate", "circuit_race_dnf_rate"),
        (
            "circuit_historical_avg_positions_gained",
            "circuit_race_avg_positions_gained",
        ),
        ("circuit_historical_pole_win_rate", "pole_won_race"),
        ("circuit_historical_avg_winner_grid", "winner_grid"),
    ):
        circuit_races[output] = _prior_mean(circuit_races, group, source)
    keep = ["raceId", "circuit_id"] + [
        column for column in NUMERIC_FEATURES if column.startswith("circuit_historical")
    ] + ["circuit_prior_races"]
    return circuit_races[keep]


def _add_pre_race_standings(
    frame: pd.DataFrame, tables: dict[str, pd.DataFrame], config: Stage2AConfig
) -> pd.DataFrame:
    races = tables["races"]
    races = races[races["year"].le(config.end_year)].sort_values(
        ["year", "round", "date", "id"]
    )
    race_order = races[races["year"].between(config.start_year, config.end_year)][
        ["id", "year", "round"]
    ].copy()
    race_order["next_race_id"] = race_order.groupby("year")["id"].shift(-1)

    driver_standings = tables["driver_standings"].merge(
        race_order[["id", "next_race_id"]],
        left_on="raceId",
        right_on="id",
        how="inner",
        validate="many_to_one",
    )
    driver_pre = driver_standings[
        ["next_race_id", "driverId", "positionNumber", "points"]
    ].rename(
        columns={
            "next_race_id": "raceId",
            "positionNumber": "driver_championship_position_pre_race",
            "points": "driver_championship_points_pre_race",
        }
    ).dropna(subset=["raceId"])
    frame = frame.merge(driver_pre, on=["raceId", "driverId"], how="left", validate="one_to_one")

    constructor_standings = tables["constructor_standings"].merge(
        race_order[["id", "next_race_id"]],
        left_on="raceId",
        right_on="id",
        how="inner",
        validate="many_to_one",
    )
    constructor_pre = constructor_standings[
        ["next_race_id", "constructorId", "positionNumber", "points"]
    ].rename(
        columns={
            "next_race_id": "raceId",
            "constructorId": "constructor_id",
            "positionNumber": "constructor_championship_position_pre_race",
            "points": "constructor_championship_points_pre_race",
        }
    ).dropna(subset=["raceId"])
    frame = frame.merge(
        constructor_pre,
        on=["raceId", "constructor_id"],
        how="left",
        validate="many_to_one",
    )

    final_races = (
        races[races["year"].between(config.start_year - 1, config.end_year - 1)]
        .sort_values(["year", "round"])
        .groupby("year")
        .tail(1)[["id", "year"]]
        .rename(columns={"year": "previous_year"})
    )
    driver_previous = tables["driver_standings"].merge(
        final_races[["id", "previous_year"]],
        left_on="raceId",
        right_on="id",
        how="inner",
    )
    driver_previous["year"] = driver_previous["previous_year"] + 1
    driver_previous = driver_previous[
        ["year", "driverId", "positionNumber", "points"]
    ].rename(
        columns={
            "positionNumber": "driver_previous_season_position",
            "points": "driver_previous_season_points",
        }
    )
    frame = frame.merge(
        driver_previous, on=["year", "driverId"], how="left", validate="many_to_one"
    )

    constructor_previous = tables["constructor_standings"].merge(
        final_races[["id", "previous_year"]],
        left_on="raceId",
        right_on="id",
        how="inner",
    )
    constructor_previous["year"] = constructor_previous["previous_year"] + 1
    constructor_previous = constructor_previous[
        ["year", "constructorId", "positionNumber", "points"]
    ].rename(
        columns={
            "constructorId": "constructor_id",
            "positionNumber": "constructor_previous_season_position",
            "points": "constructor_previous_season_points",
        }
    )
    frame = frame.merge(
        constructor_previous,
        on=["year", "constructor_id"],
        how="left",
        validate="many_to_one",
    )

    for column in (
        "driver_championship_position_pre_race",
        "driver_championship_points_pre_race",
        "driver_previous_season_position",
        "driver_previous_season_points",
        "constructor_championship_position_pre_race",
        "constructor_championship_points_pre_race",
        "constructor_previous_season_position",
        "constructor_previous_season_points",
    ):
        frame[column] = _as_number(frame[column])
    frame["driver_championship_points_pre_race"] = frame[
        "driver_championship_points_pre_race"
    ].fillna(0)
    frame["constructor_championship_points_pre_race"] = frame[
        "constructor_championship_points_pre_race"
    ].fillna(0)
    return frame


def _assign_split(year: pd.Series, config: Stage2AConfig) -> pd.Series:
    split = pd.Series(index=year.index, dtype="string")
    split.loc[year.le(config.train_end_year)] = "train"
    split.loc[year.eq(config.validation_year)] = "validation"
    split.loc[year.ge(config.test_start_year)] = "test"
    return split


def _apply_prediction_eligibility(
    history: pd.DataFrame, audit_records: list[dict[str, Any]]
) -> pd.DataFrame:
    """Apply explicit pre-start eligibility decisions to missing-grid entries."""

    frame = history.copy()
    missing_grid = frame["_has_final_grid_record"].eq(0)
    missing_keys = set(
        zip(
            frame.loc[missing_grid, "raceId"].astype(int),
            frame.loc[missing_grid, "driverId"].astype(str),
        )
    )
    audit_by_key = {
        (int(record["race_id"]), str(record["driver_id"])): record
        for record in audit_records
    }
    unresolved = sorted(missing_keys - set(audit_by_key))
    if unresolved:
        raise ValueError(
            "Missing final-grid records require an explicit prediction-timestamp "
            f"eligibility audit; unresolved race/driver keys: {unresolved}"
        )

    audited_grid_records = sorted(set(audit_by_key) - missing_keys)
    if audited_grid_records:
        raise ValueError(
            "Eligibility audit contains records that have a final-grid entry or are "
            f"outside the configured dataset: {audited_grid_records}"
        )

    frame["prediction_eligible"] = 1
    frame["eligibility_reason"] = "present_in_final_grid_source"
    frame["eligibility_source"] = "F1DB final-grid record"
    for key, record in audit_by_key.items():
        race_id, driver_id = key
        row_mask = frame["raceId"].eq(race_id) & frame["driverId"].eq(driver_id)
        if int(row_mask.sum()) != 1:
            raise ValueError(f"Eligibility audit key does not identify one row: {key}")
        actual_status = str(frame.loc[row_mask, "positionText"].iloc[0])
        documented_status = record.get("result_status")
        if documented_status is not None and actual_status != documented_status:
            raise ValueError(
                f"Eligibility audit status mismatch for {key}: "
                f"expected {documented_status}, found {actual_status}"
            )
        frame.loc[row_mask, "prediction_eligible"] = int(
            bool(record["prediction_eligible"])
        )
        frame.loc[row_mask, "eligibility_reason"] = str(record["reason"])
        frame.loc[row_mask, "eligibility_source"] = str(record["source_url"])

    frame["prediction_eligible"] = frame["prediction_eligible"].astype("int8")
    frame["classified_finish_position"] = _as_number(frame["positionNumber"])
    frame["result_status"] = frame["positionText"].astype("string")
    display_order = _as_number(frame["positionDisplayOrder"]).where(
        frame["prediction_eligible"].eq(1)
    )
    frame["finish_order"] = display_order.groupby(frame["raceId"]).rank(
        method="first", ascending=True
    ).astype("Int64")

    eligible = frame[frame["prediction_eligible"].eq(1)]
    expected_counts = eligible.groupby("raceId").size()
    actual_max = eligible.groupby("raceId")["finish_order"].max()
    if eligible["finish_order"].isna().any() or not actual_max.eq(expected_counts).all():
        raise ValueError("Eligible finish_order values are not complete and contiguous")
    return frame


def build_stage2a_dataset(
    raw_dir: Path,
    config: Stage2AConfig | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    eligibility_audit: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build one pre-race feature row per official driver/race result."""

    config = config or Stage2AConfig()
    tables = tables or load_raw_tables(raw_dir)
    _validate_raw_tables(tables, config)

    history = _prepare_history(tables, config)
    history = _add_current_pre_race_features(history, tables)
    history = _add_driver_history_features(history)
    constructor_features = _constructor_history_features(history)
    circuit_features = _circuit_history_features(history)
    history = history.merge(
        constructor_features,
        on=["raceId", "constructor_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        circuit_features,
        on=["raceId", "circuit_id"],
        how="left",
        validate="many_to_one",
    )
    history = history[
        history["year"].between(config.start_year, config.end_year)
    ].copy()
    history = _add_pre_race_standings(history, tables, config)
    history = _apply_prediction_eligibility(
        history,
        eligibility_audit
        if eligibility_audit is not None
        else load_eligibility_audit(),
    )

    dataset = history.copy()
    dataset["split"] = _assign_split(dataset["year"], config)
    dataset = dataset.rename(columns={"raceId": "race_id", "driverId": "driver_id"})
    dataset["race_date"] = dataset["race_date"].dt.strftime("%Y-%m-%d")

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Stage 2A builder did not create columns: {missing_columns}")
    dataset = dataset[OUTPUT_COLUMNS].sort_values(
        ["year", "round", "race_id", "effective_grid_position", "driver_id"],
        na_position="last",
    ).reset_index(drop=True)

    if dataset.duplicated(["race_id", "driver_id"]).any():
        raise ValueError("Stage 2A output contains duplicate race/driver records")
    if dataset["split"].isna().any():
        raise ValueError("Stage 2A output contains rows outside the configured splits")
    if dataset[FEATURE_COLUMNS].columns.intersection(FORBIDDEN_CURRENT_RACE_FEATURES).any():
        raise ValueError("A forbidden current-race outcome field was included as a feature")
    return dataset


def _feature_lineage(name: str) -> tuple[str, str]:
    race_metadata = {
        "year",
        "round",
        "race_month",
        "course_length_km",
        "circuit_turns",
        "is_sprint_weekend",
        "grand_prix_id",
        "circuit_id",
        "circuit_layout_id",
        "circuit_type",
        "circuit_direction",
        "qualifying_format",
    }
    registry_metadata = {
        "driver_nationality_country_id",
        "circuit_country_id",
        "driver_age_years",
        "is_driver_home_race",
    }
    entry_metadata = {"driver_id", "constructor_id", "engine_manufacturer_id"}
    current_grid_or_qualifying = {
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
        "current_constructor_entry_count",
        "current_constructor_avg_grid",
        "current_constructor_best_grid",
        "current_constructor_avg_qualifying",
        "grid_delta_to_constructor_avg",
        "qualifying_delta_to_constructor_avg",
    }
    previous_standings = {
        "driver_championship_position_pre_race",
        "driver_championship_points_pre_race",
        "constructor_championship_position_pre_race",
        "constructor_championship_points_pre_race",
    }
    previous_season = {
        "driver_previous_season_position",
        "driver_previous_season_points",
        "constructor_previous_season_position",
        "constructor_previous_season_points",
    }

    if name in race_metadata:
        return "F1DB race schedule and circuit metadata", "known_before_event"
    if name in registry_metadata:
        return "F1DB driver and circuit registries", "static_or_race_date_derived"
    if name in entry_metadata:
        return (
            "F1DB final-grid or qualifying entry metadata; result metadata is identity-only fallback",
            "known_before_race_start",
        )
    if name in current_grid_or_qualifying:
        return "F1DB qualifying and final starting-grid records", "current_race_pre_start"
    if name in previous_standings:
        return "F1DB standings after the previous completed race", "strictly_prior_race"
    if name in previous_season:
        return "F1DB standings after the previous season finale", "strictly_prior_season"
    if name.startswith("driver_"):
        return "Shifted official results from earlier driver entries", "strictly_prior_races"
    if name.startswith("constructor_"):
        return "Shifted constructor race aggregates from earlier races", "strictly_prior_races"
    if name.startswith("circuit_"):
        return "Shifted circuit aggregates from earlier races", "strictly_prior_races"
    raise ValueError(f"Feature lineage is not defined for {name!r}")


def _feature_definitions() -> list[dict[str, str]]:
    manual = {
        "year": "Season of the predicted race.",
        "round": "Championship round of the predicted race.",
        "race_month": "Calendar month of the predicted race.",
        "course_length_km": "Published circuit lap length for the event.",
        "circuit_turns": "Published number of circuit turns for the event.",
        "is_sprint_weekend": "1 when the event schedule includes a sprint session.",
        "driver_age_years": "Driver age on race day.",
        "is_driver_home_race": "1 when driver nationality and circuit country match.",
        "qualifying_position": "Official qualifying rank available before the race.",
        "qualifying_position_normalized": "Qualifying rank divided by confirmed grid field size.",
        "final_grid_position": "Numeric final starting-grid slot; missing for pit-lane/non-starters.",
        "effective_grid_position": "Final grid slot with pit-lane starts placed after the grid.",
        "grid_position_normalized": "Effective grid slot divided by confirmed grid field size.",
        "grid_delta_from_qualifying": "Effective grid slot minus qualifying rank.",
        "grid_penalty_positions": "Published numeric grid penalty, with no penalty encoded as zero.",
        "has_grid_penalty": "1 when the final-grid file records a grid penalty.",
        "pit_lane_start": "1 when the confirmed start is from the pit lane.",
        "missing_final_grid": "1 when no numeric final-grid slot is available.",
        "qualifying_laps": "Laps recorded in the qualifying result.",
        "reached_q2": "1 when a Q2 time is recorded.",
        "reached_q3": "1 when a Q3 time is recorded.",
        "q1_pct_off_best": "Q1 milliseconds above the session best, divided by the best.",
        "q2_pct_off_best": "Q2 milliseconds above the session best, divided by the best.",
        "q3_pct_off_best": "Q3 milliseconds above the session best, divided by the best.",
        "grid_field_size": "Number of numeric slots in the confirmed starting grid.",
        "driver_prior_entries": "Number of earlier official result entries for the driver, including DNS/DNP records.",
        "driver_season_prior_entries": "Earlier official result entries for the driver in the current season.",
        "driver_circuit_prior_entries": "Earlier official result entries for the driver at the circuit.",
        "driver_constructor_prior_entries": "Earlier official result entries for the driver-constructor pairing.",
        "constructor_prior_entries": "Earlier official driver entries for the constructor.",
        "driver_championship_position_pre_race": "Official driver standing after the previous race in the same season.",
        "driver_championship_points_pre_race": "Official driver points after the previous race; zero before round one or a debut.",
        "constructor_championship_position_pre_race": "Official constructor standing after the previous race in the same season.",
        "constructor_championship_points_pre_race": "Official constructor points after the previous race; zero before round one.",
        "circuit_prior_races": "Number of earlier races at the circuit in F1DB history.",
        "circuit_historical_dnf_rate": "Mean DNF rate at the circuit over earlier races only.",
        "circuit_historical_avg_positions_gained": "Mean positions gained at the circuit over earlier races only.",
        "circuit_historical_pole_win_rate": "Share of earlier circuit races won from pole.",
        "circuit_historical_avg_winner_grid": "Mean winner grid slot at the circuit over earlier races.",
    }
    definitions: list[dict[str, str]] = []
    for name in FEATURE_COLUMNS:
        source, temporal_scope = _feature_lineage(name)
        if name in CATEGORICAL_FEATURES:
            description = manual.get(
                name, name.replace("_", " ").capitalize() + " known before race start."
            )
            role = "categorical_feature"
        else:
            description = manual.get(name)
            if description is None:
                description = name.replace("_", " ").capitalize() + "."
                if any(
                    token in name
                    for token in ("career", "recent", "prior", "previous", "historical", "season")
                ):
                    description = description[:-1] + ", calculated from completed earlier races only."
            role = "numeric_feature"
        definitions.append(
            {
                "name": name,
                "role": role,
                "availability": "after_final_grid_before_race",
                "source": source,
                "temporal_scope": temporal_scope,
                "definition": description,
            }
        )
    return definitions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_stage2a_outputs(
    dataset: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
    config: Stage2AConfig | None = None,
) -> dict[str, Any]:
    """Write deterministic Stage 2A datasets, definitions, and provenance."""

    config = config or Stage2AConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "race_features.csv"
    dataset.to_csv(dataset_path, index=False, float_format="%.10g", lineterminator="\n")

    split_paths: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.csv"
        dataset[dataset["split"].eq(split)].to_csv(
            path, index=False, float_format="%.10g", lineterminator="\n"
        )
        split_paths[split] = path.name

    definitions_path = output_dir / "feature_definitions.json"
    definitions_payload = {
        "prediction_timestamp": "after final grid confirmation and before race start",
        "targets": {
            "points_finish": "1 when official race-result points are greater than zero",
            "podium_finish": "1 when official numeric finishing position is 1, 2, or 3",
            "race_winner": "1 when official numeric finishing position is 1",
            "finish_order": (
                "Official display order re-ranked contiguously within drivers eligible "
                "at the prediction timestamp; lower is better and DNF/DSQ/NC remain ranked."
            ),
        },
        "outcome_metadata": {
            "classified_finish_position": (
                "Nullable official numeric classification; diagnostic only because "
                "DNF/DNS/DNP records can be unclassified."
            ),
            "result_status": "Official result status such as a position, DNF, DNS, DNP, DSQ, or NC.",
            "prediction_eligible": (
                "1 only when the driver belonged to the field at the prediction timestamp."
            ),
            "eligibility_reason": "Documented basis for the eligibility decision.",
            "eligibility_source": "F1DB grid source or external contemporaneous evidence URL.",
        },
        "eligibility_policy": (
            "Missing final-grid data is never itself evidence of ineligibility; every "
            "missing record requires an explicit timestamp-based audit decision."
        ),
        "features": _feature_definitions(),
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "forbidden_current_race_outcome_features": sorted(FORBIDDEN_CURRENT_RACE_FEATURES),
    }
    definitions_path.write_text(
        json.dumps(definitions_payload, indent=2) + "\n", encoding="utf-8"
    )

    audit_path = output_dir / "eligibility_audit.json"
    audit_payload = json.loads(
        DEFAULT_ELIGIBILITY_AUDIT_PATH.read_text(encoding="utf-8")
    )
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    source_hashes = {
        filename: _sha256(raw_dir / filename) for filename in RAW_FILES.values()
    }
    split_summary = {
        split: {
            "rows": int(part.shape[0]),
            "eligible_rows": int(part["prediction_eligible"].sum()),
            "races": int(part["race_id"].nunique()),
            "years": sorted(int(value) for value in part["year"].unique()),
        }
        for split, part in dataset.groupby("split", sort=False)
    }
    missing = {
        column: int(count)
        for column, count in dataset[FEATURE_COLUMNS].isna().sum().items()
        if count
    }
    train = dataset[dataset["split"].eq("train")]
    unseen_categories = {}
    for split in ("validation", "test"):
        part = dataset[dataset["split"].eq(split)]
        unseen = {
            column: sorted(
                set(part[column].dropna().astype(str))
                - set(train[column].dropna().astype(str))
            )
            for column in CATEGORICAL_FEATURES
        }
        unseen_categories[split] = {
            column: values for column, values in unseen.items() if values
        }
    lineage_counts: dict[str, int] = {}
    for definition in _feature_definitions():
        scope = definition["temporal_scope"]
        lineage_counts[scope] = lineage_counts.get(scope, 0) + 1
    metadata = {
        "stage": "2A",
        "prediction_timestamp": "after final grid confirmation and before race start",
        "configuration": config.__dict__,
        "runtime": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
        },
        "dataset": {
            "file": dataset_path.name,
            "sha256": _sha256(dataset_path),
            "rows": int(dataset.shape[0]),
            "columns": int(dataset.shape[1]),
            "races": int(dataset["race_id"].nunique()),
            "drivers": int(dataset["driver_id"].nunique()),
            "feature_count": len(FEATURE_COLUMNS),
            "target_count": len(TARGET_COLUMNS),
        },
        "eligibility": {
            "audit_file": audit_path.name,
            "audited_records": len(audit_payload["records"]),
            "eligible_rows": int(dataset["prediction_eligible"].sum()),
            "ineligible_rows": int(dataset["prediction_eligible"].eq(0).sum()),
            "policy": audit_payload["policy"],
        },
        "splits": split_summary,
        "split_files": split_paths,
        "missing_feature_values": missing,
        "feature_temporal_scope_counts": lineage_counts,
        "unseen_categories_relative_to_train": unseen_categories,
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "source_files_sha256": source_hashes,
        "source_normalization": [
            "Shared-car-era duplicate driver/race result rows are collapsed to one appearance using the best classified record, with credited points summed.",
            "Duplicate qualifying rows are collapsed by official display order; the final-grid table remains authoritative for the start position.",
        ],
        "known_limitations": [
            "Weather, safety-car expectations, and live market data are not present in F1DB.",
            "Some official entries have no final-grid or qualifying record; indicators and missing values are retained.",
            "Prediction eligibility for missing final-grid records depends on a manually reviewed contemporaneous evidence audit.",
            "Pit-lane starters have no numeric final-grid slot and receive an effective slot after the confirmed grid.",
            "Historical and rookie features are naturally missing when no prior observation exists.",
            "Points-finish follows awarded official race points, so it is not hard-coded to finishing position <= 10.",
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
