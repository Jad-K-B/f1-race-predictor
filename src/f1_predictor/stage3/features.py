"""Unlabeled, entry-driven features. Only completed history contains outcomes."""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd

from ..stage2a import (FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                      Stage2AConfig, _prepare_history, _add_current_pre_race_features)
from .contracts import RaceSnapshot, FeatureBatch
from .sources import SourceStore, utc

DRIVER_RATES = {"avg_finish": "finish_position_history", "points_per_start": "points_history",
    "points_finish_rate": "points_finish", "podium_rate": "podium_finish", "win_rate": "race_winner",
    "dnf_rate": "dnf_history", "avg_grid": "grid_position_history", "avg_qualifying": "qualifying_position_history"}


def historical_tables(store: SourceStore, snapshot: RaceSnapshot) -> dict[str, pd.DataFrame]:
    record, payload = store.read(snapshot.history_source)
    if record["provider"] != "f1db":
        raise ValueError("Frozen historical compatibility needs a versioned F1DB import")
    manifest = json.loads(payload)
    if snapshot.evidence_mode == "prospective":
        store.available(snapshot.history_source, snapshot.cutoff, 2592000)
    tables = {}
    for name, reference in manifest["tables"].items():
        if snapshot.evidence_mode == "prospective":
            store.available(reference, snapshot.cutoff, 2592000)
        _, blob = store.read(reference)
        tables[name] = pd.read_csv(io.BytesIO(blob), low_memory=False)
    races = tables["races"]
    same_day = pd.to_datetime(races["date"]).dt.date.eq(utc(snapshot.cutoff).date()) & races["id"].isin(manifest["result_race_ids"]) & races["id"].ne(snapshot.race["race_id"])
    if snapshot.evidence_mode == "prospective" and same_day.any():
        raise ValueError("Same-day historical classification requires reviewed completion-time ledger; refresh after UTC day boundary")
    # Retrospective dates are a conservative approximation, never publication proof.
    before = pd.to_datetime(races["date"]).dt.date < utc(snapshot.cutoff).date()
    selected = races[before & races["id"].isin(manifest["result_race_ids"]) & races["id"].ne(snapshot.race["race_id"])].copy()
    if selected.empty:
        raise ValueError("No completed historical races supplied")
    tables["races"] = selected
    ids = set(selected["id"])
    for name in ("results", "qualifying", "grid", "driver_standings", "constructor_standings"):
        tables[name] = tables[name][tables[name]["raceId"].isin(ids)].copy()
    if snapshot.evidence_mode == "prospective":
        season = selected[selected.year.eq(snapshot.race["year"])].sort_values(["date", "round"])
        if len(season):
            last_id = season.iloc[-1]["id"]
            for name in ("driver_standings", "constructor_standings"):
                if not tables[name]["raceId"].eq(last_id).any():
                    raise ValueError(f"Latest completed race {name} unavailable; cannot substitute zero standings")
    # A missing immediately preceding classification must not silently become older form.
    previous_scheduled = races[(races["year"] == snapshot.race["year"]) & before]
    absent = set(previous_scheduled["id"]) - ids
    if absent:
        raise ValueError(f"History lacks earlier scheduled races {sorted(absent)}; review cancellations or update source")
    return tables


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else float("nan")


def _ratio(total: float, count: float) -> float:
    return float(total / count) if count else float("nan")


def _driver_features(history: pd.DataFrame, entry: dict, race: dict) -> dict[str, Any]:
    driver = history[history["driverId"].eq(entry["driver_id"])]
    out: dict[str, Any] = {"driver_prior_entries": len(driver),
                         "driver_prior_classified_finishes": int(driver["classified_history"].sum())}
    groups = [("driver_career", driver), ("driver_season", driver[driver["year"].eq(race["year"])]),
              ("driver_circuit", driver[driver["circuit_id"].eq(race["circuit_id"])]),
              ("driver_constructor", driver[driver["constructor_id"].eq(entry["constructor_id"])])]
    for prefix, rows in groups:
        if prefix != "driver_career":
            out[f"{prefix}_prior_entries"] = len(rows)
        for suffix, column in DRIVER_RATES.items():
            name = f"{prefix}_{suffix}"
            if name in FEATURE_COLUMNS:
                out[name] = _mean(rows, column)
        if prefix == "driver_season":
            out["driver_season_prior_points"] = float(rows["points_history"].sum())
    for suffix, column in {"finish": "finish_position_history", "points": "points_history",
                           "grid": "grid_position_history", "qualifying": "qualifying_position_history"}.items():
        out[f"driver_previous_{suffix}"] = float(driver.iloc[-1][column]) if len(driver) else np.nan
    for window in (3, 5, 10):
        for suffix, column in DRIVER_RATES.items():
            out[f"driver_recent_{window}_{suffix}"] = _mean(driver.tail(window), column)
    return out


def _constructor_features(history: pd.DataFrame, entry: dict, race: dict) -> dict[str, Any]:
    rows = history[history["constructor_id"].eq(entry["constructor_id"])].copy()
    grouped = rows.groupby(["raceId", "year", "event_index"], as_index=False).agg(
        entries=("driverId", "size"), points=("points_history", "sum"), points_finishes=("points_finish", "sum"),
        podiums=("podium_finish", "sum"), wins=("race_winner", "sum"), dnfs=("dnf_history", "sum"),
        finish_sum=("finish_position_history", "sum"), classified=("classified_history", "sum"),
        avg_finish=("finish_position_history", "mean"), avg_grid=("grid_position_history", "mean"),
        avg_qualifying=("qualifying_position_history", "mean")).sort_values("event_index")
    grouped["dnf_rate"] = grouped["dnfs"] / grouped["entries"]
    out = {"constructor_prior_races": len(grouped), "constructor_prior_entries": len(rows)}
    for prefix, subset in (("constructor_career", grouped), ("constructor_season", grouped[grouped["year"].eq(race["year"])])):
        count, entries = len(subset), float(subset["entries"].sum())
        out.update({f"{prefix}_points_per_race": _ratio(subset["points"].sum(), count),
                    f"{prefix}_points_finish_rate": _ratio(subset["points_finishes"].sum(), entries),
                    f"{prefix}_podiums_per_race": _ratio(subset["podiums"].sum(), count),
                    f"{prefix}_wins_per_race": _ratio(subset["wins"].sum(), count),
                    f"{prefix}_dnf_rate": _ratio(subset["dnfs"].sum(), entries)})
        if prefix == "constructor_career":
            out[f"{prefix}_avg_finish"] = _ratio(subset["finish_sum"].sum(), subset["classified"].sum())
        else:
            out[f"{prefix}_prior_races"] = count
            out[f"{prefix}_prior_points"] = float(subset["points"].sum())
    mapping = {"points_per_race": "points", "podiums_per_race": "podiums", "wins_per_race": "wins",
               "dnf_rate": "dnf_rate", "avg_finish": "avg_finish", "avg_grid": "avg_grid", "avg_qualifying": "avg_qualifying"}
    for window in (3, 5, 10):
        for suffix, column in mapping.items():
            out[f"constructor_recent_{window}_{suffix}"] = _mean(grouped.tail(window), column)
    return out


def _circuit_features(history: pd.DataFrame, circuit: str) -> dict[str, Any]:
    rows = history[history["circuit_id"].eq(circuit)]
    grouped = rows.groupby("raceId").agg(dnf_rate=("dnf_history", "mean"), gained=("positions_gained_history", "mean"))
    winners = rows[rows["race_winner"].eq(1)].groupby("raceId")["final_grid_position"].min()
    grouped["winner_grid"] = winners
    grouped["pole_win"] = grouped["winner_grid"].eq(1).astype(float)
    return {"circuit_prior_races": len(grouped), "circuit_historical_dnf_rate": _mean(grouped, "dnf_rate"),
            "circuit_historical_avg_positions_gained": _mean(grouped, "gained"),
            "circuit_historical_avg_winner_grid": _mean(grouped, "winner_grid"),
            "circuit_historical_pole_win_rate": _mean(grouped, "pole_win")}


def _standings(tables: dict, entry: dict, race: dict) -> dict:
    races = tables["races"].sort_values(["date", "round", "id"])
    out = {}
    for prefix, identity, table in (("driver", entry["driver_id"], "driver_standings"),
                                    ("constructor", entry["constructor_id"], "constructor_standings")):
        for previous in (False, True):
            season = races[races["year"].eq(race["year"] - int(previous))]
            subset = tables[table].iloc[:0]
            if len(season):
                subset = tables[table][tables[table]["raceId"].eq(season.iloc[-1]["id"]) & tables[table][f"{prefix}Id"].eq(identity)]
            if len(subset) > 1:
                raise ValueError("Duplicate standings join")
            pos = float(subset.iloc[0]["positionNumber"]) if len(subset) else np.nan
            points = float(subset.iloc[0]["points"]) if len(subset) else (np.nan if previous else 0.0)
            if previous:
                out[f"{prefix}_previous_season_position"] = pos
                out[f"{prefix}_previous_season_points"] = points
            else:
                out[f"{prefix}_championship_position_pre_race"] = pos
                out[f"{prefix}_championship_points_pre_race"] = points
    return out


def build_features(snapshot: RaceSnapshot, store: SourceStore) -> FeatureBatch:
    snapshot.validate(store)
    if snapshot.kind != "confirmed_grid":
        raise ValueError("Frozen models require confirmed-grid features; early-model integration is separate")
    tables = historical_tables(store, snapshot)
    # The legacy config validates train/test years; input tables are already cut
    # to completed history and the legacy routine only consumes its end_year.
    history = _prepare_history(tables, Stage2AConfig(end_year=max(2025, snapshot.race["year"])))
    history = history.sort_values(["race_date", "round", "raceId", "driverId"]).reset_index(drop=True)
    race_order = {r: i for i, r in enumerate(history["raceId"].drop_duplicates())}
    history["event_index"] = history["raceId"].map(race_order)
    race = snapshot.race
    current = []
    for entry in snapshot.entries:
        row = {k: race[k] for k in ("year", "round", "race_date", "grand_prix_id", "circuit_id", "circuit_layout_id",
                 "circuit_type", "circuit_direction", "qualifying_format", "course_length_km", "circuit_turns")}
        row.update({"raceId": race["race_id"], "driverId": entry["driver_id"], "constructor_id": entry["constructor_id"],
            "engine_manufacturer_id": entry["engine_manufacturer_id"], "sprint_race_date": None,
            "sprint_qualifying_format": None,
            "grid_position_source": entry["grid_position"], "final_grid_position": entry["grid_position"],
            "pit_lane_start": int(entry["pit_lane_start"]), "qualifying_position": entry["qualifying_position"],
            "qualifying_laps": entry["qualifying_laps"], "q1Millis": entry["q1_millis"], "q2Millis": entry["q2_millis"],
            "q3Millis": entry["q3_millis"], "grid_penalty_positions": entry["grid_penalty_positions"],
            "has_grid_penalty": int(entry["has_grid_penalty"]), "missing_final_grid": int(entry["grid_position"] is None)})
        current.append(row)
    registries = {"drivers": pd.DataFrame([{"id": e["driver_id"], "dateOfBirth": e["date_of_birth"],
                     "nationalityCountryId": e["nationality_country_id"]} for e in snapshot.entries]),
                  "circuits": pd.DataFrame([{"id": race["circuit_id"], "countryId": race["circuit_country_id"]}])}
    # Compute teammate/session context over the reviewed event participant list,
    # before excluding withdrawals, matching the historical training convention.
    frame = _add_current_pre_race_features(pd.DataFrame(current), registries)
    frame["is_sprint_weekend"] = int(race["is_sprint_weekend"])
    circuit = _circuit_features(history, race["circuit_id"])
    output = []
    missingness = []
    for index, entry in enumerate(snapshot.entries):
        if not entry["eligible"]:
            continue
        row = frame.iloc[index].to_dict()
        row.update(_driver_features(history, entry, race))
        row.update(_constructor_features(history, entry, race))
        row.update(circuit)
        row.update(_standings(tables, entry, race))
        row.update(driver_id=entry["driver_id"], race_id=race["race_id"], race_date=race["race_date"])
        missing = set(FEATURE_COLUMNS) - row.keys()
        if missing:
            raise ValueError(f"Feature implementation incomplete: {sorted(missing)}")
        output.append({k: row[k] for k in ["race_id", "race_date", *FEATURE_COLUMNS]})
        for key in FEATURE_COLUMNS:
            if pd.isna(row[key]):
                reason = entry["missing_reasons"].get(key)
                if reason is None:
                    reason = ("undefined_or_missing_completed_history" if key.startswith(("driver_", "constructor_", "circuit_historical"))
                              else "missing_pre_race_measurement_or_not_applicable")
                missingness.append({"driver_id": entry["driver_id"], "feature": key, "reason": reason})
    result = pd.DataFrame(output)
    for column in NUMERIC_FEATURES:
        # Match Stage 2A's %.10g CSV precision, which the frozen models consumed.
        result[column] = pd.to_numeric(result[column], errors="raise").map(lambda value: float(f"{value:.10g}"))
    for column in CATEGORICAL_FEATURES:
        # pandas 3 read_csv (used in training) uses NaN-backed strings, not pd.NA.
        result[column] = result[column].astype(pd.StringDtype(na_value=np.nan))
    # Match Stage 2A's tie ordering; do not let caller row ordering affect ranks.
    result = result.sort_values(["effective_grid_position", "driver_id"], na_position="last", kind="mergesort").reset_index(drop=True)
    if np.isinf(result[NUMERIC_FEATURES].to_numpy(float)).any():
        raise ValueError("Non-finite engineered feature")
    return FeatureBatch(result, snapshot, missingness, tuple(int(i) for i in history["raceId"].drop_duplicates()))
