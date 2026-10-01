"""Prospective early features from a reviewed snapshot and as-of source bytes.

This builder does not authorize model inference or publication. Research packets
cannot be passed here, and the current-race result is never an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import json

import numpy as np
import pandas as pd

from ..stage2a import CATEGORICAL_FEATURES, RAW_FILES, Stage2AConfig, _prepare_history
from ..stage3.features import _circuit_features, _constructor_features, _driver_features, _standings
from ..stage3.sources import SourceStore, utc
from .contracts import EarlySnapshot
from .policy import CANDIDATE_FEATURES

LIVE_FEATURE_SCHEMA = "early-prospective-features-v1"


@dataclass(frozen=True)
class LiveFeatureBatch:
    frame: pd.DataFrame
    missingness: list[dict[str, str]]
    historical_race_ids: tuple[int, ...]
    snapshot_id: str
    cutoff: str
    schema_version: str = LIVE_FEATURE_SCHEMA
    evidence_mode: str = "prospective"


def _source_tables(snapshot: EarlySnapshot, store: SourceStore) -> tuple[dict[str, pd.DataFrame], set[int]]:
    manifests = {}
    for role in ("history", "registry"):
        record = store.available(snapshot.sources[role], snapshot.cutoff, 2592000)
        if record["provider"] != "f1db":
            raise ValueError("Live history and registry require versioned F1DB observations")
        manifests[role] = json.loads(store.read(snapshot.sources[role])[1])
        manifest = manifests[role]
        if not manifest.get("version") or set(manifest.get("tables", {})) != set(RAW_FILES):
            raise ValueError("Complete versioned F1DB manifest required")
    history, registry = manifests["history"], manifests["registry"]
    if (history["version"] != registry["version"] or
            any(history["tables"][name] != registry["tables"][name]
                for name in ("drivers", "circuits", "races"))):
        raise ValueError("History and registry versions or identity tables differ")
    result_ids = history.get("result_race_ids")
    if (not isinstance(result_ids, list) or
            any(type(value) is not int or value <= 0 for value in result_ids) or
            len(set(result_ids)) != len(result_ids) or
            snapshot.race["race_id"] in result_ids):
        raise ValueError("Invalid completed-race ledger or current-race result in history")
    tables = {}
    for name in RAW_FILES:
        reference = (registry if name in ("drivers", "circuits", "races") else history)["tables"][name]
        store.available(reference, snapshot.cutoff, 2592000)
        _, payload = store.read(reference)
        tables[name] = pd.read_csv(io.BytesIO(payload), low_memory=False)
    if set(tables["results"].raceId.astype(int)) != set(result_ids):
        raise ValueError("Completed-race ledger differs from result table")
    return tables, set(result_ids)


def _completed_tables(snapshot: EarlySnapshot, tables: dict[str, pd.DataFrame],
                      result_ids: set[int]) -> dict[str, pd.DataFrame]:
    races = tables["races"]
    if races.id.duplicated().any():
        raise ValueError("Duplicate race identity in historical source")
    cutoff_date = utc(snapshot.cutoff).date()
    dates = pd.to_datetime(races.date, errors="raise").dt.date
    earlier = races[dates < cutoff_date]
    same_day = races[dates == cutoff_date]
    if set(same_day.id) & result_ids:
        raise ValueError("Same-day results need reviewed completion-time evidence")
    previous = earlier[earlier.year.eq(snapshot.race["year"])]
    missing = set(previous.id) - result_ids
    if missing:
        raise ValueError(f"Earlier scheduled races lack completed results: {sorted(missing)}")
    selected = earlier[earlier.id.isin(result_ids)].copy()
    if selected.empty:
        raise ValueError("No completed historical races before cutoff")
    ids = set(selected.id)
    if ids - set(tables["results"].raceId):
        raise ValueError("Completed-race ledger lacks classifications")
    prior = {**tables, "races": selected}
    for name in ("results", "qualifying", "grid", "driver_standings", "constructor_standings"):
        prior[name] = tables[name][tables[name].raceId.isin(ids)].copy()
    for name in ("drivers", "circuits"):
        if prior[name].id.duplicated().any():
            raise ValueError(f"Duplicate {name} identity")
    for name, identity in (("driver_standings", "driverId"),
                           ("constructor_standings", "constructorId")):
        relevant = selected[selected.year.isin([snapshot.race["year"] - 1, snapshot.race["year"]])]
        prior[name] = prior[name][prior[name].raceId.isin(relevant.id)].copy()
        if prior[name].duplicated(["raceId", identity]).any():
            raise ValueError(f"Duplicate {name} snapshot")
        for _, season in relevant.groupby("year"):
            latest = season.sort_values(["date", "round", "id"]).iloc[-1].id
            if not prior[name].raceId.eq(latest).any():
                raise ValueError(f"Latest completed season race lacks {name}")
    return prior


def _build_from_tables(snapshot: EarlySnapshot, tables: dict[str, pd.DataFrame],
                       result_ids: set[int]) -> LiveFeatureBatch:
    """Pure feature calculation; the public entrypoint enforces source validation."""
    current = tables["races"][tables["races"].id.eq(snapshot.race["race_id"])]
    if len(current) != 1:
        raise ValueError("Exactly one matching upcoming race configuration required")
    raw = current.iloc[0]
    race = snapshot.race
    if (int(raw.year) != race["year"] or int(raw["round"]) != race["round"] or
            raw.grandPrixId != race["grand_prix_id"] or raw.circuitId != race["circuit_id"]):
        raise ValueError("F1DB event identity differs from reviewed schedule")
    local_race_date = datetime.fromisoformat(race["race_start"].replace("Z", "+00:00")).date()
    race_date = pd.Timestamp(raw.date)
    if race_date.date() != local_race_date:
        raise ValueError("F1DB race date differs from reviewed local schedule date")
    prior = _completed_tables(snapshot, tables, result_ids)
    history = _prepare_history(prior, Stage2AConfig(end_year=max(2025, race["year"])))
    history = history.sort_values(["race_date", "round", "raceId", "driverId"]).reset_index(drop=True)
    history["event_index"] = history.raceId.map(
        {race_id: index for index, race_id in enumerate(history.raceId.drop_duplicates())}
    )
    config = {
        "year": race["year"], "round": race["round"], "race_month": race_date.month,
        "grand_prix_id": raw.grandPrixId, "circuit_id": raw.circuitId,
        "circuit_layout_id": raw.circuitLayoutId, "circuit_type": raw.circuitType,
        "circuit_direction": raw.direction, "qualifying_format": raw.qualifyingFormat,
        "course_length_km": raw.courseLength, "circuit_turns": raw.turns,
        "is_sprint_weekend": int(pd.notna(raw.sprintRaceDate) or
                                 raw.qualifyingFormat == "SPRINT_RACE" or
                                 pd.notna(raw.sprintQualifyingFormat)),
    }
    circuit = tables["circuits"][tables["circuits"].id.eq(raw.circuitId)]
    circuit_country = circuit.iloc[0].countryId if len(circuit) else np.nan
    circuit_history = _circuit_features(history, raw.circuitId)
    announced = [entry for entry in snapshot.entries if entry["status"] == "announced"]
    if not announced:
        raise ValueError("No announced entrants remain at cutoff")
    counts = pd.Series([entry["constructor_id"] for entry in announced]).value_counts()
    rows, missingness = [], []
    for entry in sorted(announced, key=lambda item: item["driver_id"]):
        drivers = tables["drivers"][tables["drivers"].id.eq(entry["driver_id"])]
        driver = drivers.iloc[0] if len(drivers) else None
        nationality = driver.nationalityCountryId if driver is not None else np.nan
        birth = pd.to_datetime(driver.dateOfBirth, errors="raise") if driver is not None else pd.NaT
        if pd.notna(birth) and birth >= race_date:
            raise ValueError("Invalid driver birth date")
        row = {
            **config, "race_id": race["race_id"], "race_date": race_date.date().isoformat(),
            "cutoff": snapshot.cutoff, "driver_id": entry["driver_id"],
            "constructor_id": entry["constructor_id"],
            "engine_manufacturer_id": entry["engine_manufacturer_id"],
            "driver_nationality_country_id": nationality,
            "circuit_country_id": circuit_country,
            "driver_age_years": (race_date - birth).days / 365.2425 if pd.notna(birth) else np.nan,
            "is_driver_home_race": int(pd.notna(nationality) and pd.notna(circuit_country)
                                       and nationality == circuit_country),
            "current_constructor_entry_count": int(counts[entry["constructor_id"]]),
        }
        row.update(_driver_features(history, entry, config))
        row.update(_constructor_features(history, entry, config))
        row.update(circuit_history)
        row.update(_standings(prior, entry, config))
        missing = set(CANDIDATE_FEATURES) - row.keys()
        if missing:
            raise ValueError(f"Early feature implementation incomplete: {sorted(missing)}")
        rows.append(row)
        for name in CANDIDATE_FEATURES:
            if pd.isna(row[name]):
                missingness.append({"driver_id": entry["driver_id"], "feature": name,
                                    "reason": "missing_registry_or_undefined_completed_history"})
    frame = pd.DataFrame(rows)[["race_id", "race_date", "cutoff", *CANDIDATE_FEATURES]]
    numeric = [name for name in CANDIDATE_FEATURES if name not in CATEGORICAL_FEATURES]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise").map(lambda value: float(f"{value:.10g}"))
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].astype(pd.StringDtype(na_value=np.nan))
    if np.isinf(frame[numeric].to_numpy(float)).any():
        raise ValueError("Infinite early feature")
    return LiveFeatureBatch(frame, missingness,
                            tuple(int(value) for value in history.raceId.drop_duplicates()),
                            snapshot.snapshot_id, snapshot.cutoff)


def _validate_current_configuration(snapshot: EarlySnapshot, tables: dict[str, pd.DataFrame],
                                    store: SourceStore) -> None:
    rows = tables["races"][tables["races"].id.eq(snapshot.race["race_id"])]
    if len(rows) != 1:
        raise ValueError("Exactly one upcoming race configuration required")
    raw = rows.iloc[0]
    sprint = bool(pd.notna(raw.sprintRaceDate) or raw.qualifyingFormat == "SPRINT_RACE"
                  or pd.notna(raw.sprintQualifyingFormat))
    schedule = store.read(snapshot.sources["schedule"])[0]["applicability"]
    if (schedule.get("qualifying_format") != raw.qualifyingFormat or
            schedule.get("is_sprint_weekend") is not sprint):
        raise ValueError("Reviewed schedule must confirm current qualifying and sprint format")
    roster = store.read(snapshot.sources["roster"])[0]["applicability"]
    expected_engines = {
        entry["driver_id"]: entry["engine_manufacturer_id"] for entry in snapshot.entries
    }
    if roster.get("engine_manufacturer_by_driver") != expected_engines:
        raise ValueError("Reviewed roster must bind each entrant's engine identity or missing value")


def build_live_features(snapshot: EarlySnapshot, store: SourceStore,
                        *, approved_team_hosts: frozenset[str] = frozenset()) -> LiveFeatureBatch:
    snapshot.validate(store, approved_team_hosts=approved_team_hosts)
    tables, result_ids = _source_tables(snapshot, store)
    _validate_current_configuration(snapshot, tables, store)
    return _build_from_tables(snapshot, tables, result_ids)
