"""Approximate historical reconstruction from pre-race tables, not outcome rosters."""
from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from .contracts import ENTRY_COLUMNS, RaceSnapshot
from .sources import SourceStore


def clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def replay_snapshot(store: SourceStore, history_source: str, race_id: int,
                    eligibility_audit: list[dict]) -> RaceSnapshot:
    """No current race-result table is read to construct entries or inputs."""
    _, raw = store.read(history_source)
    manifest = json.loads(raw)
    tables = {}
    for name in ("races", "grid", "qualifying", "drivers", "circuits"):
        _, blob = store.read(manifest["tables"][name])
        tables[name] = pd.read_csv(io.BytesIO(blob), low_memory=False)
    race = tables["races"].set_index("id").loc[race_id]
    grid = tables["grid"][tables["grid"]["raceId"].eq(race_id)].set_index("driverId")
    qualifying = tables["qualifying"][tables["qualifying"]["raceId"].eq(race_id)].sort_values("positionDisplayOrder").drop_duplicates("driverId").set_index("driverId")
    drivers = tables["drivers"].set_index("id")
    audits = {a["driver_id"]: a for a in eligibility_audit if a["race_id"] == race_id}
    entrants = sorted(set(grid.index) | set(qualifying.index) | set(audits))
    entries = []
    for driver in entrants:
        g = grid.loc[driver] if driver in grid.index else pd.Series(dtype=object)
        q = qualifying.loc[driver] if driver in qualifying.index else pd.Series(dtype=object)
        identity = g if len(g) else q
        if identity.empty:
            raise ValueError(f"No independent pre-race identity for {driver}; reviewed entry evidence required")
        entry = dict.fromkeys(ENTRY_COLUMNS)
        entry.update(driver_id=driver, constructor_id=clean(identity.get("constructorId")),
            engine_manufacturer_id=clean(identity.get("engineManufacturerId")),
            date_of_birth=clean(drivers.loc[driver]["dateOfBirth"]),
            nationality_country_id=clean(drivers.loc[driver]["nationalityCountryId"]),
            eligible=driver not in audits or bool(audits[driver]["prediction_eligible"]),
            eligibility_reason="Approximate retrospective grid/qualifying roster plus frozen eligibility audit",
            eligibility_source=manifest["tables"]["grid"],
            qualifying_position=clean(g.get("qualificationPositionNumber")),
            qualifying_laps=clean(q.get("laps")), q1_millis=clean(q.get("q1Millis")),
            q2_millis=clean(q.get("q2Millis")), q3_millis=clean(q.get("q3Millis")),
            grid_position=clean(g.get("positionNumber")), pit_lane_start=g.get("positionText") == "PL",
            grid_penalty_positions=clean(g.get("gridPenaltyPositions")) or 0,
            has_grid_penalty=pd.notna(g.get("gridPenalty")) or (clean(g.get("gridPenaltyPositions")) or 0) > 0,
            missing_reasons={"grid_position": "Retrospective source has no numeric slot; eligibility separately audited"})
        if entry["qualifying_position"] is None:
            entry["qualifying_position"] = clean(q.get("positionNumber"))
        entries.append(entry)
    date = str(race["date"])
    metadata = {"race_id": race_id, "year": int(race["year"]), "round": int(race["round"]), "race_date": date,
        "race_start": None, "grand_prix_id": race["grandPrixId"], "circuit_id": race["circuitId"],
        "circuit_layout_id": clean(race["circuitLayoutId"]), "circuit_type": clean(race["circuitType"]),
        "circuit_direction": clean(race["direction"]), "qualifying_format": clean(race["qualifyingFormat"]),
        "circuit_country_id": tables["circuits"].set_index("id").loc[race["circuitId"]]["countryId"],
        "course_length_km": clean(race["courseLength"]), "circuit_turns": clean(race["turns"]),
        "is_sprint_weekend": pd.notna(race["sprintRaceDate"]) or race["qualifyingFormat"] == "SPRINT_RACE" or pd.notna(race["sprintQualifyingFormat"])}
    return RaceSnapshot(f"replay-{race_id}", "confirmed_grid", date + "T00:00:00Z", metadata, entries,
        {"schedule": manifest["tables"]["races"], "registry": manifest["tables"]["drivers"],
         "entries": manifest["tables"]["qualifying"], "eligibility": manifest["tables"]["grid"],
         "qualifying": manifest["tables"]["qualifying"], "grid": manifest["tables"]["grid"]},
        history_source, "automated approximate replay, not contemporary evidence", "approximate_retrospective")
