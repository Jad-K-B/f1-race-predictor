"""Produce an explicitly incomplete review packet, never invented participants."""
import io
import json

import pandas as pd

from .contracts import ENTRY_COLUMNS, RaceSnapshot
from .replay import clean
from .sources import SourceStore, now


def draft_snapshot(store: SourceStore, history_source: str, race_id: int, snapshot_id: str,
                   kind: str, reviewed_entries: list[dict]) -> dict:
    _, payload = store.read(history_source)
    manifest = json.loads(payload)
    def table(name):
        return pd.read_csv(io.BytesIO(store.read(manifest["tables"][name])[1]))
    r = table("races").set_index("id").loc[race_id]
    drivers = table("drivers").set_index("id")
    entries = []
    for item in reviewed_entries:
        if set(item) - ENTRY_COLUMNS:
            raise ValueError("Outcome or unknown fields in reviewed entries")
        row = {k: None for k in sorted(ENTRY_COLUMNS)}
        row.update(missing_reasons={}, eligibility_reason="", eligibility_source="")
        if item["driver_id"] in drivers.index:
            row.update(date_of_birth=clean(drivers.loc[item["driver_id"], "dateOfBirth"]),
                       nationality_country_id=clean(drivers.loc[item["driver_id"], "nationalityCountryId"]))
        row.update(item)
        entries.append(row)
    metadata = {"race_id": race_id, "year": int(r.year), "round": int(r["round"]), "race_date": r.date,
        "race_start": None, "grand_prix_id": r.grandPrixId, "circuit_id": r.circuitId,
        "circuit_layout_id": clean(r.circuitLayoutId), "circuit_type": clean(r.circuitType),
        "circuit_direction": clean(r.direction), "qualifying_format": clean(r.qualifyingFormat),
        "circuit_country_id": clean(table("circuits").set_index("id").loc[r.circuitId, "countryId"]),
        "course_length_km": clean(r.courseLength), "circuit_turns": clean(r.turns),
        "is_sprint_weekend": bool(pd.notna(r.sprintRaceDate) or r.qualifyingFormat == "SPRINT_RACE" or pd.notna(r.sprintQualifyingFormat))}
    snapshot = RaceSnapshot(snapshot_id, kind, now(), metadata, entries,
        {"schedule": manifest["tables"]["races"], "registry": manifest["tables"]["drivers"]}, history_source, "")
    return snapshot.to_dict()
