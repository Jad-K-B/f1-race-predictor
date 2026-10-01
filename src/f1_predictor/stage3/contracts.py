"""Serving contracts distinguish missing evidence from negative evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import pandas as pd

from .sources import SourceStore, utc

SnapshotKind = Literal["pre_weekend", "provisional_grid", "confirmed_grid"]
FRESHNESS_SECONDS = {"schedule": 86400, "registry": 2592000, "entries": 86400,
                     "qualifying": 172800, "grid": 3600, "eligibility": 3600}
ENTRY_COLUMNS = {"driver_id", "constructor_id", "engine_manufacturer_id", "date_of_birth",
    "nationality_country_id", "eligible", "eligibility_reason", "eligibility_source",
    "qualifying_position", "qualifying_laps", "q1_millis", "q2_millis", "q3_millis",
    "grid_position", "pit_lane_start", "grid_penalty_positions", "has_grid_penalty",
    "missing_reasons"}
RACE_COLUMNS = {"race_id", "year", "round", "race_date", "race_start", "grand_prix_id",
    "circuit_id", "circuit_layout_id", "circuit_type", "circuit_direction", "qualifying_format",
    "circuit_country_id", "course_length_km", "circuit_turns", "is_sprint_weekend"}


@dataclass(frozen=True)
class RaceSnapshot:
    snapshot_id: str
    kind: SnapshotKind
    cutoff: str
    race: dict[str, Any]
    entries: list[dict[str, Any]]
    sources: dict[str, str]
    history_source: str
    reviewed_by: str
    evidence_mode: str = "prospective"
    revision_of: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RaceSnapshot":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, store: SourceStore) -> None:
        if self.kind not in ("pre_weekend", "provisional_grid", "confirmed_grid"):
            raise ValueError("Unknown snapshot type")
        if self.evidence_mode not in ("prospective", "approximate_retrospective"):
            raise ValueError("Unknown point-in-time evidence mode")
        if set(self.race) != RACE_COLUMNS or not self.reviewed_by.strip() or not self.snapshot_id:
            raise ValueError("Complete reviewed race metadata required")
        if pd.Timestamp(self.race["race_date"]).year != self.race["year"] or not 1 <= self.race["round"] <= 50:
            raise ValueError("Inconsistent race date/year/round")
        if not isinstance(self.race["is_sprint_weekend"], bool):
            raise ValueError("Sprint designation must be explicitly known")
        if self.race["race_start"] is None and self.evidence_mode != "approximate_retrospective":
            raise ValueError("Prospective forecast requires authoritative race start time")
        if self.race["race_start"] is not None and utc(self.cutoff) >= utc(self.race["race_start"]):
            raise ValueError("Prediction cutoff must precede race start")
        if not self.entries or len(self.entries) > 40:
            raise ValueError("Invalid entrant list")
        ids = [e["driver_id"] for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate entrant")
        required = {"schedule", "registry", "entries", "eligibility"}
        if self.kind != "pre_weekend":
            required |= {"qualifying", "grid"}
        if not required <= self.sources.keys():
            raise ValueError(f"Missing source roles: {sorted(required - self.sources.keys())}")
        for role, reference in self.sources.items():
            if role not in FRESHNESS_SECONDS:
                raise ValueError("Unknown source role")
            if self.evidence_mode == "prospective":
                record = store.available(reference, self.cutoff, FRESHNESS_SECONDS[role])
                if record["published_at"] is not None and record.get("publication_basis") not in ("document", "provider_metadata"):
                    raise ValueError("Publication timestamp has no verified basis; reimport with unknown publication time")
                if role != "registry":
                    self._validate_applicability(record, role, ids, store)
                if role in ("entries", "eligibility") or (role == "grid" and self.kind == "confirmed_grid"):
                    if record["provider"] != "fia" or not record["reviewed_by"]:
                        raise ValueError(f"Authoritative reviewed FIA evidence required for {role}")
                    allowed = {"entries": {"entry_list"}, "grid": {"final_grid"},
                               "eligibility": {"final_grid", "grid_amendment", "eligibility_review"}}
                    if record.get("document_kind") not in allowed[role]:
                        raise ValueError(f"Wrong FIA document kind for {role}")
            else:
                store.read(reference)
        grid = []
        for entry in self.entries:
            if set(entry) != ENTRY_COLUMNS:
                raise ValueError("Entry fields differ from the unlabeled allowlist")
            if not isinstance(entry["eligible"], bool) or not entry["eligibility_reason"]:
                raise ValueError("Explicit eligibility decision required; missing grid is not exclusion")
            if entry["eligibility_source"] not in self.sources.values():
                raise ValueError("Eligibility evidence must be in source snapshot")
            if self.evidence_mode == "prospective" and entry["eligibility_source"] != self.sources["eligibility"]:
                raise ValueError("Every entrant requires the authoritative eligibility decision, not missing session data")
            if not entry["driver_id"] or not entry["constructor_id"]:
                raise ValueError("Canonical driver/constructor IDs required")
            if entry["date_of_birth"] is not None and pd.Timestamp(entry["date_of_birth"]) >= pd.Timestamp(self.race["race_date"]):
                raise ValueError("Invalid driver birth date")
            if self.kind == "pre_weekend":
                for key in ("qualifying_position", "qualifying_laps", "q1_millis", "q2_millis", "q3_millis",
                            "grid_position", "pit_lane_start", "grid_penalty_positions", "has_grid_penalty"):
                    if entry[key] is not None:
                        raise ValueError("Pre-weekend snapshot must not pretend session/grid availability")
                continue
            for key in ("pit_lane_start", "has_grid_penalty"):
                if not isinstance(entry[key], bool):
                    raise ValueError(f"Explicit reviewed {key} decision required")
            position = entry["grid_position"]
            if position is not None:
                if isinstance(position, bool) or int(position) != position or position < 1:
                    raise ValueError("Invalid grid position")
                grid.append(position)
            if position is not None and entry["pit_lane_start"]:
                raise ValueError("Pit-lane starter cannot also occupy a numeric grid slot")
            if entry["grid_penalty_positions"] is None or entry["grid_penalty_positions"] < 0:
                raise ValueError("Penalty details require review; unknown is not zero")
            if entry["grid_penalty_positions"] > 0 and not entry["has_grid_penalty"]:
                raise ValueError("Contradictory grid penalty flags")
            if position is None and not entry["pit_lane_start"] and entry["eligible"]:
                if not entry["missing_reasons"].get("grid_position"):
                    raise ValueError("Missing grid needs explicit reason, not automatic exclusion")
            for key in ("q1_millis", "q2_millis", "q3_millis"):
                if entry[key] is not None and entry[key] <= 0:
                    raise ValueError("Qualifying time must be positive or missing")
        if len(set(grid)) != len(grid):
            raise ValueError("Duplicate numeric grid slot")
        if self.kind == "confirmed_grid" and not grid:
            raise ValueError("No numeric grid slots: structural grid outage cannot be imputed")
        if not any(e["eligible"] for e in self.entries):
            raise ValueError("No eligible entrants")

    def _validate_applicability(self, record: dict[str, Any], role: str, ids: list[str], store: SourceStore) -> None:
        scope = record.get("applicability", {})
        if not record.get("reviewed_by") or not record.get("reviewed_at") or not record.get("review_notes"):
            raise ValueError(f"Reviewed source applicability required for {role}")
        for field in ("race_id", "year", "round", "grand_prix_id", "circuit_id"):
            if scope.get(field) != self.race[field]:
                raise ValueError(f"Source race applicability mismatch for {role}: {field}")
        session = {"schedule": "event", "entries": "event", "qualifying": "qualifying", "grid": "race", "eligibility": "race"}[role]
        if scope.get("session") != session:
            raise ValueError(f"Source session applicability mismatch for {role}")
        if role == "schedule" and scope.get("race_start") != self.race["race_start"]:
            raise ValueError("Reviewed schedule must confirm the exact race start")
        if role in ("entries", "eligibility") and set(scope.get("entrant_ids", [])) != set(ids):
            raise ValueError(f"Source review must cover the complete entrant list for {role}")
        # Older penalties can apply to this event; preserve them without pretending they are new.
        for reference in record.get("supporting_observations", []):
            evidence = store.available(reference, self.cutoff, 10**12)
            if evidence["published_at"] is not None and evidence.get("publication_basis") not in ("document", "provider_metadata"):
                raise ValueError("Supporting evidence has an unverified publication timestamp")


@dataclass(frozen=True)
class FeatureBatch:
    frame: pd.DataFrame
    snapshot: RaceSnapshot
    missingness: list[dict[str, Any]]
    history_race_ids: tuple[int, ...]


class EarlyForecastModel(Protocol):
    def predict(self, snapshot: RaceSnapshot) -> dict[str, Any]: ...


@dataclass(frozen=True)
class WeatherContext:
    provider: str
    issued_at: str
    valid_at: str
    latitude: float
    longitude: float
    air_temperature_c: float | None = None
    track_temperature_c: float | None = None
    rain_probability: float | None = None
    precipitation_mm: float | None = None
    wind_speed_mps: float | None = None
    wind_direction_degrees: float | None = None
    track_conditions: str | None = None


class RaceSimulator(Protocol):
    def simulate(self, predictions: dict[str, Any], seed: int, runs: int) -> dict[str, Any]: ...


class PredictionExplainer(Protocol):
    def explain(self, batch: FeatureBatch, predictions: dict[str, Any]) -> dict[str, Any]: ...
