"""Entry-driven retrospective features. Not a production inference entrypoint.

Source versions are retained but their original historical publication vintages
are not proven. Live forecasts must use EarlySnapshot's stricter evidence gate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..stage2a import RAW_FILES, CATEGORICAL_FEATURES, Stage2AConfig, _prepare_history
from ..stage3.features import _driver_features, _constructor_features, _circuit_features, _standings
from ..stage3.sources import SourceStore, utc
from .contracts import _official, _scope
from .policy import CANDIDATE_FEATURES, scheduled_cutoff

RESEARCH_SCHEMA = "early-retrospective-features-v1"
ENTRY_FIELDS = {"driver_id", "constructor_id", "engine_manufacturer_id", "evidence"}
LIMITATIONS = (
    "Historical schedule, registry and result revisions are retrospective, not verified publication vintages.",
    "Roster review is an explicit research attestation, not prospective byte-level proof.",
)


@dataclass(frozen=True)
class ResearchRequest:
    race_id: int
    fp1_start: str
    cutoff: str
    entries: tuple[dict[str, Any], ...]
    reviewed_by: str
    review_notes: str
    roster_complete: bool
    unresolved_conflicts: tuple[str, ...]
    # References map review IDs to captured source SHA-256 digests. Raw sources
    # stay in the private evidence store; callers must preserve the review file.
    source_hashes: dict[str, str]
    evidence_mode: str = "retrospective_research"

    def validate(self) -> None:
        if self.evidence_mode != "retrospective_research":
            raise ValueError("Research inputs cannot authorize prospective forecasts")
        if type(self.race_id) is not int or self.race_id <= 0:
            raise ValueError("Invalid race ID")
        if utc(self.cutoff) != utc(scheduled_cutoff(self.fp1_start)):
            raise ValueError("Cutoff must equal scheduled FP1 minus 24 hours")
        if not self.reviewed_by.strip() or not self.review_notes.strip():
            raise ValueError("Explicit roster reviewer and limitations required")
        if self.roster_complete is not True or self.unresolved_conflicts:
            raise ValueError("Incomplete or conflicting early roster")
        if not 1 <= len(self.entries) <= 40:
            raise ValueError("Invalid reviewed field size")
        if not self.source_hashes or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for k, v in self.source_hashes.items()
        ):
            raise ValueError("Captured source hashes required")
        drivers = []
        for entry in self.entries:
            if set(entry) != ENTRY_FIELDS:
                raise ValueError("Only reviewed identities and evidence belong in early entries")
            for key in ("driver_id", "constructor_id"):
                if not isinstance(entry[key], str) or not entry[key].strip():
                    raise ValueError("Driver and constructor identities required")
            if entry["engine_manufacturer_id"] is not None and not isinstance(entry["engine_manufacturer_id"], str):
                raise ValueError("Engine identity must be a string or unknown")
            if not isinstance(entry["evidence"], list) or not entry["evidence"]:
                raise ValueError("Each entrant requires reviewed source references")
            if any(ref not in self.source_hashes for ref in entry["evidence"]):
                raise ValueError("Unknown entrant source reference")
            drivers.append(entry["driver_id"])
        if len(drivers) != len(set(drivers)):
            raise ValueError("Duplicate reviewed driver")


@dataclass
class ResearchBatch:
    frame: pd.DataFrame
    missingness: list[dict[str, str]]
    historical_race_ids: tuple[int, ...]
    request: ResearchRequest
    schema_version: str = RESEARCH_SCHEMA
    evidence_mode: str = "retrospective_research"
    limitations: tuple[str, ...] = LIMITATIONS


def publication_upper_bound(record: dict[str, Any]) -> datetime:
    """Conservative research claim; never backdate capture or live availability."""
    if not record.get("reviewed_by"):
        raise ValueError("Roster evidence needs a reviewed publication claim")
    claim = record.get("applicability", {}).get("research_publication_date")
    exact = record.get("published_at")
    if claim is None:
        if not exact or record.get("publication_basis") not in {"document", "provider_metadata"}:
            raise ValueError("Roster evidence publication is unknown")
        return utc(exact)
    if not isinstance(claim, dict) or set(claim) != {"date", "basis", "timezone"}:
        raise ValueError("Malformed research publication date")
    if claim["basis"] not in {"document", "provider_metadata"} or claim["timezone"] != "unknown":
        raise ValueError("Date-only research evidence requires an explicit unknown timezone")
    value = claim["date"]
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("Research publication date must be YYYY-MM-DD")
    day = date.fromisoformat(value)
    if day.isoformat() != value:
        raise ValueError("Research publication date must be YYYY-MM-DD")
    # End of the stated calendar day at UTC-12 is next-day 12:00 UTC.
    # Do not interpret a date-only page's generated midnight as an exact time.
    bound = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=36)
    if exact:
        if record.get("publication_basis") not in {"document", "provider_metadata"}:
            raise ValueError("Publication timestamp has no verified basis")
        bound = max(bound, utc(exact))
    return bound


def publication_lower_bound(record: dict[str, Any]) -> datetime:
    """Earliest possible research publication, for post-cutoff outcome evidence."""
    publication_upper_bound(record)
    claim = record.get("applicability", {}).get("research_publication_date")
    if claim is None:
        return utc(record["published_at"])
    day = date.fromisoformat(claim["date"])
    bound = datetime.combine(day, datetime.min.time(), timezone.utc) - timedelta(hours=14)
    return max(bound, utc(record["published_at"])) if record.get("published_at") else bound


def _research_evidence(store: SourceStore, reference: str, cutoff: str,
                       approved_team_hosts: frozenset[str], seen: set[str] | None = None) -> dict:
    seen = set() if seen is None else seen
    if reference in seen:
        raise ValueError("Cyclic research evidence dependency")
    seen.add(reference)
    record, _ = store.read(reference)
    if not _official(record, approved_team_hosts):
        raise ValueError("Official roster sources required; team hosts need explicit approval")
    if publication_upper_bound(record) > utc(cutoff):
        raise ValueError("Roster evidence publication is after cutoff or overlaps its uncertainty window")
    for dependency in record.get("supporting_observations", []):
        _research_evidence(store, dependency, cutoff, approved_team_hosts, seen)
    seen.remove(reference)
    return record


def load_research_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read raw history only through 2023; never open processed split files.

    Stream filters discard later event rows before creating dataframes. This
    does not turn the current F1DB files into historical publication vintages.
    """
    tables = {}
    ids: set[int] = set()
    for name, filename in RAW_FILES.items():
        with (raw_dir / filename).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            buffer = StringIO()
            writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if name == "races":
                    if int(row["year"]) > 2023:
                        continue
                    ids.add(int(row["id"]))
                elif "raceId" in row and int(row["raceId"]) not in ids:
                    continue
                writer.writerow(row)
            buffer.seek(0)
            tables[name] = pd.read_csv(buffer, low_memory=False)
    return tables


def _prior_tables(tables: dict[str, pd.DataFrame], request: ResearchRequest) -> dict[str, pd.DataFrame]:
    races = tables["races"]
    dates = pd.to_datetime(races["date"], errors="raise")
    before = dates.dt.date < utc(request.cutoff).date()
    selected = races[before & races["id"].ne(request.race_id)].copy()
    selected = selected[selected.year.le(2023)]
    if selected.empty:
        raise ValueError("No prior completed race history")
    ids = set(selected.id)
    if not ids <= set(tables["results"].raceId):
        raise ValueError("Earlier scheduled races lack results; review cancellations or missing history")
    result = {**tables, "races": selected}
    for name in ("results", "qualifying", "grid", "driver_standings", "constructor_standings"):
        result[name] = tables[name][tables[name].raceId.isin(ids)].copy()
    for name in ("races", "drivers", "circuits"):
        if result[name].id.duplicated().any():
            raise ValueError(f"Duplicate {name} identity")
    for name, identity in (("driver_standings", "driverId"), ("constructor_standings", "constructorId")):
        target_year = int(tables["races"].loc[tables["races"].id.eq(request.race_id), "year"].iloc[0])
        relevant = selected[selected.year.isin([target_year - 1, target_year])]
        # Earlier standings never enter these features; pre-modern constructor
        # tables can legitimately contain multiple engine combinations.
        result[name] = result[name][result[name].raceId.isin(relevant.id)].copy()
        if result[name].duplicated(["raceId", identity]).any():
            raise ValueError("Duplicate historical standings")
        for _, season in relevant.groupby("year"):
            latest = season.sort_values(["date", "round", "id"]).iloc[-1].id
            if not result[name].raceId.eq(latest).any():
                raise ValueError(f"Latest completed season race lacks {name}")
    return result


def build_research_features(request: ResearchRequest, tables: dict[str, pd.DataFrame], store: SourceStore,
                            *, approved_team_hosts: frozenset[str] = frozenset()) -> ResearchBatch:
    request.validate()
    records = {}
    for reference, checksum in request.source_hashes.items():
        record = _research_evidence(store, reference, request.cutoff, approved_team_hosts)
        if record["sha256"] != checksum:
            raise ValueError("Reviewed source hash mismatch")
        if record.get("document_kind") not in {"event_lineup", "season_registration", "entry_list"}:
            raise ValueError("Roster evidence must be an announcement or registration, not session results")
        records[reference] = record
    current = tables["races"][tables["races"].id.eq(request.race_id)]
    if len(current) != 1:
        raise ValueError("Expected one matching scheduled race")
    raw = current.iloc[0]
    if not 2014 <= int(raw.year) <= 2023:
        raise ValueError("Research is limited to 2014-2023; do not open the test years")
    identity = {"race_id": request.race_id, "year": int(raw.year), "round": int(raw["round"]),
                "grand_prix_id": raw.grandPrixId, "circuit_id": raw.circuitId}
    for record in records.values():
        _scope(record, identity)
    for entry in request.entries:
        for reference in entry["evidence"]:
            if [entry["driver_id"], entry["constructor_id"]] not in records[reference]["applicability"].get("driver_constructor_pairs", []):
                raise ValueError("Source review does not bind the entrant/team pairing")
    race_date = pd.Timestamp(raw.date)
    if not utc(request.fp1_start).date() <= race_date.date() or utc(request.fp1_start).year != int(raw.year):
        raise ValueError("FP1 and race calendar disagree")
    if (race_date.date() - utc(request.fp1_start).date()).days > 4:
        raise ValueError("FP1 does not belong to the reviewed race weekend")
    prior = _prior_tables(tables, request)
    # Only the already-truncated prior tables reach the unchanged history code.
    history = _prepare_history(prior, Stage2AConfig())
    history = history.sort_values(["race_date", "round", "raceId", "driverId"]).reset_index(drop=True)
    history["event_index"] = history.raceId.map({r: i for i, r in enumerate(history.raceId.drop_duplicates())})
    race = {
        "year": int(raw.year), "round": int(raw["round"]), "race_month": race_date.month,
        "grand_prix_id": raw.grandPrixId, "circuit_id": raw.circuitId,
        "circuit_layout_id": raw.circuitLayoutId, "circuit_type": raw.circuitType,
        "circuit_direction": raw.direction, "qualifying_format": raw.qualifyingFormat,
        "course_length_km": raw.courseLength, "circuit_turns": raw.turns,
        "is_sprint_weekend": int(pd.notna(raw.sprintRaceDate) or raw.qualifyingFormat == "SPRINT_RACE" or pd.notna(raw.sprintQualifyingFormat)),
    }
    circuit = tables["circuits"][tables["circuits"].id.eq(raw.circuitId)]
    country = circuit.iloc[0].countryId if len(circuit) else np.nan
    circuit_features = _circuit_features(history, raw.circuitId)
    counts = pd.Series([e["constructor_id"] for e in request.entries]).value_counts()
    rows, missingness = [], []
    for entry in sorted(request.entries, key=lambda item: item["driver_id"]):
        drivers = tables["drivers"][tables["drivers"].id.eq(entry["driver_id"])]
        driver = drivers.iloc[0] if len(drivers) else None
        nationality = driver.nationalityCountryId if driver is not None else np.nan
        birth = pd.to_datetime(driver.dateOfBirth, errors="raise") if driver is not None else pd.NaT
        if pd.notna(birth) and birth >= race_date:
            raise ValueError("Invalid driver birth date")
        row = {**race, **{k: entry[k] for k in ENTRY_FIELDS - {"evidence"}},
               "race_id": request.race_id, "race_date": race_date.date().isoformat(),
               "cutoff": request.cutoff, "split": "train" if raw.year <= 2022 else "validation",
               "driver_nationality_country_id": nationality, "circuit_country_id": country,
               "driver_age_years": (race_date - birth).days / 365.2425 if pd.notna(birth) else np.nan,
               "is_driver_home_race": int(pd.notna(nationality) and pd.notna(country) and nationality == country),
               "current_constructor_entry_count": int(counts[entry["constructor_id"]])}
        row.update(_driver_features(history, entry, race))
        row.update(_constructor_features(history, entry, race))
        row.update(circuit_features)
        row.update(_standings(prior, entry, race))
        if set(CANDIDATE_FEATURES) - row.keys():
            raise ValueError("Early feature implementation incomplete")
        rows.append(row)
        for name in CANDIDATE_FEATURES:
            if pd.isna(row[name]):
                missingness.append({"driver_id": entry["driver_id"], "feature": name,
                                    "reason": "missing_registry_or_undefined_prior_history"})
    frame = pd.DataFrame(rows)[["race_id", "race_date", "cutoff", "split", *CANDIDATE_FEATURES]]
    numeric = [c for c in CANDIDATE_FEATURES if c not in CATEGORICAL_FEATURES]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise").map(lambda v: float(f"{v:.10g}"))
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].astype(pd.StringDtype(na_value=np.nan))
    if np.isinf(frame[numeric].to_numpy(float)).any():
        raise ValueError("Infinite research feature")
    return ResearchBatch(frame, missingness, tuple(int(v) for v in history.raceId.drop_duplicates()), request)
