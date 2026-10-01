"""Evidence-only early contract, not a predictor or a confirmed-grid substitute."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any
from urllib.parse import urlparse

from ..stage3.sources import SourceStore, utc
from .policy import POLICY_VERSION, scheduled_cutoff

RACE_FIELDS = {"race_id", "year", "round", "grand_prix_id", "circuit_id",
               "fp1_start", "qualifying_start", "race_start"}
ENTRY_FIELDS = {"driver_id", "constructor_id", "engine_manufacturer_id",
                "status", "evidence", "reason"}
IDENTITY_FIELDS = ("race_id", "year", "round", "grand_prix_id", "circuit_id")
OFFICIAL_HOSTS = {"fia": {"fia.com", "www.fia.com", "api.fia.com"},
                  "f1": {"formula1.com", "www.formula1.com", "corp.formula1.com"}}
FRESHNESS = {"schedule": 86400, "roster": 86400, "history": 2592000, "registry": 2592000}


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}", value) is not None


def _evidence(store: SourceStore, reference: str, cutoff: str, max_age: int,
              seen: set[str] | None = None) -> dict:
    seen = set() if seen is None else seen
    if reference in seen:
        raise ValueError("Cyclic evidence dependency")
    seen.add(reference)
    record = store.available(reference, cutoff, max_age)
    if record.get("published_at") and record.get("publication_basis") not in ("document", "provider_metadata"):
        raise ValueError("Publication timestamp has no verified basis")
    for dependency in record.get("supporting_observations", []):
        _evidence(store, dependency, cutoff, 10**12, seen)
    seen.remove(reference)
    return record


def _official(record: dict, approved_team_hosts: frozenset[str]) -> bool:
    url = urlparse(record["uri"])
    hosts = approved_team_hosts if record["provider"] == "team" else OFFICIAL_HOSTS.get(record["provider"], set())
    return (url.scheme == "https" and url.hostname in hosts and
            not url.username and not url.password and url.port in (None, 443))


def _scope(record: dict, race: dict) -> None:
    if not all(record.get(key) for key in ("reviewed_by", "reviewed_at", "review_notes")):
        raise ValueError("Named, timestamped applicability review required")
    scope = record.get("applicability", {})
    if any(scope.get(key) != race[key] for key in IDENTITY_FIELDS) or scope.get("session") != "event":
        raise ValueError("Wrong event or session applicability")


@dataclass(frozen=True)
class EarlySnapshot:
    snapshot_id: str
    cutoff: str
    race: dict[str, Any]
    entries: list[dict[str, Any]]
    sources: dict[str, str]
    reviewed_by: str
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "EarlySnapshot":
        return cls(**value)

    def validate(self, store: SourceStore, *, approved_team_hosts: frozenset[str] = frozenset()) -> None:
        """Check provenance and roster, not historical aggregates or a model's readiness."""
        if self.policy_version != POLICY_VERSION or not _identifier(self.snapshot_id) or not self.reviewed_by.strip():
            raise ValueError("Versioned snapshot and named reviewer required")
        if set(self.race) != RACE_FIELDS:
            raise ValueError("Early race fields differ from allowlist")
        for field in ("race_id", "year", "round"):
            if type(self.race[field]) is not int or self.race[field] <= 0:
                raise ValueError("Invalid numeric race identity")
        if not all(_identifier(self.race[key]) for key in ("grand_prix_id", "circuit_id")):
            raise ValueError("Canonical race identity required")
        fp1, qualifying, start = (utc(self.race[key]) for key in ("fp1_start", "qualifying_start", "race_start"))
        if not fp1 < qualifying < start or start.year != self.race["year"]:
            raise ValueError("Inconsistent session chronology")
        if utc(self.cutoff) != utc(scheduled_cutoff(self.race["fp1_start"])):
            raise ValueError("Cutoff must equal scheduled FP1 minus 24 hours")
        if set(self.sources) != set(FRESHNESS):
            raise ValueError("Exactly schedule, roster, history and registry evidence required")
        records = {role: _evidence(store, ref, self.cutoff, FRESHNESS[role])
                   for role, ref in self.sources.items()}
        schedule, roster = records["schedule"], records["roster"]
        for record in (schedule, roster):
            _scope(record, self.race)
        if not _official(schedule, frozenset()) or schedule.get("document_kind") not in {"calendar", "event_schedule"}:
            raise ValueError("Official reviewed session schedule required")
        for key in ("fp1_start", "qualifying_start", "race_start"):
            if schedule["applicability"].get(key) != self.race[key]:
                raise ValueError("Reviewed schedule must bind every exact session timestamp")
        if roster["provider"] != "formation" or roster.get("document_kind") != "early_roster_review":
            raise ValueError("Explicit early announced-roster review required")
        if roster["applicability"].get("roster_complete") is not True:
            raise ValueError("Roster completeness must be explicitly reviewed")
        packet = json.loads(store.read(self.sources["roster"])[1])
        if set(packet) != {"policy_version", "race_id", "entries", "unresolved_conflicts"}:
            raise ValueError("Unknown roster packet fields")
        if packet["policy_version"] != POLICY_VERSION or packet["race_id"] != self.race["race_id"]:
            raise ValueError("Wrong roster policy or race")
        if packet["unresolved_conflicts"] != [] or packet["entries"] != self.entries:
            raise ValueError("Unresolved roster conflicts or entries differ from reviewed bytes")
        if not 1 <= len(self.entries) <= 40:
            raise ValueError("Invalid announced field size")
        ids = []
        announced = 0
        supports = set(roster.get("supporting_observations", []))
        for entry in self.entries:
            if set(entry) != ENTRY_FIELDS:
                raise ValueError("Entry fields differ from early allowlist; no qualifying/grid/outcomes")
            if not all(_identifier(entry[key]) for key in ("driver_id", "constructor_id")) or not entry["reason"]:
                raise ValueError("Explicit driver/team identity and decision reason required")
            if entry["engine_manufacturer_id"] is not None and not _identifier(entry["engine_manufacturer_id"]):
                raise ValueError("Engine identity must be canonical or unknown")
            if entry["status"] not in ("announced", "withdrawn"):
                raise ValueError("Ambiguous participation must block, not be inferred from missing sessions")
            if not isinstance(entry["evidence"], list) or not entry["evidence"]:
                raise ValueError("Each entry needs documentary evidence")
            kinds = set()
            for ref in entry["evidence"]:
                if ref not in supports:
                    raise ValueError("Entry evidence must belong to the reviewed dependency closure")
                record = _evidence(store, ref, self.cutoff, 10**12)
                if not _official(record, approved_team_hosts):
                    raise ValueError("Official entry evidence required; team hosts need explicit approval")
                _scope(record, self.race)
                pairs = record["applicability"].get("driver_constructor_pairs", [])
                if [entry["driver_id"], entry["constructor_id"]] not in pairs:
                    raise ValueError("Evidence does not bind the driver/team pairing")
                kind = record.get("document_kind")
                if kind not in {"entry_list", "season_registration", "event_lineup", "withdrawal"}:
                    raise ValueError("Results, qualifying and grid are not early entry evidence")
                if kind == "entry_list" and record["provider"] != "fia":
                    raise ValueError("An official entry list must be FIA evidence")
                kinds.add(kind)
            if not ("entry_list" in kinds or {"season_registration", "event_lineup"} <= kinds):
                raise ValueError("Require FIA event list or season registration plus event lineup")
            if (entry["status"] == "withdrawn") != ("withdrawal" in kinds):
                raise ValueError("Withdrawal status must agree with its pre-cutoff evidence")
            announced += entry["status"] == "announced"
            ids.append(entry["driver_id"])
        if not announced or len(ids) != len(set(ids)):
            raise ValueError("No announced entrants or duplicate driver")
        if sorted(roster["applicability"].get("entrant_ids", [])) != sorted(ids):
            raise ValueError("Roster review must bind the complete field")
        if records["history"]["provider"] != "f1db" or records["registry"]["provider"] != "f1db":
            raise ValueError("Versioned F1DB history and registry required for initial compatibility")
        # Check the historical table observations, not just the manifest timestamp.
        history = json.loads(store.read(self.sources["history"])[1])
        if not history.get("version") or not history.get("tables"):
            raise ValueError("Versioned historical table manifest required")
        for reference in history["tables"].values():
            _evidence(store, reference, self.cutoff, 2592000)

    def validate_publication_time(self, generated_at: str, published_at: str) -> None:
        deadline = min(utc(self.race[key]) for key in ("fp1_start", "qualifying_start", "race_start"))
        if not utc(self.cutoff) <= utc(generated_at) <= utc(published_at) < deadline:
            raise ValueError("Early publication must follow cutoff and precede FP1; never backdate")
