"""Append-only source observations. Network clients never fabricate fallback data."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

from ..stage2a import RAW_FILES, load_raw_tables
from .release import checked, sha256, write_json


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return result.astimezone(timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes archive")
    return path


class SourceStore:
    def __init__(self, root: Path):
        self.root = root

    def capture(self, provider: str, uri: str, payload: bytes, *, published_at: str | None = None,
                reviewed_by: str | None = None, document_kind: str | None = None,
                publication_basis: str | None = None, http_last_modified: str | None = None) -> str:
        retrieved = now()
        if published_at is not None and utc(published_at) > utc(retrieved):
            raise ValueError("Publication is in the future")
        if publication_basis not in (None, "document", "provider_metadata"):
            raise ValueError("Publication basis must identify document or provider metadata, not an HTTP header")
        if publication_basis is not None and published_at is None:
            raise ValueError("Publication basis requires a publication timestamp")
        key = digest_json({"provider": provider, "uri": uri})
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        observations = [self.read(f"{key}/{p.name}")[0] for p in directory.glob("*.json")]
        observations.sort(key=lambda x: utc(x["retrieved_at"]))
        checksum = hashlib.sha256(payload).hexdigest()
        first = next((o["first_observed_at"] for o in observations if o["sha256"] == checksum), retrieved)
        blob = directory / f"{checksum}.blob"
        if blob.exists():
            checked(blob, checksum)
        else:
            with blob.open("xb") as stream:
                stream.write(payload)
        record = {"provider": provider, "uri": uri, "published_at": published_at,
                  "first_observed_at": first, "retrieved_at": retrieved,
                  "sha256": checksum, "blob": f"{key}/{checksum}.blob",
                  "previous_observation": observations[-1]["observation_id"] if observations else None,
                  "reviewed_by": reviewed_by, "document_kind": document_kind,
                  "publication_basis": publication_basis, "http_last_modified": http_last_modified}
        record["observation_id"] = f"{key}/{digest_json(record)}.json"
        write_json(self.root / record["observation_id"], record)
        return record["observation_id"]

    def review(self, observation_id: str, *, applicability: dict[str, Any], reviewed_by: str,
               notes: str, supporting_observations: list[str] | None = None) -> str:
        """Append a review without refreshing or backdating the observed source bytes."""
        reviewed_at = now()
        record = dict(self.available(observation_id, reviewed_at, 10**12))
        if not reviewed_by.strip() or not notes.strip():
            raise ValueError("Named reviewer and applicability rationale required")
        required = {"race_id", "year", "round", "grand_prix_id", "circuit_id", "session"}
        if not required <= applicability.keys() or applicability["session"] not in ("event", "qualifying", "race"):
            raise ValueError("Explicit race and session applicability required")
        references = sorted(set(supporting_observations or []))
        for reference in references:
            self.available(reference, reviewed_at, 10**12)
        record.update(applicability=applicability, reviewed_by=reviewed_by, reviewed_at=reviewed_at,
                      review_notes=notes, supporting_observations=references,
                      previous_observation=observation_id)
        record.pop("observation_id")
        key = digest_json({"provider": record["provider"], "uri": record["uri"]})
        record["observation_id"] = f"{key}/{digest_json(record)}.json"
        write_json(self.root / record["observation_id"], record)
        return record["observation_id"]

    def read(self, observation_id: str) -> tuple[dict[str, Any], bytes]:
        record = json.loads(member(self.root, observation_id).read_text())
        unsigned = {k: v for k, v in record.items() if k != "observation_id"}
        expected = f"{digest_json({'provider': record['provider'], 'uri': record['uri']})}/{digest_json(unsigned)}.json"
        if observation_id != expected or record["observation_id"] != expected:
            raise ValueError("Observation metadata hash mismatch")
        blob = member(self.root, record["blob"])
        checked(blob, record["sha256"])
        return record, blob.read_bytes()

    def available(self, observation_id: str, cutoff: str, max_age_seconds: int) -> dict[str, Any]:
        record, _ = self.read(observation_id)
        point = utc(cutoff)
        for field in ("retrieved_at", "first_observed_at", "published_at", "reviewed_at"):
            if record.get(field) is not None and utc(record[field]) > point:
                raise ValueError(f"Source {field} is after cutoff")
        age = (point - utc(record["retrieved_at"])).total_seconds()
        if max_age_seconds < 0 or age > max_age_seconds:
            raise ValueError("Source is stale; fetch/review a new observation")
        return record


class _SameHostRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https" or urlparse(newurl).hostname != urlparse(req.full_url).hostname:
            raise ValueError("Unapproved cross-host source redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_json(store: SourceStore, provider: str, url: str) -> tuple[Any, str]:
    expected = {"jolpica": "api.jolpi.ca", "openf1": "api.openf1.org"}
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != expected.get(provider) or parsed.username or parsed.password:
        raise ValueError("Unapproved source URL")
    request = Request(url, headers={"User-Agent": "F1RacePredictor/3.0", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with build_opener(_SameHostRedirect()).open(request, timeout=30) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
                if len(payload) > 32 * 1024 * 1024:
                    raise ValueError("Source response too large")
                data = json.loads(payload)
            return data, store.capture(provider, url, payload)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Unreachable retry state")


class Jolpica:
    def __init__(self, store: SourceStore):
        self.store = store

    def fetch(self, year: int, endpoint: str = "", round_number: int | None = None) -> tuple[list[dict], list[str]]:
        if endpoint not in ("", "qualifying", "sprint", "results", "driverstandings", "constructorstandings", "circuits", "drivers", "constructors"):
            raise ValueError("Unsupported Jolpica endpoint")
        if not 1950 <= year <= 2100 or (round_number is not None and not 1 <= round_number <= 50):
            raise ValueError("Invalid year/round")
        route = f"{year}/" + (f"{round_number}/" if round_number is not None else "")
        route += endpoint + "/" if endpoint else ""
        pages, ids = [], []
        offset = 0
        total = None
        while total is None or offset < total:
            data, source = fetch_json(self.store, "jolpica", f"https://api.jolpi.ca/ergast/f1/{route}?limit=100&offset={offset}")
            mr = data["MRData"]
            count, current_total = int(mr["limit"]), int(mr["total"])
            if count < 1 or int(mr["offset"]) != offset or current_total > 100000:
                raise ValueError("Invalid pagination response")
            if total is not None and total != current_total:
                raise ValueError("Source changed during pagination; refetch complete snapshot")
            total = current_total
            pages.append(data)
            ids.append(source)
            offset += count
            if offset < total:
                time.sleep(2.1)
        return pages, ids


class OpenF1:
    def __init__(self, store: SourceStore):
        self.store = store

    def fetch(self, endpoint: str, **filters: int | str) -> tuple[list[dict], str]:
        if endpoint not in ("meetings", "sessions", "drivers", "session_result", "starting_grid"):
            raise ValueError("OpenF1 integration is restricted to session/grid gaps")
        if not filters or any(str(v) == "latest" for v in filters.values()):
            raise ValueError("Explicit stable session/year filters required")
        data, source = fetch_json(self.store, "openf1", f"https://api.openf1.org/v1/{endpoint}?{urlencode(filters)}")
        if not isinstance(data, list):
            raise ValueError("Unexpected OpenF1 response")
        return data, source


def import_f1db(store: SourceStore, raw_dir: Path, version: str) -> str:
    """Pin bytes, not an assumed publication vintage. Existing CSVs stay untouched."""
    if not version.strip():
        raise ValueError("A version/release label is required")
    tables = load_raw_tables(raw_dir)
    observations = {key: store.capture("f1db", f"f1db:{version}/{name}", (raw_dir / name).read_bytes())
                    for key, name in RAW_FILES.items()}
    result_ids = sorted(int(x) for x in tables["results"]["raceId"].unique())
    return store.capture("f1db", f"f1db:{version}/manifest", json.dumps({
        "version": version, "tables": observations, "result_race_ids": result_ids,
        "completion_evidence": "F1DB published race classifications; latest-state import, not historical publication reconstruction"
    }, sort_keys=True).encode())


def import_fia(store: SourceStore, path: Path, uri: str, published_at: str | None, reviewer: str,
               document_kind: str, publication_basis: str | None = None) -> str:
    parsed = urlparse(uri)
    if (not reviewer.strip() or parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.hostname not in ("www.fia.com", "fia.com", "api.fia.com")):
        raise ValueError("Reviewed FIA document and named reviewer required")
    if document_kind not in ("entry_list", "final_grid", "grid_amendment", "eligibility_review", "qualifying", "results", "registry", "calendar"):
        raise ValueError("Explicit reviewed FIA document kind required")
    return store.capture("fia", uri, path.read_bytes(), published_at=published_at,
                         reviewed_by=reviewer, document_kind=document_kind, publication_basis=publication_basis)
