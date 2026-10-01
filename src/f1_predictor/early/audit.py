"""Offline metadata/evidence census. Never loads model-training or test CSVs."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from ..stage3.release import sha256, write_json
from ..stage3.sources import SourceStore, utc
from .contracts import EarlySnapshot
from .policy import feature_lineage, policy_manifest

RESEARCH_YEARS = range(2014, 2024)


def inspect_inputs(root: Path, source_store: Path | None = None) -> dict:
    """Census available files, not a claim that retrospectively recorded rows are as-of valid."""
    inputs = {}

    def read_csv(relative: str) -> list[dict]:
        path = root / relative
        inputs[relative] = sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    races = [row for row in read_csv("data/raw/f1db-races.csv") if int(row["year"]) in RESEARCH_YEARS]
    practice = {}
    for number in (1, 2, 3):
        relative = f"data/raw/f1db-races-free-practice-{number}-results.csv"
        practice[number] = {int(row["raceId"]) for row in read_csv(relative)
                            if int(row["year"]) in RESEARCH_YEARS}
    coverage = []
    for year in RESEARCH_YEARS:
        rows = [row for row in races if int(row["year"]) == year]
        ids = {int(row["id"]) for row in rows}
        coverage.append({"year": year, "races": len(rows),
                         "fp1_date_and_time_present": sum(bool(row.get("freePractice1Date") and row.get("freePractice1Time")) for row in rows),
                         "qualifying_date_and_time_present": sum(bool(row.get("qualifyingDate") and row.get("qualifyingTime")) for row in rows),
                         "race_time_present": sum(bool(row.get("time")) for row in rows),
                         "practice_summary_races": {f"fp{n}": len(ids & practice[n]) for n in practice}})

    observations = []
    valid_packets = []
    packet_errors = []
    if source_store is not None:
        if not source_store.is_dir():
            raise ValueError("Supplied private source store does not exist")
        store = SourceStore(source_store)
        for path in sorted(source_store.glob("*/*.json")):
            ref = path.relative_to(source_store).as_posix()
            record, _ = store.read(ref)
            observations.append({"observation_id": ref, "sha256": record["sha256"],
                                 "first_observed_at": record["first_observed_at"],
                                 "reviewed_at": record.get("reviewed_at"),
                                 "applicability": record.get("applicability", {})})
        for path in sorted((root / "data/early/snapshots").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("race", {}).get("year") not in RESEARCH_YEARS:
                continue
            try:
                snapshot = EarlySnapshot.from_dict(value)
                snapshot.validate(store)
                valid_packets.append(snapshot)
            except (ValueError, TypeError, KeyError) as error:
                packet_errors.append({"file": path.relative_to(root).as_posix(), "reason": str(error)})
    audit_path = "config/stage2a_eligibility_audit.json"
    inputs[audit_path] = sha256(root / audit_path)
    old_audit = json.loads((root / audit_path).read_text(encoding="utf-8"))
    exceptions = []
    for old in old_audit["records"]:
        matches = [entry for packet in valid_packets if packet.race["race_id"] == old["race_id"]
                   for entry in packet.entries if entry["driver_id"] == old["driver_id"]]
        status = matches[0]["status"] if len(matches) == 1 else "unresolved"
        exceptions.append({"race_id": old["race_id"], "year": old["year"], "driver_id": old["driver_id"],
                           "post_qualifying_eligible": old["prediction_eligible"],
                           "existing_reference_url": old["source_url"], "early_status": status,
                           "decision": "Do not carry the later exclusion backward. Require pre-cutoff roster/withdrawal evidence.",
                           "legacy_reason_is_not_early_evidence": True})
    lineage_counts = dict(Counter(row["early_decision"] for row in feature_lineage()))
    packet_races = {packet.race["race_id"] for packet in valid_packets}
    historical_ids = {int(row["id"]) for row in races}
    reviewed = [row for row in observations if row["applicability"].get("race_id") in historical_ids and row["reviewed_at"]]
    return {
        "phase": 1, "status": "audit_only_not_ready_for_model_training",
        "scope": "Local 2014-2023 schedule/practice metadata, legacy eligibility audit and supplied evidence store only; no external requests.",
        "source_files_sha256": inputs,
        "research_races": len(races), "feature_decisions": lineage_counts,
        "schedule_and_practice_coverage": coverage,
        "coverage_notice": "Existing schedule/practice fields are retrospective metadata, not proof of original publication. Practice is excluded from Early v1.",
        "evidence_inventory": {
            "source_store_scanned": source_store is not None,
            "observations_integrity_checked": len(observations),
            "first_observation": min((row["first_observed_at"] for row in observations), key=utc, default=None),
            "historical_event_reviews_found": len(reviewed),
            "validated_early_evidence_packets": len(valid_packets),
            "research_races_with_early_packet": len(packet_races & historical_ids),
            "packet_errors": packet_errors,
            "inventory_sha256": hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest(),
            "note": "Availability claims are limited to this scanned store. Packet checks do not establish feature-value lineage or model readiness.",
        },
        "eligibility_reaudit": exceptions,
        "blockers": [
            "Recover original pre-practice session schedules with known timezone and publication/observation evidence.",
            "Reconstruct complete announced fields, including later withdrawals and substitutions absent from result rows.",
            "Audit prior-result and standings revisions as of each cutoff; present-day corrected tables are not publication vintages.",
            "Build and test the separate early feature builder before using any candidate features for training.",
        ],
        "activity": {"model_fit_performed": False, "test_split_loaded": False, "historical_evaluation_performed": False,
                     "network_used": False, "forecast_generated": False, "existing_files_modified": False},
    }


def write_audit(root: Path, output: Path, source_store: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Audit destination exists; never overwrite an earlier audit")
    report = inspect_inputs(root, source_store)
    output.mkdir(parents=True)
    write_json(output / "policy.json", policy_manifest())
    write_json(output / "feature_lineage.json", feature_lineage())
    write_json(output / "evidence_audit.json", report)
    write_json(output / "manifest.json", {
        "phase": 1, "input_files_sha256": report["source_files_sha256"],
        "runtime": {"python": platform.python_version(), "pandas": importlib.metadata.version("pandas")},
        "implementation_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in
                                  sorted((root / "src/f1_predictor/early").glob("*.py")) +
                                  [root / "scripts/audit_early_forecast.py", root / "src/f1_predictor/stage2a.py",
                                   root / "src/f1_predictor/stage3/sources.py", root / "src/f1_predictor/stage3/release.py"]},
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.glob("*.json"))},
    })
    return report
