"""Offline, private research datasets from reviewed announced-field packets.

Outcomes are joined only after feature construction. These artifacts are not
forecasts, and original publication vintages remain explicitly unverified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from ..stage2a import RAW_FILES
from ..stage3.sources import SourceStore, now, utc
from .contracts import _official, _scope
from .policy import CANDIDATE_FEATURES
from .research_features import ResearchBatch, ResearchRequest, build_research_features, load_research_tables, publication_lower_bound

DATASET_SCHEMA = "early-retrospective-dataset-v1"
NON_STARTS = {"DNS", "DNP", "DNQ", "DNPQ", "EX"}


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_block(race_id: int, races: pd.DataFrame) -> str:
    season = races[races.year.eq(2023)].sort_values(["date", "round", "id"])
    if len(season) != 22 or season.id.duplicated().any():
        raise ValueError("Original 22-race validation calendar required")
    ids = season.id.tolist()
    if race_id not in ids:
        return "train"
    index = ids.index(race_id)
    return "selection" if index < 8 else "calibration" if index < 15 else "report"


def validate_schedule(packet: dict, tables: dict, store: SourceStore) -> None:
    request = ResearchRequest(**packet["request"])
    rows = tables["races"][tables["races"].id.eq(request.race_id)]
    if len(rows) != 1:
        raise ValueError("Unknown or duplicate research race")
    row = rows.iloc[0]
    schedule = packet["schedule"]
    record, _ = store.read(schedule["reference"])
    if record["sha256"] != schedule["sha256"]:
        raise ValueError("Schedule hash mismatch")
    if not (_official(record, frozenset()) or
            (record["provider"] == "jolpica" and record["uri"].startswith("https://api.jolpi.ca/ergast/f1/"))):
        raise ValueError("Unapproved research schedule provider")
    if record.get("document_kind") != "event_schedule":
        raise ValueError("Reviewed schedule required")
    _scope(record, {"race_id": request.race_id, "year": int(row.year), "round": int(row["round"]),
                    "grand_prix_id": row.grandPrixId, "circuit_id": row.circuitId})
    if record["applicability"].get("fp1_start") != request.fp1_start:
        raise ValueError("FP1 differs from reviewed schedule")
    if record["applicability"].get("evidence_mode") != "retrospective_research":
        raise ValueError("Historical schedule vintage limitation required")
    race_start = record["applicability"].get("race_start")
    if race_start is not None and utc(race_start).date().isoformat() != row.date:
        raise ValueError("Race date differs from reviewed schedule; explicit research correction required")


def apply_calendar_corrections(tables: dict, corrections: list[dict], store: SourceStore) -> dict:
    """Overlay reviewed historical dates in memory; never mutate frozen raw files."""
    races = tables["races"].copy()
    seen = set()
    for correction in corrections:
        if set(correction) != {"race_id", "original_date", "reference", "sha256"}:
            raise ValueError("Unexpected calendar correction fields")
        race_id = correction["race_id"]
        rows = races[races.id.eq(race_id)]
        if type(race_id) is not int or race_id in seen or len(rows) != 1:
            raise ValueError("Unknown or duplicate calendar correction")
        seen.add(race_id)
        row = rows.iloc[0]
        if not 2014 <= int(row.year) <= 2023 or row.date != correction["original_date"]:
            raise ValueError("Calendar correction does not match research source year/date")
        record, _ = store.read(correction["reference"])
        if (record["sha256"] != correction["sha256"] or not _official(record, frozenset())
                or record.get("document_kind") != "event_schedule"):
            raise ValueError("Official reviewed calendar correction required")
        _scope(record, {"race_id": race_id, "year": int(row.year), "round": int(row["round"]),
                        "grand_prix_id": row.grandPrixId, "circuit_id": row.circuitId})
        scope = record["applicability"]
        if scope.get("evidence_mode") != "retrospective_research" or not scope.get("race_start"):
            raise ValueError("Research calendar correction requires an explicit race timestamp")
        race_start = utc(scope["race_start"])
        fp1 = utc(scope["fp1_start"])
        if (race_start.year != int(row.year) or not 0 < (race_start - fp1).total_seconds() <= 5 * 86400
                or race_start.date().isoformat() == row.date):
            raise ValueError("Invalid or redundant research calendar correction")
        races.loc[rows.index, "date"] = race_start.date().isoformat()
    return {**tables, "races": races}


def attach_outcomes(batch: ResearchBatch, results: pd.DataFrame, store: SourceStore,
                    exceptions: list[dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """Left join the announced field, never replace it with eventual starters.

Missing result rows require independent reviewed withdrawal evidence. Explicit
DNS/DNP result rows are negative binary targets with no invented ordering.
DNF/DSQ retain official display order, distinct from classified position.
"""
    if batch.evidence_mode != "retrospective_research":
        raise ValueError("Only retrospective research batches may receive outcomes")
    race_id = batch.request.race_id
    actual = results[results.raceId.eq(race_id)].copy()
    if actual.empty or actual.driverId.duplicated().any():
        raise ValueError("Missing or duplicate race results")
    classified = pd.to_numeric(actual.positionNumber, errors="raise")
    display = pd.to_numeric(actual.positionDisplayOrder, errors="raise")
    if (classified.eq(1).sum() != 1 or classified.between(1, 3).sum() != 3
            or classified.dropna().duplicated().any() or display.isna().any()
            or display.duplicated().any() or (display < 1).any()
            or (display % 1 != 0).any() or (classified.dropna() < 1).any()
            or (classified.dropna() % 1 != 0).any()):
        raise ValueError("Incomplete or inconsistent official classification")
    allowed = {}
    for exception in exceptions or []:
        if set(exception) != {"driver_id", "constructor_id", "reference", "sha256", "reason"}:
            raise ValueError("Unexpected outcome exception fields")
        driver = exception["driver_id"]
        if driver in allowed or not exception["reason"].strip():
            raise ValueError("Duplicate or unexplained outcome exception")
        record, _ = store.read(exception["reference"])
        identity = batch.frame.iloc[0]
        _scope(record, {"race_id": race_id, "year": int(identity.year), "round": int(identity["round"]),
                        "grand_prix_id": identity.grand_prix_id, "circuit_id": identity.circuit_id})
        if (record["sha256"] != exception["sha256"] or not _official(record, frozenset())
                or record.get("document_kind") != "withdrawal"
                or publication_lower_bound(record) <= utc(batch.request.cutoff)
                or [driver, exception["constructor_id"]] not in record["applicability"].get("driver_constructor_pairs", [])):
            raise ValueError("Reviewed post-cutoff withdrawal evidence required")
        allowed[driver] = exception
    indexed = actual.set_index("driverId")
    rows, used = [], set()
    for _, entrant in batch.frame.iterrows():
        driver = entrant.driver_id
        if driver not in indexed.index:
            if driver not in allowed or allowed[driver]["constructor_id"] != entrant.constructor_id:
                raise ValueError(f"Unexplained missing result: {race_id}/{driver}")
            used.add(driver)
            status, position, order, points, source = "withdrawn_after_cutoff", np.nan, np.nan, 0.0, "reviewed_withdrawal"
        else:
            result = indexed.loc[driver]
            if result.constructorId != entrant.constructor_id:
                raise ValueError("Result constructor differs from announced constructor")
            status, source = str(result.positionText), "official_result"
            position = float(result.positionNumber)
            points = 0.0 if pd.isna(result.points) else float(result.points)
            order = np.nan if status in NON_STARTS else float(result.positionDisplayOrder)
            if status in NON_STARTS and (pd.notna(position) or points != 0):
                raise ValueError("Non-start result has a classified position or points")
            if not np.isfinite(points) or points < 0:
                raise ValueError("Invalid official race points")
        rows.append({"race_id": race_id, "driver_id": driver, "result_status": status,
                     "classified_finish_position": position, "finish_order": order,
                     "ranking_available": int(pd.notna(order)), "race_winner": int(position == 1),
                     "podium_finish": int(pd.notna(position) and 1 <= position <= 3),
                     "points_finish": int(points > 0), "outcome_source": source})
    if set(allowed) != used:
        raise ValueError("Unused outcome exception; review scope rather than hiding data")
    labels = pd.DataFrame(rows)
    outside = sorted(set(actual.driverId) - set(batch.frame.driver_id))
    uncovered = actual[actual.driverId.isin(outside)]
    outside_positives = {
        "race_winner": int(uncovered.positionNumber.eq(1).sum()),
        "podium_finish": int(uncovered.positionNumber.between(1, 3).sum()),
        "points_finish": int(pd.to_numeric(uncovered.points, errors="raise").gt(0).sum()),
    }
    return labels, {"race_id": race_id, "announced_entries": len(labels),
                    "ranked_entries": int(labels.ranking_available.sum()),
                    "result_drivers_outside_announced_field": outside,
                    "outside_field_binary_positives": outside_positives,
                    "target_coverage_complete": {k: v == 0 for k, v in outside_positives.items()},
                    "reviewed_missing_result_drivers": sorted(used),
                    "binary_positives": {k: int(labels[k].sum()) for k in ("race_winner", "podium_finish", "points_finish")}}


def write_dataset(root: Path, packets_path: Path, store: SourceStore, output: Path) -> dict[str, Any]:
    """Exclusive-create private research output; no network or estimator calls."""
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(packets_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_SCHEMA or payload.get("evidence_mode") != "retrospective_research":
        raise ValueError("Versioned retrospective packet required")
    packets = payload["packets"]
    ids = [p["request"]["race_id"] for p in packets]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate research races")
    tables = load_research_tables(root / "data/raw")
    corrections = payload.get("calendar_corrections", [])
    tables = apply_calendar_corrections(tables, corrections, store)
    features, labels, coverage, missingness = [], [], [], []
    references = {c["reference"] for c in corrections}
    for packet in sorted(packets, key=lambda p: (utc(p["request"]["cutoff"]), p["request"]["race_id"])):
        validate_schedule(packet, tables, store)
        batch = build_research_features(ResearchRequest(**packet["request"]), tables, store)
        target, info = attach_outcomes(batch, tables["results"], store, packet.get("outcome_exceptions", []))
        info.update(validation_block=validation_block(batch.request.race_id, tables["races"]),
                    historical_race_ids=list(batch.historical_race_ids))
        features.append(batch.frame)
        labels.append(target)
        coverage.append(info)
        missingness.extend({"race_id": batch.request.race_id, **m} for m in batch.missingness)
        references.update(batch.request.source_hashes)
        references.add(packet["schedule"]["reference"])
        references.update(e["reference"] for e in packet.get("outcome_exceptions", []))
    frame, target = pd.concat(features, ignore_index=True), pd.concat(labels, ignore_index=True)
    if not frame[["race_id", "driver_id"]].equals(target[["race_id", "driver_id"]]):
        raise ValueError("Features and labels lost row alignment")
    # Freeze the exact reviewed dependency closure, including original captures.
    pending = list(references)
    observations = {}
    while pending:
        reference = pending.pop()
        if reference in observations:
            continue
        record, _ = store.read(reference)
        observations[reference] = record
        pending.extend(record.get("supporting_observations", []))
        if record.get("previous_observation"):
            pending.append(record["previous_observation"])
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "features.csv", index=False, lineterminator="\n", float_format="%.10g")
    target.to_csv(output / "outcomes.csv", index=False, lineterminator="\n", float_format="%.10g")
    for name, value in (("packets.json", payload), ("coverage.json", coverage), ("missingness.json", missingness)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    private_store = output / "sources"
    for reference, record in sorted(observations.items()):
        for name in (reference, record["blob"]):
            dest = private_store / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as stream:
                    stream.write((store.root / name).read_bytes())
    counts = {block: sum(c["validation_block"] == block for c in coverage)
              for block in ("train", "selection", "calibration", "report")}
    code_paths = [*sorted((root / "src/f1_predictor/early").glob("*.py")),
                  root / "src/f1_predictor/stage2a.py", root / "src/f1_predictor/stage3/features.py",
                  root / "src/f1_predictor/stage3/sources.py", root / "scripts/build_early_research.py"]
    report = {"schema_version": DATASET_SCHEMA, "evidence_mode": "retrospective_research",
              "point_in_time_verified": False, "created_at": now(),
              "rows": len(frame), "races": len(coverage), "feature_count": len(CANDIDATE_FEATURES),
              "feature_order": list(CANDIDATE_FEATURES), "columns_with_missing": frame.isna().sum()[lambda x: x.gt(0)].to_dict(),
              "blocks": counts, "original_validation_block_sizes": [8, 7, 7],
              "calendar_corrections": corrections,
              "training_roster_coverage_complete": counts["train"] == int(tables["races"].year.between(2014, 2022).sum()),
              "model_training_performed": False, "sealed_test_opened": False,
              "limitations": list(batch.limitations) + ["Partial cohorts do not authorize model selection or block reassignment."],
              "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
              "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "code_sha256": {p.relative_to(root).as_posix(): checksum(p) for p in code_paths},
              "raw_sha256": {name: checksum(root / "data/raw" / filename) for name, filename in RAW_FILES.items()},
              "packet_sha256": checksum(packets_path),
              "files_sha256": {p.relative_to(output).as_posix(): checksum(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    return report
