"""Immutable forecast archives, offline replay and separate outcome attachments."""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from ..stage2a import FEATURE_COLUMNS

from .contracts import RaceSnapshot
from .features import build_features
from .inference import FrozenPredictor
from .release import sha256, checked, write_json
from .sources import SourceStore, now, utc, digest_json, member


def source_closure(store: SourceStore, snapshot: RaceSnapshot) -> set[str]:
    return evidence_closure(store, set(snapshot.sources.values()) | {snapshot.history_source})


def evidence_closure(store: SourceStore, references: set[str]) -> set[str]:
    required = set(references)
    pending = list(required)
    while pending:
        record, raw = store.read(pending.pop())
        linked = set(record.get("supporting_observations", []))
        if record["previous_observation"]:
            linked.add(record["previous_observation"])
        if record["provider"] == "f1db" and record["uri"].endswith("/manifest"):
            linked.update(json.loads(raw)["tables"].values())
        pending.extend(sorted(linked - required))
        required.update(linked)
    return required


def export_evidence(store: SourceStore, references: list[str], destination: Path) -> dict[str, Any]:
    """Private portable evidence, including all dependencies and original observation times."""
    if not references:
        raise ValueError("Evidence references must not be empty")
    closure = evidence_closure(store, set(references))
    destination.mkdir(parents=True, exist_ok=False)
    for reference in sorted(closure):
        record, _ = store.read(reference)
        for name in (reference, record["blob"]):
            target = member(destination / "sources", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(member(store.root, name), target)
    manifest = {"version": 1, "exported_at": now(), "references": sorted(set(references)),
                "observations": sorted(closure), "redistribution": "private_pending_rights_review",
                "files_sha256": {p.relative_to(destination).as_posix(): sha256(p)
                                 for p in sorted(destination.rglob("*")) if p.is_file()}}
    write_json(destination / "manifest.json", manifest)
    return inspect_evidence(destination)


def inspect_evidence(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        checked(member(directory, name), digest)
    closure = evidence_closure(SourceStore(directory / "sources"), set(manifest["references"]))
    if closure != set(manifest["observations"]):
        raise ValueError("Evidence dependency closure differs from manifest")
    return manifest


def _copy_sources(store: SourceStore, snapshot: RaceSnapshot, destination: Path) -> None:
    for key in sorted(source_closure(store, snapshot)):
        record, _ = store.read(key)
        for name in (key, record["blob"]):
            target = member(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(member(store.root, name), target)


def archive_snapshot(snapshot: RaceSnapshot, store: SourceStore, output_root: Path,
                     release: Path | None = None) -> Path:
    snapshot.validate(store)
    generated = now()
    if snapshot.evidence_mode == "prospective":
        if utc(snapshot.cutoff) > utc(generated) or utc(generated) >= utc(snapshot.race["race_start"]):
            raise ValueError("A genuine prospective forecast must be generated before race start, after its cutoff")
        store.available(snapshot.history_source, snapshot.cutoff, 2592000)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", snapshot.snapshot_id):
        raise ValueError("Unsafe snapshot identifier")
    destination = output_root / snapshot.snapshot_id
    if destination.exists():
        raise FileExistsError("Forecasts cannot be overwritten; create a new snapshot revision")
    if snapshot.revision_of is not None:
        previous = member(output_root, snapshot.revision_of)
        inspect_archive(previous)
        prior = RaceSnapshot.from_dict(json.loads((previous / "snapshot.json").read_text()))
        if prior.race["race_id"] != snapshot.race["race_id"] or utc(prior.cutoff) > utc(snapshot.cutoff):
            raise ValueError("Snapshot revision must follow the same race chronologically")
    batch = predictions = None
    if snapshot.kind == "confirmed_grid":
        if release is None:
            raise ValueError("Confirmed-grid forecast requires approved release")
        batch = build_features(snapshot, store)
        predictions = FrozenPredictor(release).predict(batch)
    # Directory creation is exclusive. Interrupted archives lack a completion manifest.
    output_root.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    write_json(destination / "snapshot.json", snapshot.to_dict())
    _copy_sources(store, snapshot, destination / "sources")
    if batch is not None:
        features = {"columns": list(batch.frame.columns), "dtypes": batch.frame.dtypes.astype(str).to_dict(),
                    "records": json.loads(batch.frame.to_json(orient="records", double_precision=15)),
                    "missingness": batch.missingness, "history_race_ids": list(batch.history_race_ids)}
        write_json(destination / "features.json", features)
        write_json(destination / "predictions.json", predictions)
        root = Path(__file__).resolve().parents[3]
        definitions = json.loads((root / "data/processed/stage2a/feature_definitions.json").read_text())
        write_json(destination / "feature_lineage.json", [{"name": feature["name"], "definition": feature["definition"],
            "availability": ("completed_history" if index >= 42 else "pre_race_metadata" if index < 19 or index == 36 else "qualifying_or_grid"),
            "history_source": snapshot.history_source if index >= 42 else None,
            "source_observations": snapshot.sources if index < 42 else {},
            "cutoff": snapshot.cutoff, "evidence_mode": snapshot.evidence_mode}
            for index, feature in enumerate(definitions["features"])])
    completed = now()
    if snapshot.evidence_mode == "prospective" and utc(completed) >= utc(snapshot.race["race_start"]):
        raise ValueError("Race started during prediction; incomplete archive cannot be published")
    root = Path(__file__).resolve().parents[3]
    source_files = sorted((root / "src/f1_predictor/stage3").glob("*.py")) + [root / "src/f1_predictor/stage2a.py"]
    manifest = {"archive_version": 1, "snapshot_id": snapshot.snapshot_id,
        "status": "complete", "generated_at": generated, "completed_at": completed,
        "cutoff": snapshot.cutoff, "snapshot_kind": snapshot.kind, "evidence_mode": snapshot.evidence_mode,
        "prediction_status": "predicted" if predictions else "captured_no_compatible_model",
        "feature_schema_sha256": digest_json(FEATURE_COLUMNS) if batch is not None else None,
        "release_manifest_sha256": sha256(release / "manifest.json") if predictions else None,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "code_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in source_files},
        "files_sha256": {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
    write_json(destination / "manifest.json", manifest)
    return destination


def inspect_archive(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete archive")
    for name, digest in manifest["files_sha256"].items():
        checked(member(directory, name), digest)
    return manifest


def replay_archive(directory: Path, release: Path) -> dict[str, Any]:
    manifest = inspect_archive(directory)
    root = Path(__file__).resolve().parents[3]
    for name, digest in manifest["code_sha256"].items():
        checked(member(root, name), digest)
    if manifest["prediction_status"] != "predicted":
        return {"status": "capture_only", "inference_performed": False}
    checked(release / "manifest.json", manifest["release_manifest_sha256"])
    snapshot = RaceSnapshot.from_dict(json.loads((directory / "snapshot.json").read_text()))
    batch = build_features(snapshot, SourceStore(directory / "sources"))
    saved = json.loads((directory / "features.json").read_text())
    expected = pd.DataFrame(saved["records"], columns=saved["columns"]).astype(saved["dtypes"])
    pd.testing.assert_frame_equal(batch.frame, expected, check_exact=True)
    actual = FrozenPredictor(release).predict(batch)
    expected_predictions = json.loads((directory / "predictions.json").read_text())
    if actual != expected_predictions:
        raise ValueError("Archived predictions do not reproduce exactly")
    return {"status": "exact_replay", "snapshot_id": snapshot.snapshot_id, "network_used": False}


def attach_results(directory: Path, store: SourceStore, observation: str, completed_at: str,
                   results: dict[str, Any], attachment_root: Path) -> Path:
    """Attach reviewed outcomes outside the immutable prediction directory."""
    manifest = inspect_archive(directory)
    snapshot = RaceSnapshot.from_dict(json.loads((directory / "snapshot.json").read_text()))
    source, raw = store.read(observation)
    if source["provider"] not in ("jolpica", "fia", "f1db"):
        raise ValueError("Unsupported results source")
    if results.get("race_id") != snapshot.race["race_id"]:
        raise ValueError("Result race differs from forecast")
    if utc(completed_at) > utc(source["retrieved_at"]) or utc(completed_at) <= utc(snapshot.cutoff):
        raise ValueError("Invalid race completion timestamp")
    if attachment_root.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Results must be attached outside the sealed forecast")
    attachment = member(attachment_root, manifest["snapshot_id"]) / source["sha256"]
    attachment.mkdir(parents=True, exist_ok=False)
    with (attachment / "source.blob").open("xb") as stream:
        stream.write(raw)
    write_json(attachment / "results.json", {"race_id": snapshot.race["race_id"], "completed_at": completed_at,
        "source": source, "results": results, "forecast_manifest_sha256": sha256(directory / "manifest.json")})
    return attachment
