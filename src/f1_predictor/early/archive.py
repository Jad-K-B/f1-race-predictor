"""Immutable private early captures; a capture is never a model forecast."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import subprocess

import pandas as pd

from ..stage3.archive import evidence_closure
from ..stage3.release import checked, sha256, write_json
from ..stage3.sources import SourceStore, member, now, utc
from .contracts import EarlySnapshot
from .live_features import LIVE_FEATURE_SCHEMA, build_live_features
from .policy import CANDIDATE_FEATURES

CAPTURE_SCHEMA = "early-private-capture-v1"
PRIVATE_ROOT = Path(__file__).resolve().parents[3] / "data/stage3/evidence-private"


def _copy_evidence(store: SourceStore, references: set[str], destination: Path) -> None:
    for reference in sorted(evidence_closure(store, references)):
        record, _ = store.read(reference)
        for name in (reference, record["blob"]):
            target = member(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(member(store.root, name), target)


def capture_early_snapshot(snapshot: EarlySnapshot, store: SourceStore,
                           output_root: Path) -> Path:
    """Capture evidence and features before FP1, without a prediction claim."""
    if not output_root.resolve().is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError("Early source and feature captures must remain private")
    snapshot.validate(store)
    generated = now()
    if not utc(snapshot.cutoff) <= utc(generated) < utc(snapshot.race["fp1_start"]):
        raise ValueError("Early capture must happen after cutoff and before FP1")
    batch = build_live_features(snapshot, store)
    destination = member(output_root, snapshot.snapshot_id)
    if destination.exists():
        raise FileExistsError("Early captures are immutable; use a new snapshot ID")
    output_root.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    write_json(destination / "snapshot.json", snapshot.to_dict())
    _copy_evidence(store, set(snapshot.sources.values()), destination / "sources")
    write_json(destination / "features.json", {
        "schema_version": LIVE_FEATURE_SCHEMA,
        "columns": list(batch.frame.columns),
        "dtypes": batch.frame.dtypes.astype(str).to_dict(),
        "records": json.loads(batch.frame.to_json(orient="records", double_precision=15)),
        "missingness": batch.missingness,
        "historical_race_ids": list(batch.historical_race_ids),
    })
    completed = now()
    if utc(completed) >= utc(snapshot.race["fp1_start"]):
        raise ValueError("FP1 began during early capture; incomplete archive is unusable")
    root = Path(__file__).resolve().parents[3]
    code = [root / name for name in (
        "src/f1_predictor/early/contracts.py",
        "src/f1_predictor/early/policy.py",
        "src/f1_predictor/early/live_features.py",
        "src/f1_predictor/early/archive.py",
        "src/f1_predictor/stage3/sources.py",
        "src/f1_predictor/stage3/features.py",
        "src/f1_predictor/stage2a.py",
    )]
    manifest = {
        "schema_version": CAPTURE_SCHEMA, "snapshot_id": snapshot.snapshot_id,
        "status": "complete", "prediction_status": "capture_only_no_model_output",
        "evidence_mode": "prospective", "cutoff": snapshot.cutoff,
        "generated_at": generated, "completed_at": completed,
        "feature_order": list(CANDIDATE_FEATURES),
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "code_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in code},
        "files_sha256": {
            path.relative_to(destination).as_posix(): sha256(path)
            for path in sorted(destination.rglob("*")) if path.is_file()
        },
    }
    write_json(destination / "manifest.json", manifest)
    return destination


def inspect_early_capture(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != CAPTURE_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("prediction_status") != "capture_only_no_model_output"
            or manifest.get("evidence_mode") != "prospective"
            or manifest.get("feature_order") != list(CANDIDATE_FEATURES)):
        raise ValueError("Invalid or incomplete early capture")
    for name, digest in manifest["files_sha256"].items():
        checked(member(directory, name), digest)
    return manifest


def replay_early_capture(directory: Path) -> dict:
    """Verify code, copied evidence, and exact unlabeled feature reproduction."""
    manifest = inspect_early_capture(directory)
    root = Path(__file__).resolve().parents[3]
    for name, digest in manifest["code_sha256"].items():
        checked(member(root, name), digest)
    snapshot = EarlySnapshot.from_dict(json.loads((directory / "snapshot.json").read_text()))
    batch = build_live_features(snapshot, SourceStore(directory / "sources"))
    saved = json.loads((directory / "features.json").read_text())
    if saved["schema_version"] != LIVE_FEATURE_SCHEMA:
        raise ValueError("Early feature schema version changed")
    expected = pd.DataFrame(saved["records"], columns=saved["columns"]).astype(saved["dtypes"])
    pd.testing.assert_frame_equal(batch.frame, expected, check_exact=True)
    if (batch.missingness != saved["missingness"] or
            list(batch.historical_race_ids) != saved["historical_race_ids"]):
        raise ValueError("Early missingness or history differs from archive")
    return {"status": "exact_capture_replay", "snapshot_id": snapshot.snapshot_id,
            "prediction_performed": False, "network_used": False}
