"""Stage 3 CLI. Forecasts and source observations are append-only."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_predictor.stage2a import load_eligibility_audit
from f1_predictor.stage3.archive import archive_snapshot, inspect_archive, replay_archive, attach_results, export_evidence, inspect_evidence
from f1_predictor.stage3.contracts import RaceSnapshot
from f1_predictor.stage3.sources import SourceStore, Jolpica, OpenF1, import_f1db, import_fia, now
from f1_predictor.stage3.release import write_json
from f1_predictor.stage3.replay import replay_snapshot
from f1_predictor.stage3.readiness import readiness
from f1_predictor.stage3.validation import validate_2023_replay
from f1_predictor.stage3.draft import draft_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/stage3/cache")
    parser.add_argument("--release", type=Path, default=ROOT / "artifacts/stage3/releases/frozen-v2")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch-jolpica")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--round", type=int)
    p.add_argument("--endpoint", default="")
    p = commands.add_parser("fetch-openf1")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--session-key", type=int)
    p = commands.add_parser("import-f1db")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--version", required=True)
    p = commands.add_parser("import-fia")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--published-at", help="Verified publication timestamp; omit when unknown")
    p.add_argument("--publication-basis", choices=["document", "provider_metadata"])
    p.add_argument("--reviewed-by", required=True)
    p.add_argument("--document-kind", required=True)
    p = commands.add_parser("review-source")
    p.add_argument("--source", required=True)
    p.add_argument("--review", type=Path, required=True, help="JSON: applicability, reviewed_by, notes, supporting_observations")
    p = commands.add_parser("export-evidence")
    p.add_argument("--references", type=Path, required=True, help="JSON list of observation IDs")
    p.add_argument("--destination", type=Path, required=True)
    p = commands.add_parser("inspect-evidence")
    p.add_argument("--directory", type=Path, required=True)
    p = commands.add_parser("capture")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/stage3/forecasts")
    p.add_argument("--cutoff-now", action="store_true", help="Use current UTC cutoff after all source observations have been collected")
    for command in ("inspect", "replay"):
        p = commands.add_parser(command)
        p.add_argument("--archive", type=Path, required=True)
    p = commands.add_parser("prepare-replay")
    p.add_argument("--race-id", type=int, required=True)
    p.add_argument("--history-source", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("draft-snapshot")
    p.add_argument("--race-id", type=int, required=True)
    p.add_argument("--history-source", required=True)
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--kind", choices=["pre_weekend", "provisional_grid", "confirmed_grid"], required=True)
    p.add_argument("--entries", type=Path, required=True, help="Reviewed canonical entrants; no inferred roster")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("validate-replay")
    p.add_argument("--history-source", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("readiness")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("attach-results")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--completed-at", required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/stage3/result_attachments")
    args = parser.parse_args()
    store = SourceStore(args.cache)
    if args.command == "fetch-jolpica":
        pages, sources = Jolpica(store).fetch(args.year, args.endpoint, args.round)
        result = {"pages": len(pages), "sources": sources}
    elif args.command == "fetch-openf1":
        filters = {k: v for k, v in {"year": args.year, "session_key": args.session_key}.items() if v is not None}
        data, source = OpenF1(store).fetch(args.endpoint, **filters)
        result = {"rows": len(data), "source": source}
    elif args.command == "import-f1db":
        result = {"history_source": import_f1db(store, args.raw_dir, args.version)}
    elif args.command == "import-fia":
        result = {"source": import_fia(store, args.file, args.url, args.published_at, args.reviewed_by, args.document_kind, args.publication_basis)}
    elif args.command == "review-source":
        result = {"source": store.review(args.source, **json.loads(args.review.read_text()))}
    elif args.command == "export-evidence":
        result = export_evidence(store, json.loads(args.references.read_text()), args.destination)
    elif args.command == "inspect-evidence":
        result = inspect_evidence(args.directory)
    elif args.command == "capture":
        snapshot = RaceSnapshot.from_dict(json.loads(args.snapshot.read_text()))
        if args.cutoff_now:
            if snapshot.evidence_mode != "prospective":
                raise ValueError("Cannot turn an approximate replay into a live snapshot")
            snapshot = replace(snapshot, cutoff=now())
        result = {"archive": str(archive_snapshot(snapshot, store, args.output_root, args.release))}
    elif args.command == "inspect":
        result = inspect_archive(args.archive)
    elif args.command == "replay":
        result = replay_archive(args.archive, args.release)
    elif args.command == "prepare-replay":
        snapshot = replay_snapshot(store, args.history_source, args.race_id, load_eligibility_audit())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, snapshot.to_dict())
        result = {"packet": str(args.output), "evidence_mode": snapshot.evidence_mode}
    elif args.command == "draft-snapshot":
        packet = draft_snapshot(store, args.history_source, args.race_id, args.snapshot_id, args.kind, json.loads(args.entries.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, packet)
        result = {"packet": str(args.output), "status": "draft_requires_authoritative_review"}
    elif args.command == "validate-replay":
        report = validate_2023_replay(ROOT, store, args.history_source, args.output)
        result = {"passed": report["passed"], "races": len(report["races"]), "report": str(args.output)}
    elif args.command == "readiness":
        result = readiness(store, args.raw_dir, args.release, args.year, args.output)
    else:
        result = {"attachment": str(attach_results(args.archive, store, args.source, args.completed_at,
                  json.loads(args.results.read_text()), args.output_root))}
    print(json.dumps(result, indent=2))
    return 2 if result.get("status") == "blocked" or result.get("passed") is False else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, URLError, TimeoutError) as error:
        print(f"Stage 3: {error}", file=sys.stderr)
        raise SystemExit(2)
