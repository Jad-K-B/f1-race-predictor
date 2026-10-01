"""Live source checks report blockers, not invented future-race inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .sources import SourceStore, Jolpica, OpenF1, now, utc
from .release import write_json, verify_release


def readiness(store: SourceStore, raw_dir: Path, release: Path, year: int, output: Path) -> dict[str, Any]:
    instant = now()
    report: dict[str, Any] = {"checked_at": instant, "status": "blocked", "forecast_generated": False,
        "checks": {}, "blockers": [], "source_observations": []}
    try:
        report["checks"]["release"] = verify_release(release)["release_version"]
    except (ValueError, FileNotFoundError) as error:
        report["blockers"].append(f"Release verification: {error}")
    upcoming = None
    sessions = []
    try:
        pages, observations = Jolpica(store).fetch(year)
        report["source_observations"] += observations
        races = [r for page in pages for r in page["MRData"]["RaceTable"]["Races"]]
        eligible = [r for r in races if r.get("time") and utc(r["date"] + "T" + r["time"]) > utc(instant)]
        upcoming = min(eligible, key=lambda r: utc(r["date"] + "T" + r["time"])) if eligible else None
        report["checks"]["jolpica_schedule"] = {"race_count": len(races), "upcoming": upcoming}
        if upcoming is None:
            report["blockers"].append("No upcoming timed race in requested Jolpica season")
    except Exception as error:
        report["blockers"].append(f"Jolpica unavailable: {type(error).__name__}: {error}")
    try:
        sessions, observation = OpenF1(store).fetch("sessions", year=year)
        report["source_observations"].append(observation)
        future = [s for s in sessions if s["session_name"] == "Race" and not s.get("is_cancelled", False) and utc(s["date_start"]) > utc(instant)]
        next_session = min(future, key=lambda s: utc(s["date_start"])) if future else None
        report["checks"]["openf1_sessions"] = {"count": len(sessions), "upcoming": next_session}
        if upcoming and next_session and utc(upcoming["date"] + "T" + upcoming["time"]) != utc(next_session["date_start"]):
            report["blockers"].append("Schedule conflict: Jolpica and OpenF1 disagree on next race; authoritative calendar review required")
        if next_session:
            for endpoint, session in (("starting_grid", next_session), ("session_result", next((s for s in sessions if s["meeting_key"] == next_session["meeting_key"] and s["session_name"] == "Qualifying"), None))):
                if session is None:
                    continue
                try:
                    data, ref = OpenF1(store).fetch(endpoint, session_key=session["session_key"])
                    report["source_observations"].append(ref)
                    report["checks"][endpoint] = {"rows": len(data), "session_key": session["session_key"]}
                    if not data:
                        report["blockers"].append(f"Upcoming {endpoint} is not published")
                except Exception as error:
                    report["blockers"].append(f"OpenF1 {endpoint}: {type(error).__name__}: {error}")
    except Exception as error:
        report["blockers"].append(f"OpenF1 unavailable: {type(error).__name__}: {error}")
    race_table = pd.read_csv(raw_dir / "f1db-races.csv")
    results = pd.read_csv(raw_dir / "f1db-races-race-results.csv", usecols=["raceId"])
    completed = race_table[race_table.id.isin(results.raceId)].sort_values("date")
    report["checks"]["local_f1db_latest_classification"] = completed.iloc[-1][["id", "year", "round", "date", "grandPrixId"]].to_dict()
    ended_dates = {utc(s["date_start"]).date() for s in sessions if s["session_name"] == "Race"
                   and not s.get("is_cancelled", False) and utc(s["date_end"]) < utc(instant)}
    dates = pd.to_datetime(race_table.date).dt.date
    past = race_table[(race_table.year == year) & ((dates < utc(instant).date()) | dates.isin(ended_dates))]
    missing = past[~past.id.isin(results.raceId)]
    report["checks"]["earlier_races_missing_classification"] = missing[["id", "date", "grandPrixId"]].to_dict("records")
    if len(missing):
        report["blockers"].append("Local F1DB history is incomplete; update or explicitly review cancelled events")
    if upcoming:
        matching = race_table[race_table.date.eq(upcoming["date"]) & race_table.year.eq(year)]
        report["checks"]["local_race_date_candidates"] = matching[["id", "date", "grandPrixId", "circuitId"]].to_dict("records")
        if len(matching) != 1:
            report["blockers"].append("Upcoming API race has no unique local metadata match; do not guess canonical IDs")
    report["checked_completed_at"] = now()
    report["checks"]["prospective_source_observations_verified"] = all(
        store.available(ref, report["checked_completed_at"], 86400) is not None for ref in report["source_observations"])
    report["blockers"] += ["Reviewed FIA entry list, final grid/amendments and latest withdrawal/penalty review not supplied",
                            "Reviewed canonical race/driver/constructor mapping and complete snapshot packet not supplied"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # pandas scalars in to_dict are converted to native JSON types.
    write_json(output, report)
    return report
