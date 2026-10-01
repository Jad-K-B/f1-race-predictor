"""Versioned pre-practice boundary, independent of the confirmed-grid schema."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from ..stage2a import FEATURE_COLUMNS, _feature_definitions
from ..stage3.sources import utc

POLICY_VERSION = "early-pre-practice-v1"
LEAD_HOURS = 24
EXCLUDED_FEATURES = frozenset({
    "qualifying_position", "qualifying_position_normalized", "qualifying_laps",
    "reached_q2", "reached_q3", "q1_pct_off_best", "q2_pct_off_best", "q3_pct_off_best",
    "final_grid_position", "effective_grid_position", "grid_position_normalized",
    "grid_delta_from_qualifying", "grid_penalty_positions", "has_grid_penalty",
    "pit_lane_start", "missing_final_grid", "grid_field_size",
    "current_constructor_avg_grid", "current_constructor_best_grid",
    "current_constructor_avg_qualifying", "grid_delta_to_constructor_avg",
    "qualifying_delta_to_constructor_avg",
})
REBUILT_FEATURES = frozenset({"current_constructor_entry_count"})
CANDIDATE_FEATURES = tuple(name for name in FEATURE_COLUMNS if name not in EXCLUDED_FEATURES)


def scheduled_cutoff(fp1_start: str) -> str:
    return (utc(fp1_start) - timedelta(hours=LEAD_HOURS)).isoformat()


def validate_feature_names(names: Iterable[str]) -> None:
    """Boundary check only: this does not establish the provenance of values."""
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Early feature names must be nonempty and unique")
    forbidden = set(names) - set(CANDIDATE_FEATURES)
    if forbidden:
        raise ValueError(f"Unavailable or unknown early features: {sorted(forbidden)}")


def feature_lineage() -> list[dict[str, str]]:
    rows = []
    for original in _feature_definitions():
        name = original["name"]
        scope = original["temporal_scope"]
        if name in EXCLUDED_FEATURES:
            decision, source = "exclude", "none"
            requirement = "Current qualifying/grid information is unavailable at the early cutoff."
        elif name in REBUILT_FEATURES:
            decision, source = "rebuild", "announced_roster"
            requirement = "Count announced entries at cutoff, never final starters or result rows."
        else:
            decision = "candidate_requires_evidence"
            source = ({"known_before_event": "schedule",
                       "static_or_race_date_derived": "registry",
                       "known_before_race_start": "announced_roster"}.get(scope, "history"))
            requirement = {
                "schedule": "Use event configuration observed before cutoff, not later corrections.",
                "registry": "Use a version available by cutoff; unknown values remain missing.",
                "announced_roster": "Use dated driver/team announcements; no result or session identity fallback.",
                "history": "Use only completed prior races and revisions available by cutoff; audit input vintages.",
            }[source]
        rows.append({"name": name, "type": original["role"],
                     "frozen_definition": original["definition"], "frozen_scope": scope,
                     "early_decision": decision, "source_role": source,
                     "availability_requirement": requirement})
    return rows


def policy_manifest() -> dict:
    return {
        "version": POLICY_VERSION, "stage": "early", "lead_hours_before_fp1": LEAD_HOURS,
        "cutoff_policy": "Exactly scheduled FP1 minus 24 hours; a missed cutoff is not silently moved.",
        "publication_deadline": "Before FP1 and before qualifying; late generation cannot backdate evidence.",
        "roster_policy": "Reviewed announced field as of cutoff, not guaranteed starters. Later withdrawals remain in the original forecast.",
        "roster_evidence": "Reviewed FIA event entry list, or season registration plus event-applicable official team announcements/amendments.",
        "unknown_roster_policy": "Block unresolved completeness, substitutions or conflicts. Missing qualifying/grid never excludes a driver.",
        "targets": {
            "race_winner": "Official classified race position 1.",
            "podium_finish": "Official classified race positions 1-3.",
            "points_finish": "Official race points greater than zero, not a finishing-position threshold.",
            "finish_order": "Official ordering where defined; retain DNF records, never invent ranks for unranked withdrawals.",
        },
        "later_withdrawals": "Keep original announced entrants; DNS/DNP without a scoring result are negative binary outcomes; report ranking coverage separately.",
        "future_substitutes": "Do not insert drivers learned after cutoff into earlier forecasts or hide coverage gaps.",
        "research_splits": {"training": [2014, 2022], "validation": [2023, 2023],
                            "validation_blocks": [8, 7, 7], "sealed_2024_2025": "Do not load or rerun; already opened, not a fresh holdout."},
        "validation": "Race-grouped expanding folds; train-only preprocessing; calibration/selection/report separated chronologically. Prospective races provide fresh evidence.",
        "candidate_features_are_not_a_model_schema": True,
        "production_feature_builder_implemented": False,
        "practice_weather_in_v1": False, "frozen_release_modified": False,
        "authority_review": "Named review is an attestation, not proof of correct document extraction. Unknown formats/conflicts require review.",
    }
