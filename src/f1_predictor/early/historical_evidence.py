"""Assess claimed historical publication timing, never live source eligibility.

Recovered pages may have been revised. A publication window is a reviewed
claim about timing, not proof that today's bytes existed at that time.
"""

from datetime import datetime, timezone
from typing import Literal


Timing = Literal["before_or_at_cutoff", "after_cutoff", "ambiguous", "unknown"]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timezone-aware timestamps are required")
    return value.astimezone(timezone.utc)


def publication_timing(
    cutoff: datetime,
    earliest: datetime | None,
    latest: datetime | None,
) -> Timing:
    """Classify a closed uncertainty interval; never infer an unknown timezone.

For an exact timestamp, pass the same value twice. For day-only evidence,
the caller must establish the timezone and supply conservative day bounds.
Unknown publication time uses two None values.
"""
    cutoff = _utc(cutoff)
    if earliest is None and latest is None:
        return "unknown"
    if earliest is None or latest is None:
        raise ValueError("Both publication bounds are required")
    earliest, latest = _utc(earliest), _utc(latest)
    if earliest > latest:
        raise ValueError("Publication bounds are reversed")
    if latest <= cutoff:
        return "before_or_at_cutoff"
    if earliest > cutoff:
        return "after_cutoff"
    return "ambiguous"


def assess_recovered_timing(
    *, cutoff: datetime, retrieved_at: datetime,
    earliest: datetime | None, latest: datetime | None,
) -> dict:
    """Return research metadata only; this cannot authorize model inputs."""
    cutoff, retrieved_at = _utc(cutoff), _utc(retrieved_at)
    timing = publication_timing(cutoff, earliest, latest)
    if latest is not None and _utc(latest) > retrieved_at:
        raise ValueError("Publication window extends beyond retrieval")
    return {
        "mode": "retrospective_source_review",
        "claimed_publication_timing": timing,
        "retrieved_after_cutoff": retrieved_at > cutoff,
        "point_in_time_bytes_verified": False,
        "authorizes_forecast_inputs": False,
    }
