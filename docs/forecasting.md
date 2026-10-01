# Prospective forecasting

Stage 3 turns reviewed event entries and timestamped sources into unlabeled
features, then loads the frozen post-qualifying release from the private
operational checkout. That release is not included in this public source export,
so a fresh checkout cannot run the validated predictor. It does not fit or select models. The
observed Windows inference dependencies are pinned in
`requirements/stage3.txt`; the release also checks their runtime versions. The
release includes feature order, preprocessing, estimators, calibration and
reconciliation. A prior recovery candidate was not approved because the original
calibration bytes had not been serialized. The private V2 release is the approved
numerical reconstruction, not a claim of bit-identical unsaved thresholds.

## Evidence and prediction gates

Source adapters support versioned F1DB history, Jolpica, restricted OpenF1
queries and reviewed FIA imports. Immutable observations record hashes,
first-observed and retrieval times, known publication time, source identity and
revision links. Unknown publication time stays unknown. Cache hits do not count
as fresh retrievals, and a network failure never silently promotes old data.
Freshness ceilings are 24 hours for schedule/entries, 48 hours for qualifying,
one hour for grid/eligibility and 30 days for registry/history; freshness alone
does not prove completeness. The live builder refuses a missing prior scheduled
race or an unverified same-day result.

An event-specific, reviewed entrant list and eligibility decisions are required
before a pre-weekend information snapshot. That snapshot archives inputs only;
the frozen grid-dependent models do not produce an early forecast from null or
invented qualifying/grid values. A confirmed-grid forecast additionally requires
qualifying results, the authoritative final grid, all applicable amendments,
penalties and withdrawals, and a verified race-start time. A provisional grid is
capture-only. Missing grid data never implies that an entrant is ineligible.

The builder emits the exact 147 frozen feature columns, types and missingness
from completed pre-cutoff history. Rookies retain missing history; constructor
renames are not silently merged. `FrozenPredictor` returns every predetermined
NumPy/external method and baseline, raw/calibrated/reconciled marginals, ranking
and race-level consistency diagnostics. Its probabilities are separate marginals,
not a joint podium simulation. The highest winner probability can disagree with
the ranker's predicted winner; neither is retrospectively promoted based on the
opened historical test.

## Capture, replay and publication

Use `scripts/stage3.py` from a private operational checkout. Its `readiness`,
`import-f1db`, `fetch-jolpica`, `fetch-openf1`, `import-fia`, `draft-snapshot`,
`capture`, `inspect`, `replay` and `attach-results` subcommands keep source review,
inference and later outcomes separate. For a reviewed packet, the final local
gate is:

```powershell
python scripts/stage3.py capture --snapshot REVIEWED_PACKET.json --cutoff-now
python scripts/stage3.py inspect --archive PRIVATE_ARCHIVE
python scripts/stage3.py replay --archive PRIVATE_ARCHIVE
```

`--cutoff-now` does not fetch newer data or reset source timestamps. Archives
are exclusive-create and include the source closure, exact features, predictions
and code/model/schema hashes; results are attached separately after the race.
The example race 1080 replay is **approximate retrospective reconstruction**:
original publication-time evidence is incomplete, so it cannot prove a genuine
pre-race forecast. This public repository omits the private source closure and
live archive; some integration checks intentionally skip without them.

Public publication is a separate operation in
[Jad-K-B/f1-race-forecasts](https://github.com/Jad-K-B/f1-race-forecasts).
Only original prediction outputs, explanation, source links and hashes are
approved for that repository. Raw FIA/F1 documents, source payloads and feature
matrices stay private; third-party redistribution rights are not inferred from
API access. Jolpica's [data terms](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md)
also differ from its software license. A genuine forecast needs generation and
independently timestamped publication **before** the verified race start,
followed by a server receipt and downloaded-byte hash check. A Git author date
or historical replay is not publication proof. The frontend accepts only a
receipt-verified public export; missing evidence remains pending. No command
should publish a new forecast after the start or when timing is uncertain.

## Experimental Early Forecast

The separate early research path proposes a cutoff 24 hours before FP1 and 125
features without current-event qualifying, grid or outcomes. Its 2014-2023
comparison has incomplete historical publication-time vintages and a seven-race
2023 report block won entirely by one driver. The private research bundle and
live feature gate are **experimental**, not substitutes for the frozen
confirmed-grid model. The public feature builder validates a reviewed schedule,
announced field, source timestamps and completed history, but it does not publish
or authorize an early prediction. Detailed selection reports and live evidence
remain private; this code is not an authorized early production release.
