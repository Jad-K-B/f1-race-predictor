# Historical data and models

This project builds a post-qualifying, pre-race dataset from F1DB race records.
The historical CSVs, attribution and redistribution boundary are described in
[Data sources](../DATA_SOURCES.md). The exact upstream tag of the original Stage 1
snapshot was not recorded; source hashes in the Stage 2A metadata identify its
bytes. The exploratory notebook is in `notebooks/data_exploration.ipynb`.

## Dataset and targets

`scripts/build_stage2a.py` produces one row per eligible driver and race for
2014-2025, with 147 ordered pre-race features. Feature definitions, source hashes,
missingness, eligibility decisions and split metadata are saved in
`data/processed/stage2a/`. The immutable split is 2014-2022 training, 2023
validation and 2024-2025 final test, grouped by race.

- `points_finish`: official race-result points greater than zero, not simply a
  top-ten classification.
- `podium_finish`: official numeric finishing position 1-3.
- `race_winner`: official numeric finishing position 1.
- `finish_order`: official display order re-ranked among drivers eligible at the
  prediction timestamp. Eligible DNF, DSQ, NC and DNS entries remain in the
  ranking; pre-start withdrawals do not.

A missing grid row never proves ineligibility. Six historical exclusions were
reviewed individually in `config/stage2a_eligibility_audit.json`; the builder
fails on a new unaudited case. Pit-lane starters remain eligible. Driver,
constructor, circuit and pairing histories are shifted before rolling statistics;
standings come from the previous race. Current-race outcomes and F1DB all-time
summary fields are excluded from model inputs. A later race may use results from
an earlier completed race in the same split, but never its own or future result.

## Training and evaluation

Preprocessing, including numeric medians, scaling, missing indicators and
category vocabularies, is fitted on training rows only. Unknown categories have
a defined fallback. Five expanding train-era folds end in 2018-2022. The 2023
races are divided chronologically into calibration (8), method selection (7)
and an untouched validation report block (7). The 2024-2025 test split was
opened once after configuration was frozen; it was not used to select models.

The project retains a NumPy pipeline and a pinned scikit-learn/XGBoost comparison
covering regularized logistic regression, Random Forest, XGBoost classifiers
and a race-grouped XGBoost ranker. Grid, qualifying, recent-form, uniform-quota
and training-grid-frequency baselines are reported alongside them. Selected
calibration and reconciliation settings are saved, not chosen at inference time.
Reconciled race marginals sum to 1 winner, 3 podium and 10 points finishers and
obey `P(win) <= P(podium) <= P(points)`. The ten-finisher assumption must be
revisited if scoring or classification rules change.

The last-seven-race 2023 validation block was unusually homogeneous: all seven
winners were Max Verstappen. Its NumPy winner log loss of 0.024 was verified, but
did not generalize to the final test. Pooled 2024-2025 log loss was:

| Method | Points | Podium | Winner |
| --- | ---: | ---: | ---: |
| External selected | 0.476 | 0.251 | 0.224 |
| NumPy selected | 0.440 | 0.246 | 0.144 |
| Final-grid rule | 0.684 | 0.482 | 0.129 |
| Train grid-frequency | 0.471 | 0.219 | 0.107 |

The recent-form baseline also had the lowest pooled finishing-order MAE (3.058,
versus 3.123 NumPy and 3.136 external). Differences in ranking were small against
race-bootstrap uncertainty. Compact aggregate validation and final-test metrics
remain in `artifacts/stage2b/`. Per-driver predictions, detailed research
reports and the frozen serving release remain in the private operational project.
These are historical results, not genuine forecasts issued before those races.

## Reproduction boundary

With Python 3.12 and the pinned requirements installed in a local environment:

```powershell
python -m pip install -r requirements/stage2a.txt
python scripts/build_stage2a.py
python -m unittest discover -s tests -p 'test_*.py'
```

`requirements/stage2b.txt`, `requirements/stage2b_external.txt` and
`requirements/stage3.txt` contain the other stage-specific pins. The original
root `requirements-stage2b-external.txt` is intentionally retained byte-for-byte:
the private frozen release verifies that exact path and SHA-256. The folder
copy has the same dependency pins for new installs; it is not a replacement
for the protected provenance file.

The NumPy and external training entry points are `scripts/train_stage2b.py` and
`scripts/train_stage2b_external.py`. Private saved manifests contain source
hashes, candidate settings, runtime versions and seeds. The external runner uses only
the train and validation splits. The guarded final-test evaluator is retained
for audit, not as a routine reproduction step: do not rerun or overwrite the
sealed evaluation. The original F1DB release tag cannot be recovered from these
files alone, and historical source revisions limit exact point-in-time claims.
