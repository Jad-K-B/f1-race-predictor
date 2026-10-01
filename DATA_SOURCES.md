# Data provenance and redistribution boundary

The 11 retained historical CSV tables in `data/raw/` come from [F1DB](https://github.com/f1db/f1db),
which publishes its database under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit belongs to the F1DB contributors. `data/processed/race_data.csv` and the
Stage 2A split tables are project-derived transformations; their feature definitions,
eligibility audit, source-file hashes and split metadata are in
`data/processed/stage2a/`. The exact upstream release tag of the original
Stage 1 CSV snapshot was not recorded, so the source hashes, not an invented
release version, identify those bytes.

The prospective adapters can consult Jolpica, OpenF1, versioned F1DB releases
and reviewed FIA material. This public source export contains no downloaded FIA
documents, live source cache, private evidence bundle or genuine forecast archive.
Source links in code and documentation do not imply permission to redistribute
third-party event documents or imagery. The frontend's separate asset notes are
in `app/THIRD_PARTY.md`.
