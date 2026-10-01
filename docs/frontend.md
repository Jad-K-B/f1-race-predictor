# FORMATION frontend

The React/TypeScript application in `app/` is a read-only presentation layer.
Three.js renders the racing concept and GSAP controls its scene motion. It does
not fetch live race data, fit models or run Python inference in the browser.
Saved historical evaluation, approximate replay and genuine published forecasts
have distinct labels and data contracts. The current public build shows a
pending upcoming forecast until a receipt-verified publication exists.

From `app/`, use Node 24 and pnpm 11.19.0:

```powershell
pnpm install --frozen-lockfile
pnpm test
pnpm build:public
node tests/public-browser.mjs
```

The public source omits the historical replay fixture and third-party identity
images; it includes the local car, fallback posters, fonts and license notices.
Its auditor checks the allowed output files, hashes and absence of private model
or feature payloads. `pnpm dev` shows the pending state without private data;
only `pnpm build:public` creates the deployable build. Vite may warn about its separate Three.js chunk and
font resolution; the public auditor verifies that the font bytes were emitted.

Forecast cards show saved predicted rank, team and available probabilities.
Order-only baselines do not acquire invented probabilities. Starting positions
are unavailable in the limited public export, so the UI leaves them unavailable
rather than guessing. The race 1080 replay is historical and approximate, not
a forecast published before that race; 2024-2025 model-lab results are separate
held-out evaluations. Winner, podium and points probabilities are marginals,
not a full joint race simulation.

`app/scripts/forecast-package.mjs` validates a prospective archive and creates
one public JSON asset containing every frozen method. Its `receipt` command
checks GitHub's server metadata and downloaded bytes. A private catalog can
then supply a receipt-verified forecast to `pnpm build:public`; without one the
page stays pending. `app/scripts/score-prospective.mjs` reads a separately
reviewed post-race outcome attachment and writes a new score file without
rewriting the forecast. These commands do not collect evidence or publish by
themselves. The deployed website and forecast-only repository remain separate
from this source export.

The 3D concept, fonts and libraries have their own terms and attribution in
[asset credits](../app/THIRD_PARTY.md). Driver portraits and team logos in the
private local design review are not bundled here because public redistribution
rights were not established. The season-aware lookup remains in code, but no
third-party identity registry or images are bundled in this public export.
