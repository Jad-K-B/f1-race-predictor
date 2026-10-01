import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { validateForecast, validateReceipt } from "../src/forecast-contract.ts";
import {
  prepare,
  checkedArchive,
  verifyPublication,
  verifySavedReceipt,
  hash,
  jsonBytes,
  writeExclusive,
} from "../scripts/forecast-package.mjs";
import { loadCatalog } from "../scripts/catalog.mjs";
import { contractFixture } from "./forecast-fixture.mjs";

const archive = "../artifacts/stage3/forecasts/replay-1080";
async function privateArchiveAvailable() {
  try {
    await readFile(`${archive}/manifest.json`);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}
test("real private rehearsal preserves every saved method without private inputs", async (t) => {
  if (!(await privateArchiveAvailable())) {
    t.skip("private replay archive is not part of the public source export");
    return;
  }
  const original = JSON.parse(await readFile(`${archive}/predictions.json`));
  const actual = await prepare(archive, { rehearsal: true });
  assert.deepEqual(actual.models, original.models);
  assert.deepEqual(actual.baselines, original.baselines);
  assert.equal(actual.evidence_mode, "approximate_retrospective");
  assert.doesNotMatch(
    JSON.stringify(actual),
    /reviewed_by|missingness|q1_millis|date_of_birth|first_observed_at|features\.json/,
  );
  await assert.rejects(prepare(archive), /retrospective/);
});
test("3, 20 and 22-driver fields validate without fixed roster assumptions", () => {
  for (const n of [3, 20, 22]) validateForecast(contractFixture(n));
});
for (const [name, mutate] of [
  ["duplicate entrants", (f) => f.entrants.push(f.entrants[0])],
  ["missing method", (f) => delete f.baselines.recent_form],
  ["wrong model release", (f) => (f.release_manifest_sha256 = "a".repeat(64))],
  ["late generation", (f) => (f.completed_at = f.race.race_start)],
  ["ambiguous timezone", (f) => (f.cutoff = "2026-10-04T05:00:00")],
  ["raw evidence field", (f) => (f.features = [])],
  [
    "hidden diagnostic data",
    (f) => (f.models.external.diagnostics = { raw_features: [] }),
  ],
  [
    "NaN probability",
    (f) => (f.models.numpy.drivers[0].raw.p_race_winner = NaN),
  ],
  [
    "fabricated baseline probability",
    (f) =>
      (f.baselines.final_grid.drivers[0].reconciled =
        f.models.numpy.drivers[0].reconciled),
  ],
  [
    "wrong prediction identity",
    (f) => (f.models.numpy.drivers[0].driver_id = "wrong-driver"),
  ],
  [
    "broken quota",
    (f) => (f.models.numpy.drivers[0].reconciled.p_race_winner = 0),
  ],
])
  test(`rejects ${name}`, () => {
    const f = contractFixture();
    mutate(f);
    assert.throws(() => validateForecast(f));
  });

async function mockReceipt({ wrongBytes = false, late = false } = {}) {
  const forecast = contractFixture();
  forecast.notice = "Mock network verification fixture; never uploaded.";
  const bytes = jsonBytes(forecast),
    updated = late ? forecast.race.race_start : "2026-10-04T05:05:00Z";
  const api = {
    id: 123,
    draft: false,
    published_at: "2026-10-04T05:04:00Z",
    assets: [
      {
        id: 456,
        state: "uploaded",
        name: `${forecast.snapshot_id}.json`,
        updated_at: updated,
        browser_download_url: `https://github.com/Jad-K-B/f1-race-forecasts/releases/download/test/${forecast.snapshot_id}.json`,
      },
    ],
  };
  const calls = [];
  const result = await verifyPublication(bytes, 123, {
    now: "2026-10-04T05:06:00Z",
    fetcher: async (url) => {
      calls.push(url);
      return new Response(
        calls.length === 1 ? jsonBytes(api) : wrongBytes ? "changed" : bytes,
        { status: 200, headers: { date: "Sun, 04 Oct 2026 05:06:00 GMT" } },
      );
    },
  });
  return { ...result, forecast, bytes, calls };
}
test("receipt verification reads approved API once and hashes downloaded bytes", async () => {
  const x = await mockReceipt();
  assert.equal(x.calls.length, 2);
  verifySavedReceipt(x.forecast, x.receipt, x.bytes, x.apiBytes);
  assert.throws(() =>
    verifySavedReceipt(x.forecast, x.receipt, x.bytes, Buffer.from("{}")),
  );
  const altered = { ...x.receipt, payload_sha256: "a".repeat(64) };
  assert.throws(() => validateReceipt(x.forecast, altered, hash(x.bytes)));
  assert.throws(() =>
    validateReceipt(
      { ...x.forecast, evidence_mode: "approximate_retrospective" },
      x.receipt,
      hash(x.bytes),
    ),
  );
});
test("late assets and mismatched downloads cannot become public proof", async () => {
  await assert.rejects(mockReceipt({ wrongBytes: true }), /bytes differ/);
  await assert.rejects(mockReceipt({ late: true }), /late/);
});
test("public catalog excludes private replay and refuses unreceipted forecasts", async () => {
  const empty = await loadCatalog(process.cwd(), { publicBuild: true });
  assert.deepEqual(empty.events, []);
  assert.deepEqual(empty.benchmark.probability, []);
  const temp = await mkdtemp(path.join(os.tmpdir(), "formation-contract-"));
  await writeFile(path.join(temp, "f.json"), jsonBytes(contractFixture()));
  await writeFile(
    path.join(temp, "catalog.json"),
    JSON.stringify([{ file: "f.json" }]),
  );
  await assert.rejects(
    loadCatalog(process.cwd(), {
      publicBuild: true,
      catalogPath: path.join(temp, "catalog.json"),
    }),
    /publication/,
  );
  if (await privateArchiveAvailable()) {
    const local = await loadCatalog(process.cwd(), {
      catalogPath: path.join(temp, "catalog.json"),
    });
    assert.equal(local.events.at(-1).receipt, null);
  }
});
test("packaging never overwrites existing artifacts", async () => {
  const temp = await mkdtemp(path.join(os.tmpdir(), "formation-exclusive-"));
  await writeExclusive(temp, "test.json", "first");
  await assert.rejects(writeExclusive(temp, "test.json", "second"), /EEXIST/);
  assert.equal(await readFile(path.join(temp, "test.json"), "utf8"), "first");
});

test("archive packaging rejects changed bytes and unsafe member paths", async () => {
  const temp = await mkdtemp(path.join(os.tmpdir(), "formation-archive-"));
  const manifest = {
    status: "complete",
    prediction_status: "predicted",
    snapshot_kind: "confirmed_grid",
    files_sha256: { "snapshot.json": hash(Buffer.from("original")) },
  };
  await writeFile(path.join(temp, "manifest.json"), jsonBytes(manifest));
  await writeFile(path.join(temp, "snapshot.json"), "changed");
  await assert.rejects(checkedArchive(temp), /hash mismatch/);
  manifest.files_sha256 = { "../outside.json": "a".repeat(64) };
  await writeFile(path.join(temp, "manifest.json"), jsonBytes(manifest));
  await assert.rejects(checkedArchive(temp), /Unsafe archive member/);
});

test("public catalog requires matching saved receipt and rejects marked test packages", async () => {
  const x = await mockReceipt();
  const temp = await mkdtemp(path.join(os.tmpdir(), "formation-public-"));
  const catalogPath = path.join(temp, "catalog.json");
  await writeFile(path.join(temp, "f.json"), x.bytes);
  await writeFile(path.join(temp, "receipt.json"), jsonBytes(x.receipt));
  await writeFile(path.join(temp, "server.json"), x.apiBytes);
  await writeFile(
    catalogPath,
    jsonBytes([
      {
        file: "f.json",
        receipt: "receipt.json",
        server_response: "server.json",
      },
    ]),
  );
  const catalog = await loadCatalog(process.cwd(), {
    publicBuild: true,
    catalogPath,
  });
  assert.equal(catalog.events.length, 1);
  assert.equal(catalog.readiness.forecast_published, true);
  assert.deepEqual(catalog.events[0].forecast.models, x.forecast.models);
  const f = { ...x.forecast, notice: "CONTRACT TEST ONLY" };
  const bytes = jsonBytes(f);
  const receipt = { ...x.receipt, payload_sha256: hash(bytes) };
  await writeFile(path.join(temp, "f.json"), bytes);
  await writeFile(path.join(temp, "receipt.json"), jsonBytes(receipt));
  await assert.rejects(
    loadCatalog(process.cwd(), { publicBuild: true, catalogPath }),
    /Test packages are private/,
  );
  await writeFile(path.join(temp, "server.json"), "{}");
  await assert.rejects(
    loadCatalog(process.cwd(), { publicBuild: true, catalogPath }),
    /Server response hash mismatch/,
  );
});
