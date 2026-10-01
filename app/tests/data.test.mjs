import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { loadCatalog } from "../scripts/catalog.mjs";

const root = new URL("../", import.meta.url);
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");

test("public source opens with no invented forecast or benchmark", async () => {
  const catalog = await loadCatalog(new URL(".", root).pathname);
  assert.deepEqual(catalog.events, []);
  assert.deepEqual(catalog.benchmark, {
    probability: [], ranking: [], calibration: [], uncertainty: [],
  });
  assert.equal(catalog.readiness.forecast_published, false);
});

test("optimized car matches licensed provenance and transfer budget", async () => {
  const asset = await readFile(new URL("public/assets/formation-car.glb", root));
  const provenance = JSON.parse(
    await readFile(new URL("public/assets/car-provenance.json", root)),
  );
  assert.equal(hash(asset), provenance.output_sha256);
  assert.ok(asset.length < 2_000_000);
  assert.equal(asset.toString("ascii", 0, 4), "glTF");
  assert.equal(provenance.license.slug, "by");
});

test("frontend does not run models or fetch upstream race data", async () => {
  const app = await readFile(new URL("src/App.tsx", root), "utf8");
  assert.doesNotMatch(app, /fetch\(|\.fit\(|\.predict\(|api\.openf1|api\.jolpi/);
});
