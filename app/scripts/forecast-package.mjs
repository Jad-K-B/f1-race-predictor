import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import {
  validateForecast,
  validateReceipt,
  requireValue,
  timestamp,
} from "../src/forecast-contract.ts";

export const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
export const jsonBytes = (value) =>
  Buffer.from(JSON.stringify(value, null, 2) + "\n");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
export async function checkedArchive(directory) {
  const bytes = await readFile(path.join(directory, "manifest.json"));
  const manifest = JSON.parse(bytes);
  requireValue(
    manifest.status === "complete" &&
      manifest.prediction_status === "predicted" &&
      manifest.snapshot_kind === "confirmed_grid",
    "Complete confirmed-grid archive required",
  );
  for (const [file, digest] of Object.entries(manifest.files_sha256)) {
    const resolved = path.resolve(directory, file);
    const relative = path.relative(path.resolve(directory), resolved);
    requireValue(
      relative && !relative.startsWith("..") && !path.isAbsolute(relative),
      "Unsafe archive member",
    );
    requireValue(
      hash(await readFile(resolved)) === digest,
      `Archive hash mismatch: ${file}`,
    );
  }
  for (const name of ["snapshot.json", "predictions.json"])
    requireValue(
      manifest.files_sha256[name],
      `Missing archive member: ${name}`,
    );
  return {
    manifest,
    bytes,
    snapshot: await readJson(path.join(directory, "snapshot.json")),
    predictions: await readJson(path.join(directory, "predictions.json")),
  };
}

export async function prepare(
  directory,
  { rehearsal = false, now = new Date().toISOString() } = {},
) {
  const {
    manifest: m,
    bytes,
    snapshot: s,
    predictions: p,
  } = await checkedArchive(directory);
  requireValue(
    s.snapshot_id === m.snapshot_id &&
      s.evidence_mode === m.evidence_mode &&
      s.cutoff === m.cutoff &&
      s.kind === m.snapshot_kind,
    "Snapshot/manifest mismatch",
  );
  requireValue(
    p.release_manifest_sha256 === m.release_manifest_sha256,
    "Prediction release mismatch",
  );
  if (!rehearsal)
    requireValue(
      m.evidence_mode === "prospective" &&
        timestamp(now) >= timestamp(m.completed_at) &&
        timestamp(now) < timestamp(s.race.race_start),
      "Cannot prepare a late or retrospective prospective release",
    );
  const sources = [];
  for (const file of Object.keys(m.files_sha256).filter(
    (file) => file.startsWith("sources/") && file.endsWith(".json"),
  )) {
    const record = await readJson(path.join(directory, file));
    if (!record.uri?.startsWith("https://")) continue;
    const url = new URL(record.uri);
    if (url.search || url.hash || url.username || url.password) continue;
    sources.push({ url: url.href, sha256: record.sha256 });
  }
  const forecast = validateForecast({
    version: 1,
    snapshot_id: m.snapshot_id,
    race: Object.fromEntries(
      [
        "race_id",
        "year",
        "round",
        "grand_prix_id",
        "circuit_id",
        "race_start",
      ].map((key) => [key, s.race[key]]),
    ),
    entrants: s.entries
      .filter((e) => e.eligible)
      .map((e) => ({
        driver_id: e.driver_id,
        constructor_id: e.constructor_id,
      })),
    evidence_mode: m.evidence_mode,
    cutoff: m.cutoff,
    generated_at: m.generated_at,
    completed_at: m.completed_at,
    archive_sha256: hash(bytes),
    prediction_sha256: m.files_sha256["predictions.json"],
    release_manifest_sha256: m.release_manifest_sha256,
    feature_schema_sha256: m.feature_schema_sha256,
    git_commit: m.git_commit,
    runtime: m.runtime,
    code_sha256: m.code_sha256,
    sources,
    models: p.models,
    baselines: p.baselines,
    notice: rehearsal
      ? "PRIVATE REHEARSAL. Not a prospective forecast. Do not upload."
      : "Frozen experimental forecasts. All predetermined methods are retained. Probabilities are marginal estimates, not a joint race simulation or guarantees. Results will be recorded separately.",
  });
  return forecast;
}

export async function verifyPublication(
  bytes,
  releaseId,
  { fetcher = fetch, now = new Date().toISOString() } = {},
) {
  const forecast = validateForecast(JSON.parse(bytes));
  requireValue(
    forecast.evidence_mode === "prospective",
    "Cannot publish historical rehearsal",
  );
  requireValue(
    !/PRIVATE REHEARSAL|CONTRACT TEST ONLY/.test(forecast.notice),
    "Test packages cannot be published",
  );
  requireValue(
    Number.isSafeInteger(releaseId) && releaseId > 0,
    "Numeric GitHub release ID required",
  );
  const repository = "Jad-K-B/f1-race-forecasts";
  const api = `https://api.github.com/repos/${repository}/releases/${releaseId}`;
  const response = await fetcher(api, {
    headers: {
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2026-03-10",
    },
    signal: AbortSignal.timeout(20000),
  });
  requireValue(response.ok, `Receipt request failed: ${response.status}`);
  const apiBytes = Buffer.from(await response.arrayBuffer()),
    release = JSON.parse(apiBytes);
  requireValue(
    release.id === releaseId && release.draft === false && release.published_at,
    "Release is not published",
  );
  const assets = release.assets.filter(
    (asset) => asset.name === `${forecast.snapshot_id}.json`,
  );
  requireValue(
    assets.length === 1 && assets[0].state === "uploaded",
    "Expected one completed forecast asset",
  );
  const asset = assets[0];
  if (asset.digest != null)
    requireValue(
      asset.digest === `sha256:${hash(bytes)}`,
      "Server asset digest mismatch",
    );
  const receipt = {
    version: 1,
    repository,
    release_id: releaseId,
    asset_id: asset.id,
    api_url: api,
    asset_url: asset.browser_download_url,
    published_at: release.published_at,
    asset_updated_at: asset.updated_at,
    verified_at: now,
    server_date: response.headers.get("date"),
    payload_sha256: hash(bytes),
    api_sha256: hash(apiBytes),
  };
  validateReceipt(forecast, receipt, hash(bytes));
  const downloaded = await fetcher(receipt.asset_url, {
    signal: AbortSignal.timeout(20000),
  });
  requireValue(downloaded.ok, `Public download failed: ${downloaded.status}`);
  requireValue(
    hash(Buffer.from(await downloaded.arrayBuffer())) === hash(bytes),
    "Downloaded public bytes differ",
  );
  return { receipt, apiBytes };
}

export function verifySavedReceipt(forecast, receipt, bytes, apiBytes) {
  validateReceipt(forecast, receipt, hash(bytes));
  requireValue(
    hash(apiBytes) === receipt.api_sha256,
    "Server response hash mismatch",
  );
  const api = JSON.parse(apiBytes);
  const asset = api.assets?.find((item) => item.id === receipt.asset_id);
  if (asset?.digest != null)
    requireValue(
      asset.digest === `sha256:${hash(bytes)}`,
      "Saved server asset digest mismatch",
    );
  requireValue(
    api.id === receipt.release_id &&
      api.draft === false &&
      api.published_at === receipt.published_at &&
      asset?.state === "uploaded" &&
      asset.name === `${forecast.snapshot_id}.json` &&
      asset.updated_at === receipt.asset_updated_at &&
      asset.browser_download_url === receipt.asset_url,
    "Receipt differs from saved server response",
  );
}

export async function writeExclusive(directory, name, bytes) {
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, name), bytes, { flag: "wx" });
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  const [command, input, output, option] = process.argv.slice(2);
  try {
    if (command === "prepare" || command === "rehearse") {
      requireValue(
        input && output,
        "Archive and NEW output directory required",
      );
      const forecast = await prepare(input, {
        rehearsal: command === "rehearse",
      });
      await mkdir(output, { recursive: false });
      await writeExclusive(
        output,
        `${forecast.snapshot_id}.json`,
        jsonBytes(forecast),
      );
      console.log(
        JSON.stringify({
          status:
            command === "rehearse"
              ? "private_rehearsal_only"
              : "prepared_unpublished",
          output,
          sha256: hash(jsonBytes(forecast)),
          uploaded: false,
        }),
      );
    } else if (command === "receipt") {
      requireValue(
        input && output && option,
        "Forecast JSON, NEW receipt directory and release ID required",
      );
      const result = await verifyPublication(
        await readFile(input),
        Number(option),
      );
      await mkdir(output, { recursive: false });
      await writeExclusive(output, "receipt.json", jsonBytes(result.receipt));
      await writeExclusive(output, "github-response.json", result.apiBytes);
      console.log("Verified existing public asset; no upload performed.");
    } else
      throw Error(
        "Commands: prepare ARCHIVE NEW_DIRECTORY | rehearse ARCHIVE NEW_DIRECTORY | receipt FORECAST_JSON NEW_DIRECTORY RELEASE_ID. No command uploads or publishes.",
      );
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
