import { chromium } from "playwright";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { setTimeout as delay } from "node:timers/promises";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import path from "node:path";
import { contractFixture } from "./forecast-fixture.mjs";
import { hash, jsonBytes } from "../scripts/forecast-package.mjs";

const base = "http://127.0.0.1:5181";
try {
  await fetch(base);
  throw Error("Integration port 5181 is occupied");
} catch (error) {
  if (!error.cause) throw error;
}
await mkdir(".cache", { recursive: true });
const directory = await mkdtemp(path.resolve(".cache/contract-browser-"));
const unpublished = contractFixture(22),
  published = contractFixture(3);
unpublished.snapshot_id = "contract-unpublished";
published.snapshot_id = "contract-published";
published.race.year = 2025;
published.race.race_start = "2025-10-04T07:00:00Z";
for (const key of ["cutoff", "generated_at", "completed_at"])
  published[key] = published[key].replace("2026", "2025");
const asset = {
  id: 456,
  name: "contract-published.json",
  state: "uploaded",
  updated_at: "2025-10-04T05:05:00Z",
  browser_download_url:
    "https://github.com/Jad-K-B/f1-race-forecasts/releases/download/contract-test/contract-published.json",
};
const api = {
  id: 123,
  draft: false,
  published_at: "2025-10-04T05:04:00Z",
  assets: [asset],
};
const receipt = {
  version: 1,
  repository: "Jad-K-B/f1-race-forecasts",
  release_id: 123,
  asset_id: 456,
  api_url:
    "https://api.github.com/repos/Jad-K-B/f1-race-forecasts/releases/123",
  asset_url: asset.browser_download_url,
  published_at: api.published_at,
  asset_updated_at: asset.updated_at,
  verified_at: "2025-10-04T05:06:00Z",
  server_date: "Sat, 04 Oct 2025 05:06:00 GMT",
  payload_sha256: hash(jsonBytes(published)),
  api_sha256: hash(jsonBytes(api)),
};
for (const [file, value] of Object.entries({
  "unpublished.json": unpublished,
  "published.json": published,
  "receipt.json": receipt,
  "server.json": api,
  "catalog.json": [
    {
      file: "published.json",
      receipt: "receipt.json",
      server_response: "server.json",
    },
    { file: "unpublished.json" },
  ],
}))
  await writeFile(path.join(directory, file), jsonBytes(value));
const server = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "--host",
    "127.0.0.1",
    "--port",
    "5181",
    "--strictPort",
  ],
  {
    windowsHide: true,
    stdio: "pipe",
    env: {
      ...process.env,
      FORMATION_CATALOG: path.join(directory, "catalog.json"),
    },
  },
);
let browser;
const report = {
  tested_at: new Date().toISOString(),
  synthetic_contract_data_only: true,
  network_publication_performed: false,
  checks: [],
};
try {
  let ready = false;
  for (let i = 0; i < 50; i++) {
    if (server.exitCode !== null) throw Error("Integration server exited");
    try {
      ready = (await fetch(base)).ok;
    } catch {}
    if (ready) break;
    await delay(200);
  }
  assert.ok(ready);
  browser = await chromium.launch({ headless: true, channel: "msedge" });
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({
      viewport: { width, height: 950 },
      reducedMotion: "reduce",
      hasTouch: width < 700,
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(base);
    await page.getByRole("button", { name: "Explore forecasts" }).click();
    assert.ok(
      await page
        .getByText("Unpublished forecast.", { exact: true })
        .isVisible(),
    );
    assert.equal(await page.locator(".podium-card").count(), 3);
    assert.equal(await page.locator(".podium-card img").count(), 0);
    assert.equal(
      await page.locator(".podium-card").first().getAttribute("data-season"),
      "2026",
    );
    await page.getByRole("button", { name: "Full field", exact: true }).click();
    assert.equal(await page.locator(".field-table tbody tr").count(), 22);
    await page.locator(".podium-card").nth(2).getByRole("button").focus();
    await page.keyboard.press("Enter");
    assert.equal(
      await page.locator(".driver-focus h3").innerText(),
      "Test Driver 3",
    );
    await page.getByLabel("Probability stage").selectOption("raw");
    await page
      .getByRole("combobox", { name: "Forecast method", exact: true })
      .click();
    await page.getByRole("option", { name: /^Starting grid / }).click();
    assert.equal(
      await page.locator(".podium-card [data-stat=win]").first().innerText(),
      "—",
    );
    await page
      .getByLabel("Forecast archive")
      .selectOption("contract-published");
    assert.ok(
      await page
        .getByText("Published pre-race forecast.", { exact: true })
        .isVisible(),
    );
    assert.equal(await page.locator(".field-table tbody tr").count(), 3);
    assert.equal(
      await page.locator(".podium-card").first().getAttribute("data-season"),
      "2025",
    );
    await page.getByRole("button", { name: "View forecast details" }).click();
    assert.equal(
      await page
        .getByRole("link", { name: "Published forecast", exact: true })
        .getAttribute("href"),
      receipt.asset_url,
    );
    await page.keyboard.press("Escape");
    await page.getByLabel("Forecast archive").selectOption("upcoming");
    assert.equal(await page.locator(".podium-card").count(), 0);
    await page.getByLabel("Forecast archive").selectOption("replay");
    assert.ok(
      await page.getByText("Historical replay.", { exact: true }).isVisible(),
    );
    assert.equal(
      await page.locator(".podium-card").first().getAttribute("data-season"),
      "2023",
    );
    assert.equal(await page.locator(".podium-card img").count(), 3);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    assert.deepEqual(errors, []);
    await page.screenshot({ path: `test-results/integration-${width}.png` });
    report.checks.push({ width, passed: true });
    await page.close();
  }
  console.log(JSON.stringify(report));
} finally {
  await browser?.close();
  const exited = once(server, "exit");
  server.kill();
  await exited;
  await writeFile("test-results/integration-report.json", jsonBytes(report));
}
