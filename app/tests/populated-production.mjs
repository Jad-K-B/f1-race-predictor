import { build, preview } from "vite";
import { chromium } from "playwright";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import assert from "node:assert/strict";
import { contractFixture } from "./forecast-fixture.mjs";
import { validateForecast, STAGES } from "../src/forecast-contract.ts";
import {
  hash,
  jsonBytes,
  verifySavedReceipt,
} from "../scripts/forecast-package.mjs";
import { loadCatalog } from "../scripts/catalog.mjs";

assert.ok(
  !process.env.FORMATION_CATALOG,
  "Unset FORMATION_CATALOG for private tests",
);
await mkdir(".cache", { recursive: true });
await mkdir("test-results", { recursive: true });
const directory = await mkdtemp(path.resolve(".cache/private-production-"));
const catalogPath = path.join(directory, "catalog.json");
const events = [];
for (const [index, count] of [20, 22].entries()) {
  const forecast = contractFixture(count);
  forecast.snapshot_id = `contract-production-${count}`;
  forecast.race.round = index + 1;
  forecast.entrants.forEach((entry, i) => {
    entry.constructor_id = i % 2 ? "test-team-b" : "test-team-a";
  });
  for (const [methodIndex, method] of Object.values({
    ...forecast.models,
    ...forecast.baselines,
  }).entries()) {
    const shift = (index + methodIndex) % count;
    method.predicted_order = [
      ...method.predicted_order.slice(shift),
      ...method.predicted_order.slice(0, shift),
    ];
    method.drivers.forEach((row) => {
      row.predicted_order = method.predicted_order.indexOf(row.driver_id) + 1;
      if (row.raw)
        for (const [stageIndex, stage] of STAGES.entries()) {
          if (stage === "reconciled") continue;
          row[stage].p_race_winner =
            (stageIndex + 1) / 100 + row.predicted_order / 1000;
        }
    });
  }
  validateForecast(forecast);
  const bytes = jsonBytes(forecast);
  const api = {
    id: 123 + index,
    draft: false,
    published_at: "2026-10-04T05:04:00Z",
    assets: [
      {
        id: 456 + index,
        state: "uploaded",
        name: `${forecast.snapshot_id}.json`,
        updated_at: "2026-10-04T05:05:00Z",
        browser_download_url: `https://github.com/Jad-K-B/f1-race-forecasts/releases/download/contract-test/${forecast.snapshot_id}.json`,
        digest: `sha256:${hash(bytes)}`,
      },
    ],
  };
  const receipt = {
    version: 1,
    repository: "Jad-K-B/f1-race-forecasts",
    release_id: api.id,
    asset_id: api.assets[0].id,
    api_url: `https://api.github.com/repos/Jad-K-B/f1-race-forecasts/releases/${api.id}`,
    asset_url: api.assets[0].browser_download_url,
    published_at: api.published_at,
    asset_updated_at: api.assets[0].updated_at,
    verified_at: "2026-10-04T05:06:00Z",
    server_date: "Sun, 04 Oct 2026 05:06:00 GMT",
    payload_sha256: hash(bytes),
    api_sha256: hash(jsonBytes(api)),
  };
  verifySavedReceipt(forecast, receipt, bytes, jsonBytes(api));
  events.push({ id: forecast.snapshot_id, forecast, receipt });
  if (index === 0) {
    await writeFile(path.join(directory, "forecast.json"), bytes);
    await writeFile(path.join(directory, "receipt.json"), jsonBytes(receipt));
    await writeFile(path.join(directory, "response.json"), jsonBytes(api));
    await writeFile(
      catalogPath,
      jsonBytes([
        {
          file: "forecast.json",
          receipt: "receipt.json",
          server_response: "response.json",
        },
      ]),
    );
  }
}
// Test-only module injection, confined to an ignored output directory. The normal
// public catalog gate and build audit MUST reject these same synthetic records.
await assert.rejects(
  loadCatalog(process.cwd(), { publicBuild: true, catalogPath }),
  /Test packages are private/,
);
const catalog = {
  ...(await loadCatalog(process.cwd(), { publicBuild: true })),
  events,
};
catalog.readiness.forecast_published = true;
const outDir = path.join(directory, "build");
await build({
  configLoader: "runner",
  mode: "public",
  build: { outDir },
  plugins: [
    {
      name: "private-contract-test-only",
      enforce: "pre",
      load(id) {
        if (id === "\0virtual:forecast-catalog")
          return `export default ${JSON.stringify(catalog)}`;
      },
      transformIndexHtml(html) {
        return html.replace(
          "<body>",
          '<body><aside style="position:fixed;bottom:0;left:0;z-index:99999;background:#fff;color:#000;padding:8px;font:12px monospace;pointer-events:none">CONTRACT TEST ONLY / PRIVATE / NOT A REAL FORECAST</aside>',
        );
      },
    },
  ],
});
const rejected = spawnSync(
  process.execPath,
  ["scripts/audit-public-build.mjs", outDir],
  { encoding: "utf8", windowsHide: true },
);
assert.notEqual(
  rejected.status,
  0,
  "Private fixture unexpectedly passed publication audit",
);
assert.match(rejected.stderr, /CONTRACT TEST ONLY/);
const report = {
  tested_at: new Date().toISOString(),
  synthetic_only: true,
  uploaded: false,
  public_gate_rejects_fixtures: true,
  checks: [],
  passed: false,
};
let server, browser;
const labels = {
  external: "External ML",
  numpy: "NumPy",
  final_grid: "Starting grid",
  recent_form: "Recent form",
  qualifying: "Qualifying",
  final_grid_rule: "Grid rule",
  train_grid_frequency: "Train-grid frequency",
  uniform_field_quota: "Uniform field",
};
try {
  server = await preview({
    configLoader: "runner",
    mode: "public",
    build: { outDir },
    preview: { host: "127.0.0.1", port: 4175, strictPort: true },
  });
  browser = await chromium.launch({ channel: "msedge", headless: true });
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({
      viewport: { width, height: 950 },
      hasTouch: width < 700,
      reducedMotion: "reduce",
      acceptDownloads: true,
    });
    const errors = [],
      missing = [],
      external = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("response", (r) => {
      if (r.status() >= 400) missing.push(r.url());
    });
    await page.route("**/*", (route) => {
      const url = route.request().url();
      if (
        /^https?:/.test(url) &&
        new URL(url).origin !== "http://127.0.0.1:4175"
      ) {
        external.push(url);
        return route.abort();
      }
      return route.continue();
    });
    await page.goto("http://127.0.0.1:4175");
    await page.getByRole("button", { name: "Explore forecasts" }).click();
    assert.ok(
      await page
        .getByText("CONTRACT TEST ONLY / PRIVATE / NOT A REAL FORECAST", {
          exact: true,
        })
        .isVisible(),
    );
    for (const item of events) {
      await page.getByLabel("Forecast archive").selectOption(item.id);
      assert.ok(
        await page
          .getByText("Published pre-race forecast.", { exact: true })
          .isVisible(),
      );
      for (const [name, method] of Object.entries({
        ...item.forecast.models,
        ...item.forecast.baselines,
      })) {
        await page
          .getByRole("combobox", { name: "Forecast method", exact: true })
          .click();
        await page
          .getByRole("option", { name: new RegExp(`^${labels[name]} `) })
          .click();
        for (const stage of method.drivers[0].raw ? STAGES : ["reconciled"]) {
          if (method.drivers[0].raw)
            await page.getByLabel("Probability stage").selectOption(stage);
          for (let i = 0; i < 3; i++) {
            const card = page.locator(".podium-card").nth(i);
            const id = method.predicted_order[i],
              row = method.drivers.find((r) => r.driver_id === id);
            assert.equal(await card.getAttribute("data-driver"), id);
            assert.equal(
              await card.getAttribute("data-team"),
              item.forecast.entrants.find((r) => r.driver_id === id)
                .constructor_id,
            );
            for (const [stat, target] of [
              ["win", "p_race_winner"],
              ["podium", "p_podium_finish"],
            ]) {
              const value = row[stage]?.[target];
              assert.equal(
                await card.locator(`[data-stat=${stat}]`).innerText(),
                value === undefined ? "\u2014" : `${(value * 100).toFixed(1)}%`,
              );
            }
          }
        }
      }
      await page
        .getByRole("button", { name: "Full field", exact: true })
        .click();
      assert.equal(
        await page.locator(".field-table tbody tr").count(),
        item.forecast.entrants.length,
      );
      const first = page.locator(".podium-card").first().getByRole("button");
      await first.focus();
      await page.keyboard.press("Enter");
      assert.equal(await first.getAttribute("aria-pressed"), "true");
      const downloaded = page.waitForEvent("download");
      await page
        .getByRole("button", { name: "Download selected saved predictions" })
        .click();
      const payload = JSON.parse(
        await readFile(await (await downloaded).path()),
      );
      assert.deepEqual(payload.archive, item.forecast);
      assert.deepEqual(
        payload.prediction,
        item.forecast.baselines.uniform_field_quota,
      );
      await page.getByRole("button", { name: "View forecast details" }).click();
      assert.equal(
        await page
          .getByRole("link", { name: "Published forecast", exact: true })
          .getAttribute("href"),
        item.receipt.asset_url,
      );
      await page.keyboard.press("Escape");
    }
    assert.equal(await page.locator(".podium-card img").count(), 0);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    await page.screenshot({
      path: `test-results/populated-production-${width}.png`,
    });
    await page.getByLabel("Forecast archive").selectOption("upcoming");
    assert.equal(await page.locator(".podium-card").count(), 0);
    assert.deepEqual(errors, []);
    assert.deepEqual(missing, []);
    assert.deepEqual(external, []);
    report.checks.push({ width, races: 2, methods_per_race: 8, passed: true });
    await page.close();
  }
  report.passed = true;
} finally {
  await browser?.close();
  if (server) {
    const closed = once(server.httpServer, "close");
    server.httpServer.close();
    await closed;
  }
  await writeFile(
    "test-results/populated-production-report.json",
    jsonBytes(report),
  );
}
console.log(JSON.stringify(report));
