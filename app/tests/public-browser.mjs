import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { setTimeout as delay } from "node:timers/promises";
import { writeFile, mkdir } from "node:fs/promises";
import assert from "node:assert/strict";
const base = "http://127.0.0.1:4174";
await mkdir("test-results", { recursive: true });
try {
  await fetch(base);
  throw Error("Public preview port 4174 is occupied");
} catch (error) {
  if (!error.cause) throw error;
}
const server = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "preview",
    "--mode",
    "public",
    "--configLoader",
    "runner",
    "--host",
    "127.0.0.1",
    "--port",
    "4174",
    "--strictPort",
  ],
  { windowsHide: true, stdio: "pipe" },
);
const report = {
  tested_at: new Date().toISOString(),
  deployed: false,
  checks: [],
};
let browser;
try {
  let ready = false;
  for (let i = 0; i < 50; i++) {
    if (server.exitCode !== null) throw Error("Public preview exited");
    try {
      ready = (await fetch(base)).ok;
    } catch {}
    if (ready) break;
    await delay(200);
  }
  assert.ok(ready);
  for (const channel of ["msedge", "chrome"]) {
    browser = await chromium.launch({ headless: true, channel });
    for (const width of [1440, 320]) {
      const page = await browser.newPage({
        viewport: { width, height: 950 },
        hasTouch: width < 700,
        reducedMotion: "reduce",
      });
      const errors = [],
        missing = [];
      page.on("pageerror", (e) => errors.push(e.message));
      page.on("response", (r) => {
        if (r.status() >= 400) missing.push(r.url());
      });
      await page.goto(base);
      await page.locator("[data-loaded=true]").waitFor({ timeout: 60000 });
      // Asset loading precedes the first rendered frame, especially on a cold start.
      await page.waitForFunction(
        () =>
          Number(document.querySelector(".car-canvas")?.dataset.triangles) >
          50000,
        undefined,
        { timeout: 60000 },
      );
      assert.ok(
        Number(
          await page.locator(".car-canvas").getAttribute("data-triangles"),
        ) > 50000,
      );
      await page.getByRole("button", { name: "Explore forecasts" }).click();
      assert.equal(await page.locator(".podium-card").count(), 0);
      assert.equal(
        await page.getByLabel("Forecast archive").locator("option").count(),
        1,
      );
      await page.getByRole("tab", { name: /Model lab/ }).focus();
      await page.keyboard.press("Enter");
      assert.ok(
        await page
          .getByText(
            "Historical evaluation reports are not included in this public forecast release.",
          )
          .isVisible(),
      );
      await page.getByRole("tab", { name: /Forecast archive/ }).click();
      assert.ok(
        await page
          .getByText("No verified forecast has been published yet.")
          .isVisible(),
      );
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
      );
      assert.deepEqual(errors, []);
      assert.deepEqual(missing, []);
      await page.screenshot({
        path: `test-results/public-${channel}-${width}.png`,
      });
      report.checks.push({ channel, width, passed: true });
      await page.close();
    }
    const animated = await browser.newPage({
      viewport: { width: 390, height: 844 },
      reducedMotion: "no-preference",
    });
    const animationWarnings = [];
    animated.on("console", (message) => {
      if (/GSAP target.*not found/.test(message.text()))
        animationWarnings.push(message.text());
    });
    await animated.goto(base);
    await animated.waitForFunction(
      () => document.documentElement.dataset.motion === "true",
    );
    for (const name of [/Model lab/, /Forecast archive/, /Race Forecast/]) {
      await animated.getByRole("tab", { name }).click();
      await animated.evaluate(
        () =>
          new Promise((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(resolve)),
          ),
      );
    }
    const favicon = await animated
      .locator('link[rel="icon"]')
      .getAttribute("href");
    assert.ok(favicon, "A browser icon is declared");
    assert.ok(
      await animated.evaluate(async (href) => {
        const icon = new Image();
        icon.src = new URL(href, location.href).href;
        await icon.decode();
        return icon.naturalWidth > 0 && icon.naturalHeight > 0;
      }, favicon),
      "Browser icon decodes successfully",
    );
    assert.deepEqual(animationWarnings, []);
    assert.equal(await animated.locator(".podium-card").count(), 0);
    report.checks.push({
      channel,
      animatedPending: true,
      favicon: true,
      passed: true,
    });
    await animated.close();
    const fallback = await browser.newPage({
      viewport: { width: 390, height: 844 },
    });
    await fallback.addInitScript(() => {
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (type, ...args) {
        return /webgl/.test(type) ? null : original.call(this, type, ...args);
      };
    });
    await fallback.goto(base);
    await fallback.getByText("3D unavailable", { exact: true }).waitFor();
    await fallback.waitForTimeout(1500);
    assert.ok(
      await fallback
        .locator(".car-poster")
        .evaluate((img) => img.complete && img.naturalWidth > 0),
    );
    assert.ok(
      await fallback
        .getByRole("button", { name: "Lighting", exact: true })
        .isDisabled(),
    );
    await fallback.screenshot({
      path: `test-results/public-fallback-${channel}.png`,
    });
    report.checks.push({ channel, fallback: true, passed: true });
    await browser.close();
    browser = undefined;
  }
  console.log(JSON.stringify(report));
} finally {
  await browser?.close();
  if (server.exitCode === null) {
    const exited = once(server, "exit");
    server.kill();
    await exited;
  }
  await writeFile(
    "test-results/public-browser-report.json",
    JSON.stringify(report, null, 2),
  );
}
