import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";

const base = process.env.FORMATION_URL || "http://127.0.0.1:5173";
const report = { tested_at: new Date().toISOString(), checks: [] };
await mkdir("test-results", { recursive: true });
const browser = await chromium.launch({
  headless: true,
  channel: process.env.PLAYWRIGHT_CHANNEL || "msedge",
});

async function compare(page, before, after) {
  return page.evaluate(
    async (sources) => {
      const pixels = await Promise.all(
        sources.map(async (source) => {
          const image = new Image();
          image.src = `data:image/png;base64,${source}`;
          await image.decode();
          const canvas = document.createElement("canvas");
          canvas.width = image.width;
          canvas.height = image.height;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(image, 0, 0);
          // Measure the car, excluding controls whose touch-hover state persists.
          return ctx.getImageData(
            0,
            Math.floor(canvas.height * 0.25),
            canvas.width,
            Math.floor(canvas.height * 0.45),
          ).data;
        }),
      );
      let changed = 0,
        difference = 0,
        clipped = 0;
      for (let i = 0; i < pixels[0].length; i += 4) {
        const delta =
          (Math.abs(pixels[0][i] - pixels[1][i]) +
            Math.abs(pixels[0][i + 1] - pixels[1][i + 1]) +
            Math.abs(pixels[0][i + 2] - pixels[1][i + 2])) /
          3;
        if (delta > 8) changed++;
        difference += delta;
        if (
          pixels[1][i] > 250 &&
          pixels[1][i + 1] > 250 &&
          pixels[1][i + 2] > 250
        )
          clipped++;
      }
      const count = pixels[0].length / 4;
      return {
        changedFraction: changed / count,
        meanDifference: difference / count,
        clippedFraction: clipped / count,
      };
    },
    [before.toString("base64"), after.toString("base64")],
  );
}

try {
  for (const config of [
    {
      name: "desktop",
      width: 1440,
      height: 1000,
      motion: "no-preference",
      touch: false,
    },
    {
      name: "mobile",
      width: 390,
      height: 844,
      motion: "no-preference",
      touch: true,
    },
    {
      name: "narrow-reduced-motion",
      width: 320,
      height: 780,
      motion: "reduce",
      touch: true,
    },
  ]) {
    const page = await browser.newPage({
      viewport: { width: config.width, height: config.height },
      reducedMotion: config.motion,
      hasTouch: config.touch,
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(base);
    await page.locator("[data-loaded=true]").waitFor({ timeout: 60000 });
    // Hold the pose fixed so pixel changes measure lighting, not idle movement.
    await page.getByRole("button", { name: "Orbit car", exact: true }).click();
    await page.waitForTimeout(1000);
    const lighting = page.getByRole("button", {
      name: "Lighting",
      exact: true,
    });
    assert.equal(await lighting.getAttribute("title"), "Lighting");
    assert.equal(await lighting.getAttribute("aria-pressed"), "false");
    const canvas = page.locator(".car-canvas canvas");
    const before = await canvas.screenshot();
    const header = await page.locator(".site-header").screenshot();
    await page.screenshot({
      path: `test-results/lighting-${config.name}-default.png`,
    });
    if (config.touch) await lighting.tap();
    else {
      await lighting.focus();
      await page.keyboard.press("Enter");
    }
    assert.equal(await lighting.getAttribute("aria-pressed"), "true");
    await page.waitForTimeout(900);
    const bright = await canvas.screenshot();
    const change = await compare(page, before, bright);
    assert.ok(change.changedFraction > 0.01, JSON.stringify(change));
    assert.ok(change.meanDifference > 1, JSON.stringify(change));
    assert.ok(change.clippedFraction < 0.03, JSON.stringify(change));
    assert.deepEqual(await page.locator(".site-header").screenshot(), header);
    await page.screenshot({
      path: `test-results/lighting-${config.name}-bright.png`,
    });
    // Reversals during the transition must settle at the explicit button state.
    for (let i = 0; i < 3; i++) {
      if (config.touch) await lighting.tap();
      else await lighting.press("Space");
    }
    assert.equal(await lighting.getAttribute("aria-pressed"), "false");
    await page.waitForTimeout(900);
    const restored = await compare(page, before, await canvas.screenshot());
    await page.screenshot({
      path: `test-results/lighting-${config.name}-restored.png`,
    });
    assert.ok(restored.meanDifference < 0.1, JSON.stringify(restored));
    assert.deepEqual(errors, []);
    report.checks.push({
      viewport: config.name,
      change,
      restored,
      passed: true,
    });
    console.log("PASS lighting:", config.name, change);
    await page.close();
  }
} finally {
  await browser.close();
  await writeFile(
    "test-results/lighting-report.json",
    JSON.stringify(report, null, 2),
  );
}
