import { chromium } from "playwright";
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const browser = await chromium.launch({
  headless: true,
  channel: process.env.PLAYWRIGHT_CHANNEL || "msedge",
});
try {
  const outputs = [];
  const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
  for (const viewport of [
    { width: 1440, height: 1000 },
    { width: 390, height: 844 },
  ]) {
    const page = await browser.newPage({
      viewport,
      reducedMotion: "reduce",
    });
    await page.goto(process.env.FORMATION_URL || "http://127.0.0.1:5173");
    await page.locator("[data-loaded=true]").waitFor({ timeout: 60000 });
    await page.waitForTimeout(1200);
    await page.addStyleTag({
      content:
        "#cinema * { visibility: hidden !important; } #cinema .car-canvas, #cinema .car-canvas canvas { visibility: visible !important; }",
    });
    const image = await page.locator(".car-canvas canvas").screenshot();
    const file =
      viewport.width < 700 ? "car-fallback-mobile.png" : "car-fallback.png";
    await writeFile(`public/assets/${file}`, image);
    outputs.push({ file, sha256: hash(image), viewport });
    await page.close();
  }
  await writeFile(
    "public/assets/car-fallback-provenance.json",
    JSON.stringify(
      {
        source: "formation-car.glb",
        source_sha256: hash(await readFile("public/assets/formation-car.glb")),
        outputs,
        created_at: new Date().toISOString(),
        description:
          "Rendered from the application's licensed Qvist_designs geometry and existing materials at the default camera, with reduced motion and no UI overlays.",
        creator: "Qvist_designs (geometry); project rendering",
        license: "https://creativecommons.org/licenses/by/4.0/",
      },
      null,
      2,
    ) + "\n",
  );
  console.log("Rendered licensed car fallback locally.");
} finally {
  await browser.close();
}
