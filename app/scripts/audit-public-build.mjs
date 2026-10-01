import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { hash } from "./forecast-package.mjs";

const root = process.argv[2] || "dist-public";
async function files(dir) {
  const output = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    output.push(...(entry.isDirectory() ? await files(file) : [file]));
  }
  return output;
}
const assets = [
  "formation-car.glb",
  "car-fallback.png",
  "car-fallback-mobile.png",
  "car-fallback-provenance.json",
  "car-provenance.json",
  "barlow-condensed.ttf",
  "plex-mono.ttf",
  "barlow-OFL.txt",
  "plex-OFL.txt",
  "react-LICENSE.txt",
  "react-dom-LICENSE.txt",
  "three-LICENSE.txt",
  "gsap-NOTICE.txt",
  "lucide-license.txt",
];
const members = await files(root);
for (const file of members) {
  const relative = path.relative(root, file).replaceAll("\\", "/");
  const favicon = /^assets\/favicon-[\w-]+\.svg$/.test(relative);
  assert.ok(
    relative === "index.html" ||
      favicon ||
      assets.includes(relative.replace(/^assets\//, "")) ||
      /^assets\/(index|CarScene|motion|three)-[\w-]+\.(js|css)$/.test(relative),
    `Unexpected public file: ${relative}`,
  );
  const bytes = await readFile(file);
  if (favicon)
    assert.equal(
      hash(bytes),
      hash(await readFile("src/favicon.svg")),
      relative,
    );
  assert.doesNotMatch(
    bytes.toString("utf8"),
    /OpenAI|ChatGPT|\bCodex\b|\bAstra\b|Co-authored-by|PRIVATE REHEARSAL|CONTRACT TEST ONLY|local_design_review_only|reviewed_by|c2pa\.claim|MultiViewer|Dark373|driver-headshots/i,
    relative,
  );
  if (/\.(js|json|html)$/.test(file))
    assert.doesNotMatch(
      bytes.toString("utf8"),
      /features\.json|q1_millis|test_predictions\.csv|\.joblib|"fixture_mode"/,
      relative,
    );
}
for (const asset of assets)
  assert.equal(
    hash(await readFile(path.join(root, "assets", asset))),
    hash(await readFile(path.join("public/assets", asset))),
    asset,
  );
console.log(
  JSON.stringify({
    publicFiles: members.length,
    privateAssetsExcluded: true,
    aiAttributionAbsent: true,
    assetHashesMatch: true,
  }),
);
