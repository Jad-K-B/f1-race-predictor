import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  validateForecast,
  requireValue,
  timestamp,
  STAGES,
  TARGETS,
} from "../src/forecast-contract.ts";
import { hash, jsonBytes, writeExclusive } from "./forecast-package.mjs";

// Separate prospective scoring; never imports training or sealed-test code/data.
export function scoreProspective(forecast, attachment) {
  validateForecast(forecast);
  requireValue(
    forecast.evidence_mode === "prospective",
    "Prospective scoring does not accept replay",
  );
  requireValue(
    attachment.forecast_manifest_sha256 === forecast.archive_sha256 &&
      attachment.race_id === forecast.race.race_id &&
      attachment.results.race_id === forecast.race.race_id,
    "Outcomes belong to a different forecast",
  );
  requireValue(
    timestamp(attachment.completed_at) > timestamp(forecast.race.race_start) &&
      timestamp(attachment.source.retrieved_at) >=
        timestamp(attachment.completed_at),
    "Outcome chronology invalid",
  );
  requireValue(
    attachment.source.reviewed_by &&
      ["fia", "f1db", "jolpica"].includes(attachment.source.provider),
    "Reviewed authoritative outcome source required",
  );
  const rows = attachment.results.drivers;
  requireValue(
    Array.isArray(rows) && rows.length === forecast.entrants.length,
    "Complete eligible outcome field required",
  );
  const ids = forecast.entrants.map((row) => row.driver_id).sort();
  requireValue(
    rows
      .map((row) => row.driver_id)
      .sort()
      .join("|") === ids.join("|"),
    "Outcome roster mismatch",
  );
  requireValue(
    rows
      .map((row) => row.finish_order)
      .sort((a, b) => a - b)
      .every((rank, i) => rank === i + 1),
    "Outcome order must be reranked among prediction-eligible drivers",
  );
  for (const row of rows) {
    requireValue(
      Number.isFinite(row.official_points) && row.official_points >= 0,
      "Official points required",
    );
    requireValue(
      row.official_position === null ||
        (Number.isInteger(row.official_position) && row.official_position >= 1),
      "Official position must be numeric or null",
    );
  }
  const classified = rows
    .filter((row) => row.official_position !== null)
    .map((row) => row.official_position);
  requireValue(
    new Set(classified).size === classified.length,
    "Duplicate official positions",
  );
  requireValue(
    rows.filter((row) => row.official_position === 1).length === 1,
    "Exactly one official winner required",
  );
  const actual = Object.fromEntries(rows.map((row) => [row.driver_id, row]));
  const n = rows.length,
    models = {};
  for (const [name, method] of Object.entries({
    ...forecast.models,
    ...forecast.baselines,
  })) {
    const errors = method.drivers.map(
      (row) => row.predicted_order - actual[row.driver_id].finish_order,
    );
    const winner = rows.find((row) => row.official_position === 1).driver_id;
    const podium = new Set(
      rows
        .filter(
          (row) => row.official_position !== null && row.official_position <= 3,
        )
        .map((row) => row.driver_id),
    );
    const points = new Set(
      rows.filter((row) => row.official_points > 0).map((row) => row.driver_id),
    );
    const stages = {};
    for (const stage of STAGES) {
      if (!method.drivers.every((row) => row[stage])) continue;
      stages[stage] = Object.fromEntries(
        TARGETS.map((target, index) => {
          let loss = 0,
            brier = 0;
          for (const row of method.drivers) {
            const y = Number(
              index === 0
                ? row.driver_id === winner
                : index === 1
                  ? podium.has(row.driver_id)
                  : points.has(row.driver_id),
            );
            const raw = row[stage][target],
              p = Math.max(1e-12, Math.min(1 - 1e-12, raw));
            loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
            brier += (raw - y) ** 2;
          }
          return [target, { log_loss: loss / n, brier: brier / n }];
        }),
      );
    }
    models[name] = {
      ranking: {
        mae: errors.reduce((sum, v) => sum + Math.abs(v), 0) / n,
        spearman:
          n > 1
            ? 1 -
              (6 * errors.reduce((sum, v) => sum + v * v, 0)) /
                (n * (n * n - 1))
            : null,
        winner_top1: Number(method.predicted_order[0] === winner),
        podium_overlap: podium.size
          ? method.predicted_order.slice(0, 3).filter((id) => podium.has(id))
              .length / podium.size
          : null,
      },
      probability: stages,
    };
  }
  return {
    version: 1,
    race_id: forecast.race.race_id,
    snapshot_id: forecast.snapshot_id,
    forecast_manifest_sha256: forecast.archive_sha256,
    outcome_source_sha256: attachment.source.sha256,
    completed_at: attachment.completed_at,
    methods: models,
    notice:
      "One prospective race, not a new historical evaluation. No fitting or model selection. Per-race scores do not establish calibration or generalization. Publication authenticity must be checked separately against the receipt.",
  };
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  try {
    const [forecastFile, attachmentFile, output] = process.argv.slice(2);
    requireValue(
      forecastFile && attachmentFile && output,
      "Usage: score-prospective.mjs FORECAST_JSON PRIVATE_RESULTS_JSON NEW_SCORE_JSON",
    );
    const forecastBytes = await readFile(forecastFile),
      attachmentBytes = await readFile(attachmentFile);
    const attachment = JSON.parse(attachmentBytes);
    requireValue(
      hash(
        await readFile(path.join(path.dirname(attachmentFile), "source.blob")),
      ) === attachment.source.sha256,
      "Outcome evidence hash mismatch",
    );
    const report = scoreProspective(JSON.parse(forecastBytes), attachment);
    await writeExclusive(
      path.dirname(output),
      path.basename(output),
      jsonBytes({
        ...report,
        forecast_payload_sha256: hash(forecastBytes),
        attachment_sha256: hash(attachmentBytes),
      }),
    );
    console.log(
      "Separate score written. Forecast and outcomes unchanged; nothing uploaded.",
    );
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
