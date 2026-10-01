import { readFile } from "node:fs/promises";
import path from "node:path";
import { verifySavedReceipt } from "./forecast-package.mjs";
import { validateForecast, requireValue } from "../src/forecast-contract.ts";

export async function loadCatalog(
  root,
  { publicBuild = false, catalogPath } = {},
) {
  const result = {
    publicBuild,
    events: [],
    benchmark: {
      probability: [],
      ranking: [],
      calibration: [],
      uncertainty: [],
    },
    readiness: {
      recorded_at: "2026-09-27T13:29:23.442477+00:00",
      blockers: [
        "Reviewed race-weekend evidence and verified publication are required.",
      ],
      forecast_published: false,
      pre_weekend_snapshot_generated: false,
      status: "pending",
    },
  };
  if (catalogPath) {
    const entries = JSON.parse(await readFile(catalogPath));
    requireValue(
      Array.isArray(entries),
      "Catalog must be a list of file/receipt references",
    );
    for (const entry of entries) {
      requireValue(
        Object.keys(entry).every((key) =>
          ["file", "receipt", "server_response"].includes(key),
        ),
        "Unknown catalog field",
      );
      const bytes = await readFile(
        path.resolve(path.dirname(catalogPath), entry.file),
      );
      const forecast = validateForecast(JSON.parse(bytes));
      const receipt = entry.receipt
        ? JSON.parse(
            await readFile(
              path.resolve(path.dirname(catalogPath), entry.receipt),
            ),
          )
        : null;
      if (receipt) {
        requireValue(entry.server_response, "Saved server response required");
        verifySavedReceipt(
          forecast,
          receipt,
          bytes,
          await readFile(
            path.resolve(path.dirname(catalogPath), entry.server_response),
          ),
        );
      }
      if (publicBuild) {
        requireValue(
          receipt && forecast.evidence_mode === "prospective",
          "Public catalog requires verified prospective publication",
        );
        requireValue(
          !/PRIVATE REHEARSAL|CONTRACT TEST ONLY/.test(forecast.notice),
          "Test packages are private",
        );
      }
      requireValue(
        !result.events.some(
          (item) => item.forecast.snapshot_id === forecast.snapshot_id,
        ),
        "Duplicate snapshot in catalog",
      );
      result.events.push({ id: forecast.snapshot_id, forecast, receipt });
    }
  }
  result.readiness.forecast_published = result.events.some(
    (item) => item.receipt,
  );
  return result;
}
