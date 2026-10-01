import type { Method } from "./data";

export const RELEASE_HASH =
  "2b297485036298d517c961c731dc4227c7accc35caffa4bd6fd5120e6712c418";
export const SCHEMA_HASH =
  "ffa29ccd5d861005abcd25a6a844209f5adf6619373bc49ac4bef6b00c3bd90d";
export const MODEL_NAMES = ["external", "numpy"];
export const BASELINE_NAMES = [
  "final_grid",
  "recent_form",
  "qualifying",
  "final_grid_rule",
  "train_grid_frequency",
  "uniform_field_quota",
];
export const STAGES = [
  "raw",
  "calibrated",
  "rank_calibrated",
  "blended",
  "reconciled",
] as const;
export const TARGETS = [
  "p_race_winner",
  "p_podium_finish",
  "p_points_finish",
] as const;
export type Forecast = {
  version: 1;
  snapshot_id: string;
  race: {
    race_id: number;
    year: number;
    round: number;
    grand_prix_id: string;
    circuit_id: string;
    race_start: string | null;
  };
  entrants: { driver_id: string; constructor_id: string }[];
  evidence_mode: "prospective" | "approximate_retrospective";
  cutoff: string;
  generated_at: string;
  completed_at: string;
  archive_sha256: string;
  prediction_sha256: string;
  release_manifest_sha256: string;
  feature_schema_sha256: string;
  git_commit: string;
  runtime: { python: string; platform: string };
  code_sha256: Record<string, string>;
  sources: { url: string; sha256: string }[];
  models: Record<string, Method>;
  baselines: Record<string, Method>;
  notice: string;
};
export type Receipt = {
  version: 1;
  repository: "Jad-K-B/f1-race-forecasts";
  release_id: number;
  asset_id: number;
  api_url: string;
  asset_url: string;
  published_at: string;
  asset_updated_at: string;
  verified_at: string;
  server_date: string;
  payload_sha256: string;
  api_sha256: string;
};
export function requireValue(
  condition: unknown,
  message: string,
): asserts condition {
  if (!condition) throw new Error(message);
}
export function exactKeys(value: object, keys: string[]) {
  requireValue(
    value && typeof value === "object" && !Array.isArray(value),
    "Expected object",
  );
  requireValue(
    Object.keys(value).sort().join("|") === [...keys].sort().join("|"),
    "Unexpected or missing fields",
  );
}
export function timestamp(value: string) {
  requireValue(
    typeof value === "string" &&
      /(?:Z|[+-]\d\d:\d\d)$/.test(value) &&
      Number.isFinite(Date.parse(value)),
    "Timezone-aware timestamp required",
  );
  return Date.parse(value);
}
export function digest(value: string) {
  requireValue(
    typeof value === "string" && /^[a-f0-9]{64}$/.test(value),
    "Invalid SHA-256",
  );
}
function identifier(value: string) {
  requireValue(
    typeof value === "string" &&
      /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$/.test(value),
    "Invalid identifier",
  );
}

export function validateForecast(value: Forecast): Forecast {
  exactKeys(value, [
    "version",
    "snapshot_id",
    "race",
    "entrants",
    "evidence_mode",
    "cutoff",
    "generated_at",
    "completed_at",
    "archive_sha256",
    "prediction_sha256",
    "release_manifest_sha256",
    "feature_schema_sha256",
    "git_commit",
    "runtime",
    "code_sha256",
    "sources",
    "models",
    "baselines",
    "notice",
  ]);
  requireValue(value.version === 1, "Unsupported forecast version");
  identifier(value.snapshot_id);
  exactKeys(value.race, [
    "race_id",
    "year",
    "round",
    "grand_prix_id",
    "circuit_id",
    "race_start",
  ]);
  for (const key of ["race_id", "year", "round"] as const)
    requireValue(
      Number.isInteger(value.race[key]) && value.race[key] > 0,
      "Invalid race identity",
    );
  identifier(value.race.grand_prix_id);
  identifier(value.race.circuit_id);
  requireValue(
    ["prospective", "approximate_retrospective"].includes(value.evidence_mode),
    "Unknown evidence mode",
  );
  const cutoff = timestamp(value.cutoff),
    generated = timestamp(value.generated_at),
    completed = timestamp(value.completed_at);
  requireValue(
    cutoff <= generated && generated <= completed,
    "Invalid forecast chronology",
  );
  if (value.evidence_mode === "prospective")
    requireValue(
      value.race.race_start && completed < timestamp(value.race.race_start),
      "Forecast completed after race start",
    );
  for (const hash of [
    value.archive_sha256,
    value.prediction_sha256,
    value.release_manifest_sha256,
    value.feature_schema_sha256,
    ...Object.values(value.code_sha256),
  ])
    digest(hash);
  requireValue(
    value.release_manifest_sha256 === RELEASE_HASH,
    "Unapproved model release",
  );
  requireValue(
    value.feature_schema_sha256 === SCHEMA_HASH,
    "Changed feature schema",
  );
  exactKeys(value.runtime, ["python", "platform"]);
  requireValue(
    /^\d+\.\d+\.\d+$/.test(value.runtime.python) &&
      typeof value.runtime.platform === "string" &&
      value.runtime.platform.length < 200,
    "Invalid runtime metadata",
  );
  requireValue(/^[a-f0-9]{40}$/.test(value.git_commit), "Invalid code version");
  requireValue(
    typeof value.notice === "string" &&
      value.notice.length > 0 &&
      value.notice.length < 4000,
    "Notice required",
  );
  requireValue(
    value.entrants.length > 0 && value.entrants.length <= 40,
    "Invalid field size",
  );
  const ids = value.entrants.map((entry) => {
    exactKeys(entry, ["driver_id", "constructor_id"]);
    identifier(entry.driver_id);
    identifier(entry.constructor_id);
    return entry.driver_id;
  });
  requireValue(new Set(ids).size === ids.length, "Duplicate entrants");
  exactKeys(value.models, MODEL_NAMES);
  exactKeys(value.baselines, BASELINE_NAMES);
  for (const source of value.sources) {
    exactKeys(source, ["url", "sha256"]);
    digest(source.sha256);
    const url = new URL(source.url);
    requireValue(
      url.protocol === "https:" &&
        !url.username &&
        !url.password &&
        !url.search &&
        !url.hash,
      "Unsafe public source URL",
    );
  }
  for (const [name, method] of Object.entries({
    ...value.models,
    ...value.baselines,
  })) {
    requireValue(
      Object.keys(method).every((key) =>
        [
          "drivers",
          "predicted_order",
          "ranking_reference",
          "diagnostics",
          "rank_blend_alpha",
          "calibration_diagnostics",
        ].includes(key),
      ),
      "Unknown method field",
    );
    const diagnostics = method.diagnostics || {};
    const diagnosticFields = [
      "hierarchy_violation_rate",
      "highest_win_probability_driver",
      "mean_abs_podium_sum_error",
      "mean_abs_points_sum_error",
      "mean_abs_win_sum_error",
      "podium_rank_top3_overlap",
      "points_rank_top10_overlap",
      "ranking_p1",
      "winner_disagreement",
      "winner_rank_alignment",
    ];
    requireValue(
      Object.keys(diagnostics).every((key) => diagnosticFields.includes(key)),
      "Unknown diagnostic field",
    );
    for (const [key, item] of Object.entries(diagnostics)) {
      requireValue(
        ["ranking_p1", "highest_win_probability_driver"].includes(key)
          ? typeof item === "string" && ids.includes(item)
          : key === "winner_disagreement"
            ? typeof item === "boolean"
            : typeof item === "number" && Number.isFinite(item),
        "Invalid diagnostic value",
      );
    }
    if (method.ranking_reference !== undefined)
      requireValue(
        typeof method.ranking_reference === "string" &&
          method.ranking_reference.startsWith("final_grid;") &&
          method.ranking_reference.length < 200,
        "Invalid ranking reference",
      );
    const extra = method as Method & {
      rank_blend_alpha?: number;
      calibration_diagnostics?: Record<string, Record<string, number>>;
    };
    if (extra.rank_blend_alpha !== undefined)
      requireValue(
        Number.isFinite(extra.rank_blend_alpha) &&
          extra.rank_blend_alpha >= 0 &&
          extra.rank_blend_alpha <= 1,
        "Invalid blend weight",
      );
    if (extra.calibration_diagnostics) {
      exactKeys(extra.calibration_diagnostics, [...TARGETS]);
      for (const row of Object.values(extra.calibration_diagnostics)) {
        exactKeys(row, [
          "raw_mean",
          "calibrated_mean",
          "reconciled_mean",
          "max_calibration_shift",
          "max_reconciliation_shift",
        ]);
        requireValue(
          Object.values(row).every(
            (v) => Number.isFinite(v) && v >= 0 && v <= 1,
          ),
          "Invalid calibration diagnostic",
        );
      }
    }
    requireValue(
      method.drivers.length === ids.length &&
        method.predicted_order.length === ids.length,
      "Incomplete prediction field",
    );
    requireValue(
      [...method.predicted_order].sort().join("|") ===
        [...ids].sort().join("|"),
      "Prediction roster mismatch",
    );
    requireValue(
      new Set(method.drivers.map((row) => row.driver_id)).size === ids.length,
      "Duplicate prediction row",
    );
    const orderOnly = ["final_grid", "recent_form", "qualifying"].includes(
      name,
    );
    for (const row of method.drivers) {
      requireValue(
        Object.keys(row).every((key) =>
          ["driver_id", "predicted_order", "rank_score", ...STAGES].includes(
            key,
          ),
        ),
        "Unknown prediction field",
      );
      requireValue(
        Number.isInteger(row.predicted_order) &&
          row.predicted_order >= 1 &&
          method.predicted_order[row.predicted_order - 1] === row.driver_id,
        "Invalid finishing order",
      );
      requireValue(
        row.rank_score === undefined || Number.isFinite(row.rank_score),
        "Invalid ranking score",
      );
      for (const stage of STAGES) {
        const probabilities = row[stage];
        if (!probabilities) continue;
        requireValue(!orderOnly, "Order-only baseline has probabilities");
        exactKeys(probabilities, [...TARGETS]);
        for (const p of Object.values(probabilities))
          requireValue(
            Number.isFinite(p) && p >= 0 && p <= 1,
            "Invalid probability",
          );
      }
      if (!orderOnly) {
        const p = row.reconciled;
        requireValue(
          p &&
            p.p_race_winner <= p.p_podium_finish + 1e-8 &&
            p.p_podium_finish <= p.p_points_finish + 1e-8,
          "Missing or inconsistent reconciled probabilities",
        );
        if (MODEL_NAMES.includes(name))
          for (const stage of STAGES)
            requireValue(row[stage], "Missing frozen model stage");
      }
    }
    if (!orderOnly)
      TARGETS.forEach((target, i) =>
        requireValue(
          Math.abs(
            method.drivers.reduce(
              (sum, row) => sum + row.reconciled![target],
              0,
            ) - Math.min([1, 3, 10][i], ids.length),
          ) < 1e-6,
          "Race quota mismatch",
        ),
      );
  }
  return value;
}

export function validateReceipt(
  forecast: Forecast,
  receipt: Receipt,
  payloadHash: string,
) {
  validateForecast(forecast);
  exactKeys(receipt, [
    "version",
    "repository",
    "release_id",
    "asset_id",
    "api_url",
    "asset_url",
    "published_at",
    "asset_updated_at",
    "verified_at",
    "server_date",
    "payload_sha256",
    "api_sha256",
  ]);
  requireValue(
    forecast.evidence_mode === "prospective",
    "Replay cannot acquire prospective publication status",
  );
  requireValue(
    receipt.version === 1 && receipt.repository === "Jad-K-B/f1-race-forecasts",
    "Unapproved publication destination",
  );
  requireValue(
    Number.isSafeInteger(receipt.release_id) &&
      receipt.release_id > 0 &&
      Number.isSafeInteger(receipt.asset_id) &&
      receipt.asset_id > 0,
    "Invalid publication IDs",
  );
  requireValue(
    receipt.api_url ===
      `https://api.github.com/repos/${receipt.repository}/releases/${receipt.release_id}`,
    "Invalid receipt API URL",
  );
  const url = new URL(receipt.asset_url);
  requireValue(
    url.origin === "https://github.com" &&
      url.pathname.startsWith(`/${receipt.repository}/releases/download/`) &&
      !url.search &&
      !url.hash &&
      !url.username &&
      !url.password,
    "Invalid release asset URL",
  );
  digest(receipt.payload_sha256);
  digest(receipt.api_sha256);
  requireValue(
    receipt.payload_sha256 === payloadHash,
    "Published bytes do not match forecast",
  );
  const published = timestamp(receipt.published_at),
    updated = timestamp(receipt.asset_updated_at),
    verified = timestamp(receipt.verified_at),
    start = timestamp(forecast.race.race_start!);
  requireValue(
    timestamp(forecast.completed_at) <= published &&
      published < start &&
      updated >= timestamp(forecast.completed_at) &&
      updated < start &&
      verified >= Math.max(published, updated),
    "Publication timing invalid or late",
  );
  requireValue(
    Number.isFinite(Date.parse(receipt.server_date)) &&
      Math.abs(Date.parse(receipt.server_date) - verified) < 300000,
    "Server timestamp missing or inconsistent",
  );
}
