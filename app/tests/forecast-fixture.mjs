import {
  RELEASE_HASH,
  SCHEMA_HASH,
  MODEL_NAMES,
  BASELINE_NAMES,
  STAGES,
} from "../src/forecast-contract.ts";
// Synthetic contract data only. Never used as forecasts, training data or results.
export function contractFixture(count = 3) {
  const ids = Array.from({ length: count }, (_, i) => `test-driver-${i + 1}`);
  const probability = {
    p_race_winner: 1 / count,
    p_podium_finish: Math.min(3, count) / count,
    p_points_finish: Math.min(10, count) / count,
  };
  const method = (probabilities) => ({
    predicted_order: ids,
    drivers: ids.map((id, i) => ({
      driver_id: id,
      predicted_order: i + 1,
      rank_score: i,
      ...(probabilities
        ? Object.fromEntries(STAGES.map((stage) => [stage, { ...probability }]))
        : {}),
    })),
  });
  return {
    version: 1,
    snapshot_id: "contract-test-only",
    race: {
      race_id: 99999,
      year: 2026,
      round: 1,
      grand_prix_id: "contract-test-only",
      circuit_id: "test-track",
      race_start: "2026-10-04T07:00:00Z",
    },
    entrants: ids.map((driver_id) => ({
      driver_id,
      constructor_id: "test-team",
    })),
    evidence_mode: "prospective",
    cutoff: "2026-10-04T05:00:00Z",
    generated_at: "2026-10-04T05:01:00Z",
    completed_at: "2026-10-04T05:02:00Z",
    archive_sha256: "a".repeat(64),
    prediction_sha256: "b".repeat(64),
    release_manifest_sha256: RELEASE_HASH,
    feature_schema_sha256: SCHEMA_HASH,
    git_commit: "d".repeat(40),
    runtime: { python: "3.12.14", platform: "Test fixture" },
    code_sha256: { "src/f1_predictor/stage3/inference.py": "e".repeat(64) },
    sources: [],
    models: Object.fromEntries(MODEL_NAMES.map((name) => [name, method(true)])),
    baselines: Object.fromEntries(
      BASELINE_NAMES.map((name) => [
        name,
        method(!["final_grid", "recent_form", "qualifying"].includes(name)),
      ]),
    ),
    notice:
      "CONTRACT TEST ONLY. Synthetic values, not an actual race or forecast.",
  };
}
