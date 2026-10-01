import test from "node:test";
import assert from "node:assert/strict";
import { scoreProspective } from "../scripts/score-prospective.mjs";
import { contractFixture } from "./forecast-fixture.mjs";
const fixture = () => {
  const forecast = contractFixture();
  return {
    forecast,
    attachment: {
      race_id: forecast.race.race_id,
      forecast_manifest_sha256: forecast.archive_sha256,
      completed_at: "2026-10-04T09:00:00Z",
      source: {
        provider: "fia",
        sha256: "f".repeat(64),
        reviewed_by: "Test fixture",
        retrieved_at: "2026-10-04T10:00:00Z",
      },
      results: {
        race_id: forecast.race.race_id,
        drivers: forecast.entrants.map((row, i) => ({
          driver_id: row.driver_id,
          finish_order: i + 1,
          official_position: i + 1,
          official_points: [25, 18, 15][i],
        })),
      },
    },
  };
};
test("prospective scores use all methods, official points and immutable inputs", () => {
  const { forecast, attachment } = fixture(),
    before = JSON.stringify({ forecast, attachment });
  const result = scoreProspective(forecast, attachment);
  assert.equal(Object.keys(result.methods).length, 8);
  assert.equal(result.methods.external.ranking.mae, 0);
  assert.equal(result.methods.external.ranking.spearman, 1);
  assert.ok(
    Math.abs(
      result.methods.external.probability.reconciled.p_race_winner.brier -
        2 / 9,
    ) < 1e-12,
  );
  assert.deepEqual(result.methods.final_grid.probability, {});
  assert.equal(JSON.stringify({ forecast, attachment }), before);
  attachment.results.drivers[2].official_points = 0;
  attachment.results.drivers[2].official_position = null;
  assert.equal(
    scoreProspective(forecast, attachment).methods.external.probability
      .reconciled.p_points_finish.brier,
    1 / 3,
  );
});
test("mismatched, premature, unreviewed or incomplete outcomes are rejected", () => {
  for (const mutate of [
    (a) => a.race_id++,
    (a) => (a.completed_at = "2026-10-04T06:00:00Z"),
    (a) => (a.source.reviewed_by = ""),
    (a) => a.results.drivers.pop(),
    (a) => (a.results.drivers[1].finish_order = 1),
  ]) {
    const { forecast, attachment } = fixture();
    mutate(attachment);
    assert.throws(() => scoreProspective(forecast, attachment));
  }
});
