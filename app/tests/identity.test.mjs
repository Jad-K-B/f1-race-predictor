import test from "node:test";
import assert from "node:assert/strict";
import { resolveIdentity } from "../src/identity-assets.ts";

const portrait = { src: "/test/portrait.png" };
const logo = { src: "/test/logo.png" };
const registry = {
  seasons: {
    "2023": {
      drivers: { sample_driver: { sample_team: portrait } },
      teams: { sample_team: logo },
    },
  },
};

test("identity lookup requires exact season and team", () => {
  assert.deepEqual(resolveIdentity(2023, "sample_driver", "sample_team", registry), {
    portrait, logo,
  });
  assert.equal(resolveIdentity(2024, "sample_driver", "sample_team", registry).portrait, undefined);
  assert.equal(resolveIdentity(2023, "sample_driver", "other_team", registry).portrait, undefined);
  assert.equal(resolveIdentity(2023, "other_driver", "sample_team", registry).portrait, undefined);
});

test("public source defaults to no third-party identity images", () => {
  assert.deepEqual(resolveIdentity(2023, "sample_driver", "sample_team", { seasons: {} }), {
    portrait: undefined, logo: undefined,
  });
});
