import type { CSSProperties } from "react";
import { colors, fullName, percent, type Method, type Stage } from "./data";
import IdentityImage from "./IdentityImage";
import { resolveIdentity } from "./identity-assets";
import identityRegistry from "virtual:identity-registry";

export default function PredictedPodium({
  prediction,
  methodLabel,
  stage,
  selectedDriver,
  onSelectDriver,
  season,
  entrants,
  evidenceLabel = "HISTORICAL REPLAY",
}: {
  prediction: Method;
  methodLabel: string;
  stage: Stage;
  selectedDriver: string;
  onSelectDriver: (id: string) => void;
  season: number;
  entrants: { driver_id: string; constructor_id: string }[];
  evidenceLabel?: string;
}) {
  const teams = Object.fromEntries(
    entrants.map((entrant) => [entrant.driver_id, entrant.constructor_id]),
  );
  const podium = [...prediction.drivers]
    .sort((a, b) => a.predicted_order - b.predicted_order)
    .slice(0, 3);
  const probabilityAvailable = podium.some((driver) => driver[stage]);
  return (
    <section className="predicted-podium" aria-labelledby="podium-title">
      <div className="podium-heading">
        <div>
          <p className="eyebrow">
            {evidenceLabel} / {season}
          </p>
          <h3 id="podium-title">Predicted podium</h3>
        </div>
        <p className="mono">
          {methodLabel} /{" "}
          {probabilityAvailable
            ? fullName(stage.replaceAll("_", "-"))
            : "Order only"}
        </p>
      </div>
      <div className="podium-grid">
        {podium.map((driver) => {
          const identity = resolveIdentity(
            season,
            driver.driver_id,
            teams[driver.driver_id],
            identityRegistry,
          );
          return (
            <article
              className={`podium-card ${driver.predicted_order === 1 ? "podium-leader" : ""}`}
              key={driver.driver_id}
              data-driver={driver.driver_id}
              data-season={season}
              data-team={teams[driver.driver_id]}
              style={
                {
                  "--team-color": colors[teams[driver.driver_id]] || "#90999e",
                } as CSSProperties
              }
              aria-label={`Predicted P${driver.predicted_order}: ${fullName(driver.driver_id)}`}
            >
              <div className="podium-position">
                <span className="mono">PREDICTED FINISH</span>
                <strong>P{driver.predicted_order}</strong>
              </div>
              <button
                className="podium-identity"
                onClick={() => onSelectDriver(driver.driver_id)}
                aria-pressed={selectedDriver === driver.driver_id}
                aria-label={`View ${fullName(driver.driver_id)} details`}
                title={`View ${fullName(driver.driver_id)} details`}
              >
                <span className="podium-driver-name">
                  <span className="podium-team">
                    {identity.logo && (
                      <IdentityImage
                        key={`${season}-${teams[driver.driver_id]}`}
                        asset={identity.logo}
                      />
                    )}
                    {teams[driver.driver_id]
                      ? fullName(teams[driver.driver_id])
                      : "Team unavailable"}
                  </span>
                  <strong>{fullName(driver.driver_id)}</strong>
                </span>
              </button>
              <dl>
                <div>
                  <dt>Win probability</dt>
                  <dd data-stat="win">
                    {percent(driver[stage]?.p_race_winner)}
                  </dd>
                </div>
                <div>
                  <dt>Podium probability</dt>
                  <dd data-stat="podium">
                    {percent(driver[stage]?.p_podium_finish)}
                  </dd>
                </div>
                <div className="podium-start">
                  <dt>Starting position</dt>
                  <dd>Unavailable</dd>
                </div>
              </dl>
            </article>
          );
        })}
      </div>
      <p className="podium-note">
        Starting positions are not included in this frontend archive.
        {!probabilityAvailable &&
          " This baseline provides an order only, without probabilities."}
        {prediction.ranking_reference &&
          " This method uses the starting-grid baseline for its order."}
      </p>
    </section>
  );
}
