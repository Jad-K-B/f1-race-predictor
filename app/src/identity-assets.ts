export type IdentityAsset = {
  src: string;
  source: string;
  sha256: string;
  width: number;
  height: number;
  retrievedAt: string;
};
export type IdentityRegistry = {
  seasons: Record<
    string,
    {
      drivers: Record<string, Record<string, IdentityAsset>>;
      teams: Record<string, IdentityAsset>;
    }
  >;
};

// A portrait is specific to both season and constructor. Never fall back to a
// different season or a driver's other team when an exact match is absent.
export function resolveIdentity(
  season: number,
  driverId: string,
  teamId: string,
  registry: IdentityRegistry,
) {
  const entry = registry.seasons[String(season)];
  return {
    portrait: entry?.drivers[driverId]?.[teamId],
    logo: entry?.teams[teamId],
  };
}
