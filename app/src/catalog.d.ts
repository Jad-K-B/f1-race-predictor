declare module "virtual:forecast-catalog" {
  const catalog: {
    publicBuild: boolean;
    events: {
      id: string;
      forecast: import("./forecast-contract").Forecast;
      receipt: import("./forecast-contract").Receipt | null;
    }[];
    benchmark: {
      probability: Record<string, string | number>[];
      ranking: Record<string, string | number>[];
      calibration: Record<string, string | number>[];
      uncertainty: Record<string, string | number>[];
    };
    readiness: {
      recorded_at: string;
      blockers: string[];
      forecast_published: boolean;
      pre_weekend_snapshot_generated: boolean;
      status: string;
    };
  };
  export default catalog;
}
declare module "virtual:identity-registry" {
  const registry: import("./identity-assets").IdentityRegistry;
  export default registry;
}
