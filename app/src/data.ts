import catalog from "virtual:forecast-catalog";
export type Task = "p_race_winner" | "p_podium_finish" | "p_points_finish";
export type Stage =
  | "raw"
  | "calibrated"
  | "rank_calibrated"
  | "blended"
  | "reconciled";
export type Driver = {
  driver_id: string;
  predicted_order: number;
  rank_score?: number;
} & Partial<Record<Stage, Record<Task, number>>>;
export type Method = {
  ranking_reference?: string;
  drivers: Driver[];
  predicted_order: string[];
  diagnostics?: Record<string, unknown>;
};
export const data = catalog;
export const labels: Record<string, string> = {
  external: "External ML",
  numpy: "NumPy",
  external_selected: "External ML",
  numpy_selected: "NumPy",
  final_grid: "Starting grid",
  recent_form: "Recent form",
  qualifying: "Qualifying",
  final_grid_rule: "Grid rule",
  train_grid_frequency: "Train-grid frequency",
  uniform_field_quota: "Uniform field",
};
export const tasks: Record<Task, string> = {
  p_race_winner: "Winner",
  p_podium_finish: "Podium",
  p_points_finish: "Points",
};
export const stages: Stage[] = [
  "raw",
  "calibrated",
  "rank_calibrated",
  "blended",
  "reconciled",
];
export const fullName = (id: string) =>
  id
    .split("-")
    .map((v) => v.charAt(0).toUpperCase() + v.slice(1))
    .join(" ");
export const percent = (value: number | undefined) =>
  value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
export const colors: Record<string, string> = {
  "red-bull": "#6a84d9",
  ferrari: "#dd5448",
  mercedes: "#48b8a7",
  "aston-martin": "#39987c",
  mclaren: "#e49036",
  alpine: "#ce8ccc",
  williams: "#5b9cdb",
  "alfa-romeo": "#b35766",
  haas: "#90999e",
  alphatauri: "#8896b3",
};
export function download(name: string, payload: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
