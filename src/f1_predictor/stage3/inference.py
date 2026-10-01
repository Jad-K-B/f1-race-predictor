"""Transform/predict only. No fit, calibration fitting, or selection at serving time."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..stage2a import FEATURE_COLUMNS, BINARY_TARGET_COLUMNS
from ..stage2b_data import EXTERNAL_FEATURE_SETS, SplitSafePreprocessor
from ..stage2b_external import _predict_rank_scores
from ..stage2b_models import model_from_dict, calibrator_from_dict, sigmoid
from ..stage2b_metrics import TARGET_PROBABILITY_COLUMNS, reconcile_probabilities, probability_consistency_metrics
from ..stage2b_training import _grid_scores, _recent_form_scores, _qualifying_scores, _quota_probabilities
from .contracts import FeatureBatch
from .release import verify_release, sha256


class FrozenPredictor:
    def __init__(self, release: Path):
        import joblib
        from xgboost import XGBRanker

        self.release = release
        self.manifest = verify_release(release)
        self.numpy = json.loads((release / "numpy.json").read_text())
        self.selection = json.loads((release / "external_selection.json").read_text())
        self.calibrators = json.loads((release / "calibrators.json").read_text())
        self.rates = json.loads((release / "baseline_rates.json").read_text())
        self.external = {}
        for target, family in self.selection["selected_families"].items():
            saved = joblib.load(release / "external" / f"{target}__{family}.joblib")
            if family == "random_forest":
                saved["pipeline"].named_steps["classifier"].set_params(n_jobs=self.manifest["execution_policy"]["random_forest_n_jobs"])
            self.external[target] = saved
        self.rank_preprocessor = joblib.load(release / "external/finish_order__xgboost_preprocessor.joblib")
        self.ranker = XGBRanker()
        self.ranker.load_model(release / "external/finish_order__xgboost_ranker.json")

    def predict(self, batch: FeatureBatch) -> dict[str, Any]:
        if batch.snapshot.kind != "confirmed_grid":
            raise ValueError("No frozen early/provisional-grid model is available")
        frame = batch.frame
        if list(frame.columns) != ["race_id", "race_date", *FEATURE_COLUMNS]:
            raise ValueError("Feature schema/order differs from frozen release")
        if frame.empty or frame.race_id.nunique() != 1 or frame.driver_id.duplicated().any():
            raise ValueError("One complete, unique-driver field required")
        if set(frame.driver_id) != {e["driver_id"] for e in batch.snapshot.entries if e["eligible"]}:
            raise ValueError("Feature rows differ from reviewed eligible field")
        output: dict[str, Any] = {"release_version": self.manifest["release_version"],
            "release_manifest_sha256": sha256(self.release / "manifest.json"),
            "models": {}, "baselines": {}, "selection_policy": "all frozen named methods; no post-test winner selection",
            "probability_contract": "official race points > 0; official podium/winner; nested marginal quotas 1/3/10",
            "simulation_contract": "rank_score lower is better; marginals are not a joint finishing-order distribution"}
        output["data_quality"] = {
            "missing_features": {k: int(v) for k, v in frame[FEATURE_COLUMNS].isna().sum().items() if v},
            "rookies": frame.loc[frame.driver_prior_entries.eq(0), "driver_id"].tolist(),
            "evidence_mode": batch.snapshot.evidence_mode,
            "reliability_warning": "Frozen historical models may degrade under new regulations, entrants, circuits or missing-data patterns"}
        for family in ("numpy", "external"):
            raw = pd.DataFrame(index=frame.index)
            calibrated = pd.DataFrame(index=frame.index)
            for target in BINARY_TARGET_COLUMNS:
                if family == "numpy":
                    saved = self.numpy["binary_models"][target]
                    pre = SplitSafePreprocessor.from_dict(saved["preprocessor"])
                    values = model_from_dict(saved["model"]).predict_proba(pre.transform(frame))
                else:
                    saved = self.external[target]
                    values = saved["pipeline"].predict_proba(frame[EXTERNAL_FEATURE_SETS[saved["candidate"]["feature_set"]]])[:, 1]
                column = TARGET_PROBABILITY_COLUMNS[target]
                raw[column] = values
                calibrated[column] = calibrator_from_dict(self.calibrators[family]["binary"][target]).predict(values)
            if family == "numpy":
                saved = self.numpy["ranking_model"]
                pre = SplitSafePreprocessor.from_dict(saved["preprocessor"])
                scores = model_from_dict(saved["model"]).predict_score(pre.transform(frame))
            else:
                scores = _predict_rank_scores(self.rank_preprocessor, self.ranker, self.selection["selected_ranker"], frame)
            rank_base = sigmoid(-5.0 * (scores - 0.5))
            rank_probabilities = pd.DataFrame({TARGET_PROBABILITY_COLUMNS[t]: calibrator_from_dict(
                self.calibrators[family]["rank"][t]).predict(rank_base) for t in BINARY_TARGET_COLUMNS})
            alpha = self.calibrators[family]["rank_blend_alpha"]
            blended = (1.0 - alpha) * calibrated + alpha * rank_probabilities
            reconciled = reconcile_probabilities(frame, blended)
            stages = {"raw": raw, "calibrated": calibrated, "rank_calibrated": rank_probabilities,
                      "blended": blended, "reconciled": reconciled}
            output["models"][family] = self._serialize(frame, scores, stages)
            output["models"][family]["rank_blend_alpha"] = alpha
            output["models"][family]["calibration_diagnostics"] = {
                col: {"raw_mean": float(raw[col].mean()), "calibrated_mean": float(calibrated[col].mean()),
                      "reconciled_mean": float(reconciled[col].mean()),
                      "max_calibration_shift": float((raw[col] - calibrated[col]).abs().max()),
                      "max_reconciliation_shift": float((blended[col] - reconciled[col]).abs().max())}
                for col in raw.columns}
        ranks = {"final_grid": _grid_scores(frame), "recent_form": _recent_form_scores(frame), "qualifying": _qualifying_scores(frame)}
        for name, scores in ranks.items():
            output["baselines"][name] = self._serialize(frame, scores, {})
        frequencies = pd.DataFrame(index=frame.index)
        slots = np.rint(_grid_scores(frame)).astype(int)
        for target, column in TARGET_PROBABILITY_COLUMNS.items():
            saved = self.rates[target]
            frequencies[column] = [saved["rates"].get(str(slot), saved["prior"]) for slot in slots]
        probabilities = {"uniform_field_quota": _quota_probabilities(frame, np.zeros(len(frame))),
                         "final_grid_rule": _quota_probabilities(frame, ranks["final_grid"]),
                         "train_grid_frequency": reconcile_probabilities(frame, frequencies)}
        for name, values in probabilities.items():
            output["baselines"][name] = self._serialize(frame, ranks["final_grid"], {"reconciled": values})
            output["baselines"][name]["ranking_reference"] = "final_grid; baseline supplies marginals, not an independently learned order"
        return output

    @staticmethod
    def _serialize(frame: pd.DataFrame, scores: np.ndarray, stages: dict[str, pd.DataFrame]) -> dict[str, Any]:
        if not np.isfinite(scores).all() or any(not np.isfinite(p.to_numpy()).all() for p in stages.values()):
            raise ValueError("Non-finite model output")
        order = np.argsort(scores, kind="mergesort")
        positions = np.empty(len(frame), dtype=int)
        positions[order] = np.arange(1, len(frame) + 1)
        rows = []
        for index, driver in enumerate(frame.driver_id):
            rows.append({"driver_id": driver, "predicted_order": int(positions[index]), "rank_score": float(scores[index]),
                         **{name: {col: float(value) for col, value in values.iloc[index].items()} for name, values in stages.items()}})
        diagnostics = {}
        if "reconciled" in stages:
            p = stages["reconciled"]
            diagnostics = probability_consistency_metrics(frame, p, scores)
            diagnostics["mean_abs_podium_sum_error"] = abs(float(p[TARGET_PROBABILITY_COLUMNS["podium_finish"]].sum()) - min(3, len(frame)))
            diagnostics["mean_abs_points_sum_error"] = abs(float(p[TARGET_PROBABILITY_COLUMNS["points_finish"]].sum()) - min(10, len(frame)))
            win_index = int(np.argmax(p[TARGET_PROBABILITY_COLUMNS["race_winner"]].to_numpy()))
            diagnostics.update(ranking_p1=str(frame.iloc[order[0]].driver_id),
                               highest_win_probability_driver=str(frame.iloc[win_index].driver_id),
                               winner_disagreement=bool(win_index != order[0]))
        return {"drivers": rows, "predicted_order": frame.iloc[order].driver_id.tolist(), "diagnostics": diagnostics}
