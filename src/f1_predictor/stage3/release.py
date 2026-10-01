"""Recover the predetermined final-tested calibration, never base-model fitting."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..stage2a import BINARY_TARGET_COLUMNS, FEATURE_COLUMNS
from ..stage2b_data import EXTERNAL_FEATURE_SETS, SplitSafePreprocessor, load_stage2b_training_data
from ..stage2b_external import _predict_rank_scores, verify_external_environment
from ..stage2b_models import make_calibrator, model_from_dict, sigmoid
from ..stage2b_training import _grid_scores


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Missing or changed frozen artifact: {path}")


def verify_release(directory: Path, *, require_ready: bool = True) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["feature_order"] != FEATURE_COLUMNS:
        raise ValueError("Frozen feature schema changed")
    for name, digest in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Invalid release member")
        checked(path, digest)
    root = Path(__file__).resolve().parents[3]
    for name, digest in manifest["implementation_sha256"].items():
        checked(root / name, digest)
    runtime = verify_external_environment()
    for name in ("numpy", "pandas", "scikit_learn", "xgboost", "joblib"):
        if runtime[name] != manifest["runtime"][name]:
            raise ValueError(f"Frozen runtime mismatch: {name}")
    if require_ready and manifest.get("release_status") != "approved_ready":
        raise ValueError("Recovery candidate is BLOCKED pending exact-recovery approval")
    return manifest


def recover_release(root: Path, output: Path, *, approve_numerical_recovery: bool = False) -> dict[str, Any]:
    """Only calibrators are fit; the opened test CSV is never read."""
    if output.exists():
        raise FileExistsError(f"Release already exists: {output}")
    final_dir = root / "artifacts/stage2b/final_test"
    final = json.loads((final_dir / "manifest.json").read_text())
    audit = final["frozen_audit"]
    np_dir = root / "artifacts/stage2b/validation"
    ext_dir = root / "artifacts/stage2b/external_validation"
    data_dir = root / "data/processed/stage2a"
    checked(np_dir / "manifest.json", audit["numpy"]["manifest_sha256"])
    checked(np_dir / "model_bundle.json", audit["numpy"]["model_bundle_sha256"])
    checked(ext_dir / "manifest.json", audit["external"]["manifest_sha256"])
    ext = json.loads((ext_dir / "manifest.json").read_text())
    code_hashes = {**ext["implementation_sha256"], **final["evaluation_code_sha256"]}
    for name, digest in code_hashes.items():
        checked(root / name, digest)
    for name, digest in audit["external"]["model_artifacts_sha256"].items():
        checked(ext_dir / "models" / name, digest)
    for name in ("train.csv", "validation.csv"):
        checked(data_dir / name, final["data_sha256"][name])
    checked(final_dir / "FINAL_TEST_REPORT.md", final["artifacts_sha256"]["FINAL_TEST_REPORT.md"])
    runtime = verify_external_environment()
    for name in ("numpy", "pandas", "scikit_learn", "xgboost", "joblib"):
        if runtime[name] != final["runtime"][name]:
            raise ValueError(f"Exact recovery needs final-tested {name} version")
    train, validation, _ = load_stage2b_training_data(data_dir)
    bundle = json.loads((np_dir / "model_bundle.json").read_text())
    calibration: dict[str, Any] = {}
    numerical_check: dict[str, Any] = {}
    import joblib
    from xgboost import XGBRanker

    for family in ("numpy", "external"):
        binary = {}
        for target in BINARY_TARGET_COLUMNS:
            if family == "numpy":
                saved = bundle["binary_models"][target]
                pre = SplitSafePreprocessor.from_dict(saved["preprocessor"])
                raw = model_from_dict(saved["model"]).predict_proba(pre.transform(validation))
                kind = saved["validation_calibrator"]["type"]
            else:
                selected = ext["selected_families"][target]
                saved = joblib.load(ext_dir / "models" / f"{target}__{selected}.joblib")
                features = EXTERNAL_FEATURE_SETS[saved["candidate"]["feature_set"]]
                if approve_numerical_recovery and selected == "random_forest":
                    original = saved["pipeline"].predict_proba(validation[features])[:, 1]
                    saved["pipeline"].named_steps["classifier"].set_params(n_jobs=1)
                raw = np.asarray(saved["pipeline"].predict_proba(validation[features])[:, 1], dtype=float)
                kind = saved["calibrator"]["type"]
                if approve_numerical_recovery and selected == "random_forest":
                    repeated = saved["pipeline"].predict_proba(validation[features])[:, 1]
                    np.testing.assert_array_equal(raw, repeated)
                    np.testing.assert_allclose(raw, original, rtol=0, atol=1e-14)
                    prior = make_calibrator(kind).fit(original, validation[target].to_numpy(float)).to_dict()
                    sequential = make_calibrator(kind).fit(raw, validation[target].to_numpy(float)).to_dict()
                    np.testing.assert_allclose(prior["thresholds"], sequential["thresholds"], rtol=0, atol=1e-14)
                    np.testing.assert_array_equal(prior["values"], sequential["values"])
                    numerical_check[target] = {"absolute_tolerance": 1e-14,
                        "max_raw_difference": float(np.max(np.abs(original - raw))),
                        "deterministic_repeat": True, "isotonic_values_identical": True,
                        "note": "Threshold equivalence only; step-boundary outputs need not be bit-identical"}
            binary[target] = make_calibrator(kind).fit(raw, validation[target].to_numpy(float)).to_dict()
        if family == "numpy":
            saved = bundle["ranking_model"]
            scores = model_from_dict(saved["model"]).predict_score(
                SplitSafePreprocessor.from_dict(saved["preprocessor"]).transform(validation))
            kinds = {t: bundle["rank_probability_calibrators"][t]["type"] for t in BINARY_TARGET_COLUMNS}
            alpha = bundle["reconciliation"]["rank_blend_alpha"]
        else:
            ranker = XGBRanker()
            ranker.load_model(ext_dir / "models/finish_order__xgboost_ranker.json")
            pre = joblib.load(ext_dir / "models/finish_order__xgboost_preprocessor.joblib")
            scores = _predict_rank_scores(pre, ranker, ext["selected_ranker"], validation)
            kinds = {t: "platt" for t in BINARY_TARGET_COLUMNS}
            alpha = ext["selected_reconciliation_alpha"]
        base = sigmoid(-5.0 * (scores - 0.5))
        rank = {t: make_calibrator(kinds[t]).fit(base, validation[t].to_numpy(float)).to_dict()
                for t in BINARY_TARGET_COLUMNS}
        calibration[family] = {"binary": binary, "rank": rank, "rank_blend_alpha": alpha}

    # Freeze the existing training-grid baseline sufficient statistics, not new estimates.
    rates = {}
    for target in BINARY_TARGET_COLUMNS:
        prior = float(train[target].mean())
        grouped = pd.DataFrame({"grid": np.rint(_grid_scores(train)).astype(int),
                                "target": train[target].to_numpy(float)}).groupby("grid")["target"].agg(["sum", "count"])
        rates[target] = {"prior": prior, "rates": {str(k): float(v) for k, v in
                          ((grouped["sum"] + 8.0 * prior) / (grouped["count"] + 8.0)).items()}}
    output.mkdir(parents=True)
    shutil.copyfile(np_dir / "model_bundle.json", output / "numpy.json")
    models = output / "external"
    models.mkdir()
    for name in audit["external"]["model_artifacts_sha256"]:
        shutil.copyfile(ext_dir / "models" / name, models / name)
    shutil.copyfile(ext_dir / "manifest.json", output / "external_selection.json")
    write_json(output / "calibrators.json", calibration)
    write_json(output / "baseline_rates.json", rates)
    manifest = {
        "release_version": "stage3-frozen-v2" if approve_numerical_recovery else "stage3-frozen-v1", "feature_order": FEATURE_COLUMNS,
        "runtime": runtime, "calibration_rows": len(validation),
        "calibration_year": 2023, "base_model_fit_performed": False,
        "model_selection_performed": False, "test_data_read": False,
        "recovery": "Predetermined Stage 2B procedure reproduced on all eligible 2023 rows; exact original RF isotonic thresholds unavailable",
        "release_status": "approved_ready" if approve_numerical_recovery else "blocked_exact_recovery_unproven",
        "execution_policy": {"random_forest_n_jobs": 1 if approve_numerical_recovery else 8},
        "numerical_recovery_approved": approve_numerical_recovery,
        "numerical_check": numerical_check,
        "final_manifest_sha256": sha256(final_dir / "manifest.json"),
        "final_report_sha256": sha256(final_dir / "FINAL_TEST_REPORT.md"),
        "source_data_sha256": audit["train_validation_sha256"],
        "frozen_audit": audit, "implementation_sha256": code_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    return verify_release(output, require_ready=False)
