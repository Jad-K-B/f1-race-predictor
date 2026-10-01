"""Retrospective early-model comparison; never a live forecast entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..stage2a import BINARY_TARGET_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ..stage2b_data import COMPACT_CATEGORICAL_FEATURES, COMPACT_NUMERIC_FEATURES
from .policy import CANDIDATE_FEATURES, EXCLUDED_FEATURES

RESEARCH_MODEL_SCHEMA = "early-retrospective-model-comparison-v1"
FOLDS = ((2017, 2018), (2019, 2020), (2021, 2022))
VALIDATION_BLOCKS = ("selection", "calibration", "report")
IDENTITY_FEATURES = frozenset({"driver_id", "constructor_id", "engine_manufacturer_id"})
COMPACT = frozenset(COMPACT_CATEGORICAL_FEATURES + COMPACT_NUMERIC_FEATURES)
FEATURE_SETS = {
    "compact": tuple(column for column in CANDIDATE_FEATURES if column in COMPACT),
    "id_free": tuple(column for column in CANDIDATE_FEATURES if column not in IDENTITY_FEATURES),
    "full": CANDIDATE_FEATURES,
}


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResearchCohort:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    feature_sets: dict[str, tuple[str, ...]]
    source: Path

    def block(self, name: str) -> pd.DataFrame:
        if name == "train":
            return self.frame[self.frame.split.eq("train")].copy().reset_index(drop=True)
        if name not in VALIDATION_BLOCKS:
            raise ValueError(f"Unknown validation block: {name}")
        return self.frame[self.frame.validation_block.eq(name)].copy().reset_index(drop=True)


def load_research_cohort(directory: Path) -> ResearchCohort:
    """Load only the reviewed <=2023 private cohort and verify its contracts."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "early-retrospective-dataset-v1"
            or manifest.get("evidence_mode") != "retrospective_research"
            or manifest.get("model_training_performed") is not False
            or manifest.get("sealed_test_opened") is not False
            or manifest.get("blocks") != {
                "train": 162, "selection": 8, "calibration": 7, "report": 7
            }):
        raise ValueError("Expected verified 2014-2023 retrospective cohort")
    for name, digest in manifest["files_sha256"].items():
        if checksum(directory / name) != digest:
            raise ValueError(f"Research cohort hash mismatch: {name}")
    features = pd.read_csv(directory / "features.csv", low_memory=False)
    outcomes = pd.read_csv(directory / "outcomes.csv", low_memory=False)
    expected = ["race_id", "race_date", "cutoff", "split", *CANDIDATE_FEATURES]
    if features.columns.tolist() != expected:
        raise ValueError("Early feature order or schema differs from the reviewed cohort")
    if set(features.columns) & EXCLUDED_FEATURES:
        raise ValueError("Current-race qualifying/grid feature in early cohort")
    if (len(features) != len(outcomes) or
            not features[["race_id", "driver_id"]].equals(outcomes[["race_id", "driver_id"]]) or
            features.duplicated(["race_id", "driver_id"]).any()):
        raise ValueError("Early rows or targets are misaligned")
    frame = features.join(outcomes.drop(columns=["race_id", "driver_id"])).copy()
    if (set(frame.year.astype(int)) != set(range(2014, 2024))
            or not frame.loc[frame.year.le(2022), "split"].eq("train").all()
            or not frame.loc[frame.year.eq(2023), "split"].eq("validation").all()
            or frame.year.gt(2023).any()):
        raise ValueError("Research years or chronological splits changed")
    if not frame.groupby("race_id").split.nunique().eq(1).all():
        raise ValueError("Race rows cross splits")
    if not all(frame.groupby("race_id")[target].sum().eq(expected_count).all()
               for target, expected_count in (("race_winner", 1), ("podium_finish", 3))):
        raise ValueError("Winner or podium target count differs from official results")
    if not frame[BINARY_TARGET_COLUMNS].isin([0, 1]).all().all():
        raise ValueError("Binary targets must be 0/1")
    if not (frame.ranking_available.eq(frame.finish_order.notna().astype(int))).all():
        raise ValueError("Unranked outcome has an invented order")
    cutoff = pd.to_datetime(frame.cutoff, utc=True)
    race_day = pd.to_datetime(frame.race_date, utc=True)
    if (cutoff.isna().any() or race_day.isna().any() or
            not ((cutoff < race_day) & (cutoff > race_day - pd.Timedelta(days=6))).all() or
            not frame.groupby("race_id").cutoff.nunique().eq(1).all()):
        raise ValueError("Research cutoff or race date is inconsistent")
    coverage = json.loads((directory / "coverage.json").read_text(encoding="utf-8"))
    block_map = {int(item["race_id"]): item["validation_block"] for item in coverage}
    if len(block_map) != len(coverage) or set(block_map) != set(frame.race_id):
        raise ValueError("Research coverage map is incomplete")
    frame["validation_block"] = frame.race_id.map(block_map)
    if (not frame.loc[frame.split.eq("train"), "validation_block"].eq("train").all()
            or not frame.loc[frame.split.eq("validation"), "validation_block"].isin(VALIDATION_BLOCKS).all()):
        raise ValueError("Validation block assignments changed")
    ordered = frame.loc[frame.year.eq(2023), ["race_id", "race_date", "validation_block"]]
    ordered = ordered.drop_duplicates().sort_values(["race_date", "race_id"])
    if ordered.validation_block.tolist() != [
        *(["selection"] * 8), *(["calibration"] * 7), *(["report"] * 7)
    ]:
        raise ValueError("2023 validation block chronology changed")
    for end_year, validation_year in FOLDS:
        if (not frame.year.le(end_year).any() or
                not frame.year.eq(validation_year).any() or
                validation_year <= end_year):
            raise ValueError("Temporal training fold is empty or reversed")
    if any(not selected or not set(selected) <= set(CANDIDATE_FEATURES)
           for selected in FEATURE_SETS.values()):
        raise ValueError("Early feature set is empty or contains unavailable inputs")
    return ResearchCohort(frame, manifest, FEATURE_SETS, directory)


def feature_groups(feature_set: str) -> tuple[list[str], list[str]]:
    selected = FEATURE_SETS[feature_set]
    categorical = [name for name in selected if name in CATEGORICAL_FEATURES]
    numeric = [name for name in selected if name in NUMERIC_FEATURES]
    if len(categorical) + len(numeric) != len(selected):
        raise ValueError("Untyped early feature")
    return categorical, numeric


def make_preprocessor(feature_set: str, scale: bool) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical, numeric = feature_groups(feature_set)
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
            ]), categorical),
        ],
        remainder="drop", sparse_threshold=1.0
    )


@dataclass(frozen=True)
class Candidate:
    family: str
    feature_set: str
    parameters: dict[str, Any]

    @property
    def name(self) -> str:
        suffix = "_".join(f"{key}_{value}" for key, value in sorted(self.parameters.items()))
        return f"{self.family}_{self.feature_set}_{suffix}"


def classifier_candidates() -> tuple[Candidate, ...]:
    candidates = []
    for feature_set in FEATURE_SETS:
        for c_value in (0.1, 1.0):
            candidates.append(Candidate("logistic", feature_set, {"C": c_value}))
        candidates.append(Candidate("random_forest", feature_set, {
            "n_estimators": 180, "max_depth": 7, "min_samples_leaf": 4
        }))
        for depth in (2, 3):
            candidates.append(Candidate("xgboost", feature_set, {
                "n_estimators": 180, "max_depth": depth, "learning_rate": 0.05,
                "min_child_weight": 4, "subsample": 0.85,
                "colsample_bytree": 0.8, "reg_lambda": 3.0,
            }))
    return tuple(candidates)


def ranker_candidates() -> tuple[Candidate, ...]:
    return tuple(Candidate("xgboost_ranker", feature_set, {
        "objective": "rank:pairwise", "n_estimators": 180,
        "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 4,
        "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 3.0,
    }) for feature_set in FEATURE_SETS)


def fit_classifier(candidate: Candidate, train: pd.DataFrame,
                   target: str, *, seed: int = 2026, n_jobs: int = 2) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    if target not in BINARY_TARGET_COLUMNS:
        raise ValueError("Unknown early binary target")
    if train.split.ne("train").any() or train.year.gt(2022).any():
        raise ValueError("Early classifier fit requires training seasons only")
    if candidate.family == "logistic":
        estimator = LogisticRegression(
            C=float(candidate.parameters["C"]), solver="liblinear",
            max_iter=2000, random_state=seed
        )
    elif candidate.family == "random_forest":
        estimator = RandomForestClassifier(
            **candidate.parameters, max_features="sqrt", random_state=seed,
            n_jobs=n_jobs
        )
    elif candidate.family == "xgboost":
        estimator = XGBClassifier(
            **candidate.parameters, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=seed,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Unknown early classifier family")
    model = Pipeline([
        ("preprocessor", make_preprocessor(candidate.feature_set,
                                           scale=candidate.family == "logistic")),
        ("estimator", estimator),
    ])
    features = FEATURE_SETS[candidate.feature_set]
    model.fit(train[list(features)], train[target].to_numpy(int))
    return model


def classifier_probabilities(model: Any, candidate: Candidate,
                             frame: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(
        model.predict_proba(frame[list(FEATURE_SETS[candidate.feature_set])]),
        dtype=float
    )
    if probability.shape != (len(frame), 2) or not np.isfinite(probability).all():
        raise ValueError("Invalid early classifier probabilities")
    return probability[:, 1]


def _rankable(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame[frame.ranking_available.eq(1)]
            .sort_values(["race_date", "race_id", "driver_id"])
            .reset_index(drop=True))


def fit_ranker(candidate: Candidate, train: pd.DataFrame, *, seed: int = 2026,
               n_jobs: int = 2) -> tuple[Any, Any]:
    from xgboost import XGBRanker

    if candidate.family != "xgboost_ranker":
        raise ValueError("Unknown early ranker family")
    if train.split.ne("train").any() or train.year.gt(2022).any():
        raise ValueError("Early ranker fit requires training seasons only")
    ordered = _rankable(train)
    if ordered.empty or ordered.race_id.nunique() < 2:
        raise ValueError("Race-grouped ranking requires multiple training races")
    features = list(FEATURE_SETS[candidate.feature_set])
    preprocessor = make_preprocessor(candidate.feature_set, scale=False)
    transformed = preprocessor.fit_transform(ordered[features])
    field_size = ordered.groupby("race_id").finish_order.transform("max")
    relevance = (field_size - ordered.finish_order + 1).to_numpy(float)
    groups = ordered.groupby("race_id", sort=False).size().to_numpy(int)
    ranker = XGBRanker(**candidate.parameters, tree_method="hist",
                       random_state=seed, n_jobs=n_jobs)
    ranker.fit(transformed, relevance, group=groups, verbose=False)
    return preprocessor, ranker


def rank_scores(preprocessor: Any, ranker: Any, candidate: Candidate,
                frame: pd.DataFrame) -> np.ndarray:
    features = list(FEATURE_SETS[candidate.feature_set])
    scores = -np.asarray(ranker.predict(preprocessor.transform(frame[features])),
                         dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError("Invalid early ranking scores")
    return scores


def temporal_classifier_cv(cohort: ResearchCohort, *, n_jobs: int = 2) -> pd.DataFrame:
    from ..stage2b_metrics import brier_score, log_loss

    train = cohort.block("train")
    rows = []
    for target in BINARY_TARGET_COLUMNS:
        for candidate in classifier_candidates():
            for end_year, validation_year in FOLDS:
                fit_rows = train[train.year.le(end_year)]
                validation = train[train.year.eq(validation_year)]
                if fit_rows.empty or validation.empty:
                    raise ValueError("Empty temporal fold")
                model = fit_classifier(candidate, fit_rows, target, n_jobs=n_jobs)
                probability = classifier_probabilities(model, candidate, validation)
                labels = validation[target].to_numpy(int)
                rows.append(dict(
                    target=target, candidate=candidate.name,
                    family=candidate.family, feature_set=candidate.feature_set,
                    train_end_year=end_year, validation_year=validation_year,
                    validation_races=int(validation.race_id.nunique()),
                    log_loss=log_loss(labels, probability),
                    brier=brier_score(labels, probability),
                ))
    return pd.DataFrame(rows)


def select_classifier_candidates(cv: pd.DataFrame) -> dict[str, dict[str, Candidate]]:
    candidates = {candidate.name: candidate for candidate in classifier_candidates()}
    required = len(BINARY_TARGET_COLUMNS) * len(candidates) * len(FOLDS)
    if len(cv) != required or cv[["target", "candidate", "validation_year"]].duplicated().any():
        raise ValueError("Incomplete temporal classifier comparison")
    selected = {}
    for target in BINARY_TARGET_COLUMNS:
        selected[target] = {}
        for family in ("logistic", "random_forest", "xgboost"):
            subset = cv[cv.target.eq(target) & cv.family.eq(family)]
            means = subset.groupby("candidate", sort=False).log_loss.mean()
            name = sorted(means.index, key=lambda item: (means[item], item))[0]
            selected[target][family] = candidates[name]
    return selected


def temporal_ranker_cv(cohort: ResearchCohort, *, n_jobs: int = 2) -> pd.DataFrame:
    from ..stage2b_metrics import ranking_metrics

    train = cohort.block("train")
    rows = []
    for candidate in ranker_candidates():
        for end_year, validation_year in FOLDS:
            fit_rows = train[train.year.le(end_year)]
            validation = _rankable(train[train.year.eq(validation_year)])
            preprocessor, ranker = fit_ranker(candidate, fit_rows, n_jobs=n_jobs)
            metrics = ranking_metrics(validation, rank_scores(preprocessor, ranker,
                                                              candidate, validation))
            rows.append(dict(
                candidate=candidate.name, feature_set=candidate.feature_set,
                train_end_year=end_year, validation_year=validation_year,
                validation_races=int(validation.race_id.nunique()), **metrics
            ))
    return pd.DataFrame(rows)


def baseline_scores(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "recent_form":
        driver = pd.to_numeric(frame.driver_recent_5_avg_finish, errors="coerce")
        constructor = pd.to_numeric(frame.constructor_recent_5_avg_finish, errors="coerce")
        return -driver.fillna(constructor).fillna(30.0).to_numpy(float)
    if method == "constructor_form":
        constructor = pd.to_numeric(frame.constructor_recent_5_points_per_race,
                                    errors="coerce").fillna(0.0)
        driver = pd.to_numeric(frame.driver_recent_5_points_per_start,
                               errors="coerce").fillna(0.0)
        return (constructor + 0.5 * driver).to_numpy(float)
    raise ValueError("Unknown early baseline")


def _baseline_ranks(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    work = frame[["race_id", "driver_id"]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    work = work.sort_values(["race_id", "score", "driver_id"],
                            ascending=[True, False, True], kind="mergesort")
    work["rank"] = work.groupby("race_id", sort=False).cumcount() + 1
    return work.sort_index()["rank"].to_numpy(int)


@dataclass
class Baseline:
    method: str
    rates: dict[str, dict[int, float]]
    prior: dict[str, float]
    rank_scores_only: bool = False

    @classmethod
    def fit(cls, train: pd.DataFrame, method: str) -> "Baseline":
        if train.split.ne("train").any() or train.year.gt(2022).any():
            raise ValueError("Baseline fit requires training seasons only")
        prior = {target: float(train[target].mean()) for target in BINARY_TARGET_COLUMNS}
        if method == "prior_rate":
            return cls(method, {}, prior)
        ranks = _baseline_ranks(train, baseline_scores(train, method))
        rates = {}
        for target in BINARY_TARGET_COLUMNS:
            table = pd.DataFrame({"rank": ranks, "positive": train[target].to_numpy(int)})
            counts = table.groupby("rank").positive.agg(["sum", "count"])
            rates[target] = {
                int(rank): float((item["sum"] + 20 * prior[target]) / (item["count"] + 20))
                for rank, item in counts.iterrows()
            }
        return cls(method, rates, prior)

    def predict(self, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if self.method == "prior_rate":
            scores = np.zeros(len(frame), dtype=float)
            probability = {
                target: np.full(len(frame), self.prior[target], dtype=float)
                for target in BINARY_TARGET_COLUMNS
            }
            return probability, scores
        scores = baseline_scores(frame, self.method)
        ranks = _baseline_ranks(frame, scores)
        probability = {
            target: np.asarray([
                self.rates[target].get(int(rank), self.prior[target]) for rank in ranks
            ], dtype=float)
            for target in BINARY_TARGET_COLUMNS
        }
        return probability, ranks.astype(float)
