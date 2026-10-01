"""Stage 2B probabilistic, ranking, calibration, and coherence metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stage2a import BINARY_TARGET_COLUMNS
from .stage2b_models import PlattCalibrator


TARGET_PROBABILITY_COLUMNS = {
    "points_finish": "p_points_finish",
    "podium_finish": "p_podium_finish",
    "race_winner": "p_race_winner",
}


def log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities, dtype=float) - y) ** 2))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def roc_auc(y: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(probabilities, dtype=float))
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def average_precision(y: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probabilities, dtype=float), kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * sorted_y) / positives)


def expected_calibration_error(
    y: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(y, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(y[mask].mean())
            )
    return total


def binary_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    calibration = PlattCalibrator().fit(probabilities, y)
    return {
        "log_loss": log_loss(y, probabilities),
        "brier": brier_score(y, probabilities),
        "roc_auc": roc_auc(y, probabilities),
        "average_precision": average_precision(y, probabilities),
        "ece_10": expected_calibration_error(y, probabilities, bins=10),
        "calibration_slope": float(calibration.slope),
        "calibration_intercept": float(calibration.intercept),
        "prevalence": float(np.mean(y)),
        "mean_prediction": float(np.mean(probabilities)),
    }


def _race_predicted_order(scores: np.ndarray) -> np.ndarray:
    return _average_ranks(np.asarray(scores, dtype=float))


def _ndcg(actual_order: np.ndarray, predicted_order: np.ndarray, k: int) -> float:
    field_size = len(actual_order)
    relevance = field_size - np.asarray(actual_order, dtype=float) + 1.0
    predicted_indices = np.argsort(predicted_order, kind="mergesort")[:k]
    ideal_indices = np.argsort(actual_order, kind="mergesort")[:k]
    discounts = np.log2(np.arange(2, len(predicted_indices) + 2))
    dcg = float(np.sum(relevance[predicted_indices] / discounts))
    idcg = float(np.sum(relevance[ideal_indices] / discounts))
    return dcg / idcg if idcg else float("nan")


def ranking_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    if len(frame) != len(scores):
        raise ValueError("Ranking frame and score lengths differ")
    work = frame[["race_id", "finish_order"]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    race_metrics: list[dict[str, float]] = []
    for _, race in work.groupby("race_id", sort=False):
        actual = race["finish_order"].to_numpy(float)
        predicted = _race_predicted_order(race["score"].to_numpy(float))
        spearman = float(np.corrcoef(actual, predicted)[0, 1])
        concordant = 0
        discordant = 0
        for left in range(len(race)):
            for right in range(left + 1, len(race)):
                product = (actual[left] - actual[right]) * (
                    predicted[left] - predicted[right]
                )
                concordant += int(product > 0)
                discordant += int(product < 0)
        pairs = concordant + discordant
        predicted_top = np.argsort(predicted, kind="mergesort")
        actual_top = np.argsort(actual, kind="mergesort")
        race_metrics.append(
            {
                "mae": float(np.mean(np.abs(actual - predicted))),
                "spearman": spearman,
                "kendall": (concordant - discordant) / pairs if pairs else float("nan"),
                "ndcg_3": _ndcg(actual, predicted, min(3, len(race))),
                "ndcg_10": _ndcg(actual, predicted, min(10, len(race))),
                "winner_top1": float(predicted_top[0] == actual_top[0]),
                "podium_overlap": len(set(predicted_top[:3]) & set(actual_top[:3])) / 3.0,
                "podium_set_exact": float(set(predicted_top[:3]) == set(actual_top[:3])),
                "podium_order_exact": float(
                    np.array_equal(predicted_top[:3], actual_top[:3])
                ),
                "points_top10_overlap": len(
                    set(predicted_top[:10]) & set(actual_top[:10])
                )
                / min(10, len(race)),
            }
        )
    return {
        key: float(np.nanmean([race[key] for race in race_metrics]))
        for key in race_metrics[0]
    }


def _project_bounded_sum(
    values: np.ndarray, target: float, upper_bounds: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    target = float(np.clip(target, 0.0, upper_bounds.sum()))
    low = float(np.min(values - upper_bounds) - 1.0)
    high = float(np.max(values) + 1.0)
    for _ in range(80):
        midpoint = (low + high) / 2.0
        total = float(np.minimum(np.maximum(values - midpoint, 0.0), upper_bounds).sum())
        if total > target:
            low = midpoint
        else:
            high = midpoint
    return np.minimum(
        np.maximum(values - (low + high) / 2.0, 0.0), upper_bounds
    )


def _project_row_monotonic(values: np.ndarray) -> np.ndarray:
    blocks = [[float(value), 1] for value in values]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][0] <= blocks[index + 1][0]:
            index += 1
            continue
        total_count = int(blocks[index][1] + blocks[index + 1][1])
        mean = (
            blocks[index][0] * blocks[index][1]
            + blocks[index + 1][0] * blocks[index + 1][1]
        ) / total_count
        blocks[index : index + 2] = [[mean, total_count]]
        index = max(index - 1, 0)
    projected: list[float] = []
    for value, count in blocks:
        projected.extend([value] * int(count))
    return np.asarray(projected, dtype=float)


def reconcile_probabilities(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Project probabilities onto exact nested per-race target constraints."""

    columns = [
        TARGET_PROBABILITY_COLUMNS["race_winner"],
        TARGET_PROBABILITY_COLUMNS["podium_finish"],
        TARGET_PROBABILITY_COLUMNS["points_finish"],
    ]
    output = probabilities.copy()
    for _, indices in frame.groupby("race_id", sort=False).groups.items():
        positions = frame.index.get_indexer(indices)
        race_values = output.iloc[positions][columns].to_numpy(float, copy=True)
        targets = np.array([1.0, min(3.0, len(positions)), min(10.0, len(positions))])
        for row in range(len(race_values)):
            race_values[row] = _project_row_monotonic(race_values[row])
        race_values[:, 2] = _project_bounded_sum(
            race_values[:, 2], targets[2], np.ones(len(race_values))
        )
        race_values[:, 1] = _project_bounded_sum(
            race_values[:, 1], targets[1], race_values[:, 2]
        )
        race_values[:, 0] = _project_bounded_sum(
            race_values[:, 0], targets[0], race_values[:, 1]
        )
        output.iloc[positions, output.columns.get_indexer(columns)] = race_values
    return output


def probability_consistency_metrics(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    rank_scores: np.ndarray,
) -> dict[str, float]:
    p_win = probabilities[TARGET_PROBABILITY_COLUMNS["race_winner"]].to_numpy(float)
    p_podium = probabilities[TARGET_PROBABILITY_COLUMNS["podium_finish"]].to_numpy(float)
    p_points = probabilities[TARGET_PROBABILITY_COLUMNS["points_finish"]].to_numpy(float)
    hierarchy_violations = (p_win > p_podium + 1e-9) | (
        p_podium > p_points + 1e-9
    )
    work = frame[["race_id"]].copy()
    work["p_win"] = p_win
    work["p_podium"] = p_podium
    work["p_points"] = p_points
    work["rank_score"] = np.asarray(rank_scores, dtype=float)
    sum_errors = {"win": [], "podium": [], "points": []}
    winner_alignment: list[float] = []
    podium_alignment: list[float] = []
    points_alignment: list[float] = []
    for _, race in work.groupby("race_id", sort=False):
        sum_errors["win"].append(abs(float(race["p_win"].sum()) - 1.0))
        sum_errors["podium"].append(abs(float(race["p_podium"].sum()) - 3.0))
        sum_errors["points"].append(abs(float(race["p_points"].sum()) - 10.0))
        rank_order = np.argsort(race["rank_score"].to_numpy(), kind="mergesort")
        winner_order = np.argsort(-race["p_win"].to_numpy(), kind="mergesort")
        podium_order = np.argsort(-race["p_podium"].to_numpy(), kind="mergesort")
        points_order = np.argsort(-race["p_points"].to_numpy(), kind="mergesort")
        winner_alignment.append(float(rank_order[0] == winner_order[0]))
        podium_alignment.append(
            len(set(rank_order[:3]) & set(podium_order[:3])) / 3.0
        )
        points_alignment.append(
            len(set(rank_order[:10]) & set(points_order[:10])) / min(10, len(race))
        )
    return {
        "hierarchy_violation_rate": float(np.mean(hierarchy_violations)),
        "mean_abs_win_sum_error": float(np.mean(sum_errors["win"])),
        "mean_abs_podium_sum_error": float(np.mean(sum_errors["podium"])),
        "mean_abs_points_sum_error": float(np.mean(sum_errors["points"])),
        "winner_rank_alignment": float(np.mean(winner_alignment)),
        "podium_rank_top3_overlap": float(np.mean(podium_alignment)),
        "points_rank_top10_overlap": float(np.mean(points_alignment)),
    }


def evaluate_probability_frame(
    frame: pd.DataFrame, probabilities: pd.DataFrame
) -> dict[str, Any]:
    return {
        target: binary_metrics(
            frame[target].to_numpy(float),
            probabilities[TARGET_PROBABILITY_COLUMNS[target]].to_numpy(float),
        )
        for target in BINARY_TARGET_COLUMNS
    }
