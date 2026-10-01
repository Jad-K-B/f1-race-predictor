"""Deterministic NumPy estimators and calibrators used by Stage 2B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


class LogisticRegressionGD:
    """L2-regularized logistic regression optimized by full-batch Adam."""

    def __init__(
        self,
        l2: float = 0.005,
        learning_rate: float = 0.03,
        max_iter: int = 600,
        class_weight: str | None = None,
    ):
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.weights: np.ndarray | None = None
        self.intercept = 0.0
        self.iterations = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("Logistic training arrays have incompatible shapes")
        prevalence = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.intercept = float(np.log(prevalence / (1.0 - prevalence)))
        sample_weight = np.ones(len(y), dtype=float)
        if self.class_weight == "balanced":
            positives = max(float(y.sum()), 1.0)
            negatives = max(float(len(y) - y.sum()), 1.0)
            sample_weight = np.where(
                y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)
            )
        elif self.class_weight is not None:
            raise ValueError(f"Unsupported class_weight: {self.class_weight}")
        sample_weight /= np.mean(sample_weight)

        first_w = np.zeros_like(self.weights)
        second_w = np.zeros_like(self.weights)
        first_b = 0.0
        second_b = 0.0
        best_loss = float("inf")
        stale = 0
        for iteration in range(1, self.max_iter + 1):
            probabilities = sigmoid(x @ self.weights + self.intercept)
            errors = sample_weight * (probabilities - y)
            gradient_w = x.T @ errors / len(y) + self.l2 * self.weights
            gradient_b = float(np.mean(errors))
            first_w = 0.9 * first_w + 0.1 * gradient_w
            second_w = 0.999 * second_w + 0.001 * gradient_w * gradient_w
            first_b = 0.9 * first_b + 0.1 * gradient_b
            second_b = 0.999 * second_b + 0.001 * gradient_b * gradient_b
            first_w_hat = first_w / (1.0 - 0.9**iteration)
            second_w_hat = second_w / (1.0 - 0.999**iteration)
            first_b_hat = first_b / (1.0 - 0.9**iteration)
            second_b_hat = second_b / (1.0 - 0.999**iteration)
            self.weights -= self.learning_rate * first_w_hat / (
                np.sqrt(second_w_hat) + 1e-8
            )
            self.intercept -= self.learning_rate * first_b_hat / (
                np.sqrt(second_b_hat) + 1e-8
            )
            if iteration % 10 == 0:
                current = sigmoid(x @ self.weights + self.intercept)
                loss = -float(
                    np.mean(
                        sample_weight
                        * (
                            y * np.log(np.clip(current, 1e-12, 1.0))
                            + (1.0 - y)
                            * np.log(np.clip(1.0 - current, 1e-12, 1.0))
                        )
                    )
                ) + 0.5 * self.l2 * float(self.weights @ self.weights)
                if loss < best_loss - 1e-7:
                    best_loss = loss
                    stale = 0
                else:
                    stale += 1
                if stale >= 8:
                    self.iterations = iteration
                    break
        if self.iterations == 0:
            self.iterations = self.max_iter
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fit")
        return np.asarray(x, dtype=float) @ self.weights + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.predict_score(x))

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None:
            raise ValueError("Cannot serialize an unfitted model")
        return {
            "type": "logistic_regression_gd",
            "parameters": {
                "l2": self.l2,
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
            },
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "iterations": self.iterations,
        }


def _thresholds_for_columns(x: np.ndarray, max_bins: int) -> list[np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, max_bins + 2)[1:-1]
    candidates: list[np.ndarray] = []
    for column in range(x.shape[1]):
        values = np.unique(x[:, column])
        if len(values) <= 1:
            candidates.append(np.array([], dtype=float))
        elif len(values) <= max_bins + 1:
            candidates.append((values[:-1] + values[1:]) / 2.0)
        else:
            candidates.append(np.unique(np.quantile(values, quantiles)))
    return candidates


@dataclass
class Stump:
    feature: int
    threshold: float
    left_value: float
    right_value: float


class GradientBoostedStumpClassifier:
    """Small deterministic gradient-boosted stump classifier."""

    def __init__(
        self,
        n_estimators: int = 60,
        learning_rate: float = 0.08,
        l2: float = 1.0,
        max_bins: int = 12,
        max_features: int = 48,
        min_leaf: int = 20,
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.max_bins = int(max_bins)
        self.max_features = int(max_features)
        self.min_leaf = int(min_leaf)
        self.random_state = int(random_state)
        self.initial_score = 0.0
        self.stumps: list[Stump] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostedStumpClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        prevalence = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
        self.initial_score = float(np.log(prevalence / (1.0 - prevalence)))
        scores = np.full(len(y), self.initial_score, dtype=float)
        thresholds = _thresholds_for_columns(x, self.max_bins)
        rng = np.random.default_rng(self.random_state)
        self.stumps = []
        for _ in range(self.n_estimators):
            probabilities = sigmoid(scores)
            gradient = y - probabilities
            hessian = probabilities * (1.0 - probabilities)
            feature_count = min(self.max_features, x.shape[1])
            features = rng.choice(x.shape[1], size=feature_count, replace=False)
            best_gain = -np.inf
            best: Stump | None = None
            for feature in features:
                values = x[:, feature]
                for threshold in thresholds[feature]:
                    left = values <= threshold
                    left_count = int(left.sum())
                    if left_count < self.min_leaf or len(y) - left_count < self.min_leaf:
                        continue
                    right = ~left
                    gradient_left = float(gradient[left].sum())
                    gradient_right = float(gradient[right].sum())
                    hessian_left = float(hessian[left].sum())
                    hessian_right = float(hessian[right].sum())
                    gain = gradient_left**2 / (hessian_left + self.l2)
                    gain += gradient_right**2 / (hessian_right + self.l2)
                    if gain > best_gain:
                        best_gain = gain
                        best = Stump(
                            int(feature),
                            float(threshold),
                            gradient_left / (hessian_left + self.l2),
                            gradient_right / (hessian_right + self.l2),
                        )
            if best is None:
                break
            self.stumps.append(best)
            scores += self.learning_rate * np.where(
                x[:, best.feature] <= best.threshold,
                best.left_value,
                best.right_value,
            )
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        scores = np.full(len(x), self.initial_score, dtype=float)
        for stump in self.stumps:
            scores += self.learning_rate * np.where(
                x[:, stump.feature] <= stump.threshold,
                stump.left_value,
                stump.right_value,
            )
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.predict_score(x))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gradient_boosted_stump_classifier",
            "parameters": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "max_bins": self.max_bins,
                "max_features": self.max_features,
                "min_leaf": self.min_leaf,
                "random_state": self.random_state,
            },
            "initial_score": self.initial_score,
            "stumps": [stump.__dict__ for stump in self.stumps],
        }


class RidgeRanker:
    """Ridge regression on normalized official finish order."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRanker":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.intercept = float(np.mean(y))
        centered = y - self.intercept
        gram = x.T @ x
        regularized = gram + self.alpha * np.eye(x.shape[1])
        self.weights = np.linalg.solve(regularized, x.T @ centered)
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fit")
        return np.asarray(x, dtype=float) @ self.weights + self.intercept

    def to_dict(self) -> dict[str, Any]:
        if self.weights is None:
            raise ValueError("Cannot serialize an unfitted model")
        return {
            "type": "ridge_ranker",
            "parameters": {"alpha": self.alpha},
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
        }


class GradientBoostedStumpRanker:
    """Squared-error boosting model for normalized finish order."""

    def __init__(
        self,
        n_estimators: int = 80,
        learning_rate: float = 0.08,
        max_bins: int = 12,
        max_features: int = 48,
        min_leaf: int = 20,
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_bins = int(max_bins)
        self.max_features = int(max_features)
        self.min_leaf = int(min_leaf)
        self.random_state = int(random_state)
        self.initial_score = 0.0
        self.stumps: list[Stump] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostedStumpRanker":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.initial_score = float(np.mean(y))
        scores = np.full(len(y), self.initial_score, dtype=float)
        thresholds = _thresholds_for_columns(x, self.max_bins)
        rng = np.random.default_rng(self.random_state)
        self.stumps = []
        for _ in range(self.n_estimators):
            residual = y - scores
            feature_count = min(self.max_features, x.shape[1])
            features = rng.choice(x.shape[1], size=feature_count, replace=False)
            best_gain = -np.inf
            best: Stump | None = None
            for feature in features:
                values = x[:, feature]
                for threshold in thresholds[feature]:
                    left = values <= threshold
                    left_count = int(left.sum())
                    right_count = len(y) - left_count
                    if left_count < self.min_leaf or right_count < self.min_leaf:
                        continue
                    right = ~left
                    left_sum = float(residual[left].sum())
                    right_sum = float(residual[right].sum())
                    gain = left_sum**2 / left_count + right_sum**2 / right_count
                    if gain > best_gain:
                        best_gain = gain
                        best = Stump(
                            int(feature),
                            float(threshold),
                            left_sum / left_count,
                            right_sum / right_count,
                        )
            if best is None:
                break
            self.stumps.append(best)
            scores += self.learning_rate * np.where(
                x[:, best.feature] <= best.threshold,
                best.left_value,
                best.right_value,
            )
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        scores = np.full(len(x), self.initial_score, dtype=float)
        for stump in self.stumps:
            scores += self.learning_rate * np.where(
                x[:, stump.feature] <= stump.threshold,
                stump.left_value,
                stump.right_value,
            )
        return scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gradient_boosted_stump_ranker",
            "parameters": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_bins": self.max_bins,
                "max_features": self.max_features,
                "min_leaf": self.min_leaf,
                "random_state": self.random_state,
            },
            "initial_score": self.initial_score,
            "stumps": [stump.__dict__ for stump in self.stumps],
        }


class IdentityCalibrator:
    name = "none"

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name}


class PlattCalibrator:
    name = "platt"

    def __init__(self, l2: float = 1e-3, max_iter: int = 100):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        score = logit(probabilities)
        y = np.asarray(y, dtype=float)
        prevalence = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
        self.slope = 0.0
        self.intercept = float(np.log(prevalence / (1.0 - prevalence)))
        for _ in range(self.max_iter):
            predicted = sigmoid(self.slope * score + self.intercept)
            residual = predicted - y
            weight = predicted * (1.0 - predicted)
            gradient = np.array(
                [float(np.sum(residual * score)) + self.l2 * self.slope, residual.sum()]
            )
            hessian = np.array(
                [
                    [float(np.sum(weight * score * score)) + self.l2, float(np.sum(weight * score))],
                    [float(np.sum(weight * score)), float(np.sum(weight)) + 1e-8],
                ]
            )
            step = np.linalg.solve(hessian, gradient)
            self.slope -= float(step[0])
            self.intercept -= float(step[1])
            if float(np.max(np.abs(step))) < 1e-8:
                break
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.clip(
            sigmoid(self.slope * logit(probabilities) + self.intercept),
            1e-8,
            1.0 - 1e-8,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "l2": self.l2,
            "slope": self.slope,
            "intercept": self.intercept,
        }


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self):
        self.thresholds = np.array([], dtype=float)
        self.values = np.array([], dtype=float)

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        probabilities = np.asarray(probabilities, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(probabilities, kind="mergesort")
        sorted_p = probabilities[order]
        sorted_y = y[order]
        unique_p, inverse = np.unique(sorted_p, return_inverse=True)
        sums = np.bincount(inverse, weights=sorted_y).astype(float)
        counts = np.bincount(inverse).astype(float)
        block_sums: list[float] = []
        block_counts: list[float] = []
        block_ends: list[int] = []
        for index, (total, count) in enumerate(zip(sums, counts)):
            block_sums.append(float(total))
            block_counts.append(float(count))
            block_ends.append(index)
            while len(block_sums) >= 2:
                previous = block_sums[-2] / block_counts[-2]
                current = block_sums[-1] / block_counts[-1]
                if previous <= current:
                    break
                block_sums[-2] += block_sums[-1]
                block_counts[-2] += block_counts[-1]
                block_ends[-2] = block_ends[-1]
                block_sums.pop()
                block_counts.pop()
                block_ends.pop()
        fitted = np.empty(len(unique_p), dtype=float)
        start = 0
        for total, count, end in zip(block_sums, block_counts, block_ends):
            fitted[start : end + 1] = total / count
            start = end + 1
        self.thresholds = unique_p
        self.values = np.clip(fitted, 1e-8, 1.0 - 1e-8)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if len(self.thresholds) == 0:
            raise ValueError("Calibrator is not fit")
        indices = np.searchsorted(self.thresholds, probabilities, side="right") - 1
        indices = np.clip(indices, 0, len(self.values) - 1)
        return self.values[indices]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "thresholds": self.thresholds.tolist(),
            "values": self.values.tolist(),
        }


def make_calibrator(name: str) -> IdentityCalibrator | PlattCalibrator | IsotonicCalibrator:
    if name == "none":
        return IdentityCalibrator()
    if name == "platt":
        return PlattCalibrator()
    if name == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibrator: {name}")


def model_from_dict(payload: dict[str, Any]) -> Any:
    """Restore one of the project-native fitted estimators."""

    model_type = payload["type"]
    parameters = dict(payload["parameters"])
    if model_type == "logistic_regression_gd":
        model = LogisticRegressionGD(**parameters)
        model.weights = np.asarray(payload["weights"], dtype=float)
        model.intercept = float(payload["intercept"])
        model.iterations = int(payload["iterations"])
        return model
    if model_type == "gradient_boosted_stump_classifier":
        model = GradientBoostedStumpClassifier(**parameters)
        model.initial_score = float(payload["initial_score"])
        model.stumps = [Stump(**stump) for stump in payload["stumps"]]
        return model
    if model_type == "ridge_ranker":
        model = RidgeRanker(**parameters)
        model.weights = np.asarray(payload["weights"], dtype=float)
        model.intercept = float(payload["intercept"])
        return model
    if model_type == "gradient_boosted_stump_ranker":
        model = GradientBoostedStumpRanker(**parameters)
        model.initial_score = float(payload["initial_score"])
        model.stumps = [Stump(**stump) for stump in payload["stumps"]]
        return model
    raise ValueError(f"Unknown serialized model type: {model_type}")


def calibrator_from_dict(
    payload: dict[str, Any],
) -> IdentityCalibrator | PlattCalibrator | IsotonicCalibrator:
    """Restore a fitted validation calibrator."""

    calibrator_type = payload["type"]
    if calibrator_type == "none":
        return IdentityCalibrator()
    if calibrator_type == "platt":
        calibrator = PlattCalibrator(l2=float(payload["l2"]))
        calibrator.slope = float(payload["slope"])
        calibrator.intercept = float(payload["intercept"])
        return calibrator
    if calibrator_type == "isotonic":
        calibrator = IsotonicCalibrator()
        calibrator.thresholds = np.asarray(payload["thresholds"], dtype=float)
        calibrator.values = np.asarray(payload["values"], dtype=float)
        return calibrator
    raise ValueError(f"Unknown serialized calibrator type: {calibrator_type}")
