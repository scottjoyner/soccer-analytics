from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    log_loss: float
    brier_score: float
    accuracy: float
    majority_baseline_accuracy: float


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("probability rows must have positive sums")
    return probabilities / row_sums


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = normalize_probabilities(probabilities)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = normalize_probabilities(probabilities)
    clipped = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def classification_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.argmax(probabilities, axis=1)
    return float(np.mean(predictions == labels))


def majority_baseline_accuracy(labels: np.ndarray) -> float:
    counts = np.bincount(labels) if labels.dtype.kind != "f" else np.bincount(labels.astype(int))
    return float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0


def score_classification(probabilities: np.ndarray, labels: np.ndarray) -> ClassificationMetrics:
    return ClassificationMetrics(
        log_loss=multiclass_log_loss(probabilities, labels),
        brier_score=multiclass_brier_score(probabilities, labels),
        accuracy=classification_accuracy(probabilities, labels),
        majority_baseline_accuracy=majority_baseline_accuracy(labels),
    )
