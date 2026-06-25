from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    bins: list[CalibrationBin]


def confidence_bins(probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> CalibrationReport:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels
    bins: list[CalibrationBin] = []
    total = len(labels)
    ece = 0.0

    for idx in range(num_bins):
        lower = idx / num_bins
        upper = (idx + 1) / num_bins
        if idx == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        avg_confidence = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += (count / total) * abs(avg_confidence - accuracy)
        bins.append(CalibrationBin(lower, upper, count, avg_confidence, accuracy))

    return CalibrationReport(expected_calibration_error=float(ece), bins=bins)


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)
