import numpy as np
import pytest

from soccer_edge.models.calibration import confidence_bins, temperature_scale_logits
from soccer_edge.models.metrics import classification_accuracy, multiclass_brier_score, multiclass_log_loss, score_classification


def test_classification_metrics() -> None:
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    labels = np.array([0, 1])
    assert classification_accuracy(probs, labels) == 1.0
    assert multiclass_log_loss(probs, labels) > 0.0
    assert multiclass_brier_score(probs, labels) > 0.0
    report = score_classification(probs, labels)
    assert report.accuracy == 1.0


def test_confidence_bins() -> None:
    probs = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]])
    labels = np.array([0, 1])
    report = confidence_bins(probs, labels, num_bins=2)
    assert len(report.bins) == 2
    assert report.expected_calibration_error >= 0.0


def test_temperature_scale_logits() -> None:
    logits = np.array([[2.0, 1.0, 0.0]])
    probs = temperature_scale_logits(logits, temperature=1.0)
    assert probs.shape == (1, 3)
    assert round(float(probs.sum()), 6) == 1.0
    with pytest.raises(ValueError):
        temperature_scale_logits(logits, temperature=0.0)
