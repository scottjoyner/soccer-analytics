import pandas as pd

from soccer_edge.models.prediction_export import export_bundle_predictions
from soccer_edge.models.simple_classifier import fit_simple_classifier


def test_export_bundle_predictions(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"x": 0.0, "label": 0},
            {"x": 1.0, "label": 1},
            {"x": 2.0, "label": 1},
            {"x": -1.0, "label": 0},
        ]
    )
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    bundle_dir = tmp_path / "bundle"
    fit_simple_classifier(frame, ["x"], "label", bundle_dir)
    output = export_bundle_predictions(bundle_dir, train_path, tmp_path / "predictions.csv")
    assert output.exists()
    predictions = pd.read_csv(output)
    assert "prob_0" in predictions.columns
