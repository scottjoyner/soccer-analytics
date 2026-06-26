from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from soccer_edge.models.bundle import save_bundle


def fit_simple_classifier(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    output_dir: Path,
) -> dict[str, Path]:
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    x_values = frame[feature_columns]
    y_values = frame[label_column]
    model = LogisticRegression(max_iter=1000)
    model.fit(x_values, y_values)
    predictions = model.predict(x_values)
    metrics = {"accuracy": float(accuracy_score(y_values, predictions))}
    return save_bundle(
        model=model,
        output_dir=output_dir,
        name="simple_logistic_regression",
        version="v0",
        feature_names=feature_columns,
        metrics=metrics,
    )
