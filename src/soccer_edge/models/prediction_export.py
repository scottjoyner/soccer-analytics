from pathlib import Path

import pandas as pd

from soccer_edge.models.bundle import load_bundle


def export_bundle_predictions(
    bundle_dir: Path,
    source: Path,
    output: Path,
    feature_columns: list[str] | None = None,
) -> Path:
    model, metadata = load_bundle(bundle_dir)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    features = feature_columns or metadata.feature_names
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    x_values = frame[features]
    output_frame = frame.copy()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_values)
        for idx in range(probabilities.shape[1]):
            output_frame[f"prob_{idx}"] = probabilities[:, idx]
    else:
        output_frame["prediction"] = model.predict(x_values)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        output_frame.to_parquet(output, index=False)
    else:
        output_frame.to_csv(output, index=False)
    return output
