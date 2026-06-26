from pathlib import Path

import pandas as pd


def select_low_confidence_rows(
    frame: pd.DataFrame,
    confidence_column: str = "confidence",
    threshold: float = 0.5,
    limit: int | None = None,
) -> pd.DataFrame:
    if confidence_column not in frame.columns:
        raise ValueError(f"missing confidence column: {confidence_column}")
    selected = frame[frame[confidence_column] <= threshold].sort_values(confidence_column).reset_index(drop=True)
    if limit is not None:
        selected = selected.head(limit)
    return selected


def write_low_confidence_rows(
    source: Path,
    output: Path,
    confidence_column: str = "confidence",
    threshold: float = 0.5,
    limit: int | None = None,
) -> Path:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    selected = select_low_confidence_rows(frame, confidence_column=confidence_column, threshold=threshold, limit=limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        selected.to_parquet(output, index=False)
    else:
        selected.to_csv(output, index=False)
    return output
