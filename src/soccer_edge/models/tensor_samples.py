from pathlib import Path

import numpy as np
import pandas as pd


def build_npz_tensor_samples(
    frame: pd.DataFrame,
    spatial_columns: list[str],
    label_column: str,
    output_path: Path,
    sequence_length: int = 1,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> Path:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    expected = channels * height * width
    if len(spatial_columns) != expected:
        raise ValueError(f"spatial column count must equal {expected}")
    missing = [column for column in spatial_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing spatial columns: {missing}")

    values = frame[spatial_columns].to_numpy(dtype=np.float32).reshape(len(frame), channels, height, width)
    labels = frame[label_column].to_numpy(dtype=np.int64)
    samples = []
    sample_labels = []
    for end_idx in range(sequence_length - 1, len(frame)):
        start_idx = end_idx - sequence_length + 1
        samples.append(values[start_idx : end_idx + 1])
        sample_labels.append(labels[end_idx])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, spatial=np.stack(samples).astype(np.float32), labels=np.asarray(sample_labels, dtype=np.int64))
    return output_path


def build_npz_from_table(
    source: Path,
    output_path: Path,
    spatial_columns: list[str],
    label_column: str,
    sequence_length: int = 1,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> Path:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    return build_npz_tensor_samples(
        frame=frame,
        spatial_columns=spatial_columns,
        label_column=label_column,
        output_path=output_path,
        sequence_length=sequence_length,
        channels=channels,
        height=height,
        width=width,
    )
