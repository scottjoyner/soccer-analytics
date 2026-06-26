from pathlib import Path

import numpy as np
import pandas as pd


def validate_tensor_columns(
    frame: pd.DataFrame,
    spatial_columns: list[str],
    label_column: str,
    channels: int,
    height: int,
    width: int,
) -> None:
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    expected = channels * height * width
    if len(spatial_columns) != expected:
        raise ValueError(f"spatial column count must equal {expected}")
    missing = [column for column in spatial_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing spatial columns: {missing}")


def append_sequence_rows(
    frame: pd.DataFrame,
    spatial_columns: list[str],
    label_column: str,
    sequence_length: int,
    channels: int,
    height: int,
    width: int,
    samples: list[np.ndarray],
    labels: list[int],
) -> None:
    values = frame[spatial_columns].to_numpy(dtype=np.float32).reshape(len(frame), channels, height, width)
    label_values = frame[label_column].to_numpy(dtype=np.int64)
    for end_idx in range(sequence_length - 1, len(frame)):
        start_idx = end_idx - sequence_length + 1
        samples.append(values[start_idx : end_idx + 1])
        labels.append(int(label_values[end_idx]))


def ordered_frame(frame: pd.DataFrame, order_column: str | None) -> pd.DataFrame:
    if order_column is None:
        return frame.reset_index(drop=True)
    if order_column not in frame.columns:
        raise ValueError(f"missing order column: {order_column}")
    return frame.sort_values(order_column).reset_index(drop=True)


def build_npz_tensor_samples(
    frame: pd.DataFrame,
    spatial_columns: list[str],
    label_column: str,
    output_path: Path,
    sequence_length: int = 1,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
    group_column: str | None = None,
    order_column: str | None = None,
) -> Path:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    validate_tensor_columns(frame, spatial_columns, label_column, channels, height, width)
    if group_column is not None and group_column not in frame.columns:
        raise ValueError(f"missing group column: {group_column}")

    samples: list[np.ndarray] = []
    labels: list[int] = []
    if group_column is None:
        append_sequence_rows(ordered_frame(frame, order_column), spatial_columns, label_column, sequence_length, channels, height, width, samples, labels)
    else:
        for _, group in frame.groupby(group_column, sort=False):
            append_sequence_rows(ordered_frame(group, order_column), spatial_columns, label_column, sequence_length, channels, height, width, samples, labels)
    if not samples:
        raise ValueError("no tensor samples produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, spatial=np.stack(samples).astype(np.float32), labels=np.asarray(labels, dtype=np.int64))
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
    group_column: str | None = None,
    order_column: str | None = None,
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
        group_column=group_column,
        order_column=order_column,
    )
