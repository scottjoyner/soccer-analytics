from pathlib import Path

import pandas as pd


def split_annotation_rows(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
    group_column: str | None = "frame_idx",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if group_column is not None and group_column in frame.columns:
        groups = list(dict.fromkeys(frame[group_column].tolist()))
        split_idx = max(1, int(len(groups) * train_fraction)) if len(groups) > 1 else 1
        train_groups = set(groups[:split_idx])
        train = frame[frame[group_column].isin(train_groups)].reset_index(drop=True)
        val = frame[~frame[group_column].isin(train_groups)].reset_index(drop=True)
        return train, val
    split_idx = max(1, int(len(frame) * train_fraction)) if len(frame) > 1 else len(frame)
    return frame.iloc[:split_idx].reset_index(drop=True), frame.iloc[split_idx:].reset_index(drop=True)


def write_annotation_split(
    source: Path,
    train_output: Path,
    val_output: Path,
    train_fraction: float = 0.8,
    group_column: str | None = "frame_idx",
) -> dict[str, Path]:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    train, val = split_annotation_rows(frame, train_fraction=train_fraction, group_column=group_column)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_output, index=False)
    val.to_csv(val_output, index=False)
    return {"train": train_output, "val": val_output}
