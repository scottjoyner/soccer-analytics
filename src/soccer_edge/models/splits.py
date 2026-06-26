from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    validation_end: str
    date_column: str = "match_date"


def add_time_split(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    if config.date_column not in frame.columns:
        raise ValueError(f"missing date column: {config.date_column}")
    output = frame.copy()
    dates = pd.to_datetime(output[config.date_column])
    train_end = pd.Timestamp(config.train_end)
    validation_end = pd.Timestamp(config.validation_end)
    if validation_end <= train_end:
        raise ValueError("validation_end must be after train_end")

    output["split"] = "test"
    output.loc[dates <= train_end, "split"] = "train"
    output.loc[(dates > train_end) & (dates <= validation_end), "split"] = "validation"
    return output


def split_frames(frame: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    labeled = add_time_split(frame, config)
    return {name: group.reset_index(drop=True) for name, group in labeled.groupby("split")}
