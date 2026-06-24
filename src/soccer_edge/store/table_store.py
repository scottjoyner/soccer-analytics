from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def load_table(source: Path) -> pd.DataFrame:
    return pd.read_parquet(source)
