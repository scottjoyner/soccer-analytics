from pathlib import Path

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def build_model_comparison(registry: pd.DataFrame, evaluation: pd.DataFrame | None = None) -> pd.DataFrame:
    output = registry.copy()
    if evaluation is not None and not evaluation.empty:
        join_columns = [column for column in ["name", "version"] if column in output.columns and column in evaluation.columns]
        if join_columns:
            output = output.merge(evaluation, on=join_columns, how="left", suffixes=("", "_evaluation"))
    sort_columns = [column for column in ["metric_accuracy", "accuracy", "created_at_utc"] if column in output.columns]
    if sort_columns:
        output = output.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
    return output


def write_model_comparison(registry_path: Path, output_path: Path, evaluation_path: Path | None = None) -> Path:
    registry = load_table(registry_path)
    evaluation = load_table(evaluation_path) if evaluation_path is not None else None
    comparison = build_model_comparison(registry, evaluation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        comparison.to_parquet(output_path, index=False)
    else:
        comparison.to_csv(output_path, index=False)
    return output_path
