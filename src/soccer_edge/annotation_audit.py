from pathlib import Path

import pandas as pd


def count_by_column(frame: pd.DataFrame, column: str, count_name: str = "row_count") -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"missing column: {column}")
    return frame.groupby(column, dropna=False).size().reset_index(name=count_name).sort_values(count_name, ascending=False)


def annotation_audit_tables(
    frame: pd.DataFrame,
    class_column: str = "class_name",
    frame_column: str = "frame_idx",
    split_column: str = "split",
) -> dict[str, pd.DataFrame]:
    tables = {
        "by_class": count_by_column(frame, class_column),
        "by_frame": count_by_column(frame, frame_column),
    }
    if split_column in frame.columns:
        tables["by_split"] = count_by_column(frame, split_column)
        tables["by_split_class"] = (
            frame.groupby([split_column, class_column], dropna=False).size().reset_index(name="row_count").sort_values([split_column, "row_count"], ascending=[True, False])
        )
    return tables


def write_annotation_audit(
    source: Path,
    output_dir: Path,
    class_column: str = "class_name",
    frame_column: str = "frame_idx",
    split_column: str = "split",
) -> dict[str, Path]:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    tables = annotation_audit_tables(frame, class_column=class_column, frame_column=frame_column, split_column=split_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths
