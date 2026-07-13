from pathlib import Path

import pandas as pd


def counts_from_status_rows(frame: pd.DataFrame, class_column: str = "class_name", status_column: str = "status") -> pd.DataFrame:
    if class_column not in frame.columns:
        raise ValueError(f"missing class column: {class_column}")
    if status_column not in frame.columns:
        raise ValueError(f"missing status column: {status_column}")
    rows = []
    for class_name, group in frame.groupby(class_column, dropna=False):
        statuses = group[status_column].astype(str).str.lower()
        rows.append(
            {
                class_column: class_name,
                "tp": int((statuses == "tp").sum()),
                "fp": int((statuses == "fp").sum()),
                "fn": int((statuses == "fn").sum()),
            }
        )
    return pd.DataFrame(rows)


def metrics_from_counts(counts: pd.DataFrame, class_column: str = "class_name") -> pd.DataFrame:
    required = {class_column, "tp", "fp", "fn"}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = []
    for _, row in counts.iterrows():
        tp = float(row["tp"])
        fp = float(row["fp"])
        fn = float(row["fn"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                class_column: row[class_column],
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def object_eval_metrics(frame: pd.DataFrame, class_column: str = "class_name", status_column: str = "status") -> pd.DataFrame:
    if {"tp", "fp", "fn"}.issubset(frame.columns):
        return metrics_from_counts(frame, class_column=class_column)
    return metrics_from_counts(counts_from_status_rows(frame, class_column=class_column, status_column=status_column), class_column=class_column)


def write_object_eval_metrics(
    source: Path,
    output: Path,
    class_column: str = "class_name",
    status_column: str = "status",
) -> Path:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    metrics = object_eval_metrics(frame, class_column=class_column, status_column=status_column)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        metrics.to_parquet(output, index=False)
    else:
        metrics.to_csv(output, index=False)
    return output
