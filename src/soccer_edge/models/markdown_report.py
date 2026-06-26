from pathlib import Path

import pandas as pd

from soccer_edge.models.comparison import load_table


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows available."
    display = frame.head(max_rows).copy()
    return display.to_markdown(index=False)


def write_model_markdown_report(
    comparison_path: Path,
    output_path: Path,
    title: str = "Model Comparison Report",
    max_rows: int = 20,
) -> Path:
    frame = load_table(comparison_path)
    lines = [f"# {title}", "", f"Rows: {len(frame)}", "", dataframe_to_markdown(frame, max_rows=max_rows), ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
