from pathlib import Path

import pandas as pd


def confusion_matrix_table(
    frame: pd.DataFrame,
    actual_column: str = "actual_class",
    predicted_column: str = "predicted_class",
) -> pd.DataFrame:
    if actual_column not in frame.columns:
        raise ValueError(f"missing actual column: {actual_column}")
    if predicted_column not in frame.columns:
        raise ValueError(f"missing predicted column: {predicted_column}")
    matrix = pd.crosstab(frame[actual_column], frame[predicted_column], dropna=False)
    matrix.index.name = actual_column
    return matrix.reset_index()


def confusion_matrix_svg(matrix: pd.DataFrame, actual_column: str = "actual_class", cell_size: int = 56) -> str:
    if actual_column not in matrix.columns:
        raise ValueError(f"missing actual column: {actual_column}")
    labels = [column for column in matrix.columns if column != actual_column]
    width = max(320, (len(labels) + 1) * cell_size + 120)
    height = max(220, (len(matrix) + 1) * cell_size + 80)
    max_value = max([float(matrix[label].max()) for label in labels] + [1.0])
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>", "<rect width='100%' height='100%' fill='white' />"]
    parts.append("<text x='10' y='24' font-family='sans-serif' font-size='18'>Object Confusion Matrix</text>")
    for col_idx, label in enumerate(labels):
        x = 120 + col_idx * cell_size
        parts.append(f"<text x='{x}' y='55' font-family='sans-serif' font-size='11'>{label}</text>")
    for row_idx, row in matrix.iterrows():
        y = 70 + row_idx * cell_size
        actual = str(row[actual_column])
        parts.append(f"<text x='10' y='{y + 32}' font-family='sans-serif' font-size='12'>{actual}</text>")
        for col_idx, label in enumerate(labels):
            x = 120 + col_idx * cell_size
            value = float(row[label])
            shade = int(255 - (value / max_value) * 155)
            parts.append(f"<rect x='{x}' y='{y}' width='{cell_size - 4}' height='{cell_size - 4}' fill='rgb({shade},{shade},{shade})' stroke='black' />")
            parts.append(f"<text x='{x + 18}' y='{y + 32}' font-family='sans-serif' font-size='14'>{int(value)}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def write_confusion_outputs(
    source: Path,
    table_output: Path,
    svg_output: Path,
    actual_column: str = "actual_class",
    predicted_column: str = "predicted_class",
) -> dict[str, Path]:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    matrix = confusion_matrix_table(frame, actual_column=actual_column, predicted_column=predicted_column)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(table_output, index=False)
    svg_output.write_text(confusion_matrix_svg(matrix, actual_column=actual_column), encoding="utf-8")
    return {"table": table_output, "svg": svg_output}
