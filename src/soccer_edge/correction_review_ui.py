from html import escape
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = [
    "review_action",
    "corrected_class_name",
    "corrected_x1",
    "corrected_y1",
    "corrected_x2",
    "corrected_y2",
    "review_notes",
]


def correction_template(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing key columns: {missing}")
    template = frame[key_columns].copy()
    for column in REVIEW_COLUMNS:
        template[column] = ""
    return template


def image_cell(row: pd.Series, image_column: str) -> str:
    if image_column not in row or pd.isna(row[image_column]):
        return ""
    src = escape(str(row[image_column]))
    return f"<img src='{src}' alt='crop' loading='lazy' style='max-width:160px;max-height:120px' />"


def correction_review_html(
    frame: pd.DataFrame,
    key_columns: list[str],
    image_column: str = "crop_path",
    class_column: str = "class_name",
    title: str = "Correction Review",
) -> str:
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing key columns: {missing}")
    columns = [column for column in [*key_columns, class_column, "confidence", "x1", "y1", "x2", "y2"] if column in frame.columns]
    header_cells = "".join(f"<th>{escape(column)}</th>" for column in ["image", *columns, *REVIEW_COLUMNS])
    rows = []
    for _, row in frame.iterrows():
        value_cells = [f"<td>{image_cell(row, image_column)}</td>"]
        value_cells.extend(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        value_cells.extend("<td></td>" for _ in REVIEW_COLUMNS)
        rows.append("<tr>" + "".join(value_cells) + "</tr>")
    return "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8' />",
            f"<title>{escape(title)}</title>",
            "<style>body{font-family:sans-serif} table{border-collapse:collapse} th,td{border:1px solid #ccc;padding:6px;vertical-align:top} th{background:#f5f5f5}</style>",
            "</head><body>",
            f"<h1>{escape(title)}</h1>",
            "<p>Fill the companion CSV with review_action values such as keep, correct, or drop. Use corrected_* columns for class or box edits.</p>",
            "<table>",
            f"<thead><tr>{header_cells}</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</body></html>",
        ]
    )


def write_correction_review_assets(
    source: Path,
    html_output: Path,
    template_output: Path,
    key_columns: list[str],
    image_column: str = "crop_path",
    class_column: str = "class_name",
    title: str = "Correction Review",
) -> dict[str, Path]:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    template_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(correction_review_html(frame, key_columns, image_column=image_column, class_column=class_column, title=title), encoding="utf-8")
    correction_template(frame, key_columns).to_csv(template_output, index=False)
    return {"html": html_output, "template": template_output}
