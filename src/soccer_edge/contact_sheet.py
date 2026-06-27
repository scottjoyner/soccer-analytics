from html import escape
from pathlib import Path

import pandas as pd


def contact_card(row: pd.Series, image_column: str = "crop_path") -> str:
    image_path = escape(str(row.get(image_column, "")))
    class_name = escape(str(row.get("class_name", "object")))
    confidence = row.get("confidence", "")
    frame_idx = row.get("frame_idx", "")
    return (
        '<div class="card">'
        f'<img src="{image_path}" alt="{class_name}" />'
        f'<p><strong>{class_name}</strong></p>'
        f'<p>frame={escape(str(frame_idx))} confidence={escape(str(confidence))}</p>'
        "</div>"
    )


def contact_sheet_html(frame: pd.DataFrame, title: str = "Crop Review", image_column: str = "crop_path") -> str:
    cards = "\n".join(contact_card(row, image_column=image_column) for _, row in frame.iterrows())
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #ccc; padding: 0.5rem; border-radius: 0.5rem; }}
    .card img {{ max-width: 100%; height: auto; display: block; }}
    .card p {{ margin: 0.25rem 0; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p>Rows: {len(frame)}</p>
  <div class="grid">
{cards}
  </div>
</body>
</html>
"""


def write_contact_sheet(source: Path, output: Path, title: str = "Crop Review", image_column: str = "crop_path") -> Path:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contact_sheet_html(frame, title=title, image_column=image_column), encoding="utf-8")
    return output
