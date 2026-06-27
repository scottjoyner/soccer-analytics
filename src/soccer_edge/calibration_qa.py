from pathlib import Path

import pandas as pd

from soccer_edge.video.calibration_io import load_homography, point_pairs, read_mapping
from soccer_edge.video.homography import HomographyTransform


def projection_qa_frame(
    pixel_points: list[tuple[float, float]],
    pitch_points: list[tuple[float, float]],
    transform: HomographyTransform,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for idx, (pixel, expected) in enumerate(zip(pixel_points, pitch_points, strict=True)):
        projected = transform.transform_pixel(pixel[0], pixel[1])
        projected_x = projected.x_m if projected is not None else float("nan")
        projected_y = projected.y_m if projected is not None else float("nan")
        error = ((projected_x - expected[0]) ** 2 + (projected_y - expected[1]) ** 2) ** 0.5
        rows.append(
            {
                "point_idx": idx,
                "pixel_x": pixel[0],
                "pixel_y": pixel[1],
                "expected_x_m": expected[0],
                "expected_y_m": expected[1],
                "projected_x_m": projected_x,
                "projected_y_m": projected_y,
                "error_m": error,
            }
        )
    return pd.DataFrame(rows)


def write_projection_qa_csv(calibration_path: Path, output_path: Path) -> Path:
    data = read_mapping(calibration_path)
    pixel_points, pitch_points = point_pairs(data)
    frame = projection_qa_frame(pixel_points, pitch_points, load_homography(calibration_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def projection_qa_svg(frame: pd.DataFrame, width: int = 640, height: int = 420) -> str:
    if frame.empty:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    max_x = max(float(frame["expected_x_m"].max()), float(frame["projected_x_m"].max()), 1.0)
    max_y = max(float(frame["expected_y_m"].max()), float(frame["projected_y_m"].max()), 1.0)

    def sx(value: float) -> float:
        return 20.0 + (value / max_x) * (width - 40.0)

    def sy(value: float) -> float:
        return height - 20.0 - (value / max_y) * (height - 40.0)

    marks = []
    for _, row in frame.iterrows():
        ex = sx(float(row["expected_x_m"]))
        ey = sy(float(row["expected_y_m"]))
        px = sx(float(row["projected_x_m"]))
        py = sy(float(row["projected_y_m"]))
        marks.append(f"<line x1='{ex:.2f}' y1='{ey:.2f}' x2='{px:.2f}' y2='{py:.2f}' stroke='gray' />")
        marks.append(f"<circle cx='{ex:.2f}' cy='{ey:.2f}' r='4' fill='green' />")
        marks.append(f"<circle cx='{px:.2f}' cy='{py:.2f}' r='3' fill='red' />")
    return "\n".join([f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>", "<rect width='100%' height='100%' fill='white' />", *marks, "</svg>"])


def write_projection_qa_svg(calibration_path: Path, output_path: Path) -> Path:
    data = read_mapping(calibration_path)
    pixel_points, pitch_points = point_pairs(data)
    frame = projection_qa_frame(pixel_points, pitch_points, load_homography(calibration_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(projection_qa_svg(frame), encoding="utf-8")
    return output_path
