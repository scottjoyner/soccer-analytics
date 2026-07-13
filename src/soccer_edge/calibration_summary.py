from pathlib import Path

import pandas as pd


def calibration_error_stats(frame: pd.DataFrame) -> dict[str, float]:
    if "error_m" not in frame.columns:
        raise ValueError("missing error_m column")
    return {
        "count": float(len(frame)),
        "mean_error_m": float(frame["error_m"].mean()) if len(frame) else 0.0,
        "max_error_m": float(frame["error_m"].max()) if len(frame) else 0.0,
        "median_error_m": float(frame["error_m"].median()) if len(frame) else 0.0,
    }


def calibration_summary_markdown(frame: pd.DataFrame, title: str = "Calibration QA Summary") -> str:
    stats = calibration_error_stats(frame)
    lines = [
        f"# {title}",
        "",
        f"Points: {int(stats['count'])}",
        f"Mean error (m): {stats['mean_error_m']:.4f}",
        f"Median error (m): {stats['median_error_m']:.4f}",
        f"Max error (m): {stats['max_error_m']:.4f}",
        "",
    ]
    return "\n".join(lines)


def write_calibration_summary(source: Path, output: Path, title: str = "Calibration QA Summary") -> Path:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(calibration_summary_markdown(frame, title=title), encoding="utf-8")
    return output
