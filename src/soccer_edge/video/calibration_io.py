import json
from pathlib import Path
from typing import Any

from soccer_edge.video.homography import HomographyTransform, build_homography


def read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install pyyaml to read YAML calibration files.") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError("calibration file must be JSON or YAML")


def point_pairs(data: dict[str, Any]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if "pixel_points" in data and "pitch_points" in data:
        pixel_points = [(float(x), float(y)) for x, y in data["pixel_points"]]
        pitch_points = [(float(x), float(y)) for x, y in data["pitch_points"]]
        return pixel_points, pitch_points
    if "points" in data:
        pixel_points = []
        pitch_points = []
        for item in data["points"]:
            pixel = item["pixel"]
            pitch = item["pitch"]
            pixel_points.append((float(pixel[0]), float(pixel[1])))
            pitch_points.append((float(pitch[0]), float(pitch[1])))
        return pixel_points, pitch_points
    raise ValueError("calibration must contain pixel_points/pitch_points or points")


def load_homography(path: Path) -> HomographyTransform:
    pixel_points, pitch_points = point_pairs(read_mapping(path))
    return build_homography(pixel_points, pitch_points)
