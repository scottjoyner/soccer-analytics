from dataclasses import dataclass


@dataclass(frozen=True)
class PitchDimensions:
    length_m: float = 105.0
    width_m: float = 68.0


@dataclass(frozen=True)
class NormalizedCoordinate:
    x_original: float
    y_original: float
    x_m: float
    y_m: float
    coordinate_system: str


def normalize_unit_coordinate(
    x_unit: float,
    y_unit: float,
    pitch: PitchDimensions = PitchDimensions(),
) -> tuple[float, float]:
    if not 0.0 <= x_unit <= 1.0:
        raise ValueError("x_unit must be between 0 and 1")
    if not 0.0 <= y_unit <= 1.0:
        raise ValueError("y_unit must be between 0 and 1")
    return x_unit * pitch.length_m, y_unit * pitch.width_m


def normalize_with_originals(
    x_value: float,
    y_value: float,
    coordinate_system: str = "unit",
    pitch: PitchDimensions = PitchDimensions(),
) -> NormalizedCoordinate:
    if coordinate_system == "unit":
        x_m, y_m = normalize_unit_coordinate(x_value, y_value, pitch)
    elif coordinate_system == "meters":
        x_m, y_m = x_value, y_value
    else:
        raise ValueError("coordinate_system must be unit or meters")
    return NormalizedCoordinate(
        x_original=x_value,
        y_original=y_value,
        x_m=x_m,
        y_m=y_m,
        coordinate_system=coordinate_system,
    )


def is_inside_pitch(x_m: float, y_m: float, pitch: PitchDimensions = PitchDimensions()) -> bool:
    return 0.0 <= x_m <= pitch.length_m and 0.0 <= y_m <= pitch.width_m
