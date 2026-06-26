import pytest

from soccer_edge.features.coordinates import is_inside_pitch, normalize_unit_coordinate, normalize_with_originals


def test_normalize_unit_coordinate() -> None:
    x_m, y_m = normalize_unit_coordinate(0.5, 0.5)
    assert x_m == 52.5
    assert y_m == 34.0
    assert is_inside_pitch(x_m, y_m)


def test_normalize_with_originals_unit_coordinates() -> None:
    coordinate = normalize_with_originals(0.5, 0.25)
    assert coordinate.x_original == 0.5
    assert coordinate.y_original == 0.25
    assert coordinate.x_m == 52.5
    assert coordinate.y_m == 17.0
    assert coordinate.coordinate_system == "unit"


def test_normalize_with_originals_meter_coordinates() -> None:
    coordinate = normalize_with_originals(10.0, 20.0, coordinate_system="meters")
    assert coordinate.x_original == 10.0
    assert coordinate.y_original == 20.0
    assert coordinate.x_m == 10.0
    assert coordinate.y_m == 20.0


def test_normalize_unit_coordinate_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        normalize_unit_coordinate(-0.1, 0.5)
