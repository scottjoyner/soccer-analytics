import pytest

from soccer_edge.features.coordinates import is_inside_pitch, normalize_unit_coordinate


def test_normalize_unit_coordinate() -> None:
    x_m, y_m = normalize_unit_coordinate(0.5, 0.5)
    assert x_m == 52.5
    assert y_m == 34.0
    assert is_inside_pitch(x_m, y_m)


def test_normalize_unit_coordinate_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        normalize_unit_coordinate(-0.1, 0.5)
