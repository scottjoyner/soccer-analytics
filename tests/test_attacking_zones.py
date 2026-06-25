import pytest

from soccer_edge.features.attacking_zones import count_zone_entries, is_in_final_third, is_in_penalty_box, zone_flags
from soccer_edge.features.spatial import Point


def test_final_third_flags() -> None:
    assert is_in_final_third(Point(80.0, 34.0), attacking_left_to_right=True)
    assert not is_in_final_third(Point(50.0, 34.0), attacking_left_to_right=True)
    assert is_in_final_third(Point(20.0, 34.0), attacking_left_to_right=False)


def test_penalty_box_flags() -> None:
    assert is_in_penalty_box(Point(100.0, 34.0), attacking_left_to_right=True)
    assert not is_in_penalty_box(Point(100.0, 5.0), attacking_left_to_right=True)
    flags = zone_flags(Point(100.0, 34.0), attacking_left_to_right=True)
    assert flags.in_final_third
    assert flags.in_penalty_box


def test_count_zone_entries() -> None:
    points = [Point(60.0, 34.0), Point(71.0, 34.0), Point(80.0, 34.0)]
    assert count_zone_entries(points, zone="final_third") == 1
    with pytest.raises(ValueError):
        count_zone_entries(points, zone="bad")
