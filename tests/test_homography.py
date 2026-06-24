from soccer_edge.video.homography import build_homography


def test_build_homography_maps_corners() -> None:
    transform = build_homography(
        pixel_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        pitch_points=[(0, 0), (105, 0), (105, 68), (0, 68)],
    )

    center = transform.transform_pixel(50, 50)
    assert center is not None
    assert round(center.x_m, 6) == 52.5
    assert round(center.y_m, 6) == 34.0
