from soccer_edge.video.alignment import ClipAlignment


def test_clip_alignment_maps_time() -> None:
    alignment = ClipAlignment(
        clip_id="clip_1",
        match_id="match_1",
        video_start_second=10.0,
        match_start_second=1200.0,
        period="first_half",
    )
    assert alignment.video_to_match_second(15.0) == 1205.0
    assert alignment.match_to_video_second(1205.0) == 15.0
