import soccer_edge.ingest.video_discovery as video_discovery


def test_video_discovery_exposes_no_download_function() -> None:
    public_names = [name.lower() for name in dir(video_discovery)]
    assert "download" not in public_names
    assert "download_video" not in public_names


def test_video_candidate_defaults_to_pending() -> None:
    candidate = video_discovery.build_candidate(
        url="https://example.com/manual-review",
        title="Manual review",
        query="soccer",
    )
    assert candidate.rights_status == "pending"
