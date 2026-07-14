import pandas as pd

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


def test_video_candidate_rejects_invalid_rights_status() -> None:
    try:
        video_discovery.build_candidate(
            url="https://example.com/x", title="x", rights_status="downloaded"
        )
        raise AssertionError("expected ValueError for invalid rights_status")
    except ValueError:
        pass


def test_approved_status_requires_rights_reference() -> None:
    try:
        video_discovery.build_candidate(
            url="https://example.com/x", title="x", rights_status="licensed"
        )
        raise AssertionError("expected ValueError for missing rights_reference")
    except ValueError:
        pass
    candidate = video_discovery.build_candidate(
        url="https://example.com/x", title="x", rights_status="licensed", rights_reference="license-file:///perms/abc.pdf"
    )
    assert candidate.rights_reference == "license-file:///perms/abc.pdf"
    frame = video_discovery.candidates_to_frame([candidate])
    assert "rights_reference" in frame.columns



def test_append_candidate_writes_metadata_only_manifest(tmp_path) -> None:
    output = tmp_path / "discovery.csv"
    first = video_discovery.build_candidate(url="https://example.com/a", title="A", query="q")
    second = video_discovery.build_candidate(url="https://example.com/b", title="B", query="q")
    video_discovery.append_candidate(first, output)
    video_discovery.append_candidate(second, output)
    frame = pd.read_csv(output)
    assert set(frame["url"]) == {"https://example.com/a", "https://example.com/b"}
    assert set(frame["rights_status"]) == {"pending"}
    assert "local_path" not in frame.columns


def test_append_candidate_dedupes_by_url(tmp_path) -> None:
    output = tmp_path / "discovery.csv"
    video_discovery.append_candidate(
        video_discovery.build_candidate(url="https://example.com/a", title="A"), output
    )
    video_discovery.append_candidate(
        video_discovery.build_candidate(url="https://example.com/a", title="A2"), output
    )
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.iloc[0]["title"] == "A2"
