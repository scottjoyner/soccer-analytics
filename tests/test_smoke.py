from soccer_edge import __version__
from soccer_edge.config import get_settings
from soccer_edge.ingest.video_discovery import build_candidate


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_config_defaults() -> None:
    settings = get_settings()
    assert settings.env == "dev"
    assert settings.external_execution_enabled is False


def test_video_candidate_metadata_only() -> None:
    candidate = build_candidate(
        url="https://example.com/video",
        title="Example soccer clip",
        query="soccer",
    )
    assert candidate.rights_status == "pending"
    assert candidate.url.startswith("https://")
