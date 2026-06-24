"""Video discovery metadata helpers.

This module is intentionally metadata-only. It must not download, cache, mirror, or
store audiovisual content.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class VideoCandidate:
    """Candidate video metadata for manual rights review."""

    url: str
    title: str
    channel: str | None = None
    published_at: datetime | None = None
    query: str | None = None
    rights_status: str = "pending"
    notes: str | None = None
    created_at: datetime = datetime.now(timezone.utc)


def build_candidate(url: str, title: str, query: str | None = None) -> VideoCandidate:
    """Create a metadata-only video candidate record."""

    return VideoCandidate(url=url, title=title, query=query)
