"""Video discovery metadata helpers.

This module is intentionally metadata-only. It must not download, cache, mirror, or
store audiovisual content. Public URLs are kept as discovery metadata only and are
never used as processing inputs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class VideoCandidate:
    """Candidate video metadata for manual rights review.

    Approved rights statuses (owned/licensed/compatible_license) require a recorded
    ``rights_reference`` (the explicit written rights) before the footage may be used.
    """

    url: str
    title: str
    channel: str | None = None
    published_at: datetime | None = None
    query: str | None = None
    rights_status: str = "pending"
    rights_reference: str | None = None
    notes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    APPROVED_STATUSES = ("owned", "licensed", "compatible_license")

    def validate_rights_status(self) -> None:
        allowed = {"pending", "owned", "licensed", "compatible_license", "blocked"}
        if self.rights_status not in allowed:
            raise ValueError(f"rights_status must be one of {sorted(allowed)}; got {self.rights_status!r}")
        if self.rights_status in self.APPROVED_STATUSES and not self.rights_reference:
            raise ValueError(
                f"rights_status={self.rights_status!r} requires a recorded rights_reference "
                "(explicit written rights) before the footage may be used"
            )


def build_candidate(
    url: str,
    title: str,
    query: str | None = None,
    channel: str | None = None,
    published_at: datetime | None = None,
    rights_status: str = "pending",
    rights_reference: str | None = None,
    notes: str | None = None,
) -> VideoCandidate:
    """Create a metadata-only video candidate record."""

    candidate = VideoCandidate(
        url=url,
        title=title,
        channel=channel,
        published_at=published_at,
        query=query,
        rights_status=rights_status,
        rights_reference=rights_reference,
        notes=notes,
    )
    candidate.validate_rights_status()
    return candidate


def candidates_to_frame(candidates: Iterable[VideoCandidate]) -> pd.DataFrame:
    rows = [asdict(candidate) for candidate in candidates]
    return pd.DataFrame(rows)


def write_candidates_manifest(candidates: Iterable[VideoCandidate], output: Path) -> Path:
    """Persist discovery candidates as a metadata-only manifest (no audiovisual content)."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = candidates_to_frame(candidates)
    frame.to_csv(output, index=False)
    return output


def append_candidate(candidate: VideoCandidate, output: Path) -> Path:
    """Append a discovery candidate to an existing manifest, de-duplicating by URL."""

    output = Path(output)
    existing = pd.DataFrame()
    if output.exists():
        existing = pd.read_csv(output)
    frame = candidates_to_frame([candidate])
    combined = pd.concat([existing, frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"], keep="last")
    combined.to_csv(output, index=False)
    return output
