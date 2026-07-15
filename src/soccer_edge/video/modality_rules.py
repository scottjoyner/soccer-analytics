from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soccer_edge.video.manifest import VideoManifestRow

DEFAULT_BLOCKED_KEYWORDS = ("youtube", "youtu.be", "twitch", "stream")
DEFAULT_BLOCKED_SCHEMES = ("http://", "https://", "rtmp://", "rtsp://")
DEFAULT_ALLOWED_MODALITIES = ("local_licensed_file",)


@dataclass(frozen=True)
class ModalityRules:
    blocked_keywords: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BLOCKED_KEYWORDS)
    blocked_schemes: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BLOCKED_SCHEMES)
    allowed_modalities: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ALLOWED_MODALITIES)

    @classmethod
    def default(cls) -> ModalityRules:
        return cls()

    @classmethod
    def from_json(cls, path: Path) -> ModalityRules:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            blocked_keywords=tuple(data.get("blocked_keywords", DEFAULT_BLOCKED_KEYWORDS)),
            blocked_schemes=tuple(data.get("blocked_schemes", DEFAULT_BLOCKED_SCHEMES)),
            allowed_modalities=tuple(data.get("allowed_modalities", DEFAULT_ALLOWED_MODALITIES)),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> ModalityRules:
        if path is not None and Path(path).exists():
            return cls.from_json(path)
        return cls.default()

    def blocked_reason(self, row: VideoManifestRow) -> str | None:
        for text in ((row.source_url or "").strip().lower(), (row.clip_type or "").strip().lower()):
            if not text:
                continue
            for scheme in self.blocked_schemes:
                if text.startswith(scheme):
                    return scheme.rstrip(":/")
            for keyword in self.blocked_keywords:
                if keyword in text:
                    return keyword
        return None
