from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestSettings:
    """Runtime settings for ingest jobs."""

    download_root: str = os.getenv("INGEST_DOWNLOAD_ROOT", "/data/downloads")
    frame_root: str = os.getenv("INGEST_FRAME_ROOT", "/data/frames")
    command_timeout_s: int = int(os.getenv("INGEST_COMMAND_TIMEOUT_S", "1800"))
    retry_count: int = int(os.getenv("INGEST_RETRY_COUNT", "3"))
    retry_backoff_s: float = float(os.getenv("INGEST_RETRY_BACKOFF_S", "2.0"))
    allowed_hosts: tuple[str, ...] = tuple(
        h.strip().lower()
        for h in os.getenv(
            "INGEST_ALLOWED_VIDEO_HOSTS",
            "youtube.com,www.youtube.com,youtu.be,m.youtube.com",
        ).split(",")
        if h.strip()
    )
