from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .config import IngestSettings
from .downloader import CommandError


def extract_frames(video_path: Path, job_id: str, settings: IngestSettings) -> Path:
    frame_dir = Path(settings.frame_root) / job_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(frame_dir / "frame_%06d.jpg")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        output_pattern,
    ]

    last_error: CommandError | None = None
    for attempt in range(1, settings.retry_count + 1):
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=settings.command_timeout_s,
            )
            return frame_dir
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = CommandError(
                f"Attempt {attempt}/{settings.retry_count} failed for ffmpeg extraction: {exc}"
            )
            if attempt < settings.retry_count:
                time.sleep(settings.retry_backoff_s * attempt)

    raise last_error or CommandError("ffmpeg failed unexpectedly")
