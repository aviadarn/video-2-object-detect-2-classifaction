from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .config import IngestSettings


class CommandError(RuntimeError):
    pass


def _run_with_retry(command: list[str], settings: IngestSettings) -> None:
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
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = CommandError(
                f"Attempt {attempt}/{settings.retry_count} failed for command {command}: {exc}"
            )
            if attempt < settings.retry_count:
                time.sleep(settings.retry_backoff_s * attempt)

    raise last_error or CommandError("Command failed unexpectedly")


def download_video(video_url: str, job_id: str, settings: IngestSettings) -> Path:
    output_dir = Path(settings.download_root) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "source.mp4"

    command = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--no-progress",
        "--socket-timeout",
        "30",
        "--retries",
        str(settings.retry_count),
        "-f",
        "bestvideo+bestaudio/best",
        "-o",
        str(output_path),
        video_url,
    ]
    _run_with_retry(command, settings)
    return output_path
