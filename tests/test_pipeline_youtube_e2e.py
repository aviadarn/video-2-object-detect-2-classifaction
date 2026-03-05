import os
from pathlib import Path

import pytest

from services.pipeline.worker import PipelineConfig, PipelineRunner


@pytest.mark.integration
def test_youtube_pipeline_e2e(tmp_path: Path) -> None:
    if os.getenv("RUN_YOUTUBE_E2E") != "1":
        pytest.skip("Set RUN_YOUTUBE_E2E=1 to run youtube integration test")

    yt_dlp = pytest.importorskip("yt_dlp")

    video_path = tmp_path / "sample.mp4"
    ydl_opts = {
        "format": "mp4",
        "outtmpl": str(video_path),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(["https://www.youtube.com/watch?v=aqz-KE-bpKQ"])

    assert video_path.exists(), "Video should be downloaded"

    runner = PipelineRunner(config=PipelineConfig(job_id="yt-e2e", video_path=video_path))
    report = runner.run()

    assert report["job_id"] == "yt-e2e"
    assert "frames" in report
    assert "clusters" in report
    assert "summary" in report
