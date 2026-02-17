from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from .config import IngestSettings
from .downloader import download_video
from .frames import extract_frames
from .manifest import build_manifest
from .models import CreateJobRequest, CreateJobResponse, JobState
from .validators import ValidationError, validate_video_url

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    state: JobState
    manifest_uri: Optional[str] = None
    error: Optional[str] = None


class IngestService:
    def __init__(self, settings: IngestSettings | None = None) -> None:
        self.settings = settings or IngestSettings()
        self.jobs: dict[str, JobRecord] = {}

    def _set_state(self, job_id: str, state: JobState, error: str | None = None) -> None:
        record = self.jobs.setdefault(job_id, JobRecord(job_id=job_id, state=state))
        record.state = state
        record.error = error
        logger.info("job_id=%s state=%s error=%s", job_id, state.value, error)

    def create_job(self, payload: CreateJobRequest) -> CreateJobResponse:
        job_id = payload.job_id or str(uuid.uuid4())
        try:
            self._set_state(job_id, JobState.queued)
            validate_video_url(str(payload.video_url), self.settings.allowed_hosts)

            self._set_state(job_id, JobState.downloading)
            video_path = download_video(str(payload.video_url), job_id, self.settings)

            self._set_state(job_id, JobState.extracting_frames)
            frame_dir = extract_frames(video_path, job_id, self.settings)

            manifest_path, _entries = build_manifest(frame_dir)
            self._set_state(job_id, JobState.ready_for_inference)
            self.jobs[job_id].manifest_uri = str(manifest_path)

            return CreateJobResponse(
                job_id=job_id,
                state=JobState.ready_for_inference,
                manifest_uri=str(manifest_path),
                message="Job prepared and frame manifest generated",
            )
        except Exception as exc:
            self._set_state(job_id, JobState.failed, str(exc))
            if isinstance(exc, ValidationError):
                message = f"Validation failed: {exc}"
            else:
                message = f"Job failed: {exc}"
            return CreateJobResponse(
                job_id=job_id,
                state=JobState.failed,
                message=message,
            )
