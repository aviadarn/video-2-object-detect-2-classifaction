from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class JobState(str, Enum):
    queued = "queued"
    downloading = "downloading"
    extracting_frames = "extracting_frames"
    ready_for_inference = "ready_for_inference"
    failed = "failed"


class CreateJobRequest(BaseModel):
    video_url: HttpUrl
    job_id: Optional[str] = Field(default=None, min_length=3, max_length=128)


class FrameEntry(BaseModel):
    frame_index: int
    timestamp_seconds: float
    uri: str


class CreateJobResponse(BaseModel):
    job_id: str
    state: JobState
    manifest_uri: Optional[str] = None
    message: str
