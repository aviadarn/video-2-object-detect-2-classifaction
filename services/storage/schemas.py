"""Pydantic schemas mirroring MongoDB document shapes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Detection(BaseModel):
    object_id: str
    bbox: list[float] = Field(description="[x1, y1, x2, y2]")
    class_name: str = Field(alias="class")
    score: float


class SwinPrediction(BaseModel):
    object_id: str
    label: str
    confidence: float


class VideoInferenceResult(BaseModel):
    job_id: str
    video_url: str
    frame_id: int
    timestamp_sec: float
    detections: list[Detection] = Field(default_factory=list)
    swin_predictions: list[SwinPrediction] = Field(default_factory=list)
    created_at: datetime
    pipeline_version: str | None = None


class PipelineJob(BaseModel):
    job_id: str
    video_url: str | None = None
    status: str = "queued"
    progress: float = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    pipeline_version: str | None = None
