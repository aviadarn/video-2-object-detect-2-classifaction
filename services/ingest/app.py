from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .models import CreateJobRequest, CreateJobResponse, JobState
from .service import IngestService

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Video Ingest Service")
service = IngestService()


@app.post("/jobs", response_model=CreateJobResponse)
def create_job(payload: CreateJobRequest) -> CreateJobResponse:
    response = service.create_job(payload)
    if response.state == JobState.failed:
        raise HTTPException(status_code=400, detail=response.message)
    return response
