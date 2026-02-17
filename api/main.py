"""HTTP API exposing job-result read operations."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, Query

from services.storage.mongo_repo import MongoRepository, Pagination


app = FastAPI(title="Video Inference API")


@lru_cache(maxsize=1)
def get_repo() -> MongoRepository:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "video_inference")
    repo = MongoRepository.from_uri(mongo_uri, db_name)
    repo.ensure_indexes()
    return repo


@app.get("/jobs/{job_id}/results")
def get_job_results(
    job_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    repo: MongoRepository = Depends(get_repo),
) -> dict:
    """Paginated results endpoint for frame-level inference output."""
    return repo.list_results_by_job(job_id=job_id, pagination=Pagination(skip=skip, limit=limit))
