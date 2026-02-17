"""MongoDB storage repository for inference outputs and pipeline job metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database

from services.storage.schemas import PipelineJob, VideoInferenceResult


VIDEO_RESULTS_COLLECTION = "video_inference_results"
PIPELINE_JOBS_COLLECTION = "pipeline_jobs"


@dataclass(slots=True)
class Pagination:
    """Pagination request values."""

    skip: int = 0
    limit: int = 50


class MongoRepository:
    """Storage abstraction around MongoDB for pipeline result retrieval and writes."""

    def __init__(self, db: Database):
        self.db = db
        self.results: Collection = db[VIDEO_RESULTS_COLLECTION]
        self.pipeline_jobs: Collection = db[PIPELINE_JOBS_COLLECTION]

    @classmethod
    def from_uri(cls, mongo_uri: str, db_name: str) -> "MongoRepository":
        client = MongoClient(mongo_uri)
        return cls(client[db_name])

    def ensure_indexes(self) -> None:
        """Create required indexes for write idempotency and query speed."""
        self.results.create_index(
            [("job_id", ASCENDING), ("frame_id", ASCENDING)],
            unique=True,
            name="uq_job_frame",
        )
        self.results.create_index([("video_url", ASCENDING)], name="idx_video_url")
        self.results.create_index([("created_at", DESCENDING)], name="idx_created_at_desc")

        self.pipeline_jobs.create_index([("job_id", ASCENDING)], unique=True, name="uq_job_id")
        self.pipeline_jobs.create_index([("created_at", DESCENDING)], name="idx_job_created_at_desc")

    def upsert_inference_result(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Idempotent write mode: upsert by (job_id, frame_id)."""
        now = datetime.now(timezone.utc)
        payload.setdefault("created_at", now)
        doc = VideoInferenceResult.model_validate(payload).model_dump(by_alias=True)

        query = {"job_id": doc["job_id"], "frame_id": doc["frame_id"]}
        update = {
            "$set": {
                "video_url": doc["video_url"],
                "timestamp_sec": doc["timestamp_sec"],
                "detections": doc.get("detections", []),
                "swin_predictions": doc.get("swin_predictions", []),
                "pipeline_version": doc.get("pipeline_version"),
            },
            "$setOnInsert": {"created_at": doc["created_at"]},
        }
        return self.results.find_one_and_update(
            query,
            update,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def list_results_by_job(self, job_id: str, pagination: Pagination) -> dict[str, Any]:
        """Fetch paginated frame-level results for a job."""
        filter_doc = {"job_id": job_id}
        cursor = (
            self.results.find(filter_doc, {"_id": False})
            .sort([("frame_id", ASCENDING)])
            .skip(pagination.skip)
            .limit(pagination.limit)
        )
        items = list(cursor)
        total = self.results.count_documents(filter_doc)

        return {
            "job_id": job_id,
            "total": total,
            "skip": pagination.skip,
            "limit": pagination.limit,
            "items": items,
        }

    def upsert_pipeline_job(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Track run metadata in dedicated collection for status/progress."""
        now = datetime.now(timezone.utc)
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        doc = PipelineJob.model_validate(payload).model_dump()

        query = {"job_id": doc["job_id"]}
        update = {
            "$set": {
                "video_url": doc.get("video_url"),
                "status": doc.get("status", "queued"),
                "progress": doc.get("progress", 0),
                "error": doc.get("error"),
                "updated_at": now,
                "pipeline_version": doc.get("pipeline_version"),
            },
            "$setOnInsert": {
                "created_at": doc["created_at"],
            },
        }

        return self.pipeline_jobs.find_one_and_update(
            query,
            update,
            upsert=True,
            return_document=ReturnDocument.AFTER,
            projection={"_id": False},
        )

    def get_pipeline_job(self, job_id: str) -> dict[str, Any] | None:
        """Return current metadata for one job."""
        return self.pipeline_jobs.find_one({"job_id": job_id}, {"_id": False})
