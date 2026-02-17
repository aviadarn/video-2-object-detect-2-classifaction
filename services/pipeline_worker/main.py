"""Pipeline worker consuming queued jobs and driving explicit stage transitions."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import redis
from prometheus_client import Counter, Histogram, start_http_server

job_duration_seconds = Histogram(
    "job_duration_seconds",
    "End-to-end job processing duration in seconds",
    ["status"],
)
frames_processed_total = Counter(
    "frames_processed_total",
    "Total number of processed frames",
)
triton_errors_total = Counter(
    "triton_errors_total",
    "Total number of Triton inference errors",
    ["service"],
)


class JobState(str, Enum):
    RECEIVED = "received"
    DECODED = "decoded"
    DETECTED = "detected"
    CLASSIFIED = "classified"
    COMPLETED = "completed"
    FAILED = "failed"


TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.RECEIVED: {JobState.DECODED, JobState.FAILED},
    JobState.DECODED: {JobState.DETECTED, JobState.FAILED},
    JobState.DETECTED: {JobState.CLASSIFIED, JobState.FAILED},
    JobState.CLASSIFIED: {JobState.COMPLETED, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
}


@dataclass
class Job:
    job_id: str
    video_uri: str
    frames: int
    state: JobState = JobState.RECEIVED


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline_worker")
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    return logger


def _json_log(logger: logging.Logger, level: int, message: str, **extra: Any) -> None:
    payload = {"message": message, **extra}
    logger.log(level, json.dumps(payload, default=str))


def transition(job: Job, next_state: JobState) -> None:
    allowed = TRANSITIONS[job.state]
    if next_state not in allowed:
        raise ValueError(f"invalid transition {job.state} -> {next_state}")
    job.state = next_state


def run_detector(job: Job) -> None:
    # Placeholder for Triton detector inference integration.
    if job.frames <= 0:
        triton_errors_total.labels(service="detector").inc()
        raise RuntimeError("detector rejected empty frame set")


def run_classifier(job: Job) -> None:
    # Placeholder for Triton classifier inference integration.
    if job.frames <= 0:
        triton_errors_total.labels(service="classifier").inc()
        raise RuntimeError("classifier rejected empty frame set")


def process_job(logger: logging.Logger, payload: dict[str, Any]) -> None:
    started = time.monotonic()
    job = Job(
        job_id=payload["job_id"],
        video_uri=payload["video_uri"],
        frames=int(payload.get("frames", 0)),
    )

    try:
        _json_log(logger, logging.INFO, "job_received", job_id=job.job_id, state=job.state)
        transition(job, JobState.DECODED)
        _json_log(logger, logging.INFO, "job_decoded", job_id=job.job_id, state=job.state)

        run_detector(job)
        transition(job, JobState.DETECTED)
        _json_log(logger, logging.INFO, "job_detected", job_id=job.job_id, state=job.state)

        run_classifier(job)
        transition(job, JobState.CLASSIFIED)
        _json_log(logger, logging.INFO, "job_classified", job_id=job.job_id, state=job.state)

        frames_processed_total.inc(job.frames)
        transition(job, JobState.COMPLETED)
        _json_log(logger, logging.INFO, "job_completed", job_id=job.job_id, state=job.state)
        job_duration_seconds.labels(status="completed").observe(time.monotonic() - started)
    except Exception as exc:  # broad by design to mark terminal failed state
        transition(job, JobState.FAILED)
        _json_log(
            logger,
            logging.ERROR,
            "job_failed",
            job_id=job.job_id,
            state=job.state,
            error=str(exc),
        )
        job_duration_seconds.labels(status="failed").observe(time.monotonic() - started)


def consume_forever() -> None:
    logger = _setup_logger()
    metrics_port = int(os.getenv("METRICS_PORT", "9090"))
    start_http_server(metrics_port)

    redis_url = os.environ["REDIS_URL"]
    stream = os.getenv("REDIS_STREAM_NAME", "jobs")
    group = os.getenv("REDIS_CONSUMER_GROUP", "pipeline-workers")
    consumer = os.getenv("HOSTNAME", "pipeline-worker-0")

    client = redis.from_url(redis_url, decode_responses=True)
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as err:
        if "BUSYGROUP" not in str(err):
            raise

    _json_log(logger, logging.INFO, "worker_started", consumer=consumer, stream=stream, group=group)

    while True:
        items = client.xreadgroup(group, consumer, streams={stream: ">"}, count=1, block=5000)
        if not items:
            continue

        for _, messages in items:
            for msg_id, fields in messages:
                process_job(logger, fields)
                client.xack(stream, group, msg_id)


if __name__ == "__main__":
    consume_forever()
