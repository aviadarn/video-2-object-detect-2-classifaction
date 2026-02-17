# Video-to-Object Detection & Classification

This repository contains a modular video inference pipeline that:

1. Ingests a video job (for example, from a YouTube URL).
2. Extracts video frames and writes a manifest.
3. Runs object detection and classification through Triton-served models.
4. Exposes job inference results through an API backed by MongoDB.

## Repository layout

- `services/ingest/` – FastAPI ingest service that validates input URLs, downloads video files, extracts frames, and builds a `manifest.json` for downstream processing.
- `services/pipeline/` – worker/orchestration logic for frame-by-frame inference (Detectron2 detector + Swin classifier) and merged output records.
- `services/pipeline_worker/` – Redis stream consumer worker with explicit stage transitions, structured logging, and Prometheus metrics.
- `services/storage/` – MongoDB repository and schemas for storing and querying inference outputs.
- `api/` – FastAPI read API for paginated retrieval of frame-level results by `job_id`.
- `infra/k8s/` – Kubernetes manifests for deploying ingest, workers, Triton services, Redis, and related config.
- `docs/` – architecture notes and pipeline behavior documentation.
- `tests/` – unit tests for pipeline worker utilities and failure policy behavior.

## High-level data flow

1. **Ingest API** receives `POST /jobs` and prepares frame manifests.
2. A **pipeline worker** consumes queued jobs and drives state transitions:
   `received -> decoded -> detected -> classified -> completed` (or `failed`).
3. Worker calls Triton inference endpoints:
   - detector model for bounding boxes/scores/labels
   - classifier model for per-object class predictions
4. Results are persisted and made available via the read API endpoint:
   `GET /jobs/{job_id}/results`.

## Local development notes

This project is organized as Python services and workers. Typical local runs are:

```bash
uvicorn services.ingest.app:app --host 0.0.0.0 --port 8080
uvicorn api.main:app --host 0.0.0.0 --port 8001
python -m services.pipeline.worker --help
python -m services.pipeline_worker.main
```

Install dependencies from each service's `requirements.txt` as needed.

## Testing

Run tests with:

```bash
pytest
```

## Deployment

Kubernetes manifests for core components are located in `infra/k8s/`, including separate manifests for Triton-backed detector and classifier services.
