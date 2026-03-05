# Video-to-Object Detection & Classification

This repository contains a modular video inference pipeline that:

1. Ingests a video job (for example, from a YouTube URL).
2. Extracts video frames and writes a manifest.
3. Runs a scene-based pipeline: split -> object detection -> classification -> object tracking -> clustering -> JSON report.
4. Exposes job inference results through an API backed by MongoDB.

## Repository layout

- `services/ingest/` – FastAPI ingest service that validates input URLs, downloads video files, extracts frames, and builds a `manifest.json` for downstream processing.
- `services/pipeline/` – scene-based orchestration logic using PySceneDetect sampling, Detectron2 + YOLO12 detection, OpenCLIP classification, tracking, clustering, and JSON report generation.
- `services/pipeline_worker/` – Redis stream consumer worker with explicit stage transitions, structured logging, and Prometheus metrics.
- `services/storage/` – MongoDB repository and schemas for storing and querying inference outputs.
- `api/` – FastAPI read API for paginated retrieval of frame-level results by `job_id`.
- `infra/k8s/` – Kubernetes manifests for deploying ingest, workers, Triton services, Redis, and related config.
- `docs/` – architecture notes and pipeline behavior documentation.
- `tests/` – unit tests for pipeline worker utilities and failure policy behavior.

## Pipeline diagram

Requested stage order: `split -> object_detection -> classification -> object_tracking -> clustering -> report`

```mermaid
flowchart LR
    A[Video Input] --> B[Split (PySceneDetect)]
    B --> C[Sample Frames: First / Middle / Last]
    C --> D[Object Detection: Detectron2 Faster R-CNN + YOLO12]
    D --> E[Classification: OpenCLIP]
    E --> F[Object Tracking]
    F --> G[Clustering]
    G --> H[Report (JSON)]
```

## High-level data flow

1. **Ingest API** receives `POST /jobs` and prepares frame manifests.
2. A **pipeline worker** consumes queued jobs and drives state transitions:
   `split -> object_detection -> classification -> object_tracking -> clustering -> report`.
3. Worker detects scenes with PySceneDetect and samples first/middle/last frames.
4. The sampled frames are processed by Detectron2 Faster R-CNN + Ultralytics YOLO12, then classified by OpenCLIP, tracked, clustered, and emitted as a JSON report.

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
