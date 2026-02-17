# Ingest Service

Implements `POST /jobs` for creating ingest jobs from YouTube URLs.

## Behavior

1. Validate URL host against allow-list (`INGEST_ALLOWED_VIDEO_HOSTS`).
2. Download source video using `yt-dlp` with retries and timeout.
3. Extract exactly 1 frame/second with `ffmpeg -vf fps=1` into `/data/frames/{job_id}/`.
4. Emit `manifest.json` containing:
   - `frame_index`
   - `timestamp_seconds`
   - `uri`
5. Log lifecycle states: `queued`, `downloading`, `extracting_frames`, `ready_for_inference`, `failed`.

## Run

```bash
uvicorn services.ingest.app:app --host 0.0.0.0 --port 8080
```
