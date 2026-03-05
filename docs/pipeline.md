# Video Pipeline Architecture

## Data flow

The pipeline now runs in the following order:

1. **split**: PySceneDetect detects scene boundaries.
2. **object_detection**: sample the **first**, **middle**, and **last** frame from each scene and run both:
   - Detectron2 Faster R-CNN
   - Ultralytics YOLO12
3. **classification**: classify every detected object with OpenCLIP.
4. **object_tracking**: assign lightweight track IDs based on box center proximity.
5. **clustering**: group tracked objects by classification label.
6. **report**: emit a JSON report with frames, detections, clusters, and summary counters.

## JSON report format

Top-level report fields:

- `job_id`
- `pipeline` (ordered stage list)
- `frames` (per sampled frame with detections + track IDs + classifications)
- `clusters` (cluster key, track IDs, count)
- `summary` (`total_frames`, `total_detections`, `total_tracks`, `total_clusters`)

## Testing strategy (TDD)

- Unit tests verify pipeline stage behavior and cluster/report shape using fakes.
- An integration test (`tests/test_pipeline_youtube_e2e.py`) downloads a YouTube sample video and executes the pipeline end-to-end when `RUN_YOUTUBE_E2E=1`.
