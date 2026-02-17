# Video Pipeline Architecture

## Data flow

1. **Ingest API** accepts video job requests and writes a message into a Redis Stream (`jobs`) rather than doing synchronous inference.
2. **Pipeline worker** consumers in a Redis consumer group pull jobs and execute the stage machine:
   - `received -> decoded -> detected -> classified -> completed`
   - failures may transition from any non-terminal state to `failed`.
3. Worker calls dedicated **Triton Detector** and **Triton Classifier** services in sequence.
4. Job status events are emitted as structured logs including `job_id` at every transition.

This queue-based split decouples request latency from long-running GPU inference and allows independent scaling.

## Failure handling

- The worker enforces explicit allowed transitions with a transition map, preventing invalid stage jumps.
- Any exception during decode/inference marks the job as `failed` and records failure duration metric.
- Triton-call failures increment `triton_errors_total{service="detector|classifier"}` for alerting.
- Redis consumer group + ack semantics ensure jobs are acknowledged only after processing.

## Observability

Worker exposes Prometheus metrics:

- `job_duration_seconds{status}` histogram for completed/failed durations.
- `frames_processed_total` counter for throughput.
- `triton_errors_total{service}` counter for model-serving failures.

Structured JSON logs include `job_id`, `state`, and message labels (`job_received`, `job_detected`, etc.), enabling per-job tracing in log aggregation systems.

## Scaling knobs

- `ingest` deployment replicas scale request intake without impacting inference workers.
- `pipeline-worker` replicas scale queue consumers.
- `triton-detector` and `triton-classifier` replicas can be tuned independently based on model cost.
- CPU/memory requests and limits in Kubernetes manifests should be tuned with real production profiling.
