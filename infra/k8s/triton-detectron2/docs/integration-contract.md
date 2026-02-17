# Triton Detectron2 Integration Contract

This document defines the inference payload contract between frame producers/consumers and Triton-hosted Detectron2.

## Endpoint

- **HTTP**: `POST /v2/models/detectron2/infer`
- **gRPC**: `inference.GRPCInferenceService/ModelInfer`

## Request payload

### Required fields

| Field | Type | Description |
|---|---|---|
| `model_name` | string | Must be `detectron2`. |
| `inputs[0].name` | string | Must be `images`. |
| `inputs[0].datatype` | string | `FP32`. |
| `inputs[0].shape` | int[] | `[batch, 3, 720, 1280]`. |
| `inputs[0].data` | float[] | Normalized image tensor in CHW order. |
| `parameters.frame_id` | string/integer | Producer frame identifier for correlation. |
| `parameters.timestamp` | string | RFC3339 timestamp of frame capture. |

### HTTP request example

```json
{
  "model_name": "detectron2",
  "inputs": [
    {
      "name": "images",
      "datatype": "FP32",
      "shape": [1, 3, 720, 1280],
      "data": [0.0, 0.1, 0.2]
    }
  ],
  "parameters": {
    "frame_id": "frame-000001",
    "timestamp": "2026-01-01T12:00:00Z"
  }
}
```

## Response payload

### Required fields

| Field | Type | Description |
|---|---|---|
| `outputs[].name=boxes` | float[][] | Bounding boxes in `[x1, y1, x2, y2]`, per detection. |
| `outputs[].name=classes` | int[] | Predicted class index per detection. |
| `outputs[].name=scores` | float[] | Confidence score per detection. |
| `parameters.frame_id` | string/integer | Echoed input frame ID for correlation. |
| `parameters.timestamp` | string | Echoed input timestamp. |

### HTTP response example

```json
{
  "model_name": "detectron2",
  "outputs": [
    {
      "name": "boxes",
      "datatype": "FP32",
      "shape": [1, 2, 4],
      "data": [12.0, 18.0, 210.0, 280.0, 400.0, 300.0, 620.0, 700.0]
    },
    {
      "name": "classes",
      "datatype": "INT64",
      "shape": [1, 2],
      "data": [1, 3]
    },
    {
      "name": "scores",
      "datatype": "FP32",
      "shape": [1, 2],
      "data": [0.96, 0.81]
    }
  ],
  "parameters": {
    "frame_id": "frame-000001",
    "timestamp": "2026-01-01T12:00:00Z"
  }
}
```
