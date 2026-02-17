#!/usr/bin/env python3
"""Pipeline worker that orchestrates Detectron2 + Swin Triton inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("pipeline.worker")


@dataclass(frozen=True)
class WorkerConfig:
    detectron2_url: str
    detectron2_model_name: str
    swin_url: str
    swin_model_name: str
    max_fail_ratio: float = 0.2
    swin_image_size: int = 224
    detectron_score_threshold: float = 0.2


class JobAbortedError(RuntimeError):
    """Raised when failures exceed configured ratio threshold."""


class TritonHttpClient:
    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def infer(self, model_name: str, inputs: list[dict[str, Any]], outputs: list[dict[str, str]]) -> dict[str, Any]:
        import requests

        payload = {"inputs": inputs, "outputs": outputs}
        url = f"{self.base_url}/v2/models/{model_name}/infer"
        response = requests.post(url, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()


def deterministic_object_id(job_id: str, frame_id: str, box_xyxy: list[float], det_label: str) -> str:
    rounded = ",".join(f"{value:.3f}" for value in box_xyxy)
    digest = hashlib.sha1(f"{job_id}:{frame_id}:{rounded}:{det_label}".encode("utf-8")).hexdigest()
    return digest[:24]


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pil_rgb(image_path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def _crop_and_preprocess(frame_rgb, box_xyxy: list[float], out_size: int):
    import numpy as np
    from PIL import Image

    x1, y1, x2, y2 = box_xyxy
    h, w = frame_rgb.shape[:2]
    x1_i, x2_i = int(max(0, min(w, x1))), int(max(0, min(w, x2)))
    y1_i, y2_i = int(max(0, min(h, y1))), int(max(0, min(h, y2)))
    if x2_i <= x1_i:
        x2_i = min(w, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(h, y1_i + 1)

    crop = frame_rgb[y1_i:y2_i, x1_i:x2_i]
    resized = np.array(Image.fromarray(crop).resize((out_size, out_size), Image.BILINEAR), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def _as_triton_tensor(name: str, np_array) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(np_array.shape),
        "datatype": "FP32",
        "data": np_array.reshape(-1).tolist(),
    }


def _infer_detectron(client: TritonHttpClient, cfg: WorkerConfig, frame_rgb) -> list[dict[str, Any]]:
    import numpy as np

    nchw = np.transpose(frame_rgb, (2, 0, 1)).astype(np.float32)[None, ...]
    response = client.infer(
        model_name=cfg.detectron2_model_name,
        inputs=[_as_triton_tensor("input", nchw)],
        outputs=[{"name": "boxes"}, {"name": "scores"}, {"name": "labels"}],
    )
    output_map = {item["name"]: item for item in response["outputs"]}

    boxes = np.array(output_map["boxes"]["data"], dtype=np.float32).reshape(output_map["boxes"]["shape"])
    scores = np.array(output_map["scores"]["data"], dtype=np.float32).reshape(output_map["scores"]["shape"])
    labels = np.array(output_map["labels"]["data"]).reshape(output_map["labels"]["shape"])

    objects: list[dict[str, Any]] = []
    for idx in range(len(scores)):
        if float(scores[idx]) < cfg.detectron_score_threshold:
            continue
        objects.append(
            {
                "box_xyxy": [float(v) for v in boxes[idx].tolist()],
                "det_score": float(scores[idx]),
                "det_label": str(labels[idx]),
            }
        )
    return objects


def _infer_swin(client: TritonHttpClient, cfg: WorkerConfig, frame_rgb, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    if not detections:
        return []

    batch = np.stack(
        [_crop_and_preprocess(frame_rgb=frame_rgb, box_xyxy=d["box_xyxy"], out_size=cfg.swin_image_size) for d in detections],
        axis=0,
    )
    response = client.infer(
        model_name=cfg.swin_model_name,
        inputs=[_as_triton_tensor("input", batch)],
        outputs=[{"name": "logits"}],
    )
    logits_meta = next(item for item in response["outputs"] if item["name"] == "logits")
    logits = np.array(logits_meta["data"], dtype=np.float32).reshape(logits_meta["shape"])
    probs = _softmax(logits)

    results: list[dict[str, Any]] = []
    for p in probs:
        cls = int(np.argmax(p))
        results.append({"swin_class_id": cls, "swin_confidence": float(p[cls])})
    return results


def _softmax(logits):
    import numpy as np

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def merge_outputs(job_id: str, frame_id: str, detections: list[dict[str, Any]], classifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for det, cls in zip(detections, classifications):
        object_id = deterministic_object_id(job_id, frame_id, det["box_xyxy"], det["det_label"])
        merged.append({"job_id": job_id, "frame_id": frame_id, "object_id": object_id, **det, **cls})
    return merged


def process_job(manifest: dict[str, Any], config: WorkerConfig) -> dict[str, Any]:
    detectron_client = TritonHttpClient(config.detectron2_url)
    swin_client = TritonHttpClient(config.swin_url)

    job_id = manifest["job_id"]
    frames = manifest["frames"]
    allowed_failures = max(1, int(len(frames) * config.max_fail_ratio))

    merged_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for frame in frames:
        frame_id = str(frame["frame_id"])
        image_path = Path(frame["image_path"])
        try:
            frame_rgb = _pil_rgb(image_path)
            detections = _infer_detectron(detectron_client, config, frame_rgb)
            classifications = _infer_swin(swin_client, config, frame_rgb, detections)
            merged_records.extend(merge_outputs(job_id, frame_id, detections, classifications))
        except Exception as exc:  # pragma: no cover - external services
            LOGGER.exception("Frame failure for %s/%s: %s", job_id, frame_id, exc)
            failures.append({"frame_id": frame_id, "error": str(exc)})
            if len(failures) > allowed_failures:
                raise JobAbortedError(
                    f"Abort job {job_id}: {len(failures)} frame failures exceeded threshold {allowed_failures}."
                ) from exc

    return {
        "job_id": job_id,
        "records": merged_records,
        "failed_frames": failures,
        "error_policy": {
            "max_fail_ratio": config.max_fail_ratio,
            "max_fail_count": allowed_failures,
            "status": "partial_success" if failures else "success",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detectron2-url", default="http://triton-detectron2:8000")
    parser.add_argument("--detectron2-model", default="detectron2_detector")
    parser.add_argument("--swin-url", default="http://triton-swin:8000")
    parser.add_argument("--swin-model", default="swin_classifier")
    parser.add_argument("--max-fail-ratio", type=float, default=0.2)
    parser.add_argument("--swin-image-size", type=int, default=224)
    parser.add_argument("--detectron-score-threshold", type=float, default=0.2)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    result = process_job(
        manifest=load_manifest(args.manifest),
        config=WorkerConfig(
            detectron2_url=args.detectron2_url,
            detectron2_model_name=args.detectron2_model,
            swin_url=args.swin_url,
            swin_model_name=args.swin_model,
            max_fail_ratio=args.max_fail_ratio,
            swin_image_size=args.swin_image_size,
            detectron_score_threshold=args.detectron_score_threshold,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
