#!/usr/bin/env python3
"""Scene-based video pipeline: split -> detect -> classify -> track -> cluster -> report."""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger("pipeline.worker")
PIPELINE_STAGES = ["split", "object_detection", "classification", "object_tracking", "clustering", "report"]


@dataclass(frozen=True)
class PipelineConfig:
    job_id: str
    video_path: Path
    detectron2_score_threshold: float = 0.3
    yolo_score_threshold: float = 0.25


@dataclass(frozen=True)
class SampledFrame:
    scene_id: str
    frame_role: str
    frame_index: int
    timestamp_s: float
    image_path: Path


class SceneSplitter(Protocol):
    def split(self, video_path: Path) -> list[SampledFrame]: ...


class ObjectDetector(Protocol):
    name: str

    def detect(self, frame: SampledFrame) -> list[dict[str, Any]]: ...


class ObjectClassifier(Protocol):
    def classify(self, frame: SampledFrame, detection: dict[str, Any]) -> dict[str, Any]: ...


class PySceneDetectSplitter:
    """Uses PySceneDetect to detect scenes and sample first/middle/last frames."""

    def split(self, video_path: Path) -> list[SampledFrame]:
        scenedetect = __import__("scenedetect")
        from PIL import Image

        video = scenedetect.open_video(str(video_path))
        manager = scenedetect.SceneManager()
        manager.add_detector(scenedetect.ContentDetector())
        manager.detect_scenes(video, show_progress=False)
        scenes = manager.get_scene_list()

        if not scenes:
            raise RuntimeError("No scenes detected; cannot continue")

        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        sampled: list[SampledFrame] = []
        output_dir = Path(tempfile.mkdtemp(prefix="scene-samples-"))

        for idx, (start, end) in enumerate(scenes):
            start_idx = start.get_frames()
            end_idx = max(start_idx + 1, end.get_frames())
            mid_idx = start_idx + ((end_idx - start_idx) // 2)
            for role, frame_index in (("first", start_idx), ("middle", mid_idx), ("last", end_idx - 1)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = capture.read()
                if not ok:
                    continue
                image_path = output_dir / f"scene-{idx}-{role}.jpg"
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(image_path)
                sampled.append(
                    SampledFrame(
                        scene_id=f"scene-{idx}",
                        frame_role=role,
                        frame_index=frame_index,
                        timestamp_s=frame_index / fps,
                        image_path=image_path,
                    )
                )

        capture.release()
        if not sampled:
            raise RuntimeError("Scene split produced no sampled frames")
        return sampled


class Detectron2Detector:
    name = "detectron2_faster_rcnn"

    def __init__(self, score_threshold: float = 0.3) -> None:
        self.score_threshold = score_threshold

    def detect(self, frame: SampledFrame) -> list[dict[str, Any]]:
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            import cv2
        except Exception as exc:
            LOGGER.warning("Detectron2 unavailable: %s", exc)
            return []

        if not hasattr(self, "_predictor"):
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            self._predictor = DefaultPredictor(cfg)

        image = cv2.imread(str(frame.image_path))
        outputs = self._predictor(image)["instances"].to("cpu")
        boxes = outputs.pred_boxes.tensor.numpy().tolist()
        scores = outputs.scores.numpy().tolist()
        classes = outputs.pred_classes.numpy().tolist()
        return [
            {"box_xyxy": [float(v) for v in box], "score": float(score), "label": str(label)}
            for box, score, label in zip(boxes, scores, classes)
        ]


class YOLO12Detector:
    name = "yolo12_ultralytics"

    def __init__(self, score_threshold: float = 0.25) -> None:
        self.score_threshold = score_threshold

    def detect(self, frame: SampledFrame) -> list[dict[str, Any]]:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            LOGGER.warning("Ultralytics unavailable: %s", exc)
            return []

        if not hasattr(self, "_model"):
            self._model = YOLO("yolo12n.pt")

        predictions = self._model.predict(str(frame.image_path), conf=self.score_threshold, verbose=False)
        detections: list[dict[str, Any]] = []
        for pred in predictions:
            for box in pred.boxes:
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    {
                        "box_xyxy": [float(v) for v in xyxy],
                        "score": float(box.conf.item()),
                        "label": str(int(box.cls.item())),
                    }
                )
        return detections


class OpenClipClassifier:
    def classify(self, frame: SampledFrame, detection: dict[str, Any]) -> dict[str, Any]:
        try:
            import open_clip
            import torch
            from PIL import Image
        except Exception as exc:
            LOGGER.warning("OpenCLIP unavailable: %s", exc)
            return {"label": "unknown", "confidence": 0.0}

        if not hasattr(self, "_runtime"):
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            labels = ["person", "vehicle", "animal", "sports", "food", "nature", "indoor object"]
            text_tokens = tokenizer(labels)
            self._runtime = (model.eval(), preprocess, labels, text_tokens)

        model, preprocess, labels, text_tokens = self._runtime
        x1, y1, x2, y2 = [int(v) for v in detection["box_xyxy"]]
        image = Image.open(frame.image_path).convert("RGB").crop((x1, y1, x2, y2))

        with torch.no_grad():
            image_tensor = preprocess(image).unsqueeze(0)
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            idx = int(logits.argmax().item())
            return {"label": labels[idx], "confidence": float(logits[idx].item())}


class SimpleObjectTracker:
    def __init__(self, max_distance: float = 80.0) -> None:
        self.max_distance = max_distance
        self._tracks: dict[str, tuple[float, float]] = {}
        self._counter = 0

    def assign(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for det in detections:
            center = _center(det["box_xyxy"])
            matched = self._best_track(center)
            if matched is None:
                self._counter += 1
                matched = f"track-{self._counter}"
            self._tracks[matched] = center
            result.append({**det, "track_id": matched})
        return result

    def _best_track(self, center: tuple[float, float]) -> str | None:
        best_id = None
        best_distance = math.inf
        for track_id, existing in self._tracks.items():
            dist = math.dist(center, existing)
            if dist < best_distance and dist <= self.max_distance:
                best_distance = dist
                best_id = track_id
        return best_id


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        scene_splitter: SceneSplitter | None = None,
        detectors: list[ObjectDetector] | None = None,
        classifier: ObjectClassifier | None = None,
        tracker: SimpleObjectTracker | None = None,
    ) -> None:
        self.config = config
        self.scene_splitter = scene_splitter or PySceneDetectSplitter()
        self.detectors = detectors or [
            Detectron2Detector(score_threshold=config.detectron2_score_threshold),
            YOLO12Detector(score_threshold=config.yolo_score_threshold),
        ]
        self.classifier = classifier or OpenClipClassifier()
        self.tracker = tracker or SimpleObjectTracker()

    def run(self) -> dict[str, Any]:
        sampled_frames = self.scene_splitter.split(self.config.video_path)
        frame_records: list[dict[str, Any]] = []
        all_detections: list[dict[str, Any]] = []

        for frame in sampled_frames:
            detections = []
            for detector in self.detectors:
                for raw in detector.detect(frame):
                    detections.append({**raw, "detector": detector.name, "frame_id": _frame_id(frame)})

            classified = [
                {**det, "classification": self.classifier.classify(frame, det)}
                for det in detections
            ]
            tracked = self.tracker.assign(classified)
            frame_records.append(
                {
                    "frame_id": _frame_id(frame),
                    "scene_id": frame.scene_id,
                    "frame_role": frame.frame_role,
                    "timestamp_s": frame.timestamp_s,
                    "detections": tracked,
                }
            )
            all_detections.extend(tracked)

        return build_report(job_id=self.config.job_id, frame_records=frame_records, detections=all_detections)


def build_report(job_id: str, frame_records: list[dict[str, Any]], detections: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, set[str]] = defaultdict(set)
    for det in detections:
        key = det.get("classification", {}).get("label", "unknown")
        clusters[key].add(det["track_id"])

    cluster_items = [
        {"cluster_key": key, "track_ids": sorted(track_ids), "count": len(track_ids)}
        for key, track_ids in sorted(clusters.items())
    ]

    return {
        "job_id": job_id,
        "pipeline": PIPELINE_STAGES,
        "frames": frame_records,
        "clusters": cluster_items,
        "summary": {
            "total_frames": len(frame_records),
            "total_detections": len(detections),
            "total_tracks": len({d["track_id"] for d in detections}),
            "total_clusters": len(cluster_items),
        },
    }


def _center(box_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _frame_id(frame: SampledFrame) -> str:
    return f"{frame.scene_id}:{frame.frame_role}:{frame.frame_index}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    runner = PipelineRunner(config=PipelineConfig(job_id=args.job_id, video_path=args.video))
    report = runner.run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
