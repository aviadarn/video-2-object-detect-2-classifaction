#!/usr/bin/env python3
"""Export a Detectron2 model to ONNX for Triton Inference Server.

Usage:
  python scripts/export_detectron2_to_onnx.py \
    --config-file <path/to/config.yaml> \
    --weights <path/to/model_final.pth> \
    --output-dir infra/k8s/triton-detectron2/models/detectron2/1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.export import TracingAdapter
from detectron2.modeling import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, help="Detectron2 config file")
    parser.add_argument("--weights", required=True, help="Model weights file")
    parser.add_argument(
        "--output-dir",
        default="infra/k8s/triton-detectron2/models/detectron2/1",
        help="Target Triton model version directory",
    )
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    image = torch.randn(args.batch_size, 3, args.height, args.width)
    inputs = [{"image": image[i]} for i in range(args.batch_size)]

    tracer = TracingAdapter(model, inputs, allow_non_tensor=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.onnx"

    torch.onnx.export(
        tracer,
        (image,),
        out_path,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["boxes", "classes", "scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "boxes": {0: "batch", 1: "detections"},
            "classes": {0: "batch", 1: "detections"},
            "scores": {0: "batch", 1: "detections"},
        },
    )

    print(f"Exported ONNX model to {out_path}")


if __name__ == "__main__":
    main()
