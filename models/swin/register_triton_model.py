#!/usr/bin/env python3
"""Register exported Swin ONNX into a Triton model repository layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TEMPLATE = """name: \"{model_name}\"
backend: \"onnxruntime\"
max_batch_size: {max_batch_size}
input [
  {{
    name: \"input\"
    data_type: TYPE_FP32
    dims: [3, {image_size}, {image_size}]
  }}
]
output [
  {{
    name: \"logits\"
    data_type: TYPE_FP32
    dims: [{num_classes}]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{ name : \"tensorrt\" }} ]
  }}
}}
"""


def register_model(
    onnx_path: Path,
    triton_repo: Path,
    model_name: str,
    image_size: int,
    num_classes: int,
    max_batch_size: int,
    model_version: int = 1,
) -> Path:
    model_dir = triton_repo / model_name
    version_dir = model_dir / str(model_version)
    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, version_dir / "model.onnx")

    config = TEMPLATE.format(
        model_name=model_name,
        max_batch_size=max_batch_size,
        image_size=image_size,
        num_classes=num_classes,
    )
    (model_dir / "config.pbtxt").write_text(config, encoding="utf-8")
    return model_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--triton-repo", type=Path, required=True)
    parser.add_argument("--model-name", default="swin_classifier")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--model-version", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    register_model(
        onnx_path=args.onnx,
        triton_repo=args.triton_repo,
        model_name=args.model_name,
        image_size=args.image_size,
        num_classes=args.num_classes,
        max_batch_size=args.max_batch_size,
        model_version=args.model_version,
    )


if __name__ == "__main__":
    main()
