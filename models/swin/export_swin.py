#!/usr/bin/env python3
"""Export a Swin Transformer checkpoint to ONNX for Triton/TensorRT deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """Load a timm Swin model and replace classification head size."""
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise RuntimeError(
            "timm is required to export Swin models. Install with `pip install timm`."
        ) from exc

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.eval()
    return model


def _dynamic_axes(enabled: bool) -> dict[str, dict[int, str]] | None:
    if not enabled:
        return None
    return {
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }


def export_onnx(
    checkpoint_path: Path,
    output_onnx_path: Path,
    model_name: str,
    num_classes: int,
    image_size: int,
    opset: int,
    dynamic_batch: bool,
) -> None:
    model = _load_model(model_name=model_name, num_classes=num_classes)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected}")

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_onnx_path.as_posix(),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=_dynamic_axes(dynamic_batch),
            do_constant_folding=True,
            opset_version=opset,
        )

    metadata = {
        "model_name": model_name,
        "num_classes": num_classes,
        "image_size": image_size,
        "opset": opset,
        "dynamic_batch": dynamic_batch,
        "checkpoint": checkpoint_path.as_posix(),
        "onnx": output_onnx_path.as_posix(),
        "tensorrt_ready": True,
    }
    metadata_path = output_onnx_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--disable-dynamic-batch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        opset=args.opset,
        dynamic_batch=not args.disable_dynamic_batch,
    )


if __name__ == "__main__":
    main()
