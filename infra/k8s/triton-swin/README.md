# Triton Swin Deployment Bundle

This directory contains Kubernetes deployment assets and model repository skeleton
for serving the Swin object classifier through Triton.

## Files

- `triton-swin-deployment.yaml`: Deployment + Service.
- `preprocessing-contract.yaml`: Explicit contract that Swin ingests Detectron2 crops.
- `model-repository/swin_classifier/config.pbtxt`: Triton model config.

## Notes

- Place exported `model.onnx` under `model-repository/swin_classifier/1/`.
- `services/pipeline/worker.py` enforces the same preprocessing behavior used by this contract.
