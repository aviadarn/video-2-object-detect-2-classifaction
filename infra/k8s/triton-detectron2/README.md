# Triton + Detectron2 deployment assets

## Contents

- `scripts/export_detectron2_to_onnx.py`: exports a Detectron2 checkpoint to ONNX.
- `models/detectron2/`: Triton model repository layout.
- `manifests/`: Kubernetes manifests (`Deployment`, `Service`, `HPA`, `PodDisruptionBudget`).
- `docs/integration-contract.md`: request/response integration contract.

## Export model

```bash
python infra/k8s/triton-detectron2/scripts/export_detectron2_to_onnx.py \
  --config-file /path/to/config.yaml \
  --weights /path/to/model_final.pth \
  --output-dir infra/k8s/triton-detectron2/models/detectron2/1
```

## Deploy

1. Provision a PVC named `triton-model-repo-pvc` and populate it with `models/detectron2`.
2. Apply manifests:

```bash
kubectl apply -k infra/k8s/triton-detectron2/manifests
```
