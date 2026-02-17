# Swin Export and Triton Registration

## 1) Export checkpoint to ONNX (TensorRT-ready)

```bash
python models/swin/export_swin.py \
  --checkpoint /path/to/swin_checkpoint.pth \
  --output /tmp/swin_classifier.onnx \
  --num-classes 1000
```

## 2) Register ONNX model in Triton repository

```bash
python models/swin/register_triton_model.py \
  --onnx /tmp/swin_classifier.onnx \
  --triton-repo infra/k8s/triton-swin/model-repository \
  --model-name swin_classifier \
  --num-classes 1000
```

The registration step creates/updates:

- `infra/k8s/triton-swin/model-repository/swin_classifier/1/model.onnx`
- `infra/k8s/triton-swin/model-repository/swin_classifier/config.pbtxt`
