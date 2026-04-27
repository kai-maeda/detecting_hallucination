#!/usr/bin/env bash
# One-shot env setup for a fresh RunPod H100 / A100 pod (PyTorch 2.4 image recommended).
# Skips flash-attn — inference falls back to SDPA, which is plenty fast for POC.
set -euo pipefail

echo "== System info =="
nvidia-smi || true
python3 --version

echo "== Upgrading pip =="
python3 -m pip install --upgrade pip wheel

echo "== Installing torch first =="
python3 -m pip install --no-cache-dir torch==2.4.0

echo "== Installing remaining requirements =="
python3 -m pip install --no-cache-dir -r requirements.txt

echo "== Verifying CUDA =="
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "== Done. Next: copy .env.example to .env and add your OPENAI_API_KEY =="
