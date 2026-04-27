#!/usr/bin/env bash
# One-shot env setup for a fresh RunPod H100 / A100 pod (PyTorch 2.4 image recommended).
set -euo pipefail

echo "== System info =="
nvidia-smi || true
python3 --version

echo "== Upgrading pip =="
python3 -m pip install --upgrade pip wheel

echo "== Installing requirements =="
# Install torch first (matches CUDA 12.1 in the standard RunPod PyTorch image).
python3 -m pip install --no-cache-dir torch==2.4.0

# flash-attn needs torch present at install time.
python3 -m pip install --no-cache-dir -r requirements.txt

echo "== Verifying CUDA + flash-attn =="
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception as e:
    print("flash_attn import failed:", e, "(SDPA fallback will be used)")
PY

echo "== Done. Next: copy .env.example to .env and add your OPENAI_API_KEY =="
