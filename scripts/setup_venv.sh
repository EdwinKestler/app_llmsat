# app_stremlit/scripts/setup_venv.sh (snippets to edit)



#!/usr/bin/env bash
# app_stremlit/scripts/setup_venv.sh
set -euo pipefail

# --- config ---
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
# Choose your CUDA wheel channel. Common options:
#   cu121  -> https://download.pytorch.org/whl/cu121
#   cu124  -> https://download.pytorch.org/whl/cu124
TORCH_CUDA_CHANNEL="${TORCH_CUDA_CHANNEL:-https://download.pytorch.org/whl/cu124}"

# --- create venv ---
$PYTHON_BIN -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel

echo "[1/3] Installing PyTorch + TorchVision from ${TORCH_CUDA_CHANNEL}"
pip install --index-url "$TORCH_CUDA_CHANNEL" torch torchvision

echo "[2/3] Installing project requirements"
pip install -r requirements.txt

echo "[3/3] Running GPU smoke test"
python tools/gpu_smoke_test.py || {
  echo "⚠️ GPU smoke test failed. Check NVIDIA drivers / CUDA runtime / wheel channel."
  exit 1
}

echo "✅ .venv ready. Activate with: source $VENV_DIR/bin/activate"
