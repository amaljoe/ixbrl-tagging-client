#!/usr/bin/env bash
# Build a .venv with all packages needed for iXBRL tagging (vLLM inference + Streamlit client).
# Run once per node reboot (or whenever .venv is missing/broken).
# Usage: bash env.sh [--venv-path /path/to/.venv]
#
# Requirements: python3.10, uv, ~/wheels/python310/ wheel cache, ~/wheels/flash_attn*.whl
# Pinned versions: vllm==0.15.0, torch==2.9.1+cu128, torchvision==0.24.1+cu128, flash_attn==2.8.3
# Do NOT upgrade torch to 2.10+ — flash_attn 2.8.3 wheel uses ib ABI, breaks with jb ABI (torch 2.10+).

set -euo pipefail

VENV_PATH="${1:-/dev/shm/.venv}"

# Allow overriding via --venv-path flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    *) shift ;;
  esac
done

WHEELS_DIR="$HOME/wheels/python310"
FLASH_ATTN_WHL="$(ls "$HOME"/wheels/flash_attn-*-cp310-cp310-linux_x86_64.whl 2>/dev/null | head -1)"

echo "==> Creating .venv at $VENV_PATH"
rm -rf "$VENV_PATH"
python3.10 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure uv is available
if ! command -v uv &>/dev/null; then
  echo "==> Installing uv"
  pip install uv -q
fi

echo "==> Installing wheel cache packages (httpx, socksio, PySocks)"
pip install "$WHEELS_DIR"/* --no-index --find-links "$WHEELS_DIR" -q

echo "==> Installing core ML packages via uv"
uv pip install \
  vllm==0.15.0 \
  transformers==4.57.6 \
  accelerate==1.13.0 \
  peft==0.18.1 \
  trl==0.29.0 \
  datasets \
  bitsandbytes \
  einops \
  numpy \
  pandas \
  pillow \
  tqdm \
  requests \
  beautifulsoup4 \
  pyarrow \
  streamlit \
  pydantic-settings \
  python-dotenv \
  tensorboard \
  qwen-vl-utils \
  openai \
  jupyter \
  playwright \
  "setuptools==80"

echo "==> Pinning torch==2.9.1+cu128 (must come after vllm to override any torch 2.10 resolution)"
uv pip install "torch==2.9.1+cu128" --torch-backend=auto -q
uv pip install "torchvision==0.24.1+cu128" --torch-backend=auto -q

echo "==> Installing flash_attn 2.8.3 from prebuilt wheel (ib ABI, requires torch 2.9.1)"
if [[ -n "$FLASH_ATTN_WHL" ]]; then
  pip install "$FLASH_ATTN_WHL" --no-deps
else
  echo "WARNING: flash_attn wheel not found at ~/wheels/flash_attn-*-cp310-cp310-linux_x86_64.whl — skipping"
fi

echo "==> Installing iXBRL client requirements"
pip install -r "$(dirname "$0")/requirements.txt" --no-deps -q

echo ""
echo "==> Verifying installation"
python -c "import torch; print('torch:', torch.__version__, '| cuda devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm: ok')"
python -c "import flash_attn; print('flash_attn: ok')" 2>/dev/null || echo "flash_attn: not installed (non-fatal)"
python -c "import streamlit; print('streamlit: ok')"

echo ""
echo "==> Done! Activate with:"
echo "    source $VENV_PATH/bin/activate"
echo ""
echo "==> Before starting vLLM or training, run:"
echo "    export LD_PRELOAD=$VENV_PATH/lib/libstdc++.so.6  # only if CXXABI errors appear"
echo ""
echo "==> Start vLLM:"
echo "    LD_PRELOAD=$VENV_PATH/lib/libstdc++.so.6 bash scripts/vllm_ft_qwen.sh models/phase1_prod phase1_prod"
