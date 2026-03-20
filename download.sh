#!/usr/bin/env bash
# Download phase1_prod model weights from HuggingFace.
# The token is used in-memory only — never written to disk.
set -euo pipefail

REPO_ID="amaljoe88/phase1_prod"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/phase1_prod"

echo "iXBRL Tagger — model weight downloader"
echo "Repo: ${REPO_ID}"
echo "Destination: ${LOCAL_DIR}"
echo ""

# Read token without echoing it
read -rsp "Enter HuggingFace token (input hidden): " HF_TOKEN_INPUT
echo ""

if [[ -z "${HF_TOKEN_INPUT}" ]]; then
  echo "Error: no token provided." >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo "Downloading model weights..."

# HF_HUB_DISABLE_IMPLICIT_TOKEN prevents huggingface_hub from saving
# the token to ~/.cache/huggingface/token
HF_TOKEN="${HF_TOKEN_INPUT}" \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  python3 - <<PYEOF
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not found. Run: pip install huggingface_hub")
    sys.exit(1)

snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="model",
    local_dir="${LOCAL_DIR}",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["checkpoint-*", "logs/"],
)
print("Download complete: ${LOCAL_DIR}")
PYEOF

# Clear the variable from the environment immediately
unset HF_TOKEN_INPUT

echo ""
echo "Done. Weights saved to: ${LOCAL_DIR}"
echo "Start the vLLM server with:"
echo "  bash scripts/vllm_ft_qwen.sh models/phase1_prod phase1_prod"
