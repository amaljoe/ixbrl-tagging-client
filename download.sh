#!/usr/bin/env bash
# Download phase2 (prod-unified-xt) weights from HuggingFace.
set -euo pipefail

REPO_ID="amaljoe88/xbrl-model"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/phase2"

echo "iXBRL Tagger — phase2 weight downloader"
echo "Repo: ${REPO_ID}"
echo "Destination: ${LOCAL_DIR}"

read -rsp "Enter HuggingFace token (input hidden): " HF_TOKEN_INPUT
echo ""
[[ -z "${HF_TOKEN_INPUT}" ]] && { echo "Error: no token provided." >&2; exit 1; }

mkdir -p "${LOCAL_DIR}"

HF_TOKEN="${HF_TOKEN_INPUT}" \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="model",
    local_dir="${LOCAL_DIR}",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["checkpoint-*", "logs/"],
)
print("Download complete: ${LOCAL_DIR}")
PYEOF

unset HF_TOKEN_INPUT
echo ""
echo "Done. Weights saved to: ${LOCAL_DIR}"
echo "Start vLLM with:"
echo "  bash scripts/vllm_ft_qwen.sh models/phase2 phase2"
