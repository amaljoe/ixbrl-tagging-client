#!/usr/bin/env bash
# Upload prod-unified-xt checkpoint to amaljoe88/phase2 on HuggingFace Hub.
set -euo pipefail

REPO_ID="amaljoe88/xbrl-model"
SRC_DIR="${1:-/home/compiling-ganesh/24m0797/workspace/ixbrl-tagging/models/prod-unified-xt/final}"

echo "iXBRL Tagger — xbrl-model weight uploader"
echo "Source: ${SRC_DIR}"
echo "Repo:   ${REPO_ID}"

read -rsp "Enter HuggingFace token (input hidden): " HF_TOKEN_INPUT
echo ""
[[ -z "${HF_TOKEN_INPUT}" ]] && { echo "Error: no token provided." >&2; exit 1; }

HF_TOKEN="${HF_TOKEN_INPUT}" \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  python3 - <<PYEOF
import os
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
api.upload_folder(
    folder_path="${SRC_DIR}",
    repo_id="${REPO_ID}",
    repo_type="model",
    commit_message="prod-unified-xt: unified numeric + text iXBRL tagger (Qwen3-VL-2B SFT)",
)
print("Uploaded ${SRC_DIR} -> ${REPO_ID}")
PYEOF

unset HF_TOKEN_INPUT
echo "Done. Run: bash download.sh  (on the target machine)"
