#!/usr/bin/env bash
# Start the iXBRL tagger model server (OpenAI-compatible vLLM).
python3 -m vllm.entrypoints.openai.api_server \
  --model models/phase2 \
  --served-model-name phase2 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 20000 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
