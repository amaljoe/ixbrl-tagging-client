#!/usr/bin/env bash
# Serve the prod-unified-xt (phase2) checkpoint on 2xA40.
# Max context covers training's MAX_SEQ_LEN_PAIR (14000) + output headroom.
python3 -m vllm.entrypoints.openai.api_server \
  --model ${1:-models/phase2} \
  --served-model-name ${2:-phase2} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 20000 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
