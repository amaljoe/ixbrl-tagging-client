python3 -m vllm.entrypoints.openai.api_server \
  --model ${1:-models/phase1_prod} \
  --served-model-name ${2:-phase1_prod} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 12000 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
