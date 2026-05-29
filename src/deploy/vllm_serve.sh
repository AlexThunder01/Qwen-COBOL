#!/bin/bash
# Start vLLM inference server (AWQ or FP8) on Lightning L40S or local Docker.
# Edit MODEL_ID and QUANTIZATION before running.
#
# Usage:
#   bash src/deploy/vllm_serve.sh awq   # AWQ 4-bit (default, ~15GB)
#   bash src/deploy/vllm_serve.sh fp8   # FP8 (~27GB, L40S / H100 only)

set -euo pipefail

MODE="${1:-awq}"
HF_USERNAME="YOUR_HF_USERNAME"

if [ "$MODE" = "fp8" ]; then
    MODEL_ID="${HF_USERNAME}/qwen-cobol-27b-fp8"
    QUANTIZATION="fp8"
    DTYPE="float8_e4m3fn"
    GPU_MEM=0.92
else
    MODEL_ID="${HF_USERNAME}/qwen-cobol-27b-awq"
    QUANTIZATION="awq_marlin"
    DTYPE="auto"
    GPU_MEM=0.90
fi

echo "Starting vLLM server: model=${MODEL_ID} quant=${QUANTIZATION}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --dtype "${DTYPE}" \
    --quantization "${QUANTIZATION}" \
    --max-model-len 16384 \
    --gpu-memory-utilization "${GPU_MEM}" \
    --tensor-parallel-size 1 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-prefix-caching \
    --trust-remote-code
