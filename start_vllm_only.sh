#!/bin/bash

# Start app in vLLM-only mode (no training, just inference)
export USE_VLLM=true
export USE_MEGATRON=false

# vLLM configuration
export VLLM_MODEL=${VLLM_MODEL:-"HuggingFaceTB/SmolLM2-135M-Instruct"}
export VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-"1"}
export VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-"0.2"}
export VLLM_ENABLE_LORA=${VLLM_ENABLE_LORA:-"true"}
export VLLM_MAX_LORAS=${VLLM_MAX_LORAS:-"4"}
export VLLM_MAX_LORA_RANK=${VLLM_MAX_LORA_RANK:-"64"}

# Flask port
export PORT=${PORT:-"8000"}

echo "Starting Tinker API Server with vLLM-only mode"
echo "  Model: $VLLM_MODEL"
echo "  GPU Memory: $VLLM_GPU_MEMORY_UTIL"
echo "  LoRA: $VLLM_ENABLE_LORA (max $VLLM_MAX_LORAS adapters)"
echo ""

# Use py312 Python (with all dependencies installed)
PYTHON=${PYTHON:-"/home/green/py312/bin/python"}

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Please set PYTHON environment variable or activate py312 conda environment"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# Run without torchrun - just plain python
$PYTHON src/app.py
