#!/bin/bash
#
# Enable Megatron-Bridge Backend
#
# This script sets up environment variables to use Megatron-Bridge
# instead of the default HuggingFace backend
#

echo "Setting up Megatron-Bridge backend..."

# Enable Megatron backend
export USE_MEGATRON=true

# Optional: Configure Megatron-Bridge model
export MEGATRON_MODEL="${MEGATRON_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
export MEGATRON_TP_SIZE="${MEGATRON_TP_SIZE:-1}"
export MEGATRON_PP_SIZE="${MEGATRON_PP_SIZE:-1}"

# Also configure vLLM if desired
export USE_VLLM="${USE_VLLM:-true}"
export VLLM_MODEL="${VLLM_MODEL:-$MEGATRON_MODEL}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.4}"
export VLLM_ENABLE_LORA=true

echo "✓ Megatron-Bridge backend enabled"
echo "  USE_MEGATRON=$USE_MEGATRON"
echo "  MEGATRON_MODEL=$MEGATRON_MODEL"
echo "  MEGATRON_TP_SIZE=$MEGATRON_TP_SIZE"
echo ""
echo "✓ vLLM configured"
echo "  USE_VLLM=$USE_VLLM"
echo "  VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE"
echo ""
echo "Now start the server:"
echo "  python3.12 src/app.py"
echo ""
echo "Or use the multi-GPU script:"
echo "  ./scripts/run_server_4gpu.sh"
